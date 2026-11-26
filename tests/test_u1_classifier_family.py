"""Real tiny CPU models validate routing, not CUDA compatibility."""
from copy import deepcopy
from pathlib import Path
import tempfile
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from u1_benchmark.core import digest
from u1_benchmark.family_preparation import ClassificationPreparer, validate_classifier
from u1_benchmark.family_adapter import FamilyAdapter
from u1_benchmark.classifier_family import ClassificationAdapter, classify
from u1_benchmark.precision import PrecisionError, PrecisionTracker, validate_precision
from u1_benchmark.v4_preparation import validate_settings


def spec():
    source = {'repo': 'fixture/classifier', 'revision': 'a' * 40}
    return validate_classifier({'adapter': 'hf_sequence_classification', 'weights': dict(source),
        'tokenizer': dict(source), 'template': None, 'preparation': {'limit': 512},
        'scoring': {'labels': ['SAFE', 'INJECTION']}, 'execution': {}})


def preparer(settings=None):
    from u1_fixtures import tiny_tokenizer
    p = ClassificationPreparer.__new__(ClassificationPreparer)
    p.spec = settings or spec()
    p.name, p.settings = p.spec['adapter'], p.spec['preparation']
    p.limit, p.tokenizer = p.settings['limit'], tiny_tokenizer()
    p.input_changed, p.metadata = False, {'adapter': p.name}
    return p


def adapter(directory, family='bert'):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, DebertaV2Config, DebertaV2ForSequenceClassification
    p = preparer()
    cls, config = (BertForSequenceClassification, BertConfig) if family == 'bert' else (
        DebertaV2ForSequenceClassification, DebertaV2Config)
    model = cls(config(vocab_size=len(p.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=512,
        id2label={0: 'SAFE', 1: 'INJECTION'}, attention_probs_dropout_prob=0,
        hidden_dropout_prob=0, relative_attention=family != 'bert')).eval()
    a = ClassificationAdapter.__new__(ClassificationAdapter)
    FamilyAdapter.__init__(a, p.spec, directory, p, model, device='cpu')
    a.labels = ['SAFE', 'INJECTION']
    a._native_forward = model.forward
    a._ort_shapes = set()
    a.metadata['labels'] = a.labels
    a.check_device()
    return a


class Configuration(unittest.TestCase):
    def test_settings_apply_and_reject_ignored(self):
        base = spec()
        self.assertEqual(validate_settings(base), base)
        for repo in ('different/model', 'another/classifier'):
            changed = deepcopy(base)
            changed['weights']['repo'] = repo
            self.assertEqual(validate_settings(changed)['weights']['repo'], repo)
        for section, key, value in [('preparation', 'truncation', True), ('execution', 'linear_attention', 'fla'),
                                    ('execution', 'device', 'cpu'), ('scoring', 'threshold', .4)]:
            changed = deepcopy(base)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate_settings(changed)
        changed = deepcopy(base)
        changed['template'] = {'text': 'ignored'}
        with self.assertRaises(ValueError):
            validate_settings(changed)

    def test_labels_threshold_and_finiteness(self):
        import torch
        for count in (2, 3, 12):
            labels = [str(i) for i in range(count)]
            result = classify(torch.zeros(2, count), labels, {'method': 'softmax_argmax'})
            self.assertEqual(result[0]['label'], '0')
            self.assertEqual(len(result[0]['scores']), count)
        result = classify(torch.zeros(1, 2), ['SAFE', 'INJECTION'],
            {'method': 'softmax_threshold', 'positive_label': 'INJECTION', 'threshold': .5})
        self.assertEqual(result[0]['label'], 'INJECTION')
        for v in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                classify(torch.tensor([[v, 0.]]), ['a', 'b'], {'method': 'softmax_argmax'})

    def test_precision_v2_roles_and_worker_consistency(self):
        p = {'policy': 'native_sdk', 'parameter_dtypes': ['bfloat16'], 'autocast_enabled': False,
             'autocast_dtype': None, 'version': 2,
             'tensor_roles': {'readout': {'dtypes': ['float32'], 'device_type': 'cuda'},
                              'backbone': {'dtypes': ['bfloat16'], 'device_type': 'cuda'}}}
        self.assertEqual(validate_precision(p), p)
        tracker = PrecisionTracker('engineering')
        tracker.add('baseline', {'precision_policy_version': 2, 'metadata': {'precision': p}})
        other = deepcopy(p)
        other['tensor_roles']['readout']['dtypes'] = ['bfloat16']
        with self.assertRaises(PrecisionError):
            tracker.add('candidate', {'precision_policy_version': 2, 'metadata': {'precision': other}})
        with self.assertRaises(PrecisionError):
            tracker.add('baseline', {'precision_policy_version': 1, 'metadata': {'precision': p}})

    def test_capabilities_do_not_claim_runtime_proof(self):
        from u1_benchmark.capabilities import describe
        value = describe(spec())
        self.assertFalse(value['gpu_verified'])
        self.assertEqual(value['status'], 'runtime_check_required')
        self.assertIn('ort', value['runtime_paths'])


class NativeCPU(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_preflight_boundaries_follow_configured_role(self):
        from u1_benchmark.v4_worker import preflight
        for selection in ('user', 'assistant', 'assistant_else_user'):
            with self.subTest(selection=selection):
                a = adapter(self.tmp.name)
                self.addCleanup(a.close)
                a.preparer.settings['text_selection'] = selection
                role = 'assistant' if selection == 'assistant' else 'user'
                rows = [{'id': str(i), 'messages': [{'role': role, 'content': text}]}
                        for i, text in enumerate(('library.', 'The public library.', 'library public'))]
                for row in rows:
                    row['index'] = a.index(row['messages'])
                original = deepcopy(rows)
                edge, above = a.boundary(512)
                self.assertEqual(a.index(edge)['total_tokens'], 512)
                self.assertEqual(a.index(above)['total_tokens'], 513)
                self.assertEqual(a.index(edge)['target_role'], role)
                self.assertFalse(a.index(above)['supported'])
                result = {}
                guard = SimpleNamespace(call=lambda function, *args: function(*args))
                with patch.object(a, 'prove_batch_backend', return_value={'scope': 'CPU fixture'}):
                    preflight(a, rows, {'batch_size': 2, 'seed': 17, 'bucket_max': 512}, guard, result)
                self.assertTrue(result['over_limit_rejected'])
                self.assertTrue(result['aba_passed'])
                self.assertEqual(rows, original)

    def test_boundary_does_not_hide_preparation_failure(self):
        from u1_benchmark.messages import UnsupportedInput
        a = adapter(self.tmp.name)
        self.addCleanup(a.close)
        with patch.object(a, 'index', side_effect=UnsupportedInput('unsupported mode')) as index:
            with self.assertRaisesRegex(UnsupportedInput, 'No supported role format'):
                a.boundary(512)
            self.assertEqual(index.call_count, 3)
        with patch.object(a, 'index', side_effect=ValueError('broken tokenizer')) as index:
            with self.assertRaisesRegex(ValueError, 'broken tokenizer'):
                a.boundary(512)
            index.assert_called_once()

    def test_roles_special_tokens_and_ignored_context(self):
        p = preparer()
        pair = [{'role': 'user', 'content': 'ignored'}, {'role': 'assistant', 'content': 'library'}]
        a, b = p.index(pair), p.index([pair[1]])
        self.assertEqual(a['input_sha256'], b['input_sha256'])
        self.assertEqual(a['target_index'], 1)
        self.assertEqual(a['target_role'], 'assistant')
        self.assertEqual(a['context_indices'], [])
        for content in ('', 'Привет\n世界', '[SEP] library'):
            data = p.prepare(content)
            self.assertEqual(p.index(content)['input_sha256'], digest(data))
            self.assertTrue(all(v == 1 for v in data['attention_mask']))
        p.settings['text_selection'] = 'assistant'
        self.assertFalse(p.index('user only')['supported'])

    def test_two_architectures_real_batch_scores_and_boundaries(self):
        import torch
        for family in ('bert', 'deberta'):
            a = adapter(self.tmp.name, family)
            self.addCleanup(a.close)
            payloads = ['library.', 'The public library.']
            with patch.object(a.preparer, 'prepare', wraps=a.preparer.prepare) as prepare:
                batch = a.request_batch(payloads)
                self.assertEqual(prepare.call_count, 2)
            expected = [a.index(v) for v in payloads]
            self.assertEqual(a.verify_batch(payloads, expected), batch)
            for v, result in zip(payloads, batch):
                single = a.request(v)
                torch.testing.assert_close(torch.tensor(list(single['scores'].values())),
                                           torch.tensor(list(result['scores'].values())), rtol=1e-5, atol=1e-6)
            self.assertEqual(a.metadata['batch_verification']['shape'][0], 2)
            fits = 'library ' * 512
            over = 'library ' * 513
            # The fixture tokenizer adds no post-processor special tokens.
            self.assertEqual(a.index(fits)['total_tokens'], 512)
            self.assertTrue(a.index(fits)['supported'])
            self.assertFalse(a.index(over)['supported'])
            a.reject_batch_over_limit([payloads[0], over])
            wrong = deepcopy(expected)
            wrong[0]['ids_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                a.verify_batch(payloads, wrong)

    def test_compile_routes_scores_and_restores(self):
        a = adapter(self.tmp.name, 'deberta')
        self.addCleanup(a.close)
        texts = ['library.', 'The public library.']
        expected = a.request_batch(texts)
        original = a.model.forward
        a.backend = a.spec['execution']['runtime'] = 'compile'
        a.spec['execution']['compile_backend'] = 'inductor'
        a._backend_installed = False
        with patch('torch._dynamo.backends.registry.lookup_backend',
                   side_effect=lambda name: name if callable(name) else lambda gm, args: gm.forward):
            a.prepare_batch_backend([texts, [texts[0]]])
        self.assertGreater(a.compile_count, 0)
        self.assertEqual(a.request_batch(texts), expected)
        self.assertEqual(a.verify_batch(texts, [a.index(v) for v in texts]), expected)
        model = a.model
        a.close()
        self.assertEqual(model.forward, original)

    def test_parameter_precision_and_batch_policy_changes_fail(self):
        import torch
        a = adapter(self.tmp.name)
        self.addCleanup(a.close)
        a.model.to(dtype=torch.float64)
        with self.assertRaises(ValueError):
            a.check_device()
        a.tokenizer.padding_side = 'left'
        with self.assertRaises(ValueError):
            a.request_batch(['library'])

    def test_ort_scores_route_shapes_profile_and_failure_restore(self):
        from followup import onnx_service
        for fail in (False, True):
            a = adapter(self.tmp.name, 'deberta')
            self.addCleanup(a.close)
            texts = ['library.', 'The public library.']
            expected = a.request_batch(texts)
            original = a.model.forward
            a.backend = a.spec['execution']['runtime'] = 'ort'
            a._backend_installed = False
            sessions, visits = [], []

            class Session:
                def __init__(self, path, sess_options, providers):
                    self.profile, self.disabled = sess_options.enable_profiling, False
                    sessions.append(self)

                def disable_fallback(self):
                    self.disabled = True

                def get_providers(self):
                    return ['CUDAExecutionProvider', 'CPUExecutionProvider']

                def end_profiling(self):
                    events = [{'cat': 'Node', 'args': {'provider': 'CUDAExecutionProvider'}}]
                    if fail:
                        events.append({'cat': 'Node', 'name': 'activation_kernel_time', 'args': {
                            'provider': 'CPUExecutionProvider', 'op_name': 'MatMul',
                            'input_type_shape': [{'float': [2, 8]}], 'output_type_shape': [{'float': [2, 8]}]}})
                    path = Path(self_test.tmp.name) / 'mock-ort-profile.json'
                    path.write_text(json.dumps(events))
                    return str(path)

            self_test = self

            def binding(session, allowed_shapes=None):
                def forward(input_ids=None, attention_mask=None, **kwargs):
                    if allowed_shapes is not None and tuple(input_ids.shape) not in allowed_shapes:
                        raise ValueError('Unvalidated ORT classifier shape')
                    visits.append(session.profile)
                    return original(input_ids=input_ids, attention_mask=attention_mask)
                return forward

            fake_ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(enable_profiling=False),
                GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99), InferenceSession=Session,
                get_available_providers=lambda: ['CUDAExecutionProvider'])
            fake_onnx = SimpleNamespace(load=lambda *args, **kwargs: SimpleNamespace(graph=SimpleNamespace(node=[])))
            with patch.dict(sys.modules, {'onnxruntime': fake_ort, 'onnx': fake_onnx}), \
                    patch('followup.team_smoke.preload_ort'), patch('torch.cuda.synchronize'), \
                    patch('torch.onnx.export') as export, \
                    patch.object(onnx_service, 'prove_metadata_nodes', return_value={'proven': []}), \
                    patch.object(a, '_binding', side_effect=binding):
                if fail:
                    with self.assertRaisesRegex(ValueError, 'metadata-cpu-v1'):
                        a.prepare_batch_backend([texts])
                    self.assertEqual(a.model.forward, original)
                    self.assertFalse(a._backend_installed)
                    self.assertIsNone(a.ort_session)
                else:
                    a.prepare_batch_backend([texts])
                    self.assertTrue(visits[0])
                    self.assertFalse(visits[-1])
                    self.assertEqual(a.request_batch(texts), expected)
                    self.assertFalse(visits[-1])
                    a.prepare_batch_backend([texts, [texts[0]]])
                    self.assertEqual(export.call_count, 1)
                    self.assertEqual(len(a._ort_shapes), 2)
                    self.assertTrue(all(s.disabled for s in sessions))
                    model = a.model
                    a.close()
                    self.assertEqual(model.forward, original)

    def test_external_helper_speed_quality_and_intentional_candidate_changes(self):
        import torch
        from u1_benchmark.precision import PrecisionGuard
        from u1_benchmark import v4_worker, v4_planning
        from u1_benchmark.v4_privacy import clean_evidence, private_values
        from test_u1_v4_execution import FakeSampler

        class Extension(FamilyAdapter):
            changed = False

            def forward_batch(self, values):
                logits = self.model(**self._hf_inputs(values)).logits
                # Interface fixture, not a real quantized artifact.
                if self.changed:
                    logits = -logits
                return classify(logits, ['SAFE', 'INJECTION'], {'method': 'softmax_argmax'})

            def prove_batch_backend(self, values):
                self.verify_batch(values, [self.index(v) for v in values])
                return {'scope': 'CPU test only', 'gpu_verified': False}

        base = adapter(self.tmp.name)
        a = Extension(base.spec, self.tmp.name, base.preparer, base.model, device='cpu')
        a.metadata['labels'] = ['SAFE', 'INJECTION']
        self.addCleanup(a.close)
        a.check_device()
        guard = PrecisionGuard(a, required=True)
        texts = ['PRIVATE-input library', 'PRIVATE-input opens', 'PRIVATE-input morning']
        rows = [{'id': str(i), 'messages': [{'role': 'user', 'content': text}],
                 'index': a.index(text), 'target': 'SAFE', 'target_role': 'user', 'target_index': 0,
                 'slices': {'language': 'en'}} for i, text in enumerate(texts)]
        plans = v4_planning.smoke_plans(rows, 2)[2]
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
            max_memory_allocated=lambda *_: 0, max_memory_reserved=lambda *_: 0,
            memory_allocated=lambda *_: 0, memory_reserved=lambda *_: 0))
        mapping = {'labels': ['SAFE', 'INJECTION'], 'targets': {'SAFE': 'SAFE', 'INJECTION': 'INJECTION'},
                   'verdicts': {'SAFE': 'SAFE', 'INJECTION': 'INJECTION'}, 'positive_label': 'INJECTION'}
        outcomes = []
        for name, changed in [('baseline', False), ('candidate', True)]:
            a.changed = changed
            result = {}
            v4_worker.latency(a, rows, {'batches': [plans]}, guard, result, fake_torch,
                              FakeSampler('cpu-fixture'), lambda: None)
            self.assertEqual(len(result['rows']), 1)
            directory = Path(self.tmp.name) / name
            directory.mkdir()
            quality = {'result_version': 5}
            v4_worker.quality(a, rows, {'quality_batch_size': 2, 'seed': 17, 'label_mapping': mapping},
                guard, quality, directory, lambda: None, spec=a.spec)
            self.assertEqual(quality['quality']['overall']['evaluated'], 3)
            plan = json.loads((directory / 'quality_plan.json').read_text())
            self.assertEqual([b['actual_batch_size'] for b in plan['batches']], [2, 1])
            saved = (directory / 'predictions.jsonl').read_text()
            self.assertNotIn('PRIVATE-input', saved)
            outcomes.append([json.loads(line)['score']['label'] for line in saved.splitlines()])
            self.assertEqual(clean_evidence(result, private_values(rows), a.metadata['labels']), result)
        self.assertTrue(all(x != y for x, y in zip(*outcomes)))
