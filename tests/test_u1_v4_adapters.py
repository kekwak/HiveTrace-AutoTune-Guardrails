"""Offline v4 contracts and real tiny CPU SDK fixtures; never CUDA acceptance."""
from copy import copy, deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from u1_benchmark.core import digest, sha
from u1_benchmark.v4_preparation import (NATIVE_KEYS, Preparer, create_preparer, external_module,
                                        source_path, validate_settings)
from u1_benchmark.v4_adapters import NativeAdapter, load_adapter


def spec_for(name):
    source = {'repo': 'fixture/configured-checkpoint', 'revision': 'a' * 40}
    spec = {'adapter': name, 'weights': dict(source), 'tokenizer': dict(source), 'template': None,
            'preparation': {'limit': 512}, 'scoring': {}, 'execution': {}}
    if name == 'halo':
        spec['preparation']['system_prompt'] = 'Explicit safety policy'
    if name in ('gliner', 'laya'):
        spec['sdk_assets'] = {'config': dict(source)}
    return spec


class Settings(unittest.TestCase):
    def test_defaults_independent_and_registry_unchanged(self):
        from u1_benchmark.contracts import MODEL_SPECS, DTYPES
        before = deepcopy((MODEL_SPECS, DTYPES))
        for name in NATIVE_KEYS:
            source = spec_for(name)
            normalized = validate_settings(source)
            self.assertEqual(source['execution'], {})
            self.assertEqual(normalized['execution']['runtime'], 'eager')
            self.assertEqual(normalized['weights']['repo'], 'fixture/configured-checkpoint')
        self.assertEqual((MODEL_SPECS, DTYPES), before)
        self.assertEqual(validate_settings(spec_for('halo'))['scoring']['threshold'], .65)
        self.assertEqual(validate_settings(spec_for('open'))['scoring']['max_new_tokens'], 10)
        self.assertEqual(validate_settings(spec_for('laya'))['execution']['dtype'], 'native_sdk')

    def test_supported_parameter_and_backend_settings(self):
        for dtype in ('float16', 'float32', 'bfloat16'):
            spec = spec_for('gliner')
            spec['execution']['dtype'] = dtype
            self.assertEqual(validate_settings(spec)['execution']['dtype'], dtype)
        for runtime in ('compile', 'ort'):
            spec = spec_for('gliner')
            spec['execution']['runtime'] = runtime
            self.assertEqual(validate_settings(spec)['execution']['runtime'], runtime)
        spec = spec_for('halo')
        spec['execution']['linear_attention'] = 'fla'
        self.assertEqual(validate_settings(spec)['execution']['linear_attention'], 'fla')

    def test_unsupported_overrides_fail(self):
        cases = [('llama', 'execution', 'runtime', 'compile'),
                 ('koala', 'execution', 'linear_attention', 'fla'),
                 ('gliner', 'execution', 'attention', 'sdpa'),
                 ('gliner', 'scoring', 'threshold', .7),
                 ('laya', 'execution', 'dtype', 'float16'),
                 ('laya', 'execution', 'attention', 'eager'),
                 ('hive', 'scoring', 'do_sample', True),
                 ('open', 'preparation', 'truncation', True),
                 ('shield', 'execution', 'device', 'cpu'),
                 ('gliner', 'execution', 'compile_backend', 'eager'),
                 ('halo', 'preparation', 'limit', True)]
        for name, section, key, value in cases:
            with self.subTest(name=name, key=key):
                spec = spec_for(name)
                spec[section][key] = value
                with self.assertRaises(ValueError):
                    validate_settings(spec)
        spec = spec_for('gliner')
        spec['execution'] = {'runtime': 'ort', 'dtype': 'float16'}
        with self.assertRaises(ValueError):
            validate_settings(spec)
        for name in ('gliner', 'koala', 'llama', 'laya'):
            spec = spec_for(name)
            spec['template'] = {'text': 'ignored'}
            with self.assertRaises(ValueError):
                validate_settings(spec)

    def test_native_loading_does_not_force_attention_or_kernel(self):
        from u1_benchmark.v4_models import hf_load_options, attention_context
        from contextlib import nullcontext
        import torch
        for name in ('halo', 'hive', 'open', 'koala', 'llama'):
            e = validate_settings(spec_for(name))['execution']
            self.assertEqual(e['attention'], 'native')
            self.assertIsNone(e['sdpa_backend'])
            self.assertNotIn('attn_implementation', hf_load_options(e))
            self.assertIsInstance(attention_context(e), nullcontext)
        for name in ('halo', 'hive', 'open', 'shield'):
            e = validate_settings(spec_for(name))['execution']
            self.assertEqual(hf_load_options(e)['torch_dtype'], torch.bfloat16)
        e = validate_settings(spec_for('laya'))['execution']
        self.assertIsInstance(attention_context(e), nullcontext)
        explicit = {**e, 'dtype': 'float32', 'sdpa_backend': 'math'}
        self.assertEqual(hf_load_options(explicit)['attn_implementation'], 'sdpa')
        with patch('torch.nn.attention.sdpa_kernel', return_value=nullcontext()) as kernel:
            attention_context(explicit)
            kernel.assert_called_once_with(torch.nn.attention.SDPBackend.MATH)
        invalid = spec_for('open')
        invalid['execution'] = {'attention': 'native', 'sdpa_backend': 'math'}
        with self.assertRaisesRegex(ValueError, 'requires SDPA'):
            validate_settings(invalid)

    def test_native_halo_does_not_patch_upstream_dispatch(self):
        from u1_benchmark.v4_models import halo_context
        a = SimpleNamespace(spec=validate_settings(spec_for('halo')))
        with patch('u1_benchmark.models.quiet_dispatch') as patched:
            with halo_context(a):
                pass
            patched.assert_not_called()
        a.spec['execution']['linear_attention'] = 'reference'
        a.qwen, a.quiet_implementations = object(), {}
        with patch('u1_benchmark.models.quiet_dispatch') as patched:
            with halo_context(a):
                pass
            patched.assert_called_once_with(a.qwen, a.quiet_implementations)

    def test_canonical_presets_and_smoke_share_author_baseline(self):
        root = Path(__file__).resolve().parents[1] / 'configs' / 'v4'
        presets = {name: json.loads((root / (name + '.json')).read_text())['baseline']
                   for name in NATIVE_KEYS}
        for name, spec in presets.items():
            self.assertEqual(validate_settings(spec)['execution'],
                             validate_settings(spec_for(name))['execution'])
        for path in root.glob('smoke/*.json'):
            spec = json.loads(path.read_text())['baseline']
            self.assertEqual(spec, presets[spec['adapter']], str(path))
        for name in ('halo_fla', 'gliner_compile', 'gliner_ort'):
            config = json.loads((root / (name + '.json')).read_text())
            self.assertEqual(config['baseline'], presets[config['baseline']['adapter']])
            self.assertEqual(config['baseline']['execution']['dtype'], config['candidate']['execution']['dtype'])

    def test_loaded_attention_is_observed_and_must_not_drift(self):
        a = NativeAdapter.__new__(NativeAdapter)
        a.spec, a.name, a.limit = validate_settings(spec_for('open')), 'open', 512
        config = SimpleNamespace(_attn_implementation='sdpa', max_position_embeddings=4096)
        a.target, a.metadata = SimpleNamespace(config=config), {}
        a._check_checkpoint()
        self.assertEqual(a.metadata['attention'], 'sdpa')
        config._attn_implementation = 'eager'
        with self.assertRaisesRegex(ValueError, 'changed'):
            a._check_checkpoint()
        a.metadata = {}
        a.spec['execution']['attention'] = 'sdpa'
        with self.assertRaisesRegex(ValueError, 'requested attention'):
            a._check_checkpoint()

    def test_laya_observes_sdk_encoder_attention(self):
        a = NativeAdapter.__new__(NativeAdapter)
        a.spec, a.name, a.limit = validate_settings(spec_for('laya')), 'laya', 512
        config = SimpleNamespace(_attn_implementation='sdpa', max_position_embeddings=8192)
        a.model = SimpleNamespace(encoder=SimpleNamespace(config=config))
        a.metadata = {}
        a._check_checkpoint()
        self.assertEqual(a.metadata['attention'], 'sdpa')
        config._attn_implementation = 'eager'
        with self.assertRaisesRegex(ValueError, 'requested attention'):
            a._check_checkpoint()

    def test_sdk_assets_required_and_no_ignored_assets(self):
        for name in ('gliner', 'laya'):
            spec = spec_for(name)
            del spec['sdk_assets']
            with self.assertRaisesRegex(ValueError, 'sdk_assets'):
                validate_settings(spec)
        spec = spec_for('halo')
        del spec['preparation']['system_prompt']
        with self.assertRaisesRegex(ValueError, 'sdk_assets'):
            validate_settings(spec)
        spec['sdk_assets'] = {'prompt': dict(spec['tokenizer'])}
        validate_settings(spec)

    def test_local_inventory_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'config.json'
            file.write_text('{}')
            self.assertEqual(source_path({'path': str(file), 'sha256': sha(file)}), file.resolve())
            source = {'path': tmp, 'inventory': {'config.json': sha(file)}}
            self.assertEqual(source_path(source), Path(tmp).resolve())
            source['inventory']['config.json'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'inventory'):
                source_path(source)
            with self.assertRaises(ValueError):
                source_path({'path': tmp, 'inventory': {'../outside': '0' * 64}})

    def test_external_registration_hash_deps_and_explicit_factories(self):
        code = '''from dataclasses import dataclass
@dataclass
class Prep:
    value: int = 1
    def index(self, payload): return {"total_tokens": self.value}
    def prepare(self, payload): return {"input_ids": [1], "attention_mask": [1]}
    def parity(self): return {"status": "passed", "checks": [{"fixture": True}]}
def create_preparer(spec): return Prep()
def create(spec, directory): return Prep()
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'external.py'
            path.write_text(code)
            spec = {'adapter': {'module': str(path), 'source_sha256': sha(path), 'dependencies': {}}}
            module = external_module(spec)
            self.assertIs(sys.modules[module.__name__], module)
            self.assertEqual(create_preparer(spec).index('payload')['total_tokens'], 1)
            with self.assertRaisesRegex(ValueError, 'required v4 batch method'):
                load_adapter(spec, tmp)
            path.write_text(code + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                create_preparer(spec)
            path.write_text(code)
            spec['adapter']['dependencies'] = {'fixture-package': 'wanted'}
            with patch('importlib.metadata.version', return_value='wrong'):
                with self.assertRaisesRegex(ValueError, 'dependency mismatch'):
                    create_preparer(spec)

    def test_exact_hub_file_uses_pinned_cache_without_tree_lookup(self):
        from u1_benchmark.v4_config import preparation_identity
        with tempfile.TemporaryDirectory() as tmp:
            revision = 'c' * 40
            snapshot = Path(tmp) / 'snapshots' / revision
            cached = snapshot / 'assets' / 'templates' / 'chat_template.jinja'
            cached.parent.mkdir(parents=True)
            cached.write_text('cached pinned template')
            source = {'repo': 'fixture/task-template', 'revision': revision, 'subfolder': 'assets'}
            spec = spec_for('open')
            spec['template'] = {**source, 'file': 'templates/chat_template.jinja'}
            before = deepcopy(spec)
            identity = preparation_identity(spec)
            exact = Mock(return_value=str(cached))
            tree = Mock(side_effect=AssertionError('Repository tree must not be requested'))
            hub = SimpleNamespace(hf_hub_download=exact, snapshot_download=tree)
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                for entry, patterns in ((source, ['templates/chat_template.jinja']),
                                        (spec['template'], None)):
                    root = source_path(entry, patterns)
                    self.assertEqual(root, snapshot / 'assets')
                    self.assertEqual((root / 'templates/chat_template.jinja').read_text(), 'cached pinned template')
                self.assertEqual(exact.call_count, 2)
                exact.assert_called_with(source['repo'], filename='assets/templates/chat_template.jinja', revision=revision)
                tree.assert_not_called()
                exact.side_effect = FileNotFoundError('Pinned file unavailable')
                with self.assertRaises(FileNotFoundError):
                    source_path(spec['template'])
                tree.assert_not_called()
            self.assertEqual(spec, before)
            self.assertEqual(preparation_identity(spec), identity)

    def test_multifile_and_glob_sources_keep_snapshot_resolution(self):
        source = {'repo': 'fixture/tokenizer', 'revision': 'd' * 40, 'subfolder': 'tokenizer'}
        exact = Mock(side_effect=AssertionError('Not an exact-file request'))
        tree = Mock(return_value='/cache/snapshot')
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(
                hf_hub_download=exact, snapshot_download=tree)}):
            for patterns in (['*.json'], ['tokenizer.json', 'config.json']):
                self.assertEqual(source_path(source, patterns), Path('/cache/snapshot/tokenizer'))
                tree.assert_called_with(source['repo'], revision=source['revision'],
                                        allow_patterns=['tokenizer/' + p for p in patterns])
            exact.assert_not_called()


HAS_CPU_DEPS = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'gliner2', 'laya'))


@unittest.skipUnless(HAS_CPU_DEPS, 'Requires torch/transformers/gliner2/laya CPU fixture environment')
class NativeCPU(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def preparer(self, name):
        from u1_fixtures import tiny_tokenizer, tiny_gliner_adapter, tiny_laya_agent
        spec = validate_settings(spec_for(name))
        p = Preparer.__new__(Preparer)
        p.name, p.spec, p.settings, p.limit = name, spec, spec['preparation'], 512
        p.tokenizer = tiny_tokenizer()
        p.tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
        p.input_changed, p.assets, p.versions, p.metadata = False, {}, {}, {}
        if name == 'halo':
            p.prompt = p.settings['system_prompt']
        if name == 'gliner':
            old = tiny_gliner_adapter()
            p.processor, p.schema, p.tokenizer = old.model.processor, old.schema, old.tokenizer
            p.fixture_model = old.model
        if name == 'laya':
            p.agent = tiny_laya_agent()
            p.tokenizer = p.agent.tok
            p.questions = spec['preparation']['questions']
            p.internal = {k: p.agent._to_internal(v) for k, v in p.questions.items()}
        return p

    def adapter(self, name):
        import torch
        from test_u1_batch_adapters import ToyModel
        p = self.preparer(name)
        a = NativeAdapter.__new__(NativeAdapter)
        a.name, a.spec, a.preparer, a.tokenizer = name, p.spec, p, p.tokenizer
        a.device, a.directory, a.limit, a.backend = torch.device('cpu'), Path(self.tmp.name), p.limit, 'eager'
        a._builtin, a.adapter, a.legacy = True, a, None
        a._compile_count, a._prepared, a._in_request, a._backend_installed = 0, False, False, True
        a._restore_encoder = a._restore_processor = a._restore_laya_encoding = None
        a.metadata = {}
        a.check_device = Mock()  # Only fixtures bypass production CUDA/precision enforcement.
        a._check_laya_runtime = Mock()
        a._configure_padding()
        a.model = ToyModel(9 if name == 'koala' else 15 if name in ('halo', 'shield') else 2,
                           causal=name in ('halo', 'shield'))
        if name in ('koala', 'llama'):
            a.labels = ['native-' + str(i) for i in range(a.model.classes)]
        if name in ('halo', 'shield'):
            a.label_ids = [5, 6]
        if name == 'halo':
            a.qwen, a.quiet_implementations = SimpleNamespace(), {}
        if name == 'laya':
            from u1_benchmark.adapters import Builtin
            a.agent, a.model = copy(p.agent), p.agent.model
            Builtin._install_laya_input_guard(a)
        if name == 'gliner':
            from experiments.adapters import GlinerAdapter
            a.model = p.fixture_model
            a.model.strict_extraction = True
            a.gliner = GlinerAdapter.__new__(GlinerAdapter)
            a.gliner.model, a.gliner.tokenizer, a.gliner.schema = a.model, a.tokenizer, p.schema
            a.gliner.device, a.gliner.limit, a.gliner.threshold = a.device, a.limit, .5
        a.target = a.model.encoder if name == 'gliner' else a.model
        a._record_labels()
        self.addCleanup(a.close)
        return a

    def test_native_verdict_vocabularies_are_explicit(self):
        expected = {'halo': ['safe', 'unsafe'], 'gliner': ['safe', 'unsafe'],
                    'hive': ['safe', 'unsafe'], 'open': ['safe', 'unsafe'],
                    'laya': ['A', 'B'], 'shield': ['Yes', 'No'],
                    'koala': ['native-' + str(i) for i in range(9)],
                    'llama': ['native-0', 'native-1']}
        for name, labels in expected.items():
            with self.subTest(name=name):
                a = self.adapter(name)
                self.assertEqual(a.metadata['labels'], labels)
                self.assertIsNot(a.metadata['labels'], a.labels)

    def test_native_labels_matching_private_inputs_survive_sanitizer(self):
        from u1_benchmark.v4_privacy import clean_evidence, private_values
        from u1_benchmark.v4_worker import safe_metadata
        for name, text in (('hive', 'safe'), ('open', 'safe'), ('shield', 'No')):
            with self.subTest(name=name):
                a = self.adapter(name)
                response = a.request_batch([text])[0]
                self.assertEqual(response['label'], text)
                texts = private_values([{'text': text}])
                allowed = safe_metadata(a)['labels']
                self.assertEqual(clean_evidence(response, texts, allowed), response)
                echoed = {'label': text + ' echoed request', 'scores': {text: 1.0}}
                if len(text) >= 4:
                    self.assertNotEqual(clean_evidence(echoed, texts, allowed), echoed)
        echoed = {'label': 'private request to echo'}
        self.assertNotEqual(clean_evidence(echoed, ['private request to echo']), echoed)

    def test_all_native_batches_observe_indexed_rows_in_order(self):
        for name in sorted(NATIVE_KEYS):
            with self.subTest(name=name):
                a = self.adapter(name)
                payloads = ['library.', 'The public library.']
                expected = [a.index(t) for t in payloads]
                result = a.verify_batch(payloads, expected)
                self.assertEqual(len(result), 2)
                self.assertTrue(all('label' in r for r in result))
                self.assertEqual(a.metadata['batch_verification']['shape'][0], 2)
                self.assertEqual(a.metadata['batch_verification']['lengths'], [r['total_tokens'] for r in expected])
                self.assertEqual(len(a.target._forward_pre_hooks), 0)

    def test_cpu_constructor_no_weights_and_explicit_templates(self):
        from u1_fixtures import tiny_tokenizer
        tokenizer = tiny_tokenizer()
        spec = spec_for('open')
        spec['template'] = {'text': "{% for m in messages %}{{ m['content'] }} {% endfor %}safe"}
        with patch('u1_benchmark.v4_preparation.source_path', return_value=Path(self.tmp.name)) as resolve, \
                patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=AssertionError('weights loaded')):
            p = create_preparer(spec)
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(resolve.call_args.args[0], spec['tokenizer'])
        self.assertEqual(p.render('library').strip(), 'library safe')
        self.assertEqual(p.index('library')['input_sha256'], digest(p.prepare('library')))
        self.assertNotIn('library', json.dumps(p.metadata))
        self.assertNotIn('safe', json.dumps(p.metadata))

    def test_roles_and_source_hashes(self):
        value = [{'role': 'user', 'content': 'library'}, {'role': 'assistant', 'content': 'public'}]
        for name in NATIVE_KEYS - {'open'}:
            p = self.preparer(name)
            row = p.index({'id': 'fixture', 'messages': value})
            self.assertEqual(row['source_sha256'], digest(value))
            self.assertEqual(row['target_index'], 0 if name == 'halo' else 1)
            self.assertEqual(row['context_indices'], [0] if name in ('hive', 'shield') else [])
            self.assertNotIn('effective_messages', row)
        self.assertFalse(self.preparer('open').index(value)['supported'])
        self.assertFalse(self.preparer('halo').index([value[1]])['supported'])

    def test_parity_checks_native_paths_and_detects_mismatch(self):
        for name in NATIVE_KEYS:
            with self.subTest(name=name):
                p = self.preparer(name)
                proof = p.parity()
                self.assertEqual(proof['status'], 'passed')
                self.assertTrue(any(row['status'] == 'passed' for row in proof['checks']))
                self.assertNotIn('content', json.dumps(proof))
                with patch.object(p, 'prepare', return_value={'input_ids': [999], 'attention_mask': [1]}):
                    with self.assertRaises(ValueError):
                        p.parity()

    def test_precision_passport_exact_and_detects_drift(self):
        import torch
        from u1_benchmark.precision import validate_precision, PrecisionGuard
        a = self.adapter('llama')
        a.model = torch.nn.Linear(2, 2).float()
        NativeAdapter.check_device(a)
        self.assertEqual(validate_precision(a.metadata['precision']), {
            'policy': 'explicit', 'parameter_dtypes': ['float32'],
            'autocast_enabled': False, 'autocast_dtype': None})
        PrecisionGuard(a, required=True).check()
        a.model.half()
        with self.assertRaisesRegex(ValueError, 'Physical parameter precision'):
            NativeAdapter.check_device(a)

    def test_laya_lightweight_runtime_guard(self):
        import torch
        a = self.adapter('laya')
        a.device, a.agent.device = torch.device('cuda:0'), torch.device('cuda:0')
        a.agent.dtype, a._laya_autocast_dtype, a.agent.amp_enabled = torch.float16, torch.float16, True
        NativeAdapter._check_laya_runtime(a)
        a.agent.amp_enabled = False
        with self.assertRaisesRegex(ValueError, 'autocast changed'):
            NativeAdapter._check_laya_runtime(a)
        a.agent.device = torch.device('cpu')
        with self.assertRaisesRegex(ValueError, 'CPU fallback'):
            NativeAdapter._check_laya_runtime(a)

    def test_weight_inventory_is_hashed_once_during_load(self):
        from u1_benchmark import v4_models
        a = self.adapter('llama')
        weight = Path(self.tmp.name) / 'model.safetensors'
        weight.write_bytes(b'not-real-weights-only-a-constructor-fixture')
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config = SimpleNamespace(id2label={0: 'benign', 1: 'malicious'}, num_labels=2)
        with patch.object(v4_models, 'source_path', return_value=Path(self.tmp.name)), \
                patch.object(v4_models, 'sha', wraps=sha) as hashed, \
                patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=(model, {})):
            v4_models.load_model(a)
        self.assertEqual(hashed.call_count, 1)
        inventory = a.metadata['artifacts']['weight_files']
        self.assertEqual(inventory, {'model.safetensors': {'sha256': sha(weight), 'bytes': weight.stat().st_size}})
        self.assertEqual(a.metadata['artifacts']['weight_inventory_sha256'], digest(inventory))

    def test_laya_single_encoding_and_no_preflight_encoding(self):
        a = self.adapter('laya')
        native = a.agent._encode_state
        with patch.object(a.agent, '_encode_state', wraps=native) as encode, \
                patch.object(a.preparer, 'prepare', side_effect=AssertionError('double encoding')):
            a.request_batch(['library', 'The public library'])
        self.assertEqual(encode.call_count, 2)
        # SDK guard rejects the full untruncated sequence before the batch forward.
        overhead = len(native('', list(a.preparer.questions), a.preparer.internal)[0]['ids'])
        a.reject_batch_over_limit(['library', 'library ' * (513 - overhead)])

    def test_bad_indices_padding_and_limit_rejected(self):
        a = self.adapter('llama')
        expected = [a.index('library')]
        expected[0]['target_role'] = 'assistant'
        with self.assertRaisesRegex(ValueError, 'message/context'):
            a.verify_batch(['library'], expected)
        a.reject_batch_over_limit(['library', 'library ' * 513])
        a.tokenizer.padding_side = 'left'
        with self.assertRaisesRegex(ValueError, 'padding policy'):
            a.request_batch(['library'])

    def test_eager_preparation_has_no_forward_prepass_or_timed_parameter_scan(self):
        for name in NATIVE_KEYS:
            with self.subTest(name=name):
                a = self.adapter(name)
                with patch.object(a, 'request_batch', wraps=a.request_batch) as request:
                    a.prepare_batch_backend([['library.', 'The public library.']] * 100)
                    request.assert_not_called()
                a.request_batch(['library.', 'The public library.'])
                a.check_device.assert_not_called()
                self.assertEqual(a._check_laya_runtime.call_count, 1 if name == 'laya' else 0)

    def test_boundary_is_bucket_edge_and_above_model_limit(self):
        a = self.adapter('llama')
        edge, above = a.boundary(10)
        self.assertEqual(a.index(edge)['total_tokens'], 10)
        self.assertEqual(a.index(above)['total_tokens'], 513)
        a.reject_batch_over_limit([edge, above])

    def test_compile_native_gliner_route_and_restore(self):
        import torch
        a = self.adapter('gliner')
        original = a.model.encoder.forward
        batches = [['library.', 'The public library.'], ['public library.']]
        reference = [a.request_batch(batch) for batch in batches]
        a.backend, a._backend_installed = 'compile', False
        a.spec['execution']['runtime'] = 'compile'
        a.spec = validate_settings(a.spec)
        with self.assertRaisesRegex(ValueError, 'not been prepared'):
            a.request_batch(batches[0])
        with patch('torch._dynamo.backends.registry.lookup_backend',
                   side_effect=lambda name: name if callable(name) else lambda graph, inputs: graph.forward):
            with patch.object(a, 'request_batch', wraps=a.request_batch) as request:
                a.prepare_batch_backend(batches * 100)
                self.assertEqual(request.call_count, len(batches))
        self.assertGreater(a.compile_count, 0)
        before = a.compile_count
        for batch, expected in zip(batches, reference):
            self.assertEqual(a.verify_batch(batch, [a.index(t) for t in batch]), expected)
        self.assertEqual(a.compile_count, before)
        model = a.model
        a.close()
        self.assertEqual(model.encoder.forward, original)
        torch._dynamo.reset()

    def test_ort_profile_separation_shape_extension_and_rollback(self):
        import torch
        from u1_benchmark import v4_models
        from followup import onnx_service
        for fail in (False, True):
            with self.subTest(fail=fail):
                a = self.adapter('gliner')
                a.backend, a._backend_installed = 'ort', False
                a.spec['execution']['runtime'] = 'ort'
                a.spec = validate_settings(a.spec)
                original = a.model.encoder.forward
                batches = [['library.', 'The public library.']]
                sessions, visits = [], []
                old_global = onnx_service.VALIDATION_TEXTS

                class Session:
                    def __init__(self, path, sess_options, providers):
                        self.profile = sess_options.enable_profiling
                        self.options, self.providers = sess_options, providers
                        self.ended, self.fallback_disabled = False, False
                        self.ordinal = len(sessions)
                        sessions.append(self)

                    def disable_fallback(self):
                        self.fallback_disabled = True

                    def get_providers(self):
                        return ['CUDAExecutionProvider', 'CPUExecutionProvider']

                    def end_profiling(self):
                        self.ended = True
                        path = a.directory / ('profile-' + str(self.ordinal) + '.json')
                        events = [{'cat': 'Node', 'args': {'provider': 'CUDAExecutionProvider'}}]
                        if fail:
                            events.append({'cat': 'Node', 'name': 'activation_kernel_time', 'args': {
                                'provider': 'CPUExecutionProvider', 'op_name': 'MatMul',
                                'input_type_shape': [{'float': [2, 8]}],
                                'output_type_shape': [{'float': [2, 8]}]}})
                        path.write_text(json.dumps(events))
                        return str(path)

                def binding(session, encoder, allowed, checked):
                    def forward(input_ids=None, attention_mask=None, **kwargs):
                        shape = tuple(input_ids.shape)
                        if checked() and shape not in allowed:
                            raise ValueError('Unvalidated ORT batch shape')
                        visits.append((session.ordinal, shape, session.profile))
                        return original(input_ids=input_ids, attention_mask=attention_mask)
                    return forward

                fake_ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(enable_profiling=False),
                    GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99), InferenceSession=Session,
                    get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                fake_onnx = SimpleNamespace(load=lambda *args, **kwargs: SimpleNamespace(graph=None))
                with patch.dict(sys.modules, {'onnxruntime': fake_ort, 'onnx': fake_onnx}), \
                        patch('followup.team_smoke.preload_ort'), patch('torch.cuda.synchronize'), \
                        patch('torch.onnx.export') as export, \
                        patch.object(onnx_service, 'prove_metadata_nodes', return_value={'proven': []}), \
                        patch('u1_benchmark.batch_ort.binding_forward', side_effect=binding):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'metadata-cpu-v1'):
                            a.prepare_batch_backend(batches)
                        self.assertEqual(a.model.encoder.forward, original)
                        self.assertFalse(a._backend_installed)
                        self.assertIsNone(a.ort_session)
                    else:
                        a.prepare_batch_backend(batches * 100)
                        self.assertEqual(len(visits), 2)  # One profile and one measured-session check per shape.
                        self.assertTrue(sessions[0].ended)
                        self.assertTrue(all(s.fallback_disabled for s in sessions))
                        self.assertFalse(sessions[-1].profile)
                        self.assertFalse(a.metadata['ort_profile_active_during_latency'])
                        a.verify_batch(batches[0], [a.index(t) for t in batches[0]])
                        self.assertFalse(visits[-1][2])
                        count = len(sessions)
                        a.prepare_batch_backend(batches)
                        self.assertEqual(len(sessions), count)
                        a.prepare_batch_backend([['The public library opens every morning.', 'library.']])
                        self.assertEqual(len(sessions), count + 2)
                        self.assertEqual(export.call_count, 1)
                        self.assertEqual(len(a._ort_shapes), 2)
                        with self.assertRaisesRegex(ValueError, 'Unvalidated ORT batch shape'):
                            a.request_batch(['library'])
                        model = a.model
                        a.close()
                        self.assertEqual(model.encoder.forward, original)
                self.assertIs(onnx_service.VALIDATION_TEXTS, old_global)


if __name__ == '__main__':
    unittest.main()
