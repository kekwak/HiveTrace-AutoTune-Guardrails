"""CPU-only d3 contracts. Fixtures never stand in for CUDA kernel acceptance."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from u1_benchmark import d3_family as d3
from u1_benchmark.core import digest, sha
from u1_benchmark.family_adapter import FamilyAdapter
from u1_benchmark.precision import validate_precision


# Small local SDK double tests our boundary, not real model numerical quality.
RUNTIME = '''
from dataclasses import dataclass
from types import SimpleNamespace
import json
import math
from . import d3_format

def answer_codes(tokenizer):
    return ['A', 'B'], [65, 66]

def normalize_question(q):
    return SimpleNamespace(kind=q['type'], rendered=q,
        keys=list(q['criteria']) if q['type']=='choice' else ['false','true'])

def render(tokenizer, prompt, state, question, codes):
    return tokenizer.apply_chat_template([{'role':'system','content':'native'},
        {'role':'user','content':json.dumps([state,question],ensure_ascii=False)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

def product_answer(q, row):
    if len(row)!=len(q.keys) or not all(math.isfinite(x) and x>=0 for x in row) or sum(row)<=0:
        raise ValueError('invalid probabilities')
    row=[x/sum(row) for x in row]
    if q.kind=='noul': return {'type':'noul','noul':row[1]}
    return {'type':'choice','choice':q.keys[row.index(max(row))], 'probabilities':dict(zip(q.keys,row))}

@dataclass
class Prepared:
    questions: dict
    sequences: dict
    errors: dict
    runnable: list

class D3:
    def __init__(self,*args,**kwargs):
        raise AssertionError('CPU preparer attempted weight loading')
    def prepare(self,state,questions):
        normalized={key:normalize_question(q) for key,q in questions.items()}
        texts=[render(self.tokenizer,self.prompt,state,q.rendered,self.codes) for q in normalized.values()]
        ids=self.tokenizer(texts,add_special_tokens=False)['input_ids']
        return Prepared(normalized,dict(zip(questions,ids)),{},list(questions))
'''


class NativeEnvironment(unittest.TestCase):
    def test_hidden_sdpa_override_is_rejected(self):
        with patch.dict(os.environ, {'D3_CUDNN_SDPA': '1'}):
            with self.assertRaisesRegex(ValueError, 'configured native SDPA'):
                d3.check_native_attention_environment()
        with patch.dict(os.environ, {'D3_CUDNN_SDPA': '0'}):
            d3.check_native_attention_environment()


class Tokenizer:
    padding_side = 'left'
    pad_token_id = 0
    calls = 0

    def apply_chat_template(self, value, **kwargs):
        if kwargs != {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}:
            raise AssertionError('Native template kwargs changed')
        return ''.join(m['role'] + ':' + m['content'] for m in value) + 'assistant:'

    def encode(self, text, **kwargs):
        return [ord(c) + 1 for c in text]

    def __call__(self, text, **kwargs):
        self.calls += 1
        if kwargs.get('truncation') or kwargs.get('padding'):
            raise AssertionError('Unexpected truncation/padding in preparation')
        return {'input_ids': [self.encode(t) for t in text] if isinstance(text, list) else self.encode(text)}


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(70000, 2, dtype=torch.bfloat16)
        with torch.no_grad():
            self.embedding.weight.fill_(.25)
        self.config = SimpleNamespace(text_config=SimpleNamespace(_attn_implementation='sdpa'))

    def get_input_embeddings(self):
        return self.embedding


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.code = {'d3_runtime.py': RUNTIME, 'd3_format.py': 'FORMAT_ID="fixture"\n',
                     'd3_fast.py': '', 'd3_kernels.py': ''}
        config = {'format_version': 1, 'format_id': 'd3-code-readout-v1', 'prompt': 'd3',
            'pooling': 'last', 'attention_mode': 'noncausal_full_attention', 'readout_dtype': 'float32',
            'max_length': None, 'temperature': 1.0, 'codes': ['A', 'B'], 'token_ids': [65, 66]}
        files = {**self.code, 'decision_config.json': json.dumps(config), 'config.json': '{}',
                 'tokenizer.json': '{}', 'tokenizer_config.json': '{}'}
        for name, text in files.items():
            (self.root / name).write_text(text)
        self.manifest = {name: sha(self.root / name) for name in files}
        self.codes = {name: self.manifest[name] for name in self.code}
        patcher = patch.object(d3, 'CODE_HASHES', self.codes)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tokenizer = Tokenizer()
        patcher = patch('transformers.AutoTokenizer.from_pretrained', return_value=self.tokenizer)
        patcher.start()
        self.addCleanup(patcher.stop)
        source = {'path': str(self.root)}
        self.spec = {'adapter': 'd3_decision', 'weights': source, 'tokenizer': source,
            'sdk_assets': {'runtime': source, 'manifest': self.manifest},
            'preparation': {'limit': 4096}, 'execution': {'trust_remote_code': True}}

    def preparer(self, spec=None):
        value = d3.D3Preparer(spec or self.spec)
        self.addCleanup(value.close)
        return value

    def adapter(self, spec=None):
        preparer = self.preparer(spec)
        adapter = d3.D3Adapter.__new__(d3.D3Adapter)
        FamilyAdapter.__init__(adapter, preparer.spec, self.root, preparer, TinyBackbone(), device='cpu')
        adapter.native = SimpleNamespace(encoder=lambda embeds, mask: embeds * mask.unsqueeze(-1),
            readout=torch.ones((2, 2), dtype=torch.float32), backbone=adapter.model)
        adapter.extra_tensors['readout'] = adapter.native.readout
        adapter._original_encoder = adapter.native.encoder
        adapter._linear_originals = adapter._linear_functions = adapter._quiet_linear = {}
        adapter.qwen = SimpleNamespace()
        adapter.probability_calls = []

        def probabilities(sequences, counts):
            adapter.probability_calls.append((deepcopy(sequences), list(counts)))
            width = max(map(len, sequences))
            ids = torch.zeros((len(sequences), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(sequences):
                ids[i, -len(row):] = torch.tensor(row)
                mask[i, -len(row):] = 1
            embeds = adapter.model.get_input_embeddings()(ids)
            hidden = adapter.native.encoder(embeds, mask)
            logits = hidden[:, -1].float() @ adapter.native.readout.T
            return [row[:count] for row, count in zip(logits.softmax(-1).tolist(), counts)]

        adapter.native.probabilities = probabilities
        self.addCleanup(adapter.close)
        return adapter


class ValidationTests(Fixture):
    def test_defaults_do_not_mutate_config(self):
        before = deepcopy(self.spec)
        result = d3.validate_d3(self.spec)
        self.assertEqual(before, self.spec)
        self.assertEqual(result['execution']['dtype'], 'native_sdk')
        self.assertEqual(result['scoring']['threshold'], .5)
        self.assertEqual(result['preparation']['questions']['user'], d3.BLOCK_QUESTION)

    def test_explicit_permission_hashes_and_unknown_settings(self):
        for update in ({'trust_remote_code': False}, {'trust_remote_code': 1},
                {'dtype': 'bfloat16'}, {'runtime': 'ort'}, {'sdpa_backend': 'math'}, {'unused': True}):
            with self.subTest(update=update):
                spec = deepcopy(self.spec)
                spec['execution'].update(update)
                with self.assertRaises(ValueError):
                    d3.validate_d3(spec)
        spec = deepcopy(self.spec)
        spec['sdk_assets']['manifest']['d3_runtime.py'] = 'f' * 64
        with self.assertRaises(ValueError):
            d3.validate_d3(spec)

    def test_manifest_rejects_weights_and_path_traversal(self):
        for name in ('../escape.py', '/escape.py', 'readout.safetensors'):
            spec = deepcopy(self.spec)
            spec['sdk_assets']['manifest'][name] = 'f' * 64
            with self.assertRaises(ValueError):
                d3.validate_d3(spec)

    def test_unknown_remote_runtime_rejected(self):
        spec = deepcopy(self.spec)
        spec['sdk_assets']['runtime'] = {'repo': d3.REPOSITORY, 'revision': 'f' * 40}
        with self.assertRaises(ValueError):
            d3.validate_d3(spec)

    def test_choice_config_no_ignored_threshold(self):
        spec = deepcopy(self.spec)
        spec['preparation']['questions'] = {'user': {'type': 'choice', 'criteria': {'C': None, 'A': None}}}
        self.assertEqual(d3.validate_d3(spec)['scoring'], {})
        spec['scoring'] = {'threshold': .5}
        with self.assertRaises(ValueError):
            d3.validate_d3(spec)


class PreparationTests(Fixture):
    def test_no_weights_and_exact_native_parity(self):
        preparer = self.preparer()
        self.assertFalse(hasattr(preparer.native, 'backbone'))
        self.assertEqual(preparer.parity()['status'], 'passed')
        self.assertIn('dependencies', preparer.metadata)

    def test_roles_and_no_fabricated_user(self):
        preparer = self.preparer()
        pair = [{'role': 'user', 'content': 'query'}, {'role': 'assistant', 'content': 'answer'}]
        record = preparer.index({'id': 'pair', 'messages': pair})
        self.assertEqual((record['id'], record['target_role'], record['context_indices']), ('pair', 'assistant', [0]))
        _, state, _, _ = d3._selection(pair[1:], preparer.questions)
        self.assertEqual(state, {'assistant': 'answer'})
        _, state, _, _ = d3._selection('query', preparer.questions)
        self.assertEqual(state, 'query')

    def test_one_native_encoding_per_prepare_and_no_cache(self):
        preparer = self.preparer()
        self.tokenizer.calls = 0
        a = preparer.prepare('A')
        b = preparer.prepare('B')
        self.assertEqual(a, preparer.prepare('A'))
        self.assertNotEqual(a, b)
        self.assertEqual(self.tokenizer.calls, 3)

    def test_limit_boundaries_keep_full_ids(self):
        preparer = self.preparer()
        overhead = len(preparer.prepare('')['input_ids'])
        for edge in (512, 1024, 2048, 4096):
            preparer.limit = edge
            for length in (edge, edge + 1):
                value = 'a' * (length - overhead)
                record = preparer.index(value)
                self.assertEqual(record['total_tokens'], length)
                self.assertEqual(record['supported'], length <= edge)

    def test_unicode_special_tokens_not_removed(self):
        preparer = self.preparer()
        value = '\u041f\u0440\u0438\u0432\u0435\u0442\n<|im_end|>\u200b'
        ids = preparer.prepare(value)['input_ids']
        self.assertIn(ord('\u200b') + 1, ids)
        self.assertEqual(preparer.index(value)['ids_sha256'], digest(ids))

    def test_unsupported_role_record(self):
        spec = deepcopy(self.spec)
        spec['preparation']['questions'] = {'user': d3.BLOCK_QUESTION}
        preparer = self.preparer(spec)
        record = preparer.index([{'role': 'assistant', 'content': 'answer'}])
        self.assertFalse(record['supported'])
        self.assertIsNone(record['total_tokens'])

    def test_loader_isolated_registered_dataclasses_and_cleanup(self):
        before_path = list(sys.path)
        before_modules = set(sys.modules)
        a, b = self.preparer(), self.preparer()
        self.assertNotEqual(a.package.name, b.package.name)
        self.assertTrue(a.runtime.Prepared.__module__.startswith(a.package.name))
        self.assertEqual(sys.path, before_path)
        name = a.package.name
        a.close()
        self.assertFalse(any(k == name or k.startswith(name + '.') for k in sys.modules))
        self.assertIn(b.package.name, sys.modules)
        self.assertFalse({'d3_runtime', 'd3_format'} & (set(sys.modules) - before_modules))

    def test_checksum_failure_before_execution(self):
        (self.root / 'd3_runtime.py').write_text('raise AssertionError("untrusted code")')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.preparer()

    def test_failed_import_cleans_only_own_namespace(self):
        (self.root / 'd3_runtime.py').write_text('raise RuntimeError("fixture")')
        self.manifest['d3_runtime.py'] = sha(self.root / 'd3_runtime.py')
        before = set(sys.modules)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            d3._RuntimePackage(self.root, self.manifest)
        self.assertEqual({k for k in sys.modules if k.startswith('_u1_d3_')},
                         {k for k in before if k.startswith('_u1_d3_')})


class RuntimeTests(Fixture):
    def test_preflight_boundaries_follow_configured_role_mode(self):
        from u1_benchmark.v4_worker import preflight
        for mode in ('user', 'assistant', 'user_assistant'):
            with self.subTest(mode=mode):
                spec = deepcopy(self.spec)
                spec['preparation']['questions'] = {
                    mode: d3.BLOCK_QUESTION if mode == 'user' else d3.RESPONSE_QUESTION}
                a = self.adapter(spec)
                rows = []
                for i, text in enumerate(('One.', 'Two words.', 'Three short words.')):
                    messages = [{'role': role, 'content': text} for role in mode.split('_')]
                    rows.append({'id': str(i), 'messages': messages, 'index': a.index(messages)})
                original = deepcopy(rows)
                for target in (512, 4096):
                    edge, above = a.boundary(target)
                    self.assertEqual(a.index(edge)['total_tokens'], target)
                    self.assertEqual(a.index(above)['total_tokens'], 4097)
                    self.assertEqual(a.index(edge)['source_mode'], d3.MODES[mode])
                    self.assertFalse(a.index(above)['supported'])
                    result = {}
                    guard = SimpleNamespace(call=lambda function, *args: function(*args))
                    with patch.object(a, 'prove_batch_backend', return_value={'scope': 'CPU fixture'}):
                        preflight(a, rows, {'batch_size': 2, 'seed': 17, 'bucket_max': target}, guard, result)
                    self.assertTrue(result['over_limit_rejected'])
                    self.assertTrue(result['aba_passed'])
                self.assertEqual(rows, original)

    def test_single_real_batch_and_response_order(self):
        adapter = self.adapter()
        batch = ['A', 'longer input']
        self.tokenizer.calls = 0
        results = adapter.request_batch(batch)
        self.assertEqual(len(adapter.probability_calls), 1)
        self.assertEqual(self.tokenizer.calls, 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['label'] == 'block' and r['generated_tokens'] == 0 for r in results))
        self.assertEqual(set(results[0]['scores']), {'allow', 'block'})
        self.assertLess(len(adapter.probability_calls[0][0][0]), len(adapter.probability_calls[0][0][1]))

    def test_actual_embedding_ids_masks_and_padding(self):
        adapter = self.adapter()
        batch = ['A', 'much longer']
        result = adapter.verify_batch(batch, [adapter.index(p) for p in batch])
        self.assertEqual(len(result), 2)
        self.assertGreater(adapter.metadata['batch_verification']['padding_tokens'], 0)
        self.assertIs(adapter.native.encoder, adapter._original_encoder)
        self.assertFalse(adapter.model.embedding._forward_pre_hooks)

    def test_stale_index_rejected(self):
        adapter = self.adapter()
        expected = adapter.index('A')
        expected['ids_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            adapter.verify_batch(['A'], [expected])
        self.assertEqual(adapter.probability_calls, [])

    def test_limit_rejected_before_forward_restores_guard(self):
        adapter = self.adapter()
        adapter.limit = adapter.preparer.limit = len(adapter.preparer.prepare('A')['input_ids'])
        original = adapter.native.probabilities
        adapter.request_batch(['A'])
        adapter.reject_batch_over_limit(['AA', 'A'])
        self.assertEqual(len(adapter.probability_calls), 1)
        self.assertIs(adapter.native.probabilities, original)

    def test_invalid_probabilities_rejected(self):
        adapter = self.adapter()
        for row in ([float('nan'), .5], [float('inf'), 0], [-1, 2], [0, 0], [.5]):
            adapter.native.probabilities = lambda sequences, counts, row=row: [row]
            with self.assertRaises(ValueError):
                adapter.request('A')

    def test_verify_restores_after_exception(self):
        adapter = self.adapter()
        original = adapter.native.encoder
        adapter.native.probabilities = lambda *args: (_ for _ in ()).throw(RuntimeError('fixture'))
        with self.assertRaises(RuntimeError):
            adapter.verify_batch(['A'], [adapter.index('A')])
        self.assertIs(adapter.native.encoder, original)
        self.assertFalse(adapter.model.embedding._forward_pre_hooks)

    def test_a_b_a_and_close(self):
        adapter = self.adapter()
        a = adapter.request_batch(['A', 'BB'])
        adapter.request_batch(['CC', 'D'])
        self.assertEqual(a, adapter.request_batch(['A', 'BB']))
        adapter.close()
        self.assertIsNone(adapter.model)
        self.assertIsNone(adapter.native)

    def test_cpu_runtime_rejected_not_silently_offloaded(self):
        adapter = self.adapter()
        adapter.native.device = torch.device('cpu')
        with self.assertRaisesRegex(ValueError, 'CPU/offloaded'):
            adapter.check_device()

    def test_compile_routes_real_scores_and_counts_new_graphs(self):
        adapter = self.adapter()
        adapter.backend = 'compile'
        adapter.spec['execution']['runtime'] = 'compile'
        adapter._backend_installed = False
        seen = []

        def compile_fn(fn, **options):
            self.assertTrue(options['fullgraph'])
            self.assertTrue(options['dynamic'])

            def compiled(embeds, mask):
                shape = tuple(mask.shape)
                if shape not in seen:
                    seen.append(shape)
                    options['backend'](None, [])
                return fn(embeds, mask)
            return compiled

        with patch('torch.compile', side_effect=compile_fn), patch(
                'torch._dynamo.backends.registry.lookup_backend', return_value=lambda *args: None):
            adapter.prepare_batch_backend([['A'], ['AA', 'B']])
            self.assertEqual(adapter.compile_count, 2)
            adapter.request_batch(['A'])
            self.assertEqual(adapter.compile_count, 2)
            adapter.request_batch(['new shape'])
            self.assertEqual(adapter.compile_count, 3)

    def test_actual_cpu_compile_tensor_forward(self):
        from torch._dynamo.backends.registry import lookup_backend
        eager = lookup_backend('eager')
        adapter = self.adapter()
        expected = adapter.request_batch(['A', 'BB'])
        adapter.backend = adapter.spec['execution']['runtime'] = 'compile'
        adapter._backend_installed = False
        # Dynamo/fullgraph is real; only code generation uses the CPU eager backend.
        with patch('torch._dynamo.backends.registry.lookup_backend',
                   side_effect=lambda name: eager if name == 'inductor' else lookup_backend(name)):
            adapter.prepare_batch_backend([['A', 'BB']])
        self.assertGreater(adapter.compile_count, 0)
        count = adapter.compile_count
        self.assertEqual(adapter.request_batch(['A', 'BB']), expected)
        self.assertEqual(adapter.compile_count, count)

    def test_quiet_linear_wrapper_identity_and_restoration(self):
        adapter = self.adapter()
        calls = []

        def original(*args, **kwargs):
            return 'reference'

        def quiet(*args, **kwargs):
            return 'fla'

        adapter.qwen = SimpleNamespace(torch_chunk_gated_delta_rule=original)
        adapter._linear_originals = {'torch_chunk_gated_delta_rule': original}
        adapter._linear_functions = {'torch_chunk_gated_delta_rule': quiet}
        adapter._quiet_linear = dict(adapter._linear_functions)
        for _ in range(2):
            with adapter._linear_dispatch():
                self.assertIs(adapter.qwen.torch_chunk_gated_delta_rule, quiet)
        with self.assertRaises(RuntimeError):
            with adapter._linear_dispatch(calls):
                self.assertEqual(adapter.qwen.torch_chunk_gated_delta_rule(), 'fla')
                raise RuntimeError('fixture')
        self.assertIs(adapter.qwen.torch_chunk_gated_delta_rule, original)
        self.assertEqual(len(calls), 1)

    def test_tensor_inventory_includes_readout(self):
        for name in ('readout.safetensors', 'model.safetensors'):
            (self.root / name).write_bytes(name.encode())
        source = {'path': str(self.root), 'inventory': {name: sha(self.root / name)
            for name in ('readout.safetensors', 'model.safetensors')}}
        result = d3._weight_inventory(self.root, source)
        self.assertEqual(result, source['inventory'])
        (self.root / 'readout.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            d3._weight_inventory(self.root, source)

    def test_size_includes_raw_readout_storage(self):
        adapter = self.adapter()
        expected = sum(p.untyped_storage().nbytes() for p in adapter.model.parameters())
        expected += adapter.native.readout.untyped_storage().nbytes()
        self.assertEqual(adapter.size()['unique_parameter_buffer_storage_bytes'], expected)

    def precision_fixture(self):
        adapter = self.adapter()
        device = torch.device('cuda:0')

        def tensor(dtype):
            return SimpleNamespace(device=device, dtype=dtype, is_floating_point=lambda: True)

        parameters = [tensor(torch.bfloat16)]
        adapter.device = device
        adapter.native.device = device
        adapter.native.readout = tensor(torch.float32)
        adapter.native.readout_dtype = 'float32'
        adapter.native.fast = None
        adapter.native.permutation_average = False
        adapter.native.max_length = adapter.limit
        adapter.native.prompt = 'd3'
        adapter.native.temperature = adapter.preparer.config['temperature']
        adapter.native.attention_mode = 'noncausal_full_attention'
        adapter.native.pad_id = adapter.tokenizer.pad_token_id
        adapter.model = SimpleNamespace(parameters=lambda: iter(parameters), buffers=lambda: iter([]),
            config=SimpleNamespace(text_config=SimpleNamespace(_attn_implementation='sdpa')))
        return adapter, parameters

    def test_native_mixed_precision_passport(self):
        adapter, params = self.precision_fixture()
        adapter.check_device()
        passport = validate_precision(adapter.metadata['precision'])
        self.assertEqual(passport['version'], 2)
        self.assertEqual(passport['tensor_roles']['backbone']['dtypes'], ['bfloat16'])
        self.assertEqual(passport['tensor_roles']['readout']['dtypes'], ['float32'])
        before = deepcopy(passport)
        adapter.check_device()
        self.assertEqual(adapter.metadata['precision'], before)
        params[0].dtype = torch.float32
        with self.assertRaisesRegex(ValueError, 'precision'):
            adapter.check_device()

    def test_readout_cpu_offload_and_autocast_rejected(self):
        adapter, _ = self.precision_fixture()
        adapter.native.readout.device = torch.device('cpu')
        with self.assertRaisesRegex(ValueError, 'CUDA device'):
            adapter.check_device()
        adapter.native.readout.device = adapter.device
        with patch('torch.is_autocast_enabled', return_value=True):
            with self.assertRaisesRegex(ValueError, 'precision'):
                adapter.check_device()


if __name__ == '__main__':
    unittest.main()
