import functools
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from followup.team_halo import FUNCTIONS, TeamHaloAdapter, reference_functions, dispatch, check_dispatch


class TeamHaloDispatchTests(unittest.TestCase):
    def module(self):
        qwen = SimpleNamespace(__name__='transformers.models.qwen3_5.modeling_qwen3_5')
        for name in FUNCTIONS:
            def reference(value, scale=1):
                return value * scale
            reference.__name__, reference.__module__ = name, qwen.__name__
            @functools.wraps(reference)
            def automatic(*args, **kwargs):
                raise AssertionError('Implicit optional kernel must not execute')
            setattr(qwen, name, automatic)
        return qwen

    def test_unwrap_and_call_reference_then_restore(self):
        qwen, calls = self.module(), []
        original = qwen.torch_chunk_gated_delta_rule
        with dispatch(qwen, reference_functions(qwen), calls):
            self.assertEqual(qwen.torch_chunk_gated_delta_rule(3, scale=2, ignored=True), 6)
            self.assertEqual(qwen.causal_conv1d_fn(5), 5)
        self.assertIs(qwen.torch_chunk_gated_delta_rule, original)
        self.assertEqual(check_dispatch(calls, 'none')['linear_calls'], 1)
        with self.assertRaises(RuntimeError):
            check_dispatch(calls, 'linear')

    def test_fla_only_and_error_restoration(self):
        qwen, calls = self.module(), []
        originals = {name: getattr(qwen, name) for name in FUNCTIONS}
        implementations = reference_functions(qwen)
        def fast(value):
            return value + 1
        fast.__module__ = 'fla.ops.gated_delta_rule'
        implementations['torch_chunk_gated_delta_rule'] = fast
        with self.assertRaises(ValueError):
            with dispatch(qwen, implementations, calls):
                self.assertEqual(qwen.torch_chunk_gated_delta_rule(3), 4)
                qwen.causal_conv1d_fn(5)
                raise ValueError('forward failed')
        for name, fn in originals.items():
            self.assertIs(getattr(qwen, name), fn)
        check_dispatch(calls, 'linear')
        with self.assertRaises(RuntimeError):
            check_dispatch(calls, 'none')

    def test_unknown_dispatch_cannot_pass(self):
        for calls in ([], [{'slot': 'torch_chunk_gated_delta_rule', 'function': 'fla.fake'}]):
            with self.assertRaises(RuntimeError):
                check_dispatch(calls, 'linear')
        qwen = self.module()
        qwen.causal_conv1d_fn = lambda x: x
        with self.assertRaises(RuntimeError):
            reference_functions(qwen)

    def test_real_5201_adapter_with_tiny_local_checkpoint(self):
        import transformers
        if transformers.__version__ != '5.20.1' or os.environ.get('USE_HUB_KERNELS') != 'NO':
            self.skipTest('Run separately with Transformers 5.20.1 and USE_HUB_KERNELS=NO')
        import torch
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration as Model
        from followup.team_smoke import read
        torch.set_num_threads(1)
        config = Qwen3_5Config(
            text_config=dict(vocab_size=16, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
                             linear_key_head_dim=8, linear_value_head_dim=8,
                             layer_types=['linear_attention', 'full_attention']),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                               out_hidden_size=32, num_position_embeddings=16))
        class Tokenizer:
            pad_token_id, chat_template = 0, 'CPU fixture'
            def apply_chat_template(self, *args, **kwargs):
                return 'Prefix '
            def encode(self, value, **kwargs):
                return {'safe': [1], 'unsafe': [2]}.get(value, [3, 4])
            def decode(self, ids, **kwargs):
                return 'Prefix Safety: ' + ('safe' if ids[-1] == 1 else 'unsafe')
            def __call__(self, values, **kwargs):
                return {'input_ids': torch.tensor([[3, 4, 5, 6]] * len(values)),
                        'attention_mask': torch.ones(len(values), 4, dtype=torch.long)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            Model(config).save_pretrained(path / 'checkpoint')
            card = path / 'README.md'
            card.write_text('```python\nGUARD_SYSTEM_PROMPT = "CPU fixture"\n```')
            loader = Model.from_pretrained
            def load(_name, **kwargs):
                kwargs.pop('revision')
                return loader(path / 'checkpoint', **kwargs)
            with patch('huggingface_hub.hf_hub_download', return_value=str(card)), \
                    patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                    patch.object(Model, 'from_pretrained', side_effect=load):
                adapter = TeamHaloAdapter({'dtype': 'float32', 'fast': 'none'}, path, device='cpu')
                try:
                    with torch.inference_mode():
                        scores = adapter.scores(['CPU fixture.'])
                    self.assertEqual(len(scores), 1)
                    check_dispatch(adapter.calls, 'none')
                    self.assertFalse(read(path / 'loading.json')['missing_keys'])
                finally:
                    adapter.close()
