"""Explicit Halo dispatch for Transformers 5.20.1; historical adapters stay frozen."""

from contextlib import contextmanager
import inspect
import os
from pathlib import Path

from experiments.adapters import HaloAdapter, HALO_ID, HALO_REVISION, guard_prompt
from experiments.common import digest


FUNCTIONS = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
             'causal_conv1d_fn', 'causal_conv1d_update')


def reference_functions(qwen):
    result = {}
    for name in FUNCTIONS:
        fn = inspect.unwrap(getattr(qwen, name))
        if fn.__module__ != qwen.__name__ or fn.__name__ != name:
            raise RuntimeError('Cannot establish upstream reference implementation: ' + name)
        result[name] = fn
    return result


@contextmanager
def dispatch(qwen, implementations, calls):
    """Limit module-global dispatch to one synchronous forward, restoring on errors."""
    originals = {name: getattr(qwen, name) for name in implementations}
    try:
        for name, fn in implementations.items():
            parameters = tuple(inspect.signature(fn).parameters)
            def counted(*args, _fn=fn, _name=name, _parameters=parameters, **kwargs):
                calls.append({'slot': _name, 'function': _fn.__module__ + '.' + _fn.__name__})
                # Match the upstream optional-package wrapper's keyword filtering.
                return _fn(*args, **{k: v for k, v in kwargs.items() if k in _parameters})
            setattr(qwen, name, counted)
        yield
    finally:
        for name, fn in originals.items():
            setattr(qwen, name, fn)


def check_dispatch(calls, fast):
    chunks = [c['function'] for c in calls if c['slot'] == 'torch_chunk_gated_delta_rule']
    conv = [c['function'] for c in calls if c['slot'] == 'causal_conv1d_fn']
    reference = 'transformers.models.qwen3_5.modeling_qwen3_5.'
    if not chunks or not conv or any(not n.startswith(reference) for n in conv):
        raise RuntimeError('Missing linear/conv evidence or non-reference convolution')
    expected = 'fla.' if fast == 'linear' else reference
    if any(not n.startswith(expected) for n in chunks):
        raise RuntimeError('Requested linear backend was not actually called')
    return {'linear_calls': len(chunks), 'conv_calls': len(conv),
            'linear_functions': sorted(set(chunks)), 'conv_functions': sorted(set(conv))}


class TeamHaloAdapter(HaloAdapter):
    def __init__(self, spec, directory, device='cuda:0'):
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        from transformers.integrations import hub_kernels

        if transformers.__version__ != '5.20.1':
            raise RuntimeError('This adapter requires Transformers 5.20.1')
        if os.environ.get('USE_HUB_KERNELS') != 'NO' or hub_kernels._kernels_enabled:
            raise RuntimeError('Disable hub kernels before importing Transformers')
        fast = spec.get('fast', 'none')
        if fast not in ('none', 'linear') or spec.get('attention', 'eager') != 'eager':
            raise ValueError('Smoke supports only reference or linear-only FLA with eager attention')
        self.device, self.spec = torch.device(device), spec
        self.qwen, self.calls = qwen, []
        self.implementations = reference_functions(qwen)
        if fast == 'linear':
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
            self.implementations.update(torch_chunk_gated_delta_rule=chunk_gated_delta_rule,
                                        torch_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule)
        card = Path(hf_hub_download(HALO_ID, 'README.md', revision=HALO_REVISION)).read_text()
        self.prompt = guard_prompt(card)
        self.tokenizer = AutoTokenizer.from_pretrained(HALO_ID, revision=HALO_REVISION, trust_remote_code=False)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        labels = [self.tokenizer.encode(s, add_special_tokens=False) for s in ('safe', 'unsafe')]
        if any(len(x) != 1 for x in labels) or labels[0] == labels[1]:
            raise ValueError('Labels must be distinct single tokens')
        self.label_ids = [x[0] for x in labels]
        prefix = self.render('Library hours?')
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        for label, token in zip(('safe', 'unsafe'), self.label_ids):
            if self.tokenizer.decode(ids + [token], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False) != prefix + label:
                raise ValueError('Label does not continue the official cue')
        self.model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            HALO_ID, revision=HALO_REVISION, dtype=getattr(torch, spec['dtype']),
            attn_implementation='eager', trust_remote_code=False, use_kernels=False,
            output_loading_info=True)
        # No silently random weights after an upstream checkpoint/API migration.
        from .team_smoke import write
        write(Path(directory) / 'loading.json',
              {k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()})
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            self.close()
            raise RuntimeError('Checkpoint loading differs; see loading.json')
        self.model = self.model.to(self.device).eval()
        layers = [m for m in self.model.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]
        if not layers or any(type(m.norm) is not qwen.Qwen3_5RMSNormGated for m in layers):
            self.close()
            raise RuntimeError('Unexpected GatedDeltaNet layers or non-reference gated norm')
        self.contract = {'model': HALO_ID, 'revision': HALO_REVISION, 'prompt': self.prompt,
                         'template': self.tokenizer.chat_template, 'cue': 'Safety: ',
                         'thinking': False, 'threshold': self.threshold, 'label_ids': self.label_ids}
        self.metadata = {'contract_hash': digest(self.contract), 'dtype': spec['dtype'],
                         'attention': self.model.config.text_config._attn_implementation,
                         'norm': sorted({type(m.norm).__name__ for m in layers}),
                         'fast_requested': fast, 'adapter': 'team_halo_transformers_5.20.1',
                         'hub_kernels': False, 'linear_layers': len(layers),
                         'functions': {k: v.__module__ + '.' + v.__name__
                                       for k, v in self.implementations.items()}}
        if self.metadata['attention'] != 'eager':
            self.close()
            raise RuntimeError('Requested eager full attention was not selected')
        self.tensor_call = self._tensor

    def _tensor(self, inputs):
        with dispatch(self.qwen, self.implementations, self.calls):
            return super()._tensor(inputs)
