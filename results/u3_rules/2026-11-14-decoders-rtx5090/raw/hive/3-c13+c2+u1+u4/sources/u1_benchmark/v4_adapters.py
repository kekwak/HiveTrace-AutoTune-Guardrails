"""Config-driven native batch adapters. Historical constructors are never called."""
from copy import deepcopy
from pathlib import Path

from .adapters import Builtin, parse_generated
from .batch_adapters import BatchAdapter, _payloads, _responses, generated_ids
from .core import digest
from .messages import UnsupportedInput, select
from .v4_preparation import create_preparer, external_module, validate_settings
from . import v4_models


class NativeAdapter:
    # These diagnostics depend on actual observed tensors, not model registries.
    verify_batch = BatchAdapter.verify_batch
    reject_batch_over_limit = BatchAdapter.reject_batch_over_limit
    _hf_inputs = BatchAdapter._hf_inputs

    def __init__(self, spec, directory):
        import torch
        self.spec = validate_settings(spec)
        self.name, self.directory = self.spec['adapter'], Path(directory)
        self.limit = self.spec['preparation']['limit']
        self.backend = self.spec['execution']['runtime']
        self.device = torch.device('cuda:0')
        self._builtin, self.adapter = True, self
        self._compile_count, self._prepared, self._in_request = 0, False, False
        self._restore_encoder = self._restore_processor = self._restore_laya_encoding = None
        self._backend_installed = self.backend == 'eager'
        self.metadata = {'schema_version': 4, 'adapter': self.name, 'variant_sha256': digest(self.spec),
                         'backend_requested': self.backend, 'dtype': self.spec['execution']['dtype']}
        self.preparer = create_preparer(self.spec)
        self.tokenizer = self.preparer.tokenizer
        self.metadata['preparation'] = self.preparer.metadata
        self.legacy = None
        self.model = None
        try:
            self._configure_padding()
            v4_models.load_model(self)
            self._record_labels()
            self.target = self.model.encoder if self.name == 'gliner' else self.model
            if self.name == 'laya':
                Builtin._install_laya_input_guard(self)
            self._check_checkpoint()
            self.check_device()
        except BaseException:
            self.close()
            raise

    def _record_labels(self):
        if self.name not in ('koala', 'llama'):
            self.labels = list({'halo': ('safe', 'unsafe'), 'gliner': ('safe', 'unsafe'),
                               'hive': ('safe', 'unsafe'), 'open': ('safe', 'unsafe'),
                               'laya': ('A', 'B'), 'shield': ('Yes', 'No')}[self.name])
        self.metadata['labels'] = list(self.labels)

    def _configure_padding(self):
        side = 'left' if self.name in ('halo', 'hive', 'open', 'shield') else 'right'
        self.tokenizer.padding_side = side
        if self.tokenizer.pad_token_id is None and side == 'left':
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Decoder tokenizer has neither PAD nor EOS')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if type(self.tokenizer.pad_token_id) is not int or self.tokenizer.pad_token_id < 0:
            raise ValueError('Native tokenizer requires a nonnegative integer pad_token_id')
        if self.name == 'gliner' and self.tokenizer.pad_token_id != 0:
            raise ValueError('Native GLiNER collator requires pad_token_id=0')
        self.batch_policy = {'padding': 'longest', 'padding_side': side, 'truncation': False,
                             'sort_by_length': False, 'implementation': 'native',
                             'pad_token_id': self.tokenizer.pad_token_id, 'generation_eos_in_count': True}
        self.metadata['batch_policy'] = self.batch_policy

    def _check_checkpoint(self):
        target = self.model.encoder if self.name == 'laya' else self.target
        config = getattr(target, 'config', None)
        config = getattr(config, 'text_config', config)
        attention = getattr(config, '_attn_implementation', None)
        requested = self.spec['execution']['attention']
        if not isinstance(attention, str) or not attention or (requested != 'native' and attention != requested):
            raise ValueError('Loaded checkpoint did not select requested attention implementation')
        previous = self.metadata.get('attention')
        if previous is not None and attention != previous:
            raise ValueError('Loaded attention implementation changed')
        maximum = getattr(config, 'max_position_embeddings', None)
        if isinstance(maximum, int) and maximum < self.limit:
            raise ValueError('Configured input limit exceeds checkpoint positional capacity')
        self.metadata['attention'] = attention
        self.metadata['linear_attention'] = self.spec['execution']['linear_attention']

    @property
    def compile_count(self):
        return self._compile_count

    def check_device(self):
        import torch
        if 'attention' in self.metadata:
            self._check_checkpoint()
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        if any(t.device != self.device for t in tensors):
            raise ValueError('Unexpected CPU/offloaded tensor')
        parameters = list(self.model.parameters())
        wanted = 'float32' if self.name == 'laya' else self.spec['execution']['dtype']
        if any(p.is_floating_point() and p.dtype != getattr(torch, wanted) for p in parameters):
            raise ValueError('Physical parameter precision differs from requested policy')
        if self.name != 'laya' and torch.is_autocast_enabled('cuda'):
            raise ValueError('Explicit precision cannot run inside ambient CUDA autocast')
        precision = {'policy': 'native_sdk' if self.name == 'laya' else 'explicit',
                     'parameter_dtypes': sorted({str(p.dtype).removeprefix('torch.') for p in parameters}),
                     'autocast_enabled': False, 'autocast_dtype': None}
        self.metadata['buffer_dtypes'] = sorted({str(t.dtype).removeprefix('torch.') for t in self.model.buffers()})
        if self.name == 'laya':
            self._check_laya_runtime()
            precision.update(autocast_enabled=True, autocast_dtype=str(self.agent.dtype).removeprefix('torch.'))
        self.metadata['precision'] = precision

    def _check_laya_runtime(self):
        if self.agent.device.type != 'cuda' or self.agent.device.index not in (None, self.device.index):
            raise ValueError('Laya SDK CPU fallback/device changed')
        if not self.agent.amp_enabled or self.agent.dtype != self._laya_autocast_dtype:
            raise ValueError('Laya native autocast changed')

    def legacy_text(self, payload):
        return select(self.name, payload)['effective_messages'][0]['content']

    def index(self, payload):
        value = self.preparer.index(payload)
        if value.get('total_tokens') is None:
            raise UnsupportedInput(value['reason'])
        return value

    def request_batch(self, payloads):
        import torch
        payloads = _payloads(payloads)
        if not self._backend_installed:
            raise ValueError('Batch backend has not been prepared')
        if (self.tokenizer.pad_token_id != self.batch_policy['pad_token_id'] or
                self.tokenizer.padding_side != self.batch_policy['padding_side']):
            raise ValueError('Native batch padding policy changed')
        with torch.inference_mode(), v4_models.attention_context(self.spec['execution']):
            self._in_request = True
            try:
                values = self._request(payloads)
                return _responses(values, len(payloads))
            finally:
                self._in_request = False
                if self.name == 'laya':
                    self._check_laya_runtime()

    def _request(self, payloads):
        import torch
        if self.name == 'gliner':
            values = self.gliner.scores([self.legacy_text(t) for t in payloads])
            return [{**value, 'scores': {'safe': value['p_safe'], 'unsafe': value['p_unsafe']},
                     'score_type': 'probabilities'} for value in values]
        if self.name == 'laya':
            # Do not call preparer here: SDK guard observes the one native encoding.
            values = self.agent.predict_batch([self.legacy_text(t) for t in payloads],
                self.spec['preparation']['questions'], batch_size=len(payloads), sort_by_length=False)
            answers = [v['answers']['danger'] for v in values]
            if any(v['choice'] not in ('A', 'B') for v in answers):
                raise ValueError('Invalid native Laya choice')
            return [{'label': v['choice'], 'scores': v['probabilities'], 'score_type': 'probabilities'} for v in answers]
        inputs = self._hf_inputs(payloads)
        if self.name == 'halo':
            from experiments.adapters import score_from_logits
            with v4_models.halo_context(self):
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1, :]
            values = [score_from_logits(row[self.label_ids], self.spec['scoring']['threshold']) for row in logits]
            return [{**v, 'scores': {'safe': v['p_safe'], 'unsafe': v['p_unsafe']},
                     'score_type': 'probabilities'} for v in values]
        if self.name in ('koala', 'llama', 'shield'):
            if self.name == 'shield':
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, self.label_ids].float()
                labels = ['Yes', 'No']
            else:
                logits = self.model(**inputs).logits.float()
                labels = self.labels
            if logits.shape != (len(payloads), len(labels)) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Unexpected or nonfinite native logits')
            probabilities = logits.softmax(-1).cpu().tolist()
            return [{'label': ('Yes' if p[0] > p[1] else 'No') if self.name == 'shield'
                     else labels[p.index(max(p))], 'scores': dict(zip(labels, p)),
                     'score_type': 'probabilities'} for p in probabilities]
        output = self.model.generate(**inputs, do_sample=False,
            max_new_tokens=self.spec['scoring']['max_new_tokens'], pad_token_id=self.tokenizer.pad_token_id)
        eos = self.model.generation_config.eos_token_id
        answers = []
        for row in output[:, inputs['input_ids'].shape[1]:].cpu().tolist():
            ids = generated_ids(row, eos)
            answers.append(parse_generated(self.tokenizer.decode(ids, skip_special_tokens=True), self.name, len(ids)))
        return answers

    def prepare_batch_backend(self, batches):
        batches = [_payloads(batch) for batch in batches]
        if not batches:
            raise ValueError('No batches supplied for backend preparation')
        if self.backend == 'compile' and not self._backend_installed:
            v4_models.install_compile(self)
        if self.backend == 'compile':
            shapes = {}
            payload_shapes = {}
            for batch in batches:
                key = digest(batch)
                if key not in payload_shapes:
                    payload_shapes[key] = tuple(self.gliner.prepare([self.legacy_text(t) for t in batch])['input_ids'].shape)
                shapes.setdefault(payload_shapes[key], batch)
            for batch in shapes.values():
                self.request_batch(batch)
        elif self.backend == 'ort':
            v4_models.prepare_ort(self, batches)
        if self.backend == 'compile' and not self.compile_count:
            raise ValueError('Compiled encoder never executed')
        self._prepared = True
        self.metadata.update(prepared_compile_count=self.compile_count, backend_actual=self.backend)
        self.metadata['backend'] = {'runtime': self.backend, 'attention': self.metadata.get('attention'),
                                    'attention_requested': self.spec['execution']['attention'],
                                    'linear_attention': self.spec['execution']['linear_attention'],
                                    'compile_count': self.compile_count}
        return {'compile_count': self.compile_count, 'runtime': self.backend}

    def prove_batch_backend(self, batch):
        import torch
        batch = _payloads(batch)
        self.verify_batch(batch, [self.index(t) for t in batch])
        if self.name == 'halo' and self.spec['execution']['linear_attention'] != 'native':
            from followup.team_halo import dispatch, check_dispatch
            from .models import quiet_dispatch
            inputs = self._hf_inputs(batch)
            originals = {k: getattr(self.qwen, k) for k in self.implementations}
            self.calls.clear()
            with torch.inference_mode():
                with dispatch(self.qwen, self.implementations, self.calls):
                    expected = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits.detach().clone()
                evidence = check_dispatch(self.calls, 'linear' if self.spec['execution']['linear_attention'] == 'fla' else 'none')
                with quiet_dispatch(self.qwen, self.quiet_implementations):
                    actual = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits
            if not torch.equal(actual, expected) or any(getattr(self.qwen, k) is not v for k, v in originals.items()):
                raise ValueError('Halo diagnostic/quiet dispatch differs or global dispatch was not restored')
            self.calls.clear()
            self.metadata['batch_dispatch'] = evidence
        if self.backend == 'compile' and not self.compile_count:
            raise ValueError('Compiled encoder never executed')
        if self.backend == 'ort' and not getattr(self, '_ort_ready', False):
            raise ValueError('ORT provider/shape validation missing')
        proof = v4_models.profile_backend(self, batch)
        if self.name == 'halo' and self.spec['execution']['linear_attention'] != 'native':
            proof['dispatch'] = deepcopy(self.metadata['batch_dispatch'])
        if self.backend == 'ort':
            proof['providers'] = deepcopy(self.metadata['ort_evidence'])
            proof['checked_shapes'] = deepcopy(self.metadata['batch_ort_checked_shapes'])
            proof['profile_active_during_latency'] = False
        if self.backend == 'compile':
            proof['compile'] = {**self.metadata['compile'], 'compile_count': self.compile_count}
        self.metadata['backend_proof'] = proof
        self.metadata['batch_backend_proof'] = proof
        return proof

    def boundary(self, bucket_max):
        if type(bucket_max) is not int or not 0 < bucket_max <= self.limit:
            raise ValueError('Bucket edge must be inside model input limit')

        def find(target):
            for unit in ('a ', 'hello ', 'text ', '0 '):
                lo, hi = 0, target * 2
                while lo <= hi:
                    mid = (lo + hi) // 2
                    value = unit * mid + '.'
                    n = self.index(value)['total_tokens']
                    if n == target:
                        return value
                    if n < target:
                        lo = mid + 1
                    else:
                        hi = mid - 1
            raise ValueError('Could not construct exact token boundary')

        return find(bucket_max), find(self.limit + 1)

    def size(self):
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
        checkpoint = {str(p.relative_to(self.weight_path)): p.stat().st_size for p in self.weight_path.rglob('*')
                      if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pt', '.pth')}
        if not checkpoint:
            raise ValueError('Loaded checkpoint weight files missing')
        exports = {p.name: p.stat().st_size for p in self.directory.glob('*')
                   if p.is_file() and (p.suffix in ('.onnx', '.data') or '.onnx.' in p.name)}
        return {'unique_parameter_buffer_storage_bytes': sum(storages.values()),
                'dtypes': sorted({str(t.dtype) for t in tensors}), 'checkpoint_files_bytes': checkpoint,
                'export_files_bytes': exports, 'original_encoder_retained': getattr(self, 'ort_session', None) is not None}

    def close(self):
        for name in ('_restore_laya_encoding', '_restore_processor', '_restore_encoder'):
            restore = getattr(self, name, None)
            if restore is not None:
                restore()
                setattr(self, name, None)
        self.ort_session = None
        if getattr(self, '_compile_count', 0):
            import torch
            torch._dynamo.reset()
        if hasattr(self, 'gliner'):
            self.gliner.model = None
        if hasattr(self, 'agent'):
            self.agent.model = None
        self.target = self.model = None
        self._backend_installed = False


def load_adapter(spec, directory):
    if not isinstance(spec.get('adapter'), dict):
        return NativeAdapter(spec, directory)
    adapter = external_module(spec).create(deepcopy(spec), Path(directory))
    methods = ('index', 'request_batch', 'verify_batch', 'prepare_batch_backend', 'prove_batch_backend',
               'reject_batch_over_limit', 'boundary', 'size', 'close')
    try:
        if any(not callable(getattr(adapter, method, None)) for method in methods):
            raise ValueError('External adapter is missing a required v4 batch method')
        if not isinstance(getattr(adapter, 'metadata', None), dict) or type(getattr(adapter, 'compile_count', None)) is not int:
            raise ValueError('External adapter requires metadata and integer compile_count')
        policy = getattr(adapter, 'batch_policy', adapter.metadata.get('batch_policy'))
        if (not isinstance(policy, dict) or policy.get('padding') != 'longest'
                or policy.get('padding_side') not in ('left', 'right') or policy.get('truncation') is not False
                or policy.get('sort_by_length') is not False or type(policy.get('pad_token_id')) is not int
                or policy['pad_token_id'] < 0):
            raise ValueError('External adapter requires explicit native batch padding policy')
    except BaseException:
        if callable(getattr(adapter, 'close', None)):
            adapter.close()
        raise
    return adapter
