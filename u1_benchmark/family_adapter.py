"""Reusable measurement boundary for native single-call batch implementations."""
from pathlib import Path

from .batch_adapters import BatchAdapter, _payloads, _responses
from .core import digest
from .v4_adapters import NativeAdapter


class FamilyAdapter:
    """Subclass forward_batch; expose the module that receives real IDs and masks.

    External modules keep create_preparer/create. They may use this helper instead
    of copying indexing, input proofs, limit checks and lifecycle bookkeeping.
    """
    verify_batch = BatchAdapter.verify_batch
    reject_batch_over_limit = BatchAdapter.reject_batch_over_limit
    _hf_inputs = BatchAdapter._hf_inputs

    def __init__(self, spec, directory, preparer, model, *, device='cuda:0', extra_tensors=None):
        import torch
        self.spec, self.directory = spec, Path(directory)
        self.name = spec['adapter'] if isinstance(spec['adapter'], str) else 'external'
        self.preparer, self.tokenizer = preparer, preparer.tokenizer
        self.model = self.target = model
        self.device, self.limit = torch.device(device), preparer.limit
        self.adapter, self._builtin = self, True
        self._compile_count, self._restore_forward = 0, None
        self.backend = spec['execution']['runtime']
        self._backend_installed = self.backend == 'eager'
        self.extra_tensors = dict(extra_tensors or {})
        self.ort_session = None
        self.batch_policy = {'padding': 'longest', 'padding_side': self.tokenizer.padding_side,
            'pad_token_id': self.tokenizer.pad_token_id, 'truncation': False,
            'sort_by_length': False, 'implementation': 'native'}
        if type(self.batch_policy['pad_token_id']) is not int or self.batch_policy['pad_token_id'] < 0:
            raise ValueError('Native batching requires a valid pad token')
        self.metadata = {'adapter': self.name, 'variant_sha256': digest(spec),
            'dtype': spec['execution']['dtype'], 'preparation': preparer.metadata,
            'batch_policy': self.batch_policy, 'backend_requested': self.backend}
        from .capabilities import describe
        self.metadata['capabilities'] = describe(spec)
        self._expected_precision = None

    @property
    def compile_count(self):
        return self._compile_count

    def index(self, payload):
        from .messages import UnsupportedInput
        value = self.preparer.index(payload)
        if value.get('total_tokens') is None:
            raise UnsupportedInput(value['reason'])
        return value

    def boundary(self, bucket_max):
        from .messages import UnsupportedInput
        # Only synthetic boundary probes choose a role format; corpus inputs stay unchanged.
        factories = (
            lambda text: text,
            lambda text: [{'role': 'assistant', 'content': text}],
            lambda text: [{'role': 'user', 'content': 'Boundary probe.'},
                          {'role': 'assistant', 'content': text}],
        )
        for factory in factories:
            try:
                self.index(factory('Boundary probe.'))
            except UnsupportedInput:
                continue
            return NativeAdapter.boundary(self, bucket_max, payload_factory=factory)
        raise UnsupportedInput('No supported role format for boundary diagnostics')

    def request(self, payload):
        return self.request_batch([payload])[0]

    def request_batch(self, payloads):
        import torch
        from .v4_models import attention_context
        if not self._backend_installed:
            raise ValueError('Backend has not been prepared')
        if (self.tokenizer.padding_side != self.batch_policy['padding_side'] or
                self.tokenizer.pad_token_id != self.batch_policy['pad_token_id']):
            raise ValueError('Native padding policy changed')
        payloads = _payloads(payloads)
        with torch.inference_mode(), attention_context(self.spec['execution']):
            return _responses(self.forward_batch(payloads), len(payloads))

    def forward_batch(self, payloads):
        raise NotImplementedError('Provide a native batched forward and ordered response parser')

    def check_device(self):
        import torch
        if self.model is None or not hasattr(self.model, 'parameters'):
            raise ValueError('Actual model tensors are required for precision/device proof')
        parameters = list(self.model.parameters())
        tensors = parameters + list(self.model.buffers()) + list(self.extra_tensors.values())
        if not parameters or any(t.device != self.device for t in tensors):
            raise ValueError('Missing parameters or unexpected CPU/offloaded tensors')
        wanted = getattr(torch, self.spec['execution']['dtype'])
        if any(t.is_floating_point() and t.dtype != wanted for t in parameters):
            raise ValueError('Physical parameter precision differs from configured precision')
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError('Unexpected ambient autocast')
        precision = {'policy': 'explicit', 'parameter_dtypes': sorted({str(t.dtype).removeprefix('torch.')
                     for t in parameters}), 'autocast_enabled': False, 'autocast_dtype': None}
        if self.extra_tensors:
            precision.update(version=2, tensor_roles={name: {'dtypes': [str(t.dtype).removeprefix('torch.')],
                'device_type': t.device.type} for name, t in sorted(self.extra_tensors.items())})
        if self._expected_precision is not None and precision != self._expected_precision:
            raise ValueError('Observed precision changed')
        self._expected_precision = precision
        self.metadata['precision'] = precision

    def prepare_batch_backend(self, batches):
        if self.backend != 'eager':
            raise ValueError('External helper requires explicit backend preparation for non-eager execution')
        if not batches:
            raise ValueError('No backend preparation inputs')
        self.metadata.update(backend_actual='eager', backend={'runtime': 'eager'})
        return {'runtime': 'eager', 'compile_count': 0}

    def prove_batch_backend(self, batch):
        from .v4_models import profile_backend
        self.verify_batch(batch, [self.index(t) for t in batch])
        self.check_device()
        proof = profile_backend(self, batch)
        self.metadata.update(backend_proof=proof, batch_backend_proof=proof)
        return proof

    def size(self):
        if self.model is None:
            raise ValueError('Actual model tensors are required for size accounting')
        tensors = list(self.model.parameters()) + list(self.model.buffers()) + list(self.extra_tensors.values())
        storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
        root = getattr(self, 'weight_path', None)
        files = {} if root is None else {str(p.relative_to(root)): p.stat().st_size for p in root.rglob('*')
            if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pt', '.pth')}
        exports = {p.name: p.stat().st_size for p in self.directory.glob('*')
            if p.is_file() and (p.suffix in ('.onnx', '.data') or '.onnx.' in p.name)}
        return {'unique_parameter_buffer_storage_bytes': sum(storages.values()),
                'dtypes': sorted({str(t.dtype) for t in tensors}), 'checkpoint_files_bytes': files,
                'export_files_bytes': exports, 'original_encoder_retained': self.ort_session is not None}

    def close(self):
        if getattr(self, '_restore_forward', None):
            self._restore_forward()
            self._restore_forward = None
        if getattr(self, '_compile_count', 0):
            import torch
            torch._dynamo.reset()
        self.ort_session = self.target = self.model = None
        self.extra_tensors = {}
        self.preparer.close()
