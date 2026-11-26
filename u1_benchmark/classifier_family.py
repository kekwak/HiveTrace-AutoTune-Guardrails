"""HF sequence classifiers with native logits, compile and ORT CUDA execution."""
import inspect
import json
from pathlib import Path

from .core import digest, sha, write
from .family_adapter import FamilyAdapter
from .family_preparation import ClassificationPreparer, validate_classifier
from .v4_preparation import source_path


def classify(logits, labels, scoring):
    import torch
    if logits.ndim != 2 or logits.shape[1] != len(labels) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Unexpected or nonfinite classification logits')
    probabilities = logits.float().softmax(-1).cpu().tolist()
    answers = []
    for p in probabilities:
        if scoring['method'] == 'softmax_argmax':
            label = labels[p.index(max(p))]
        else:
            positive = labels.index(scoring['positive_label'])
            label = labels[positive if p[positive] >= scoring['threshold'] else 1 - positive]
        answers.append({'label': label, 'scores': dict(zip(labels, p)), 'score_type': 'probabilities'})
    return answers


class ClassificationAdapter(FamilyAdapter):
    def __init__(self, spec, directory):
        from transformers import AutoModelForSequenceClassification
        from .v4_models import hf_load_options
        spec = validate_classifier(spec)
        preparer = ClassificationPreparer(spec)
        preparer.tokenizer.padding_side = 'right'
        weights = source_path(spec['weights'])
        model = None
        try:
            model, loading = AutoModelForSequenceClassification.from_pretrained(str(weights),
                **hf_load_options(spec['execution']))
            if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError('Classification checkpoint loading mismatch')
            super().__init__(spec, directory, preparer, model.to('cuda:0').eval())
            self.weight_path = weights
            self.labels = [str(model.config.id2label[i]) for i in range(model.config.num_labels)]
            expected = spec['scoring'].get('labels')
            if (len(self.labels) < 2 or len(set(self.labels)) != len(self.labels) or
                    (expected is not None and self.labels != expected)):
                raise ValueError('Loaded native labels differ from configuration')
            params = inspect.signature(model.forward).parameters
            if not {'input_ids', 'attention_mask'} <= params.keys():
                raise ValueError('Classifier requires a native input_ids/attention_mask forward')
            self._native_forward = model.forward
            self._ort_shapes = set()
            inventory = {str(p.relative_to(weights)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted(weights.rglob('*')) if p.is_file() and p.suffix in ('.safetensors', '.bin')}
            if not inventory:
                raise ValueError('Loaded checkpoint weight files missing')
            self.metadata.update(labels=self.labels, artifacts={'weight_files': inventory,
                'weight_inventory_sha256': digest(inventory)}, linear_attention='reference')
            self.check_device()
        except BaseException:
            if hasattr(self, 'preparer'):
                self.close()
            else:
                preparer.close()
            raise

    def check_device(self):
        super().check_device()
        requested = self.spec['execution']['attention']
        actual = getattr(self.model.config, '_attn_implementation', None)
        if not isinstance(actual, str) or (requested != 'native' and requested != actual):
            raise ValueError('Classifier attention implementation mismatch')
        if self.metadata.get('attention', actual) != actual:
            raise ValueError('Classifier attention changed')
        capacity = getattr(self.model.config, 'max_position_embeddings', None)
        if isinstance(capacity, int) and self.limit > capacity:
            raise ValueError('Input limit exceeds classifier positional capacity')
        self.metadata['attention'] = actual

    def forward_batch(self, payloads):
        inputs = self._hf_inputs(payloads)
        return classify(self.model(**inputs).logits, self.labels, self.spec['scoring'])

    def _shapes(self, batches):
        shapes = {}
        for batch in batches:
            inputs = self._hf_inputs(batch)
            shapes.setdefault(tuple(inputs['input_ids'].shape), batch)
        return shapes

    def _install_compile(self):
        import torch
        from torch._dynamo.backends.registry import lookup_backend
        from transformers.modeling_outputs import SequenceClassifierOutput
        if torch._dynamo.config.suppress_errors or self._restore_forward is not None:
            raise ValueError('Compile requires unpatched classifier and no silent fallback')
        original = self._native_forward
        backend = lookup_backend(self.spec['execution']['compile_backend'])

        def counted(graph, inputs):
            self._compile_count += 1
            return backend(graph, inputs)

        compiled = torch.compile(lambda ids, mask: original(input_ids=ids, attention_mask=mask).logits,
                                 backend=counted, dynamic=True, fullgraph=True)

        def forward(input_ids=None, attention_mask=None, **kwargs):
            if kwargs:
                raise ValueError('Unexpected compiled classifier arguments')
            return SequenceClassifierOutput(logits=compiled(input_ids, attention_mask))

        self.model.forward = forward
        self._restore_forward = lambda: setattr(self.model, 'forward', original)
        self._backend_installed = True
        self.metadata['compile'] = {'backend': 'inductor', 'mode': 'default', 'dynamic': True, 'fullgraph': True}

    def prepare_batch_backend(self, batches):
        if not batches:
            raise ValueError('No classifier backend preparation inputs')
        if self.backend == 'compile':
            if not self._backend_installed:
                self._install_compile()
            for batch in self._shapes(batches).values():
                self.request_batch(batch)
            if not self.compile_count:
                raise ValueError('Compiled classifier never executed')
        elif self.backend == 'ort':
            self._prepare_ort(batches)
        self.metadata.update(backend_actual=self.backend, prepared_compile_count=self.compile_count,
            backend={'runtime': self.backend, 'attention': self.metadata['attention'],
                     'compile_count': self.compile_count})
        return {'runtime': self.backend, 'compile_count': self.compile_count}

    def prove_batch_backend(self, batch):
        proof = super().prove_batch_backend(batch)
        if self.backend == 'compile':
            if not self.compile_count:
                raise ValueError('No compiled classifier execution')
            proof['compile'] = {**self.metadata['compile'], 'compile_count': self.compile_count}
        if self.backend == 'ort':
            if self.ort_session is None or not self._ort_shapes:
                raise ValueError('No validated classifier ORT execution')
            proof.update(providers=self.metadata['ort_evidence'],
                         checked_shapes=self.metadata['batch_ort_checked_shapes'],
                         profile_active_during_latency=False)
        return proof

    def _binding(self, session, allowed_shapes=None):
        def forward(input_ids=None, attention_mask=None, **kwargs):
            import numpy as np
            import torch
            from transformers.modeling_outputs import SequenceClassifierOutput
            if kwargs or input_ids is None or attention_mask is None:
                raise ValueError('ORT classifier needs explicit IDs and masks only')
            shape = tuple(input_ids.shape)
            if (len(shape) != 2 or tuple(attention_mask.shape) != shape or not 0 < shape[1] <= self.limit
                    or (allowed_shapes is not None and shape not in allowed_shapes)):
                raise ValueError('Unvalidated ORT classifier shape')
            inputs = {'input_ids': input_ids.contiguous(), 'attention_mask': attention_mask.contiguous()}
            if any(t.dtype != torch.int64 or t.device != self.device for t in inputs.values()):
                raise ValueError('ORT classifier input device/dtype changed')
            output = torch.empty((shape[0], len(self.labels)), device=self.device, dtype=torch.float32)
            binding = session.io_binding()
            for name, tensor in inputs.items():
                binding.bind_input(name, 'cuda', 0, np.int64, shape, tensor.data_ptr())
            binding.bind_output('logits', 'cuda', 0, np.float32, tuple(output.shape), output.data_ptr())
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            return SequenceClassifierOutput(logits=output)
        return forward

    def _prepare_ort(self, batches):
        import onnx
        import onnxruntime as ort
        import torch
        from followup.onnx_service import prove_metadata_nodes, validate_events
        from followup.team_smoke import preload_ort
        shapes = {s: b for s, b in self._shapes(batches).items() if s not in self._ort_shapes}
        if not shapes:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        preload_ort(self.directory)
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise ValueError('ORT CUDA provider unavailable')
        original, session = self._native_forward, None
        path = self.directory / 'classifier.onnx'
        try:
            self.model.forward = original
            if self.ort_session is None:
                model = self.model

                class Export(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.classifier = model

                    def forward(self, input_ids, attention_mask):
                        return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

                sample = self._hf_inputs(next(iter(shapes.values())))
                torch.onnx.export(Export().eval(), (sample['input_ids'], sample['attention_mask']), str(path),
                    input_names=['input_ids', 'attention_mask'], output_names=['logits'],
                    dynamic_axes={'input_ids': {0: 'batch', 1: 'tokens'},
                                  'attention_mask': {0: 'batch', 1: 'tokens'}, 'logits': {0: 'batch'}},
                    opset_version=17, dynamo=False)
            providers = [('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0}), 'CPUExecutionProvider']
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_profiling = True
            options.profile_file_prefix = str(self.directory / 'classifier-shapes')
            options.optimized_model_filepath = str(self.directory / 'classifier.optimized.onnx')
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            session.disable_fallback()
            proof = prove_metadata_nodes(onnx.load(options.optimized_model_filepath, load_external_data=False).graph)
            write(self.directory / 'metadata_dataflow.json', proof)
            self.model.forward = self._binding(session)
            self._backend_installed = True
            for batch in shapes.values():
                self.verify_batch(batch, [self.index(v) for v in batch])
            torch.cuda.synchronize()
            profile = Path(session.end_profiling())
            session = None
            evidence = validate_events(json.loads(profile.read_text()), {r['name'] for r in proof['proven']})
            if not evidence['passed']:
                raise ValueError('Classifier ORT violates metadata-cpu-v1')
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_profiling = False
            measured = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            measured.disable_fallback()
            if 'CUDAExecutionProvider' not in measured.get_providers():
                raise ValueError('Measured classifier ORT session has no CUDA provider')
            self._ort_shapes.update(shapes)
            self.model.forward = self._binding(measured, self._ort_shapes)
            self._restore_forward = lambda: setattr(self.model, 'forward', original)
            self.ort_session = measured
            # Optimized graph records actual fusion products, not requested optimization level alone.
            ops = sorted({node.op_type for node in onnx.load(str(self.directory / 'classifier.optimized.onnx'),
                                                            load_external_data=False).graph.node})
            self.metadata.update(ort_evidence=evidence, ort_profile_active_during_latency=False,
                batch_ort_checked_shapes=[list(s) for s in sorted(self._ort_shapes)],
                ort={'providers': measured.get_providers(), 'cpu_policy': 'metadata-cpu-v1',
                     'optimized_op_types': ops, 'original_encoder_retained': True})
            for batch in shapes.values():
                self.request_batch(batch)
        except BaseException:
            self.model.forward = original
            self.ort_session, self._restore_forward = None, None
            self._backend_installed = False
            self._ort_shapes.clear()
            if session is not None:
                try:
                    session.end_profiling()
                except Exception:
                    pass
            raise
