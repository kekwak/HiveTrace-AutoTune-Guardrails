"""Explicit constructors and backend activation without model-registry mutation."""
from contextlib import nullcontext
import gzip
import json
from pathlib import Path

from .core import digest, sha, write
from .v4_preparation import source_path


def attention_context(execution):
    if execution['attention'] != 'sdpa' or execution['sdpa_backend'] is None:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH if execution['sdpa_backend'] == 'math' else SDPBackend.EFFICIENT_ATTENTION)


def hf_load_options(execution):
    import torch
    options = {'torch_dtype': getattr(torch, execution['dtype']),
               'trust_remote_code': False, 'output_loading_info': True}
    if execution['attention'] != 'native':
        options['attn_implementation'] = execution['attention']
    return options


def halo_context(adapter):
    if adapter.spec['execution']['linear_attention'] == 'native':
        return nullcontext()
    from .models import quiet_dispatch
    return quiet_dispatch(adapter.qwen, adapter.quiet_implementations)


def load_model(adapter):
    import torch
    a = adapter
    e = a.spec['execution']
    a.weight_path = source_path(a.spec['weights'])
    if not a.weight_path.is_dir():
        raise ValueError('Native weights require a checkpoint directory')
    weights = {str(p.relative_to(a.weight_path)): {'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted(a.weight_path.rglob('*'))
               if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pt', '.pth')}
    if not weights:
        raise ValueError('Configured checkpoint contains no native weight files')
    a.metadata['artifacts'] = {'weight_files': weights, 'weight_inventory_sha256': digest(weights)}
    if a.name == 'laya':
        import laya
        a.agent = laya.load(str(a.weight_path), device=str(a.device), fast=False)
        a.agent.tok = a.tokenizer
        if a.agent.cfg != a.preparer.agent.cfg:
            raise ValueError('Laya loaded configuration differs from CPU preparation')
        a.model = a.agent.model
        a._laya_autocast_dtype = a.agent.dtype
        if not a.agent.amp_enabled or a.agent.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('Laya native autocast is unavailable')
        return
    dtype = getattr(torch, e['dtype'])
    if a.name == 'gliner':
        from gliner2 import GLiNER2
        a.model = GLiNER2.from_pretrained(str(a.weight_path), attention_backend='standard',
            quantize=False, compile=False, map_location='cpu').to(device=a.device, dtype=dtype).eval()
        a.model.strict_extraction = True
        a.model.processor = a.preparer.processor
        a.schema = a.preparer.schema
        from experiments.adapters import GlinerAdapter
        # Reuse only arithmetic/native extraction; never the pinned old constructor.
        a.gliner = GlinerAdapter.__new__(GlinerAdapter)
        a.gliner.model, a.gliner.tokenizer, a.gliner.schema = a.model, a.tokenizer, a.schema
        a.gliner.limit, a.gliner.device, a.gliner.threshold = a.limit, a.device, .5
        original = a.model.processor.collate_fn_inference

        def guarded(batch, *args, **kwargs):
            prepared = original(batch, *args, **kwargs)
            if a._in_request and prepared.original_texts != [item[0] for item in batch]:
                raise ValueError('GLiNER processor changed text')
            if a._in_request and any(not 0 < n <= a.limit for n in prepared.attention_mask.sum(-1).tolist()):
                raise ValueError('Input exceeds limit before encoder')
            return prepared

        a.model.processor.collate_fn_inference = guarded
        a._restore_processor = lambda: setattr(a.model.processor, 'collate_fn_inference', original)
        return
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    loader = AutoModelForSequenceClassification if a.name in ('koala', 'llama') else AutoModelForCausalLM
    options = hf_load_options(e)
    if a.name == 'halo':
        import os
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.integrations import hub_kernels
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        from followup.team_halo import reference_functions
        if os.environ.get('USE_HUB_KERNELS') != 'NO' or hub_kernels._kernels_enabled:
            raise ValueError('Halo requires USE_HUB_KERNELS=NO before importing Transformers')
        loader = Qwen3_5ForConditionalGeneration
        options['use_kernels'] = False
        a.qwen, a.calls = qwen, []
        if e['linear_attention'] != 'native':
            a.implementations = reference_functions(qwen)
            if e['linear_attention'] == 'fla':
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
                a.implementations.update(torch_chunk_gated_delta_rule=chunk_gated_delta_rule,
                                        torch_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule)
            from .models import filtered
            a.quiet_implementations = {k: filtered(fn) for k, fn in a.implementations.items()}
    a.model, loading = loader.from_pretrained(str(a.weight_path), **options)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Checkpoint loading has missing/unexpected/mismatched weights')
    a.model = a.model.to(a.device).eval()
    if a.name == 'halo':
        layers = [m for m in a.model.modules() if isinstance(m, a.qwen.Qwen3_5GatedDeltaNet)]
        if not layers or (e['linear_attention'] != 'native' and
                          any(type(m.norm) is not a.qwen.Qwen3_5RMSNormGated for m in layers)):
            raise ValueError('Halo checkpoint requires native GatedDeltaNet/reference gated norm')
        a.metadata['linear_layers'] = len(layers)
    if a.name in ('koala', 'llama'):
        a.labels = [str(a.model.config.id2label[i]) for i in range(a.model.config.num_labels)]
        if len(a.labels) != len(set(a.labels)) or len(a.labels) != (9 if a.name == 'koala' else 2):
            raise ValueError('Incompatible native classification labels')
        a.metadata['labels'] = a.labels
    if a.name in ('halo', 'shield'):
        labels = ('safe', 'unsafe') if a.name == 'halo' else ('Yes', 'No')
        ids = [a.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(row) != 1 for row in ids) or ids[0] == ids[1]:
            raise ValueError('Native labels must be distinct single tokens')
        a.label_ids = [row[0] for row in ids]
        if a.name == 'halo':
            prefix = a.preparer.render('Library hours?')
            prefix_ids = a.tokenizer.encode(prefix, add_special_tokens=False)
            for label, token in zip(labels, a.label_ids):
                if a.tokenizer.decode(prefix_ids + [token], skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False) != prefix + label:
                    raise ValueError('Halo label does not continue the configured cue')


def install_compile(a):
    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from transformers.modeling_outputs import BaseModelOutput
    if torch._dynamo.config.suppress_errors or a._restore_encoder is not None:
        raise ValueError('Compile requires unpatched encoder and no silent fallback')
    original = a.model.encoder.forward
    backend = lookup_backend(a.spec['execution']['compile_backend'])

    def counted(graph, inputs):
        a._compile_count += 1
        return backend(graph, inputs)

    compiled = torch.compile(lambda ids, mask: original(input_ids=ids, attention_mask=mask).last_hidden_state,
                             backend=counted, dynamic=True, fullgraph=True)

    def forward(input_ids=None, attention_mask=None, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError('Unexpected compiled encoder arguments')
        return BaseModelOutput(last_hidden_state=compiled(input_ids, attention_mask))

    a.model.encoder.forward = forward
    a._restore_encoder = lambda: setattr(a.model.encoder, 'forward', original)
    a._backend_installed = True
    a.metadata['compile'] = {'backend': 'inductor', 'mode': 'default', 'dynamic': True, 'fullgraph': True}


def prepare_ort(a, batches):
    """Export/configure per-instance ORT; no global validation-text substitutions."""
    import onnx
    import onnxruntime as ort
    import torch
    from followup.onnx_service import prove_metadata_nodes, validate_events
    from followup.team_smoke import preload_ort
    from .batch_ort import binding_forward
    ready = getattr(a, '_ort_ready', False)
    old_shapes = set(getattr(a, '_ort_shapes', ()))
    shapes_to_batches, payload_shapes = {}, {}
    for batch in batches:
        key = digest(batch)
        if key not in payload_shapes:
            payload_shapes[key] = tuple(a.gliner.prepare([a.legacy_text(t) for t in batch])['input_ids'].shape)
        shape = payload_shapes[key]
        if shape not in old_shapes:
            shapes_to_batches.setdefault(shape, batch)
    if ready and not shapes_to_batches:
        return
    if a._restore_encoder is not None and not ready:
        raise ValueError('ORT requires unpatched encoder')
    a.directory.mkdir(parents=True, exist_ok=True)
    preload_ort(a.directory)
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise ValueError('ORT CUDA provider unavailable')
    encoder = a.model.encoder
    original = a._ort_original if ready else encoder.forward
    path = a.directory / 'encoder.onnx'
    session = None
    try:
        encoder.forward = original
        if not ready:
            class Export(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder

                def forward(self, input_ids, attention_mask):
                    return original(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            sample = a.gliner.prepare([a.legacy_text(t) for t in batches[0]])
            torch.onnx.export(Export().eval(), (sample['input_ids'], sample['attention_mask']), str(path),
                input_names=['input_ids', 'attention_mask'], output_names=['hidden'],
                dynamic_axes={k: {0: 'batch', 1: 'tokens'} for k in ('input_ids', 'attention_mask', 'hidden')},
                opset_version=a.spec['execution']['opset'], dynamo=False)
        providers = [('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0}), 'CPUExecutionProvider']
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(a.directory / 'ort-shapes')
        options.optimized_model_filepath = str(a.directory / 'encoder.optimized.onnx')
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        session.disable_fallback()
        proof = prove_metadata_nodes(onnx.load(options.optimized_model_filepath, load_external_data=False).graph)
        write(a.directory / 'metadata_dataflow.json', proof)
        encoder.forward = binding_forward(session, encoder, set(), lambda: False)
        a._backend_installed = True
        shapes = set()

        def capture(module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            if ids is None:
                raise ValueError('Cannot observe ORT input shape')
            shapes.add(tuple(ids.shape))

        handle = encoder.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            for batch in shapes_to_batches.values():
                a.request_batch(batch)
            torch.cuda.synchronize()
        finally:
            handle.remove()
        profile = Path(session.end_profiling())
        session = None
        evidence = validate_events(json.loads(profile.read_text()), {r['name'] for r in proof['proven']})
        if not evidence['passed'] or shapes != set(shapes_to_batches):
            raise ValueError('ORT metadata-cpu-v1 or shape validation failed')
        shapes.update(old_shapes)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = False
        measured = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if 'CUDAExecutionProvider' not in measured.get_providers():
            raise ValueError('Measured ORT session has no CUDA provider')
        measured.disable_fallback()
        encoder.forward = binding_forward(measured, encoder, shapes, lambda: True)
        a.ort_session = measured
        a._ort_original, a._ort_shapes, a._ort_ready = original, shapes, True
        a._restore_encoder = lambda: setattr(encoder, 'forward', original)
        a.metadata.update(ort_evidence=evidence, ort_profile_active_during_latency=False,
                          batch_ort_checked_shapes=[list(s) for s in sorted(shapes)],
                          ort={'providers': measured.get_providers(), 'cpu_policy': 'metadata-cpu-v1',
                               'original_encoder_retained': True})
        for batch in shapes_to_batches.values():
            a.request_batch(batch)
    except BaseException:
        encoder.forward = original
        a.ort_session, a._restore_encoder = None, None
        a._ort_ready, a._backend_installed, a._ort_shapes = False, False, set()
        if session is not None:
            try:
                session.end_profiling()
            except Exception:
                pass
        raise


def profile_backend(a, batch):
    import torch
    before = a.compile_count
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        a.request_batch(batch)
        torch.cuda.synchronize()
    if a.compile_count != before:
        raise ValueError('New compilation during backend proof')
    a.directory.mkdir(parents=True, exist_ok=True)
    trace = a.directory / 'batch-backend-proof.trace.json'
    profiler.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())['traceEvents']
    kernels = sorted({e['name'] for e in events if e.get('cat') == 'kernel'})
    operators = sorted({e.key for e in profiler.key_averages()})
    if not kernels:
        raise ValueError('No CUDA kernels in backend proof')
    e = a.spec['execution']
    required = {'math': 'aten::_scaled_dot_product_attention_math',
                'efficient': 'aten::_scaled_dot_product_efficient_attention'}.get(e['sdpa_backend'])
    if required and required not in operators:
        raise ValueError('Requested SDPA backend was not observed')
    sdpa_operators = [op for op in operators if op.startswith('aten::_scaled_dot_product_')]
    if a.metadata.get('attention') == 'sdpa' and not sdpa_operators:
        raise ValueError('Loaded SDPA implementation was not observed')
    with gzip.open(str(trace) + '.gz', 'wb') as output:
        output.write(trace.read_bytes())
    trace.unlink()
    return {'kernels': kernels, 'operators': operators, 'trace': trace.name + '.gz',
            'input': a.metadata['batch_verification'], 'runtime': e['runtime'],
            'attention': a.metadata.get('attention'), 'sdpa_operators': sdpa_operators,
            'linear_attention': e['linear_attention']}
