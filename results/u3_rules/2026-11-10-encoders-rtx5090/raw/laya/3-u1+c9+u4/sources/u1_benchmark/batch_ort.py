"""Batched IOBinding around the pinned export and metadata-cpu-v1 validators."""
import json
from pathlib import Path

from .core import write


def binding_forward(session, encoder, allowed_shapes, checked):
    """Keep input/output allocations alive until ORT has finished using their pointers."""
    def forward(input_ids=None, attention_mask=None, **kwargs):
        import numpy as np
        import torch
        from transformers.modeling_outputs import BaseModelOutput
        from .batch_adapters import UnsupportedBatch
        if any(v is not None for v in kwargs.values()):
            raise UnsupportedBatch("Unexpected ORT encoder arguments")
        if input_ids is None or attention_mask is None:
            raise UnsupportedBatch("ORT requires explicit IDs and masks")
        shape = tuple(input_ids.shape)
        if (len(shape) != 2 or shape[0] < 1 or not 0 < shape[1] <= 512 or
                tuple(attention_mask.shape) != shape):
            raise UnsupportedBatch("Invalid ORT batch shape")
        if any(t.dtype != torch.int64 or t.device.type != "cuda" or t.device.index != 0
               for t in (input_ids, attention_mask)):
            raise UnsupportedBatch("ORT inputs must be int64 on CUDA 0")
        if checked() and shape not in allowed_shapes:
            raise UnsupportedBatch("Unvalidated ORT batch shape: " + str(shape))
        binding = session.io_binding()
        kept = {"input_ids": input_ids.contiguous(), "attention_mask": attention_mask.contiguous()}
        for name, tensor in kept.items():
            binding.bind_input(name, "cuda", 0, np.int64, tuple(tensor.shape), tensor.data_ptr())
        hidden = torch.empty((*shape, encoder.config.hidden_size), dtype=torch.float32, device=input_ids.device)
        binding.bind_output("hidden", "cuda", 0, np.float32, tuple(hidden.shape), hidden.data_ptr())
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return BaseModelOutput(last_hidden_state=hidden)
    return forward


def prepare(wrapper, batches):
    import onnx
    import onnxruntime as ort
    import torch
    from followup import onnx_service
    from followup.team_smoke import preload_ort
    from .batch_adapters import UnsupportedBatch

    a = wrapper.legacy
    ready = getattr(a, "batch_ort_ready", False)
    old_shapes = set(a.service_checked_shapes) if ready else set()
    if ready:
        batches = [b for b in batches if tuple(a.prepare([wrapper.adapter.legacy_text(t) for t in b])[
            "input_ids"].shape) not in old_shapes]
        if not batches:
            return
    if hasattr(a, "restore_encoder") and not ready:
        raise UnsupportedBatch("Batch ORT requires an unpatched encoder")
    directory = Path(wrapper.directory)
    directory.mkdir(parents=True, exist_ok=True)
    preload_ort(directory)
    original = a.batch_ort_original_forward if ready else a.model.encoder.forward
    old_texts = onnx_service.VALIDATION_TEXTS
    session = None
    try:
        if not ready:
            # Reuse the existing dynamic-axis export, graph inventory and dataflow proof.
            onnx_service.VALIDATION_TEXTS = [wrapper.adapter.legacy_text(batches[0][0])]
            onnx_service.install(a, directory)
            a.batch_ort_original_forward = original
            session = a.ort_session
        else:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_profiling = True
            options.profile_file_prefix = str(directory / "ort-batch-shapes")
            options.optimized_model_filepath = str(directory / "encoder.optimized.onnx")
            a.model.encoder.forward = original
            a.ort_session = None
            session = ort.InferenceSession(str(directory / "encoder.onnx"), sess_options=options,
                providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"])
            proof = onnx_service.prove_metadata_nodes(onnx.load(options.optimized_model_filepath,
                                                               load_external_data=False).graph)
            a.service_proven_names = {r["name"] for r in proof["proven"]}
            write(directory / "metadata_dataflow.json", proof)
        session.disable_fallback()
        a.model.encoder.forward = binding_forward(session, a.model.encoder, set(), lambda: False)
        wrapper._backend_installed = True
        shapes = set()

        def capture(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None:
                raise UnsupportedBatch("Cannot observe actual ORT encoder shape")
            shapes.add(tuple(ids.shape))

        handle = a.model.encoder.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            for batch in batches:
                wrapper.request_batch(batch)
            torch.cuda.synchronize()
        finally:
            handle.remove()
        profile = Path(session.end_profiling())
        session = None  # Profiling is ended; preserve the primary error on later failure.
        result = onnx_service.validate_events(json.loads(profile.read_text()), a.service_proven_names)
        result.update(profile=profile.name, checked_shapes=[list(s) for s in sorted(shapes)])
        write(directory / "batch-ort-providers.json", result)
        if not result["passed"] or not shapes:
            raise UnsupportedBatch("Batched ORT violates metadata-cpu-v1; see batch-ort-providers.json")
        validations = list(a.metadata.get("batch_ort_validations", [])) + [result]
        shapes.update(old_shapes)
        write(directory / "batch-ort-validations.json", validations)
        a.ort_session = None
        a.model.encoder.forward = original
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = False
        measured = ort.InferenceSession(str(directory / "encoder.onnx"), sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"])
        if "CUDAExecutionProvider" not in measured.get_providers():
            raise UnsupportedBatch("Measured ORT session has no CUDA provider")
        measured.disable_fallback()
        a.service_checked_shapes = shapes
        a.model.encoder.forward = binding_forward(measured, a.model.encoder, shapes, lambda: True)
        a.ort_session = measured
        a.ort_profile_checked = True
        a.batch_ort_ready = True
        a.metadata["ort_evidence"] = result
        a.metadata["batch_ort_validations"] = validations
        a.metadata["batch_ort_checked_shapes"] = [list(s) for s in sorted(shapes)]
        a.metadata["ort_profile_active_during_latency"] = False
        a.metadata["ort"]["providers"] = measured.get_providers()
        for batch in batches:
            wrapper.request_batch(batch)
        wrapper.metadata.update(a.metadata)
    except BaseException:
        a.model.encoder.forward = original
        a.ort_session = None
        a.ort_profile_checked = False
        a.batch_ort_ready = False
        wrapper._backend_installed = False
        if hasattr(a, "restore_encoder"):
            del a.restore_encoder
        if session is not None:
            try:
                session.end_profiling()
            except Exception:
                pass  # Cleanup must not replace an OOM or provider-validation failure.
        raise
    finally:
        onnx_service.VALIDATION_TEXTS = old_texts
