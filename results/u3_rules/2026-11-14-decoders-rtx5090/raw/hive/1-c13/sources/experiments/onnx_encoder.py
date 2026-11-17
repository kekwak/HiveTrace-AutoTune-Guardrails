"""Strict CUDA-only ONNX encoder; leaves GLiNER classification heads unchanged."""

from pathlib import Path

from .adapters import UnsupportedExperiment


def install_onnx(adapter, directory):
    import numpy as np
    import onnxruntime as ort
    import torch
    import onnx
    from collections import Counter
    from transformers.modeling_outputs import BaseModelOutput
    from .common import write_json

    # Prefer the CUDA/cuDNN libraries already shipped with this PyTorch runtime.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise UnsupportedExperiment("ONNX Runtime CUDA EP unavailable")
    encoder, original = adapter.model.encoder, adapter.model.encoder.forward
    sample = adapter.prepare(["The public library opens every morning."])

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            return original(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    path = Path(directory) / "encoder.onnx"
    torch.onnx.export(Export().eval(), (sample["input_ids"], sample["attention_mask"]), str(path),
                      input_names=["input_ids", "attention_mask"], output_names=["hidden"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                                    "attention_mask": {0: "batch", 1: "tokens"},
                                    "hidden": {0: "batch", 1: "tokens"}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(Path(directory) / "encoder.optimized.onnx")
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(directory) / "ort-profile")
    session = ort.InferenceSession(str(path), sess_options=options,
                                  providers=[("CUDAExecutionProvider", {"device_id": 0})])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise UnsupportedExperiment("Unexpected ORT primary provider")
    def inventory(file):
        graph = onnx.load(str(file), load_external_data=False).graph
        return dict(Counter((node.domain or "ai.onnx") + "::" + node.op_type for node in graph.node))
    before, after = inventory(path), inventory(options.optimized_model_filepath)
    write_json(Path(directory) / "onnx_graph.json", {
        "before": before, "after": after,
        "added_or_increased": {k: n - before.get(k, 0) for k, n in after.items() if n > before.get(k, 0)},
        "note": "Operator counts, not a claim that every added operator is a beneficial fusion"})

    def forward(input_ids=None, attention_mask=None, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise UnsupportedExperiment("Unexpected ONNX encoder arguments")
        binding = session.io_binding()
        kept = {}
        for name, value in {"input_ids": input_ids, "attention_mask": attention_mask}.items():
            kept[name] = value.contiguous()
            binding.bind_input(name, "cuda", 0, np.int64, tuple(value.shape), kept[name].data_ptr())
        hidden = torch.empty((*input_ids.shape, encoder.config.hidden_size), device=input_ids.device,
                             dtype=next(encoder.parameters()).dtype)
        np_dtype = np.float32 if hidden.dtype == torch.float32 else np.float16
        binding.bind_output("hidden", "cuda", 0, np_dtype, tuple(hidden.shape), hidden.data_ptr())
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return BaseModelOutput(last_hidden_state=hidden)

    encoder.forward = forward
    adapter.restore_encoder = lambda: setattr(encoder, "forward", original)
    adapter.ort_session = session
    adapter.metadata["ort"] = {"version": ort.__version__, "providers": session.get_providers(),
                               "cpu_fallback_disabled": True,
                               "memory_note": "PyTorch allocator excludes ORT allocations; original Torch weights retained"}
