"""Explicit, independently selectable changes. No implicit fallback."""

import copy

from .adapters import UnsupportedExperiment


SPECS = {
    "B0": {"dtype": "float32"},
    "P1": {"dtype": "float16"},
    "H1": {"dtype": "float32", "attention": "sdpa"},
    "H1_fp16": {"dtype": "float16", "attention": "sdpa"},
    "H1_repeat": {"dtype": "float16", "attention": "sdpa", "repeat_kv": True},
    "H2_linear": {"dtype": "float16", "fast": "linear"},
    "H2_conv": {"dtype": "float16", "fast": "conv"},
    "H2_norm": {"dtype": "float16", "fast": "norm"},
    "H2_all": {"dtype": "float16", "fast": "all"},
    "C1": {"dtype": "float32", "compile": "default"},
    "C1_dynamic": {"dtype": "float32", "compile": "default", "dynamic": True},
    "C2_reduce": {"dtype": "float32", "compile": "reduce-overhead"},
    "C2_graph": {"dtype": "float32", "graph": True},
    "G1_service": {"dtype": "float32", "onnx": True, "cpu_policy": "metadata-cpu-v1"},
    "G1": {"dtype": "float32", "onnx": True},
    "G2": {"dtype": "float32", "projection_cache": True},
    "H4_block": {"dtype": "float32", "block_compile": True},
}


def spec_for(name, model, custom=None):
    spec = copy.deepcopy(custom if name == "X1" else SPECS[name])
    if not isinstance(spec, dict) or spec.get("dtype") not in ("float32", "float16"):
        raise ValueError("Invalid candidate specification")
    allowed = {"dtype", "attention", "fast", "compile", "graph", "onnx", "projection_cache", "block_compile", "dynamic", "repeat_kv"}
    allowed.add("cpu_policy")
    if spec.get("cpu_policy") and (name != "G1_service" or model != "gliner"):
        raise ValueError("CPU policy is explicit and GLiNER-only")
    if set(spec) - allowed:
        raise ValueError("Unknown optimization option")
    if model == "gliner" and any(spec.get(k) for k in ("attention", "fast", "block_compile", "repeat_kv")):
        raise ValueError("Halo-only optimization")
    if model == "halo" and any(spec.get(k) for k in ("onnx", "projection_cache")):
        raise ValueError("GLiNER-only optimization")
    if sum(bool(spec.get(k)) for k in ("compile", "graph", "onnx")) > 1:
        raise ValueError("Compare compile, eager graph and ONNX independently")
    if spec.get("onnx") and spec.get("projection_cache"):
        raise ValueError("Python projection cache is not an ONNX optimization")
    if spec.get("dynamic") and not spec.get("compile"):
        raise ValueError("Dynamic shapes require compile")
    if spec.get("repeat_kv") and spec.get("attention") != "sdpa":
        raise ValueError("Explicit KV repeat requires SDPA")
    return spec


class ProjectionMemo:
    """Cache only projections of the exact relative-position tensor, in eval mode."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.originals, self.entries = [], {}
        self.rel = None
        self.hits = self.misses = 0
        original_train = encoder.train
        def train(mode=True):
            if mode != encoder.training:
                self.clear()
            return original_train(mode)
        self.originals.append((encoder, "train", original_train))
        encoder.train = train
        for layer in encoder.encoder.layer:
            attention = layer.attention.self
            if not attention.relative_attention or not attention.share_att_key:
                raise UnsupportedExperiment("Projection cache requires shared relative attention keys")
            original_bias = attention.disentangled_attention_bias
            def bias(*args, _original=original_bias, **kwargs):
                relative = kwargs.get("rel_embeddings", args[3] if len(args) > 3 else None)
                if relative is None:
                    raise ValueError("Cannot locate relative embeddings")
                self.rel = relative
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.rel = None
            self.originals.append((attention, "disentangled_attention_bias", original_bias))
            attention.disentangled_attention_bias = bias
            for projection in (attention.query_proj, attention.key_proj):
                original = projection.forward
                def projected(x, _p=projection, _original=original):
                    import torch
                    eligible = (not self.encoder.training and not torch.is_grad_enabled() and self.rel is not None and
                                x.untyped_storage().data_ptr() == self.rel.untyped_storage().data_ptr())
                    if not eligible:
                        if self.encoder.training or torch.is_grad_enabled():
                            self.entries.clear()
                        return _original(x)
                    # Include relative embedding / LayerNorm weights: normalization creates a fresh tensor per call.
                    params = list(self.encoder.encoder.rel_embeddings.parameters())
                    norm = getattr(self.encoder.encoder, "LayerNorm", None)
                    if norm is not None:
                        params += list(norm.parameters())
                    params += list(_p.parameters())
                    signature = tuple((id(p), p._version, str(p.device), str(p.dtype)) for p in params)
                    key = (signature, tuple(x.shape), str(x.dtype), str(x.device))
                    entry = self.entries.get(id(_p))
                    if entry is not None and entry[0] == key:
                        self.hits += 1
                        return entry[1]
                    self.misses += 1
                    result = _original(x)
                    self.entries[id(_p)] = (key, result.detach())
                    return result
                self.originals.append((projection, "forward", original))
                projection.forward = projected

    def clear(self):
        self.entries.clear()

    def remove(self):
        for obj, attr, original in reversed(self.originals):
            setattr(obj, attr, original)
        self.originals.clear()
        self.clear()


class CudaReplay:
    """Shape-specific capture. Copies fresh input and clones output on every call."""
    def __init__(self, fn, max_shapes=12):
        self.fn, self.max_shapes, self.entries = fn, max_shapes, {}

    def __call__(self, inputs):
        import torch
        key = tuple((k, tuple(v.shape), v.dtype, v.device) for k, v in sorted(inputs.items()))
        if key not in self.entries:
            if len(self.entries) >= self.max_shapes:
                raise UnsupportedExperiment("CUDA Graph shape budget exhausted; no eager fallback")
            static = {k: v.clone() for k, v in inputs.items()}
            stream = torch.cuda.Stream(device=next(iter(inputs.values())).device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.fn(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.fn(static)
            self.entries[key] = (static, graph, output)
        static, graph, output = self.entries[key]
        for key, value in inputs.items():
            static[key].copy_(value)
        graph.replay()
        return output.clone()


def apply_variant(adapter, spec, directory):
    import torch
    model = adapter.model
    if spec.get("projection_cache"):
        adapter.projection_memo = ProjectionMemo(model.encoder)
    if spec.get("block_compile"):
        count = 0
        for layer in model.modules():
            if hasattr(layer, "chunk_gated_delta_rule"):
                layer.chunk_gated_delta_rule = torch.compile(layer.chunk_gated_delta_rule, fullgraph=True, dynamic=False)
                count += 1
        if not count:
            raise UnsupportedExperiment("No reference linear attention blocks")
    if spec.get("onnx"):
        from .onnx_encoder import install_onnx
        install_onnx(adapter, directory)
    if not (spec.get("compile") or spec.get("graph")):
        return
    gliner = adapter.boundary == "encoder_forward_gpu_inputs"
    if gliner:
        original = model.encoder.forward
        fn = lambda inputs: original(**inputs).last_hidden_state
    else:
        fn = adapter.tensor_call
    if spec.get("compile"):
        optimized = torch.compile(fn, mode=spec["compile"], dynamic=spec.get("dynamic", False), fullgraph=True)
        adapter.metadata["compile"] = {"mode": spec["compile"], "dynamic": spec.get("dynamic", False),
                                       "fullgraph": True, "boundary": adapter.boundary}
    else:
        optimized = CudaReplay(fn)
    if gliner:
        from transformers.modeling_outputs import BaseModelOutput
        def forward(input_ids=None, attention_mask=None, **kwargs):
            if any(v is not None for v in kwargs.values()):
                raise UnsupportedExperiment("Unexpected GLiNER encoder arguments")
            output = optimized({"input_ids": input_ids, "attention_mask": attention_mask})
            return BaseModelOutput(last_hidden_state=output)
        model.encoder.forward = forward
        adapter.restore_encoder = lambda: setattr(model.encoder, "forward", original)
    else:
        adapter.tensor_call = optimized
