"""Explicit ORT CUDA + bounded metadata-CPU policy; separate from strict G1."""

from collections import Counter
import json
import math
from pathlib import Path

CPU_OPS = {"Unsqueeze", "Slice", "Squeeze", "Concat", "Gather", "Cast", "Mul", "Sqrt",
           "Equal", "Reshape", "Where", "Div", "Not", "Shape", "Size", "Constant", "Identity"}
POLICY = "metadata-cpu-v1"
VALIDATION_TEXTS = []


def validate_events(events, proven_names):
    counts, failures, cpu = Counter(), [], []
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider")
        if event.get("cat") != "Node" or not provider:
            continue
        counts[provider] += 1
        op = args.get("op_name")
        name = event.get("name", "").removesuffix("_kernel_time")
        if provider == "CUDAExecutionProvider":
            continue
        if provider != "CPUExecutionProvider":
            failures.append(f"Unexpected provider: {provider}")
            continue
        reason = []
        if op not in CPU_OPS:
            reason.append("operator_not_allowed")
        if name not in proven_names:
            reason.append("metadata_dataflow_not_proven")
        for field in ("input_type_shape", "output_type_shape"):
            shapes = args.get(field)
            if not shapes:
                reason.append("missing_shapes")
                continue
            for entry in shapes:
                if not isinstance(entry, dict) or len(entry) != 1:
                    reason.append("unknown_shape_encoding")
                    continue
                dtype, shape = next(iter(entry.items()))
                if (dtype not in ("int64", "int32", "float", "bool") or not isinstance(shape, list)
                        or not all(isinstance(d, int) and d >= 0 for d in shape)
                        or math.prod(shape) > 4):
                    reason.append("tensor_exceeds_metadata_policy")
        cpu.append({"name": name, "op": op, "inputs": args.get("input_type_shape"),
                    "outputs": args.get("output_type_shape"), "failures": reason})
        if reason:
            failures.append(name + ": " + ",".join(reason))
    if not counts.get("CUDAExecutionProvider"):
        failures.append("no_executed_cuda_nodes")
    return {"policy": POLICY, "passed": not failures, "executed_providers": dict(counts),
            "cpu_events": cpu, "failures": failures}


def prove_metadata_nodes(graph):
    """Trace CPU-eligible outputs back to shapes or small constants, never activations."""
    contexts, names = [], Counter()

    def visit(g, parent=None, scope="main"):
        ctx = {"parent": parent, "scope": scope, "nodes": {o: n for n in g.node for o in n.output},
               "initializers": {t.name: math.prod(t.dims) for t in g.initializer}, "graph": g}
        contexts.append(ctx)
        for node in g.node:
            names[node.name] += 1
            for attr in node.attribute:
                children = [attr.g] if attr.type == 5 else list(attr.graphs) if attr.type == 10 else []
                for i, child in enumerate(children):
                    visit(child, ctx, scope + "/" + node.name + "/" + attr.name + str(i))
    visit(graph)

    def value_is_metadata(value, ctx, seen):
        if not value:  # Optional ONNX input.
            return True
        token = (id(ctx), value)
        if token in seen:
            return False
        seen = seen | {token}
        if value in ctx["initializers"]:
            return ctx["initializers"][value] <= 4
        node = ctx["nodes"].get(value)
        if node is None:
            return ctx["parent"] is not None and value_is_metadata(value, ctx["parent"], seen)
        if node.domain not in ("", "ai.onnx"):
            return False
        if node.op_type in ("Shape", "Size"):
            return True
        if node.op_type == "Constant":
            return bool(node.attribute) and all(
                math.prod(a.t.dims) <= 4 if a.type == 4 else
                len(a.ints) <= 4 if a.type == 7 else len(a.floats) <= 4 if a.type == 6 else
                a.type in (1, 2) for a in node.attribute)
        return node.op_type in CPU_OPS and bool(node.input) and all(
            value_is_metadata(v, ctx, seen) for v in node.input)

    proven, unresolved = [], []
    for ctx in contexts:
        for node in ctx["graph"].node:
            if node.op_type not in CPU_OPS:
                continue
            passed = bool(node.name) and names[node.name] == 1 and all(
                value_is_metadata(o, ctx, set()) for o in node.output)
            record = {"name": node.name, "scope": ctx["scope"], "op": node.op_type}
            (proven if passed else unresolved).append(record)
    return {"proven": proven, "unresolved": unresolved}


def install(adapter, directory):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from experiments.adapters import UnsupportedExperiment
    from experiments.common import write_json

    if adapter.spec.get("dtype") != "float32" or not VALIDATION_TEXTS:
        raise UnsupportedExperiment("Service CPU policy requires FP32 and development shape probes")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise UnsupportedExperiment("ORT CUDA unavailable")
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    directory = Path(directory)
    encoder, original = adapter.model.encoder, adapter.model.encoder.forward

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            return original(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    sample = adapter.prepare(["The public library opens every morning."])
    path = directory / "encoder.onnx"
    torch.onnx.export(Export().eval(), (sample["input_ids"], sample["attention_mask"]), str(path),
                      input_names=["input_ids", "attention_mask"], output_names=["hidden"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                                    "attention_mask": {0: "batch", 1: "tokens"},
                                    "hidden": {0: "batch", 1: "tokens"}}, opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(directory / "encoder.optimized.onnx")
    options.enable_profiling = True
    options.profile_file_prefix = str(directory / "ort-service")
    session = ort.InferenceSession(str(path), sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"])
    optimized = onnx.load(options.optimized_model_filepath, load_external_data=False)
    proof = prove_metadata_nodes(optimized.graph)
    write_json(directory / "metadata_dataflow.json", proof)
    def inventory(g):
        counts = Counter()
        for n in g.node:
            counts[(n.domain or "ai.onnx") + "::" + n.op_type] += 1
            for a in n.attribute:
                for child in ([a.g] if a.type == 5 else list(a.graphs) if a.type == 10 else []):
                    counts.update(inventory(child))
        return counts
    write_json(directory / "onnx_graph.json", {
        "before": dict(inventory(onnx.load(str(path), load_external_data=False).graph)),
        "after": dict(inventory(optimized.graph)), "note": "Counts are not measured speedups"})
    del optimized
    adapter.service_proven_names = {r["name"] for r in proof["proven"]}
    adapter.service_checked_shapes = set()

    def forward(input_ids=None, attention_mask=None, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise UnsupportedExperiment("Unexpected encoder arguments")
        shape = tuple(input_ids.shape)
        if len(shape) != 2 or shape[0] != 1 or shape[1] > 512:
            raise UnsupportedExperiment("Only batch 1, at most 512 tokens")
        if getattr(adapter, "ort_profile_checked", False) and shape not in adapter.service_checked_shapes:
            raise UnsupportedExperiment("Unvalidated ONNX input shape: " + str(shape))
        binding = session.io_binding()
        kept = {"input_ids": input_ids.contiguous(), "attention_mask": attention_mask.contiguous()}
        for name, tensor in kept.items():
            binding.bind_input(name, "cuda", 0, np.int64, tuple(tensor.shape), tensor.data_ptr())
        hidden = torch.empty((*shape, encoder.config.hidden_size), device=input_ids.device, dtype=torch.float32)
        binding.bind_output("hidden", "cuda", 0, np.float32, tuple(hidden.shape), hidden.data_ptr())
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return BaseModelOutput(last_hidden_state=hidden)

    encoder.forward = forward
    adapter.restore_encoder = lambda: setattr(encoder, "forward", original)
    adapter.ort_session = session
    adapter.metadata["ort"] = {"version": ort.__version__, "providers": session.get_providers(),
        "provider_options": session.get_provider_options(), "cpu_policy": POLICY,
        "cuda_only": False, "memory_note": "Torch allocator excludes ORT; original weights retained"}


def finish(adapter, directory, text):
    import torch
    from experiments.adapters import UnsupportedExperiment
    from experiments.common import write_json
    if getattr(adapter, "ort_profile_checked", False):
        return adapter.metadata["ort_evidence"]
    with torch.inference_mode():
        for item in list(VALIDATION_TEXTS) + [text]:
            shape = tuple(adapter.prepare([item])["input_ids"].shape)
            if shape not in adapter.service_checked_shapes:
                adapter.scores([item])
                adapter.service_checked_shapes.add(shape)
    profile = Path(adapter.ort_session.end_profiling())
    events = json.loads(profile.read_text())
    result = validate_events(events, adapter.service_proven_names)
    result.update(profile=profile.name, checked_shapes=sorted(adapter.service_checked_shapes))
    write_json(Path(directory) / "ort_providers.json", result)
    if not result["passed"]:
        raise UnsupportedExperiment("ORT CPU policy rejected: " + "; ".join(result["failures"][:5]))
    adapter.metadata["ort_evidence"] = result
    adapter.ort_profile_checked = True
    return result
