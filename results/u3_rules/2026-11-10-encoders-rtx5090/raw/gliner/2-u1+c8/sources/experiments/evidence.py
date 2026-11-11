"""Backend evidence and operator/layer attribution, never inside latency timing."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path

from .adapters import UnsupportedExperiment
from .common import write_json


def category(name, cls):
    if cls == "Qwen3_5GatedDeltaNet":
        return "linear_attention"
    if cls == "Qwen3_5Attention":
        return "full_attention"
    if "MLP" in cls:
        return "mlp"
    if "Norm" in cls or "Conv" in cls:
        return "norm_conv"
    if "DisentangledSelfAttention" in cls:
        return "relative_attention"
    if name.endswith(("query_proj", "key_proj")):
        return "query_key_projection"  # Includes content AND relative projections.
    if name == "encoder":
        return "encoder"
    if "classifier" in name:
        return "head"
    return None


@contextmanager
def layer_scopes(adapter):
    import torch
    handles, stacks, originals = [], defaultdict(list), []
    compiled = bool(adapter.spec.get("compile"))
    try:
        for name, module in adapter.model.named_modules():
            group = category(name, type(module).__name__)
            # Hooks inside a compiled function cause graph breaks; annotate its external boundary only.
            if not group or (compiled and name != "encoder" and "classifier" not in name):
                continue
            label = "layer/" + group + "/" + name
            def before(mod, args, _label=label):
                scope = torch.profiler.record_function(_label)
                scope.__enter__()
                stacks[id(mod)].append(scope)
            def after(mod, args, result):
                if stacks[id(mod)]:
                    stacks[id(mod)].pop().__exit__(None, None, None)
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
            # Tag the relative-only callsite; projection weights are otherwise shared with content.
            if not compiled and hasattr(module, "disentangled_attention_bias"):
                original = module.disentangled_attention_bias
                def relative(*args, _fn=original, _name=name, **kwargs):
                    with torch.profiler.record_function("layer/relative_position/" + _name):
                        return _fn(*args, **kwargs)
                originals.append((module, original))
                module.disentangled_attention_bias = relative
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, original in originals:
            module.disentangled_attention_bias = original
        for stack in stacks.values():
            while stack:
                stack.pop().__exit__(None, None, None)


def attribute(events):
    """One owner per kernel; unmatched correlations explicitly remain unknown."""
    cpu = defaultdict(list)
    runtimes = {}
    scopes = defaultdict(list)
    for event in events:
        args = event.get("args", {})
        if event.get("cat") == "cpu_op" and "External id" in args:
            cpu[args["External id"]].append(event)
        if event.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in args:
            runtimes[args["correlation"]] = event
        if event.get("name", "").startswith("layer/") and event.get("ph") == "X":
            scopes[(event.get("pid"), event.get("tid"))].append(event)
    totals = defaultdict(lambda: {"kernels": 0, "gpu_us": 0.0})
    for kernel in (e for e in events if e.get("cat") == "kernel"):
        args = kernel.get("args", {})
        runtime = runtimes.get(args.get("correlation"), {})
        ext = args.get("External id", runtime.get("args", {}).get("External id"))
        candidates = cpu.get(ext, [])
        if runtime:
            candidates = [e for e in candidates if e.get("tid") == runtime.get("tid") and
                          e.get("ts", 0) <= runtime.get("ts", 0) <= e.get("ts", 0) + e.get("dur", 0)]
        op = min(candidates, key=lambda e: e.get("dur", 0)) if candidates else None
        owner, opname = "unknown", "unknown"
        if op:
            opname = op["name"]
            parents = [s for s in scopes[(op.get("pid"), op.get("tid"))]
                       if s["ts"] <= op["ts"] and s["ts"] + s["dur"] >= op["ts"] + op.get("dur", 0)]
            if parents:
                owner = min(parents, key=lambda e: e["dur"])["name"]
                relative = [s for s in parents if s["name"].startswith("layer/relative_position/")]
                if relative and owner.startswith("layer/query_key_projection/"):
                    owner = "layer/relative_projection/" + owner.split("/", 2)[-1]
        key = (owner, opname, kernel.get("name", "unknown"))
        totals[key]["kernels"] += 1
        totals[key]["gpu_us"] += kernel.get("dur", 0)
    def component(layer, op):
        if any(word in op.lower() for word in ("copy", "clone", "contiguous", "_to_copy")):
            return "copies"
        if "conv" in op.lower() or "norm" in op.lower():
            return "norm_conv"
        return layer.split("/", 2)[1] if layer.startswith("layer/") else "unknown"
    return [{"layer": layer, "component": component(layer, op), "operator": op, "kernel": kernel, **value}
            for (layer, op, kernel), value in sorted(totals.items(), key=lambda pair: -pair[1]["gpu_us"])]


def capture(adapter, text, directory, name, iterations=3, annotate=True):
    import torch
    from contextlib import nullcontext
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for _ in range(2):
            adapter.scores([text])
        torch.cuda.synchronize()
        with layer_scopes(adapter) if annotate else nullcontext():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as profiler:
                for _ in range(iterations):
                    with torch.profiler.record_function("text_to_verdict_" + name):
                        adapter.scores([text])
                        torch.cuda.synchronize()
    raw = directory / (name + ".json")
    profiler.export_chrome_trace(str(raw))
    payload = raw.read_bytes()
    events = json.loads(payload)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise UnsupportedExperiment("No CUDA kernels observed in backend probe")
    zipped = gzip.compress(payload, mtime=0)
    raw.with_suffix(".json.gz").write_bytes(zipped)
    raw.unlink()
    write_json(directory / (name + "_manifest.json"), {
        "raw_sha256": hashlib.sha256(payload).hexdigest(), "gz_sha256": hashlib.sha256(zipped).hexdigest(),
        "kernel_count": len(kernels), "iterations": iterations, "compressed_bytes": len(zipped)})
    write_json(directory / (name + "_attribution.json"), {
        "rows": attribute(events), "scope": "sum kernel time under profiler, not wall time or SM utilization",
        "compiled_layer_detail": "encoder boundary only" if adapter.spec.get("compile") else "module scopes"})
    (directory / (name + "_operators.txt")).write_text(
        profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=60))
    return events


def finish_ort(adapter, directory, text):
    import torch
    if getattr(adapter, "ort_profile_checked", False):
        return adapter.metadata["ort_evidence"]
    with torch.inference_mode():
        adapter.scores([text])
    path = adapter.ort_session.end_profiling()
    events = json.loads(Path(path).read_text())
    counts = Counter(e.get("args", {}).get("provider") for e in events if e.get("cat") == "Node"
                     and e.get("args", {}).get("provider"))
    result = {"executed_providers": dict(counts), "profile": Path(path).name}
    write_json(Path(directory) / "ort_providers.json", result)
    if not counts.get("CUDAExecutionProvider") or set(counts) != {"CUDAExecutionProvider"}:
        raise UnsupportedExperiment("ORT must execute CUDA nodes only; observed " + str(dict(counts)))
    adapter.metadata["ort_evidence"] = result
    adapter.ort_profile_checked = True
    return result


def verify_backend(adapter, cases, directory):
    """Called outside measurements. Evidence is retained even when a candidate is rejected."""
    spec = adapter.spec
    originals, calls = [], Counter()
    if spec.get("fast") == "linear":
        for module in adapter.model.modules():
            if hasattr(module, "chunk_gated_delta_rule"):
                original = module.chunk_gated_delta_rule
                def counted(*args, _fn=original, **kwargs):
                    calls[_fn.__module__ + "." + _fn.__name__] += 1
                    return _fn(*args, **kwargs)
                originals.append((module, original))
                module.chunk_gated_delta_rule = counted
    result = {"spec": spec, "shapes": {}, "fast_calls": {}}
    try:
        for name in ("legacy_short", "legacy_medium", "legacy_long"):
            events = capture(adapter, cases[name][0], directory, name, iterations=1, annotate=False)
            ops = {e.get("name", "") for e in events}
            math_sdpa = any("_scaled_dot_product_attention_math" in op for op in ops)
            efficient = any("_scaled_dot_product_efficient_attention" in op for op in ops)
            compiled = any("Compiled" in op or "CompiledFunction" in op for op in ops)
            result["shapes"][name] = {"math_sdpa": math_sdpa, "efficient_sdpa": efficient,
                                      "compiled_region": compiled,
                                      "kernels": sorted({e["name"] for e in events if e.get("cat") == "kernel"})}
        result["fast_calls"] = dict(calls)
        if spec.get("onnx"):
            result["ort"] = finish_ort(adapter, directory, cases["legacy_short"][0])
        failures = []
        if spec.get("attention") == "sdpa":
            if any(x["math_sdpa"] or not x["efficient_sdpa"] for x in result["shapes"].values()):
                failures.append("sdpa_not_efficient")
        if spec.get("compile") and not all(x["compiled_region"] for x in result["shapes"].values()):
            failures.append("compiled_region_not_observed")
        if spec.get("fast") == "linear":
            if not calls or any(not name.startswith("fla.") for name in calls):
                failures.append("fla_not_called")
            if any("torch_chunk_gated_delta_rule" in x for x in adapter.metadata.get("linear", [])):
                failures.append("reference_linear_path")
            if adapter.metadata.get("norm") != ["Qwen3_5RMSNormGated"]:
                failures.append("norm_changed_in_linear_only_experiment")
            if any(getattr(m, "causal_conv1d_fn", None) is not None for m, _ in originals):
                failures.append("conv_changed_in_linear_only_experiment")
        result.update(passed=not failures, failures=failures)
        write_json(Path(directory) / "backend.json", result)
        if failures:
            raise UnsupportedExperiment(", ".join(failures))
        return result
    finally:
        for module, original in originals:
            module.chunk_gated_delta_rule = original


def detailed_profile(adapter, cases, directory):
    for name in ("legacy_short", "legacy_long"):
        capture(adapter, cases[name][0], directory, name)
