"""One variant in one fresh process; preparation is inside the request timer."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

from .core import digest, read, sha, write
from .contracts import validate_response
from .messages import payload
from .precision import PrecisionGuard
from .v4_evidence import assert_environment, check_sources, configure_gpu
from .v4_planning import materialize, smoke_plans
from .v4_config import PROTOCOL_VERSION, verify_local_sources
from .v4_privacy import clean_evidence, contains_private, private_values, public_config


def responses(values, expected):
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError("Native batch returned wrong response count")
    return [validate_response(value) for value in values]


def error_status(error):
    name = type(error).__name__
    if name in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
        return "oom"
    if name in {"UnsupportedBatch", "UnsupportedExperiment", "NotImplementedError"}:
        return "unsupported"
    return "timeout" if isinstance(error, TimeoutError) else "failed"


def public_index(index):
    keys = ("user_tokens", "total_tokens", "input_sha256", "ids_sha256", "mask_sha256", "target_role",
            "target_index", "context_indices", "source_mode", "source_sha256")
    return {k: index[k] for k in keys if k in index}


def safe_metadata(adapter):
    # Extension code must not inject prompts or input strings into public evidence.
    keys = ("precision", "backend", "backend_proof", "batch", "attention", "linear_attention",
            "batch_policy", "preparation", "backend_requested", "backend_actual", "variant_sha256", "buffer_dtypes",
            "batch_backend_proof", "batch_dispatch", "ort_evidence", "batch_ort_checked_shapes", "ort",
            "ort_profile_active_during_latency", "prepared_compile_count",
            "runtime", "dtype", "padding_side", "implementation", "source_inventory", "artifacts", "labels")
    value = {k: deepcopy(v) for k, v in adapter.metadata.items() if k in keys}
    return value


def public_labels(adapter, spec):
    if not isinstance(spec, dict) or "adapter" not in spec:
        return ()
    labels = (spec.get("scoring", {}).get("labels", ()) if isinstance(spec.get("adapter"), dict)
              else adapter.metadata.get("labels", ()))
    return tuple(v for v in labels if isinstance(v, str)) if isinstance(labels, (list, tuple)) else ()


def torch_peaks(torch):
    return {"allocated_bytes": torch.cuda.max_memory_allocated(0),
            "reserved_bytes": torch.cuda.max_memory_reserved(0)}


def check_indices(adapter, rows, guard):
    for row in rows:
        actual = public_index(guard.call(adapter.index, payload(row)))
        expected = public_index(row["index"])
        if actual != expected:
            raise ValueError("Actual prepared input differs from fixed index")


def preflight(adapter, rows, job, guard, result):
    check_indices(adapter, rows, guard)
    plans = smoke_plans(rows, job["batch_size"], job["seed"])
    groups = [[payload(r) for r in materialize(plan, rows)] for plan in plans]
    guard.call(adapter.prepare_batch_backend, groups)
    proof = guard.call(adapter.prove_batch_backend, groups[1])
    result["backend_proof"] = proof if isinstance(proof, dict) else adapter.metadata.get("backend_proof")
    if not result["backend_proof"]:
        raise ValueError("Actual backend proof missing")
    observed = []
    for plan, group in zip(plans, groups, strict=True):
        expected = [r["index"] for r in materialize(plan, rows)]
        values = responses(guard.call(adapter.verify_batch, group, expected), len(group))
        observed.append(values)
    guard.call(adapter.request_batch, groups[0])
    repeated = responses(guard.call(adapter.request_batch, groups[2]), len(groups[2]))
    if [s["label"] for s in repeated] != [s["label"] for s in observed[2]]:
        raise ValueError("A-B-A verdicts are unstable")
    edge, above = guard.call(adapter.boundary, job["bucket_max"])
    if guard.call(adapter.index, edge)["total_tokens"] != job["bucket_max"]:
        raise ValueError("Exact input boundary not demonstrated")
    edge_group = [edge] + groups[0][1:]
    guard.call(adapter.prepare_batch_backend, [edge_group])
    responses(guard.call(adapter.request_batch, edge_group), len(edge_group))
    guard.call(adapter.reject_batch_over_limit, [above] + groups[0][1:])
    result.update(plans=plans, boundary_checked=job["bucket_max"], over_limit_rejected=True,
                  aba_passed=True, responses=observed, unique_inputs=len(rows))


def latency(adapter, rows, job, guard, result, torch, sampler, checkpoint):
    plans = job["batches"]
    groups = [[payload(r) for r in materialize(p, rows)] for p in plans]
    start = time.monotonic()
    guard.call(adapter.prepare_batch_backend, groups)
    result["backend_preparation_seconds"] = time.monotonic() - start
    start = time.monotonic()
    for i in range(20):
        responses(guard.call(adapter.request_batch, groups[i % len(groups)]), len(groups[i % len(groups)]))
    torch.cuda.synchronize()
    result["warmup_seconds"] = time.monotonic() - start
    result["compile_before"] = adapter.compile_count
    torch.cuda.reset_peak_memory_stats(0)
    sampler.inference()
    result["rows"] = []
    for plan, group in zip(plans, groups, strict=True):
        torch.cuda.synchronize()
        stamp = time.time()
        start = time.perf_counter_ns()
        values = guard.call(adapter.request_batch, group)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if callable(getattr(adapter, "check_device", None)):
            adapter.check_device()
            guard.check()
        values = responses(values, len(group))
        result["rows"].append({"batch_id": plan["id"], "ids": plan["ids"], "latency_ms": elapsed,
            "scores": values, "input_sha256": plan["input_sha256"], "total_tokens": plan["total_tokens"],
            "padded_tokens": plan["padded_tokens"], "useful_tokens": plan["useful_tokens"], "timestamp": stamp})
        result["torch_inference"] = torch_peaks(torch)
        result["compile_after"] = adapter.compile_count
        if result["compile_after"] != result["compile_before"]:
            raise ValueError("New compilation inside latency series")
        checkpoint()


def quality(adapter, rows, job, guard, result, directory, checkpoint, spec=None):
    from .quality import metrics
    from .quality_batches import prepare_plan
    plan = prepare_plan(adapter, rows, job["quality_batch_size"], job["seed"], guard)
    write(directory / "quality_plan.json", plan)
    result.update(quality_batch_size=job["quality_batch_size"], quality_plan_sha256=digest(plan))
    by_id = {r["id"]: r for r in rows}
    predictions = {key: {"id": key, "status": plan["excluded"].get(key, "not_run")}
                   for key in plan["ordered_ids"]}
    groups = [[payload(by_id[key]) for key in batch["ids"]] for batch in plan["batches"]]

    def save():
        texts = private_values(rows, spec)
        allowed = public_labels(adapter, spec)
        values = []
        for prediction in predictions.values():
            value = deepcopy(prediction)
            if "score" in value and clean_evidence(value["score"], texts, allowed) != value["score"]:
                value = {"id": value["id"], "status": "private_output_rejected"}
            values.append(value)
        # IDs and responses only; source messages and rendered prompts stay private.
        pending = directory / "predictions.jsonl.tmp"
        with pending.open("w") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        pending.replace(directory / "predictions.jsonl")
        result["quality"] = metrics(rows, values, job["label_mapping"])
        result["quality_evaluated"] = result["quality"]["overall"]["evaluated"] > 0
        checkpoint()

    active = []
    try:
        if groups:
            guard.call(adapter.prepare_batch_backend, groups)
            result["backend_proof"] = guard.call(adapter.prove_batch_backend, groups[0])
        for batch, group in zip(plan["batches"], groups, strict=True):
            active = batch["ids"]
            values = responses(guard.call(adapter.verify_batch, group, [plan["indices"][i] for i in active]), len(group))
            for key, value in zip(active, values, strict=True):
                predictions[key] = {"id": key, "status": "passed", "score": value}
            active = []
            save()
    except BaseException as error:
        for key in active:
            predictions[key] = {"id": key, "status": error_status(error), "error_type": type(error).__name__}
        raise
    finally:
        save()


def profile(adapter, rows, job, guard, result, directory, torch):
    plans = smoke_plans(rows, job["batch_size"], job["seed"])[:2]
    paths = []
    for name, plan in zip(("short", "long"), plans, strict=True):
        group = [payload(r) for r in materialize(plan, rows)]
        guard.call(adapter.prepare_batch_backend, [group])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            guard.call(adapter.request_batch, group)
            torch.cuda.synchronize()
        raw = directory / (name + ".trace.json")
        prof.export_chrome_trace(str(raw))
        with raw.open("rb") as source, gzip.open(str(raw) + ".gz", "wb") as dest:
            import shutil
            shutil.copyfileobj(source, dest)
        raw.unlink()
        paths.append(raw.name + ".gz")
    result["profiles"] = paths


def execute(job_path, *, adapter_factory=None, runtime_factory=None, torch_module=None, sampler_factory=None):
    job_path = Path(job_path)
    directory, job = job_path.parent, read(job_path)
    result = {"result_version": 4, "status": "running", "job": job, "rows": [],
              "quality_evaluated": False, "precision_policy_version": 1,
              "resolved_sha256": job["resolved_sha256"]}
    adapter, sampler, rows = None, None, []
    config = {}
    start = time.monotonic()

    def checkpoint():
        if adapter is not None:
            result["metadata"] = safe_metadata(adapter)
        snapshot = deepcopy(result)
        texts = private_values(rows, config)
        allowed = public_labels(adapter, config.get(job["variant"])) if adapter is not None else ()
        for field in ("metadata", "backend_proof", "size", "responses"):
            if field in snapshot:
                cleaned = clean_evidence(snapshot[field], texts, allowed)
                if field == "responses" and cleaned != snapshot[field]:
                    snapshot.update(status="failed", error_type="PrivateOutputError")
                snapshot[field] = cleaned
        for row in snapshot.get("rows", []):
            if "scores" in row:
                cleaned = clean_evidence(row["scores"], texts, allowed)
                if cleaned != row["scores"]:
                    snapshot.update(status="failed", error_type="PrivateOutputError")
                row["scores"] = cleaned
        if sampler is not None and callable(getattr(sampler, "checkpoint", None)):
            try:
                current_memory = sampler.checkpoint(directory / "memory.json")
                if "memory" not in snapshot:
                    snapshot["memory"] = current_memory
            except Exception:
                pass  # Final sampler status is separate from inference failures.
        write(directory / "result.json", snapshot)

    checkpoint()
    try:
        resolved_path = Path(job["resolved_path"])
        if sha(resolved_path) != job["resolved_sha256"]:
            raise ValueError("Resolved snapshot changed")
        resolved = read(resolved_path)
        if resolved.get("schema_version") != 4 or resolved.get("result_version") != 4 or resolved.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported resolved snapshot format")
        check_sources(resolved["source_inventory"])
        config = resolved["config"]
        if "config_sha256" in resolved:
            if sha(job["private_config_path"]) != job["private_config_sha256"]:
                raise ValueError("Private configuration changed")
            config = read(job["private_config_path"])
            if digest(config) != resolved["config_sha256"] or public_config(config) != resolved["config"]:
                raise ValueError("Private configuration differs from public identity")
            if {k: digest(config[k]) for k in ("baseline", "candidate") if k in config} != resolved["variant_identities"]:
                raise ValueError("Variant identities changed")
        if sha(job["input_path"]) != job["input_file_sha256"]:
            raise ValueError("Private input snapshot changed")
        rows = read(job["input_path"])
        actual = (runtime_factory or configure_gpu)(config["environment"])
        assert_environment(actual, resolved["environment"])
        result["runtime"] = actual
        verify_local_sources(config[job["variant"]])
        if torch_module is None:
            import torch as torch_module
        if adapter_factory is None:
            from .v4_adapters import load_adapter
            adapter_factory = load_adapter
        if sampler_factory is None:
            from .v4_memory import Sampler
            sampler_factory = Sampler
        if job["stage"] == "monitor":
            from .v4_monitoring import DisabledSampler
            sampler_factory = DisabledSampler
        sampler = sampler_factory(gpu_uuid=actual["gpu_uuid"])
        sampler.start()
        torch_module.cuda.reset_peak_memory_stats(0)
        load_start = time.monotonic()
        adapter = adapter_factory(config[job["variant"]], directory / ".private")
        result["setup_seconds"] = time.monotonic() - load_start
        result["torch_setup"] = torch_peaks(torch_module)
        result["size"] = adapter.size()
        guard = PrecisionGuard(adapter, required=True)
        if job["stage"] == "preflight":
            torch_module.cuda.reset_peak_memory_stats(0)
            sampler.inference()
            preflight(adapter, rows, job, guard, result)
            result["torch_inference"] = torch_peaks(torch_module)
        elif job["stage"] == "latency":
            proof = read(job["preflight_path"])
            if sha(job["preflight_path"]) != job["preflight_sha256"] or proof["status"] != "passed" or proof["resolved_sha256"] != job["resolved_sha256"]:
                raise ValueError("Preflight evidence unavailable or changed")
            if any(proof.get("job", {}).get(k) != job[k] for k in ("variant", "bucket", "batch_size")):
                raise ValueError("Preflight proof belongs to a different variant/cell")
            if proof["metadata"]["precision"] != adapter.metadata["precision"]:
                raise ValueError("Precision changed after preflight")
            for field in ("artifacts", "preparation", "batch_policy", "attention", "linear_attention"):
                if proof["metadata"].get(field) != safe_metadata(adapter).get(field):
                    raise ValueError("Execution, artifact or preparation changed after preflight")
            result["backend_proof"] = proof["backend_proof"]
            latency(adapter, rows, job, guard, result, torch_module, sampler, checkpoint)
        elif job["stage"] == "quality":
            torch_module.cuda.reset_peak_memory_stats(0)
            sampler.inference()
            quality(adapter, rows, job, guard, result, directory, checkpoint, config[job["variant"]])
            result["torch_inference"] = torch_peaks(torch_module)
        elif job["stage"] == "profile":
            profile(adapter, rows, job, guard, result, directory, torch_module)
        elif job["stage"] == "monitor":
            from .v4_monitoring import measure
            result["monitoring"] = measure(adapter, rows, job["batches"], guard, torch_module,
                                            actual["gpu_uuid"], directory)
        else:
            raise ValueError("Unknown worker stage")
        if callable(getattr(adapter, "check_device", None)):
            adapter.check_device()
            guard.check()
        result["status"] = "passed"
    except BaseException as error:
        result.update(status=error_status(error), error_type=type(error).__name__)
    finally:
        result["seconds"] = time.monotonic() - start
        if sampler is not None:
            try:
                result["memory"] = sampler.stop(directory / "memory.json")
            except Exception as error:
                result["memory"] = {"status": "unavailable", "error_type": type(error).__name__}
        if adapter is not None:
            try:
                result["size"] = adapter.size()
            except Exception as error:
                result["size_error_type"] = type(error).__name__
            checkpoint()
            try:
                adapter.close()
            except Exception as error:
                result["close_error_type"] = type(error).__name__
                if result["status"] == "passed":
                    result["status"] = "failed"
        checkpoint()
    return read(directory / "result.json")
