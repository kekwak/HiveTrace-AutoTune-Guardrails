"""Fresh-process execution of one static-batch cell; diagnostics never enter timing."""
import gzip
import json
import math
import os
import shutil
import time
from pathlib import Path

from .contracts import BUCKETS, model_contract, parse_config, validate_response
from .core import digest, read, sha, write
from .datasets import load_speed
from .messages import payload, private_strings
from .precision import PrecisionGuard


def responses(values, size):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError("Batch response count/order contract violated")
    return [validate_response(value) for value in values]


def repeat_check(first, second):
    if len(first) != len(second):
        raise ValueError("Batch repeat changed response count")
    for a, b in zip(first, second, strict=True):
        if (a["label"], a.get("category"), a.get("generated_tokens")) != (
                b["label"], b.get("category"), b.get("generated_tokens")):
            raise ValueError("Batch A-B-A verdict instability")
        if set(a.get("scores", {})) != set(b.get("scores", {})):
            raise ValueError("Batch A-B-A score keys changed")
        if any(abs(v - b["scores"][k]) > 1e-6 for k, v in a.get("scores", {}).items()):
            raise ValueError("Batch A-B-A numeric instability")


def cell_boundary(adapter, bucket):
    """Do not stress a short-input batch at the decoder's global 4096 limit."""
    target = BUCKETS[bucket]["max"]
    model_edge, above = adapter.boundary()
    if target == adapter.limit:
        return model_edge, above
    for unit in ("a ", "hello ", "text ", "0 ", "x"):
        low, high = 0, target * 2
        while low <= high:
            middle = (low + high) // 2
            text = unit * middle + "."
            length = adapter.index(text)["total_tokens"]
            if length == target:
                return text, above
            if length < target:
                low = middle + 1
            else:
                high = middle - 1
    raise ValueError("Could not construct exact batch bucket boundary")


def verify_evidence(job, directory, adapter):
    from .report_v2 import identity
    evidence = job["preflight"]
    root = Path(evidence["path"])
    if sha(root / "result.json") != evidence["sha256"]:
        raise ValueError("Batch preflight evidence changed")
    proof = read(root / "result.json")
    for key, expected in (("status", "passed"), ("batch_protocol_version", 1),
                          ("config_sha256", digest(job["config"])), ("variant", job["variant"]),
                          ("inputs_sha256", job["inputs_sha256"]), ("sources_sha256", job["sources_sha256"]),
                          ("contract", adapter.contract), ("bucket", job["measurement"]["bucket"]),
                          ("batch_size", job["measurement"]["batch_size"])):
        if proof.get(key) != expected:
            raise ValueError("Batch preflight identity mismatch: " + key)
    if identity(root) != identity(directory):
        raise ValueError("Batch worker environment differs from preflight")
    for key in ("precision", "batch_policy"):
        if proof["metadata"].get(key) != adapter.metadata.get(key):
            raise ValueError("Batch execution policy differs from preflight")


def execute(job_path):
    from .worker import telemetry
    from .worker_v2 import private_output, redact_files

    job = read(job_path)
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    measurement = job["measurement"]
    result = {"schema_version": job["schema_version"], "batch_protocol_version": 1,
              "status": "running", "phase": job["phase"], "variant": job["variant"],
              "bucket": measurement["bucket"], "batch_size": measurement["batch_size"],
              "job": measurement, "pid": os.getpid(), "quality_evaluated": False,
              "config_sha256": digest(job["config"]), "inputs_sha256": job["inputs_sha256"],
              "sources_sha256": job["sources_sha256"]}
    write(directory / "result.json", result)
    start = time.monotonic()
    adapter = raw = sampler = None
    rows = []
    with private_output(job["private"]):
        try:
            if job["batch_protocol_version"] != 1:
                raise ValueError("Unknown batch protocol")
            cfg = parse_config(job["config"])
            from .environment import probe
            probe(Path(cfg["environment"]), directory)
            import torch
            from .adapters import check_index, load
            from .batch_adapters import wrap
            from .batching import materialize
            from .memory import Sampler

            initial = telemetry()
            write(directory / "telemetry_before.json", initial)
            data, _ = load_speed(job["dataset"], cfg["model"], cfg["mode"], cfg["data"].get("manifest_sha256"))
            rows = data[measurement["bucket"]]
            plans = measurement["batches"]
            size = measurement["batch_size"]
            if not plans or any(len(p["ids"]) != size or len(set(p["ids"])) != size for p in plans):
                raise ValueError("Incomplete or duplicate batch members")
            members = [materialize(p, rows) for p in plans]
            batches = [[payload(row) for row in group] for group in members]
            indices = [[row["index"] for row in group] for group in members]
            if job["phase"] in ("batch_memory", "batch_preflight"):
                sampler = Sampler(initial["uuid"])
                sampler.start()
            diag = directory / (".private" if job["private"] else "diagnostics")
            diag.mkdir(exist_ok=True)
            with torch.inference_mode():
                t = time.monotonic()
                raw = load(cfg["model"], cfg[job["variant"]], diag, input_version=cfg["schema_version"])
                adapter = wrap(raw)
                if adapter.contract != model_contract(cfg["model"], cfg["schema_version"]):
                    raise ValueError("Batch inference contract changed")
                precision = PrecisionGuard(adapter, required=cfg["model"] == "laya")
                if cfg["model"] == "laya":
                    result["precision_policy_version"] = 1
                torch.cuda.synchronize()
                result.update(load_seconds=time.monotonic() - t, contract=adapter.contract,
                              adapter_spec=cfg[job["variant"]], metadata=adapter.metadata)
                if job["phase"] == "batch_preflight":
                    for row in rows:
                        check_index(adapter.index(payload(row)), row["index"])
                t = time.monotonic()
                # Shape preparation is required anew for fresh compiled/ORT processes.
                precision.call(adapter.prepare_batch_backend, batches)
                torch.cuda.synchronize()
                result.update(backend_preparation_seconds=time.monotonic() - t,
                              prepared_compile_count=adapter.compile_count,
                              prepared_shapes=sorted({(size, p["padded_length"]) for p in plans}),
                              metadata=adapter.metadata, size=adapter.size(), setup_seconds=time.monotonic() - start)

                def request(batch):
                    count = adapter.compile_count
                    answer = responses(precision.call(adapter.request_batch, batch), len(batch))
                    if adapter.compile_count != count:
                        raise RuntimeError("Compilation inside batch measurement")
                    return answer

                if job["phase"] != "batch_preflight":
                    verify_evidence(job, directory, adapter)
                if job["phase"] == "batch_preflight":
                    verified = [responses(precision.call(adapter.verify_batch, batch, expected), size)
                                for batch, expected in zip(batches, indices, strict=True)]
                    first = request(batches[-1])
                    request(batches[0])
                    second = request(batches[-1])
                    repeat_check(first, second)
                    edge, above = precision.call(cell_boundary, adapter, measurement["bucket"])
                    edge_batch = [edge] + batches[0][1:]
                    above_batch = [above] + batches[0][1:]
                    precision.call(adapter.prepare_batch_backend, [edge_batch])
                    responses(precision.call(adapter.verify_batch, edge_batch,
                              [adapter.index(value) for value in edge_batch]), size)
                    precision.call(adapter.reject_batch_over_limit, above_batch)
                    precision.call(adapter.prove_batch_backend, batches[-1])
                    result.update(aba_passed=True, boundary={"tokens": BUCKETS[measurement["bucket"]]["max"],
                                  "model_limit": adapter.limit, "over_limit_rejected": True},
                                  verified_batches=[{"batch": p, "scores": a} for p, a in zip(plans, verified, strict=True)])
                    sampler.inference()
                    timings = []
                    for p, batch in zip(plans, batches, strict=True):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter_ns()
                        values = request(batch)
                        torch.cuda.synchronize()
                        timings.append({"batch": p, "scores": values, "latency_ms": (time.perf_counter_ns() - t0) / 1e6})
                    result["timings"] = timings
                else:
                    t = time.monotonic()
                    for i in range(20):
                        request(batches[i % len(batches)])
                    torch.cuda.synchronize()
                    result.update(warmup_seconds=time.monotonic() - t, started_unix=time.time())
                    if job["phase"] == "batch_latency":
                        values = []
                        with (directory / "latencies.jsonl").open("w") as stream:
                            for plan, batch in zip(plans, batches, strict=True):
                                before = adapter.compile_count
                                torch.cuda.synchronize()
                                started = time.time()
                                t0 = time.perf_counter_ns()
                                answer = precision.call(adapter.request_batch, batch)
                                torch.cuda.synchronize()
                                elapsed = (time.perf_counter_ns() - t0) / 1e6
                                if adapter.compile_count != before:
                                    raise RuntimeError("Compilation inside batch latency")
                                responses(answer, size)
                                if not math.isfinite(elapsed) or elapsed <= 0:
                                    raise ValueError("Invalid batch latency")
                                item = {"batch": plan, "scores": answer, "latency_ms": elapsed,
                                        "started_unix": started, "finished_unix": time.time()}
                                values.append(item)
                                stream.write(json.dumps(item, allow_nan=False) + "\n")
                                stream.flush()
                        result["rows"] = values
                    elif job["phase"] == "batch_memory":
                        result["torch_setup"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
                        torch.cuda.reset_peak_memory_stats()
                        sampler.inference()
                        for batch in batches:
                            request(batch)
                        torch.cuda.synchronize()
                        result["torch_inference"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
                    elif job["phase"] == "batch_profile":
                        for label, batch in zip(("short", "long"), batches, strict=True):
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                                request(batch)
                                torch.cuda.synchronize()
                            trace = directory / (label + ".trace.json")
                            profiler.export_chrome_trace(str(trace))
                            with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            trace.unlink()
                            (directory / (label + ".operators.txt")).write_text(profiler.key_averages().table(sort_by="self_device_time_total", row_limit=40))
                    else:
                        raise ValueError("Unknown batch phase")
                precision.check()
                if job["phase"] == "batch_preflight":
                    result["prepared_compile_count"] = adapter.compile_count
                result.update(status="passed", finished_unix=time.time(), metadata=adapter.metadata,
                              final_compile_count=adapter.compile_count)
        except BaseException as error:
            kind = type(error).__name__
            result.update(status="oom" if kind in ("OutOfMemoryError", "CUDAOutOfMemoryError") else
                          "unsupported" if kind in ("UnsupportedBatch", "UnsupportedExperiment") else "failed", error_type=kind)
            if not job["private"]:
                import traceback
                (directory / "error.txt").write_text(traceback.format_exc())
        finally:
            if sampler is not None:
                result["memory"] = sampler.stop(directory / "memory.json")
            if raw is not None:
                try:
                    (adapter or raw).close()
                except Exception as error:
                    result.update(status="failed", cleanup_error_type=type(error).__name__)
            result["seconds"] = time.monotonic() - start
            write(directory / "result.json", result)
            try:
                write(directory / "telemetry_after.json", telemetry())
            except Exception:
                pass
            if job["private"]:
                redact_files(directory, [text for row in rows for text in private_strings(payload(row))])
    return result
