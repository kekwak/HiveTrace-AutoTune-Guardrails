"""One model configuration per process; never imported by CPU-only reporting."""
import gzip
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from .core import PRESETS, digest, load_data, read, score_check, write


def telemetry():
    p = subprocess.run(["nvidia-smi", "--id=0",
                        "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm,clocks.mem",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if p.returncode:
        raise RuntimeError("GPU telemetry unavailable: " + p.stderr)
    values = [v.strip() for v in p.stdout.strip().split(",")]
    if len(values) != 6:
        raise RuntimeError("Unexpected GPU telemetry")
    return dict(zip(("uuid", "name", "driver", "temperature_c", "sm_mhz", "memory_mhz"), values))


def verify_actual_request(a, row):
    """Observe the tensors actually entering the model in the full scores path."""
    from .models import normalized, stable_call
    captured = []
    target = a.model if hasattr(a, "qwen") else a.model.encoder
    def observe(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        if mask is None or ids is None:
            raise ValueError("Cannot observe actual IDs/mask")
        captured.append(digest(normalized({"input_ids": ids, "attention_mask": mask})))
    hook = target.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        result = stable_call(a, row["text"])
    finally:
        hook.remove()
    if captured != [row["index"]["input_sha256"]]:
        raise ValueError("Full scoring path input mismatch: " + row["id"])
    return result


def preparation_rows(model, phase, all_rows, measurement=None):
    """Warm every shape used by this Halo worker, not unrelated future series."""
    if model != "halo" or phase not in ("latency", "profile"):
        return all_rows, "full_corpus_and_limit"
    if phase == "latency":
        by_id = {row["id"]: row for row in all_rows}
        return [by_id[key] for key in measurement["ids"]], "measurement_shapes"
    ordered = sorted(all_rows, key=lambda row: row["index"]["total_tokens"])
    return [ordered[0], ordered[-1]], "profile_shapes"


def execute(job_path):
    job = read(job_path)
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    result = {"job": job.get("measurement"), "status": "running", "phase": job["phase"],
              "variant": job["variant"], "pid": os.getpid(), "quality_evaluated": False}
    result["config_sha256"] = digest(job["config"])
    write(directory / "result.json", result)
    adapter, sampler = None, None
    start = time.monotonic()
    try:
        initial = telemetry()
        write(directory / "telemetry_before.json", initial)
        if job["phase"] == "memory":
            from .memory import Sampler
            sampler = Sampler(initial["uuid"])
            sampler.start()
        from followup.team_skeleton import probe
        probe(Path(job["environment"]), directory)
        import torch
        from .models import activate, boundary_text, check_input, load, model_size, stable_call
        model, candidate = PRESETS[job["config"]["preset"]]
        data, manifest = load_data(job["dataset"], model)
        all_rows = [r for rows in data.values() for r in rows]
        rows_by_id = {r["id"]: r for r in all_rows}
        result["dataset_contract_sha256"] = digest(manifest["indices"][model]["contract"])
        with torch.inference_mode():
            stage_start = time.monotonic()
            adapter = load(model, job["variant"], candidate, directory)
            torch.cuda.synchronize()
            result["model_load_seconds"] = time.monotonic() - stage_start
            if adapter.contract["model"] != manifest["indices"][model]["contract"]["repo"] or adapter.contract["revision"] != manifest["indices"][model]["contract"]["revision"]:
                raise ValueError("Wrong checkpoint contract")
            stage_start = time.monotonic()
            for row in all_rows:
                check_input(adapter, row)
            # Input validation stays corpus-wide; warmup covers this worker's calls.
            warm_rows, warm_scope = preparation_rows(model, job["phase"], all_rows, job.get("measurement"))
            representatives = {}
            for row in warm_rows:
                representatives.setdefault(row["index"]["total_tokens"], row["text"])
            edge = boundary_text(adapter, adapter.limit)
            above = boundary_text(adapter, adapter.limit + 1)
            if warm_scope == "full_corpus_and_limit":
                representatives[adapter.limit] = edge
            result["input_validation_seconds"] = time.monotonic() - stage_start
            stage_start = time.monotonic()
            activate(adapter, candidate, job["variant"], list(representatives.values()), directory)
            torch.cuda.synchronize()
            result["backend_preparation_seconds"] = time.monotonic() - stage_start
            result.update(setup_seconds=time.monotonic() - start, metadata=adapter.metadata,
                          contract=adapter.contract, size=model_size(adapter, directory),
                          input_validation_count=len(all_rows), prepared_shapes=sorted(representatives),
                          preparation_scope=warm_scope)
            write(directory / "prepared.json", result)

            if job["phase"] == "preflight":
                records = []
                with (directory / "scores.jsonl").open("w") as stream:
                    for row in all_rows:
                        record = {"id": row["id"], "bucket": row["bucket"],
                                  "score": verify_actual_request(adapter, row)}
                        record["distance_to_threshold"] = abs(record["score"]["p_unsafe"] - adapter.threshold)
                        records.append(record)
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                first, last = all_rows[0]["text"], all_rows[-1]["text"]
                a1 = stable_call(adapter, first)
                stable_call(adapter, last)
                a2 = stable_call(adapter, first)
                if a1["label"] != a2["label"] or abs(a1["p_unsafe"] - a2["p_unsafe"]) > 1e-6:
                    raise ValueError("A-B-A repeat instability")
                boundary_score = stable_call(adapter, edge)
                calls = []
                target = adapter.model if model == "halo" else adapter.model.encoder
                hook = target.register_forward_pre_hook(lambda *args: calls.append(1))
                try:
                    try:
                        adapter.scores([above])
                    except ValueError as error:
                        if "limit" not in str(error) and "exceeds" not in str(error):
                            raise
                    else:
                        raise RuntimeError("Over-limit input was accepted")
                finally:
                    hook.remove()
                if calls:
                    raise RuntimeError("Over-limit request reached the model")
                result.update(scores_count=len(records), aba=[a1, a2],
                              boundary={"tokens": adapter.limit, "score": boundary_score,
                                        "over_limit_rejected_before_forward": True})

            elif job["phase"] == "latency":
                ids = job["measurement"]["ids"]
                warm_start = time.monotonic()
                for i in range(20):
                    stable_call(adapter, rows_by_id[ids[i % len(ids)]]["text"])
                torch.cuda.synchronize()
                result["warmup_seconds"] = time.monotonic() - warm_start
                result["started_unix"] = time.time()
                records = []
                with (directory / "latencies.jsonl").open("w") as stream:
                    for i in ids:
                        row = rows_by_id[i]
                        if row["index"]["total_tokens"] not in representatives:
                            raise RuntimeError("Unprepared shape in latency series")
                        before = adapter.compile_count
                        torch.cuda.synchronize()
                        t0 = time.perf_counter_ns()
                        scores = adapter.scores([row["text"]])
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns() - t0) / 1e6
                        if adapter.compile_count != before:
                            raise RuntimeError("Compilation inside latency series")
                        if len(scores) != 1:
                            raise ValueError("Missing response")
                        score_check(scores[0])
                        record = {"id": i, "latency_ms": elapsed, "score": scores[0],
                                  "finished_unix": time.time(),
                                  "input_sha256": row["index"]["input_sha256"],
                                  "total_tokens": row["index"]["total_tokens"]}
                        records.append(record)
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                result.update(rows=records, finished_unix=time.time())

            elif job["phase"] == "memory":
                warm_start = time.monotonic()
                for rows in data.values():
                    for i in range(20):
                        stable_call(adapter, rows[i % len(rows)]["text"])
                torch.cuda.synchronize()
                result["warmup_seconds"] = time.monotonic() - warm_start
                result["torch_setup"] = {"allocated": torch.cuda.max_memory_allocated(),
                                         "reserved": torch.cuda.max_memory_reserved()}
                torch.cuda.reset_peak_memory_stats()
                result["torch_at_inference_start"] = {"allocated": torch.cuda.memory_allocated(),
                                                      "reserved": torch.cuda.memory_reserved()}
                sampler.inference()
                for row in all_rows:
                    stable_call(adapter, row["text"])
                torch.cuda.synchronize()
                result["torch_inference"] = {"allocated": torch.cuda.max_memory_allocated(),
                                             "reserved": torch.cuda.max_memory_reserved()}
                result["memory"] = sampler.stop(directory / "memory.json")
                sampler = None

            elif job["phase"] == "profile":
                ordered = sorted(all_rows, key=lambda r: r["index"]["total_tokens"])
                for label, row in (("short", ordered[0]), ("long", ordered[-1])):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as p:
                        with torch.profiler.record_function(f"{model}/{job['variant']}/text_to_verdict"):
                            stable_call(adapter, row["text"])
                        torch.cuda.synchronize()
                    trace = directory / f"{label}.trace.json"
                    p.export_chrome_trace(str(trace))
                    with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dest:
                        import shutil
                        shutil.copyfileobj(src, dest)
                    trace.unlink()
                    (directory / f"{label}.operators.txt").write_text(p.key_averages().table(
                        sort_by="self_device_time_total", row_limit=40))
                    write(directory / f"{label}.input.json", {"id": row["id"], "tokens": row["index"]["total_tokens"],
                          "note": "Separate profile; unproven kernel-to-layer attribution remains unknown"})
            else:
                raise ValueError("Unknown worker phase")
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        (directory / "error.txt").write_text(traceback.format_exc())
    finally:
        if sampler is not None:
            result["memory"] = sampler.stop(directory / "memory.json")
        if adapter is not None:
            try:
                adapter.close()
            except Exception as e:
                result.update(status="failed", cleanup_error=str(e))
        if result["status"] == "passed":
            removed, failures = [], []
            for path in directory.iterdir():
                if path.is_file() and not path.is_symlink() and (
                    path.suffix in (".onnx", ".data") or ".onnx." in path.name):
                    try:
                        record = {"name": path.name, "bytes": path.stat().st_size}
                        path.unlink()
                        removed.append(record)
                    except OSError as error:
                        failures.append({"name": path.name, "error": str(error)})
            if removed or failures:
                write(directory / "export_cleanup.json", {"removed": removed, "failures": failures,
                      "note": "Regenerable exports after worker completion; sizes and graph/provider evidence retained"})
        result["seconds"] = time.monotonic() - start
        write(directory / "result.json", result)
        try:
            write(directory / "telemetry_after.json", telemetry())
        except Exception:
            pass
    return result
