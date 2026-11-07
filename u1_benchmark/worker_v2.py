"""One full-request adapter per fresh process; no cached tokenized inputs in timing."""
import contextlib
import gzip
import json
import os
import shutil
import time
from pathlib import Path

from .contracts import model_contract, parse_config, validate_response
from .core import digest, read, write
from .datasets import jsonl, load_speed, verified
from .messages import payload, private_strings, UnsupportedInput
from .precision import PrecisionError, PrecisionGuard


@contextlib.contextmanager
def private_output(enabled):
    """Suppress native/SDK stdout too: Python redirect alone misses C-level writes."""
    saved = []
    try:
        if enabled:
            for fd in (1, 2):
                saved.append((fd, os.dup(fd)))
            with open(os.devnull, "w") as sink:
                for fd, _ in saved:
                    os.dup2(sink.fileno(), fd)
        yield
    finally:
        for fd, old in saved:
            os.dup2(old, fd)
            os.close(old)


def current_rows(phase, data, measurement=None):
    rows = [r for values in data.values() for r in values]
    if phase == "latency":
        by_id = {r["id"]: r for r in rows}
        return [by_id[key] for key in measurement["ids"]]
    if phase in ("smoke", "profile"):
        selected = []
        for values in data.values():
            ordered = sorted(values, key=lambda r: r["index"]["total_tokens"])
            selected += ordered[:1] + ordered[-1:]
        return list({r["id"]: r for r in selected}.values())
    return rows


def stable_request(adapter, text, precision=None):
    count = adapter.compile_count
    value = precision.call(adapter.request, text) if precision is not None else adapter.request(text)
    if adapter.compile_count != count:
        raise RuntimeError("New compilation during measurement")
    return validate_response(value)


def redact_files(directory, texts):
    """Guard against accidental plaintext echoes in adapter metadata/diagnostics."""
    needles = sorted({s for s in texts if s}, key=len, reverse=True)
    for path in Path(directory).rglob("*"):
        if not path.is_file() or ".private" in path.parts or path.suffix not in (".json", ".jsonl", ".txt", ".log"):
            continue
        def clean(value, key=None):
            # These are validated protocol fields, not arbitrary diagnostic prose.
            # A private input equal to "safe" must not rewrite a native verdict.
            if key in {"id", "label", "scores", "score_type", "category", "generated_tokens",
                       "p_safe", "p_unsafe", "logit_margin", "public_confidence", "contract",
                       "adapter_spec", "job", "config_sha256", "inputs_sha256", "text_sha256",
                       "input_sha256", "mask_sha256", "ids_sha256", "variant", "phase", "status"}:
                return value
            if key == "precision":
                from .precision import validate_precision
                try:
                    return validate_precision(value)
                except PrecisionError:
                    pass
            if isinstance(value, dict):
                return {k: clean(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            if isinstance(value, str):
                for text in needles:
                    if len(text) >= 4:
                        value = value.replace(text, "[redacted-input]")
            return value
        # Never replace bytes in serialized JSON: quotes/newlines can corrupt it.
        if path.suffix == ".json":
            write(path, clean(read(path)))
        elif path.suffix == ".jsonl":
            jsonl(path, [clean(json.loads(line)) for line in path.read_text().splitlines() if line])
        else:
            path.write_text(clean(path.read_text()))


def execute(job_path):
    from .worker import telemetry
    job = read(job_path)
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 2, "status": "running", "phase": job["phase"], "variant": job["variant"],
              "job": job.get("measurement"), "pid": os.getpid(), "quality_evaluated": False,
              "config_sha256": digest(job["config"]), "inputs_sha256": job["inputs_sha256"]}
    write(directory / "result.json", result)
    adapter = sampler = None
    start = time.monotonic()
    rows = []
    with private_output(job["private"]):
        try:
            cfg = parse_config(job["config"])
            if cfg["model"] == "laya":
                result["precision_policy_version"] = 1
            from .environment import probe
            probe(Path(cfg["environment"]), directory)
            import torch
            from .adapters import check_index, load
            from .memory import Sampler
            initial = telemetry()
            write(directory / "telemetry_before.json", initial)
            phase = job["phase"]
            if phase == "quality":
                q = cfg["quality_data"]
                rows, manifest = verified(q["path"], q["manifest_sha256"])
                data = {}
            else:
                data, manifest = load_speed(job["dataset"], cfg["model"], cfg["mode"], cfg["data"].get("manifest_sha256"))
                rows = current_rows(phase, data, job.get("measurement"))
            if phase in ("memory", "smoke"):
                sampler = Sampler(initial["uuid"])
                sampler.start()
            diag = directory / (".private" if job["private"] else "diagnostics")
            diag.mkdir(exist_ok=True)
            with torch.inference_mode():
                t = time.monotonic()
                adapter = (load(cfg["model"], cfg[job["variant"]], diag, input_version=3) if cfg["schema_version"] == 3
                           else load(cfg["model"], cfg[job["variant"]], diag))
                torch.cuda.synchronize()
                result["load_seconds"] = time.monotonic() - t
                if adapter.contract != model_contract(cfg["model"], cfg["schema_version"]):
                    raise ValueError("Adapter contract changed")
                result["contract"] = adapter.contract
                result["adapter_spec"] = cfg[job["variant"]]
                precision = PrecisionGuard(adapter, required=cfg["model"] == "laya")
                result["metadata"] = adapter.metadata
                if phase == "quality":
                    values = []
                    quality_indices, quality_errors, shapes = {}, {}, {}
                    for row in rows:
                        try:
                            index = adapter.index(payload(row))
                            if "target_role" in row and (index.get("target_role"), index.get("target_index")) != (row["target_role"], row["target_index"]):
                                quality_errors[row["id"]] = "target_mismatch"
                                continue
                            quality_indices[row["id"]] = index
                            if 0 < index["total_tokens"] <= adapter.limit:
                                shapes.setdefault(index["total_tokens"], payload(row))
                        except PrecisionError:
                            raise
                        except UnsupportedInput:
                            quality_errors[row["id"]] = "not_applicable"
                        except Exception as error:
                            quality_errors[row["id"]] = type(error).__name__
                    if shapes:
                        precision.call(adapter.prepare_backend, list(shapes.values()))
                    result["metadata"] = adapter.metadata
                    for row in rows:
                        item = {k: row[k] for k in ("id", "text_sha256", "content_sha256") if k in row}
                        try:
                            if row["id"] in quality_errors:
                                reason = quality_errors[row["id"]]
                                item.update(status=reason if reason in ("target_mismatch", "not_applicable") else "error", error_type=reason)
                                values.append(item)
                                continue
                            index = quality_indices[row["id"]]
                            item["total_tokens"] = index["total_tokens"]
                            if not 0 < index["total_tokens"] <= adapter.limit:
                                item["status"] = "over_limit"
                            else:
                                # Quality has no latency; all required preparation stays separate.
                                item.update(status="passed", score=precision.call(adapter.verify, payload(row), index))
                        except PrecisionError:
                            raise
                        except Exception as error:
                            item.update(status="error", error_type=type(error).__name__)
                        values.append(item)
                        jsonl(directory / "predictions.jsonl", values)
                    from .quality import metrics
                    jsonl(directory / "predictions.jsonl", values)
                    result.update(quality=metrics(rows, values, cfg["label_mapping"]), quality_evaluated=True)
                else:
                    for row in rows:
                        check_index(adapter.index(payload(row)), row["index"])
                    shapes = {}
                    for row in rows:
                        shapes.setdefault(row["index"]["total_tokens"], payload(row))
                    edge = above = None
                    if phase in ("preflight", "smoke"):
                        edge, above = precision.call(adapter.boundary)
                        shapes[adapter.limit] = edge
                    t = time.monotonic()
                    precision.call(adapter.prepare_backend, list(shapes.values()))
                    if hasattr(adapter, "prove_backend"):
                        precision.call(adapter.prove_backend, payload(rows[0]))
                    elif not adapter.metadata.get("test_fixture"):
                        raise ValueError("External adapter must provide backend proof")
                    torch.cuda.synchronize()
                    result.update(backend_preparation_seconds=time.monotonic() - t, prepared_shapes=sorted(shapes),
                                  setup_seconds=time.monotonic() - start, metadata=adapter.metadata,
                                  size=adapter.size(), prepared_compile_count=adapter.compile_count)
                    write(directory / "prepared.json", result)
                    if phase in ("preflight", "smoke"):
                        values = []
                        for row in rows:
                            values.append({"id": row["id"], "bucket": row["bucket"], "score": precision.call(adapter.verify, payload(row), row["index"])})
                            jsonl(directory / "scores.jsonl", values)
                        first, last = payload(rows[0]), payload(rows[-1])
                        a = stable_request(adapter, first, precision)
                        stable_request(adapter, last, precision)
                        b = stable_request(adapter, first, precision)
                        if a["label"] != b["label"]:
                            raise ValueError("A-B-A instability")
                        for key in a.get("scores", {}):
                            if abs(a["scores"][key] - b["scores"][key]) > 1e-6:
                                raise ValueError("A-B-A numeric instability")
                        boundary = stable_request(adapter, edge, precision)
                        precision.call(adapter.reject_over_limit, above)
                        result.update(aba=[a, b], boundary={"tokens": adapter.limit, "score": boundary, "over_limit_rejected": True})
                        if phase == "smoke":
                            result["timings"] = []
                            sampler.inference()
                            for row in rows:
                                torch.cuda.synchronize()
                                t0 = time.perf_counter_ns()
                                score = stable_request(adapter, payload(row), precision)
                                torch.cuda.synchronize()
                                result["timings"].append({"id": row["id"], "bucket": row["bucket"], "latency_ms": (time.perf_counter_ns() - t0) / 1e6, "score": score})
                    elif phase == "latency":
                        t = time.monotonic()
                        for i in range(20):
                            stable_request(adapter, payload(rows[i % len(rows)]), precision)
                        torch.cuda.synchronize()
                        result.update(warmup_seconds=time.monotonic() - t, started_unix=time.time())
                        values = []
                        with (directory / "latencies.jsonl").open("w") as stream:
                            for row in rows:
                                before = adapter.compile_count
                                torch.cuda.synchronize()
                                t0 = time.perf_counter_ns()
                                score = precision.call(adapter.request, payload(row))
                                torch.cuda.synchronize()
                                elapsed = (time.perf_counter_ns() - t0) / 1e6
                                if adapter.compile_count != before:
                                    raise RuntimeError("Compilation inside latency")
                                validate_response(score)
                                item = {"id": row["id"], "score": score, "latency_ms": elapsed,
                                        "input_sha256": row["index"]["input_sha256"], "total_tokens": row["index"]["total_tokens"], "finished_unix": time.time()}
                                values.append(item)
                                stream.write(json.dumps(item, allow_nan=False) + "\n")
                                stream.flush()
                        result.update(rows=values, finished_unix=time.time())
                    elif phase == "memory":
                        for i in range(20):
                            stable_request(adapter, payload(rows[i % len(rows)]), precision)
                        torch.cuda.synchronize()
                        result["torch_setup"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
                        torch.cuda.reset_peak_memory_stats()
                        sampler.inference()
                        for row in rows:
                            stable_request(adapter, payload(row), precision)
                        torch.cuda.synchronize()
                        result["torch_inference"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
                    elif phase == "profile":
                        ordered = sorted(rows, key=lambda r: r["index"]["total_tokens"])
                        for name, row in (("short", ordered[0]), ("long", ordered[-1])):
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                                stable_request(adapter, payload(row), precision)
                                torch.cuda.synchronize()
                            trace = directory / (name + ".trace.json")
                            profiler.export_chrome_trace(str(trace))
                            with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            trace.unlink()
                            (directory / (name + ".operators.txt")).write_text(profiler.key_averages().table(sort_by="self_device_time_total", row_limit=40))
                    else:
                        raise ValueError("Unknown phase")
                precision.check()
                result["status"] = "passed"
        except BaseException as error:
            result.update(status="failed", error_type=type(error).__name__)
            if not job["private"]:
                import traceback
                (directory / "error.txt").write_text(traceback.format_exc())
        finally:
            if sampler is not None:
                result["memory"] = sampler.stop(directory / "memory.json")
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as error:
                    result.update(status="failed", cleanup_error_type=type(error).__name__)
            result["seconds"] = time.monotonic() - start
            write(directory / "result.json", result)
            try:
                write(directory / "telemetry_after.json", telemetry())
            except Exception:
                pass
            if job["private"]:
                redact_files(directory, [text for r in rows for text in private_strings(payload(r))])
    return result
