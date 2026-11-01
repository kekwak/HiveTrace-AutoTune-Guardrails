"""Offline acceptance and baseline aggregation; never imports GPU libraries."""
import contextlib
import gzip
import json
import math
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from .core import digest, lines, read, sha, write
from .quality import agreement
from .runner_v2 import schedule
from .statistics import assess, paired, paired_rows, quantile, baseline_diagnostics


def validate_worker(path, cfg, variant, inputs_hash):
    result = read(path / "result.json")
    if (result["status"] != "passed" or result["config_sha256"] != digest(cfg)
            or result["variant"] != variant or result["inputs_sha256"] != inputs_hash):
        raise ValueError("Incomplete/mismatched worker: " + str(path.name))
    return result


def identity(path):
    runtime, gpu = read(path / "runtime.json"), read(path / "telemetry_before.json")
    values = {k: runtime[k] for k in ("python", "torch", "cuda", "cudnn", "gpu", "seed", "cpu_threads", "tf32", "hub_kernels")}
    values.update(uuid=gpu["uuid"], driver=gpu["driver"], packages=read(path / "packages.json"))
    return digest(values)


def build(directory):
    root = Path(directory)
    cfg, state = read(root / "config.json"), read(root / "status.json")
    result = {"schema_version": cfg["schema_version"], "status": "incomplete", "mode": cfg["mode"], "model": cfg["model"],
              "comparison": cfg["comparison"], "quality_evaluated": False, "quality_preserved": None,
              "independently_confirmed": False, "buckets": {}, "memory": [], "errors": [],
              "speed_status": "not_measured", "verdict_status": "not_measured", "quality_status": "not_evaluated"}
    variants = ["baseline"] + (["candidate"] if cfg.get("candidate") else [])
    try:
        if state["status"] != "completed":
            raise ValueError("Execution incomplete: " + state["status"] + "/" + state.get("phase", "unknown"))
        ih = sha(root / "inputs.json")
        data = defaultdict(list)
        for row in read(root / "inputs.json"):
            data[row["bucket"]].append(row)
        for name, checksum in read(root / "source_files.json").items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != checksum:
                raise ValueError("Archived source hash mismatch")
        jobs = schedule(cfg, data)
        if read(root / "schedule.json") != jobs:
            raise ValueError("Schedule mismatch")
        expected_contract = None
        identities = set()
        fixtures = []
        def checked(path, variant):
            nonlocal expected_contract
            value = validate_worker(path, cfg, variant, ih)
            if expected_contract is None:
                expected_contract = value["contract"]
            if value["contract"] != expected_contract:
                raise ValueError("Inference contract changed")
            if value.get("metadata", {}).get("test_fixture"):
                fixtures.append(str(path))
            identities.add(identity(path))
            return value
        if cfg["mode"] != "quality":
            preflight = {}
            for variant in variants:
                path = root / "preflight" / variant
                pf = checked(path, variant)
                scores = lines(path / "scores.jsonl")
                from .worker_v2 import current_rows
                expected_ids = [r["id"] for r in current_rows("smoke" if cfg["mode"] == "smoke" else "preflight", data)]
                if [r["id"] for r in scores] != expected_ids:
                    raise ValueError("Incomplete preflight responses")
                if cfg["mode"] == "smoke":
                    timings = pf.get("timings", [])
                    if [r["id"] for r in timings] != expected_ids or any(not math.isfinite(r["latency_ms"]) or r["latency_ms"] <= 0 for r in timings):
                        raise ValueError("Incomplete smoke timings")
                    if "memory" not in pf:
                        raise ValueError("Missing smoke memory collection")
                if not pf["boundary"]["over_limit_rejected"] or pf["aba"][0]["label"] != pf["aba"][1]["label"]:
                    raise ValueError("Boundary/ABA checks failed")
                preflight[variant] = {r["id"]: r["score"] for r in scores}
            if cfg["mode"] == "smoke" or state.get("preflight_only"):
                result["status"] = "smoke_passed" if cfg["mode"] == "smoke" else "preflight_passed"
            else:
                outcomes = []
                indexed = {r["id"]: r["index"] for rows in data.values() for r in rows}
                for i, job in enumerate(jobs):
                    path = root / "latency" / f"{i:03d}"
                    out = checked(path, job["variant"])
                    if out["rows"] != lines(path / "latencies.jsonl"):
                        raise ValueError("Raw latencies differ from worker summary")
                    for record in out["rows"]:
                        if (record["input_sha256"], record["total_tokens"]) != (indexed[record["id"]]["input_sha256"], indexed[record["id"]]["total_tokens"]):
                            raise ValueError("Measured inputs differ from dataset")
                    outcomes.append(out)
                for bucket, rows in data.items():
                    if cfg.get("candidate"):
                        series, labels = paired_rows(jobs, outcomes, bucket)
                        for variant in variants:
                            for row in rows:
                                labels[variant][row["id"]].append(preflight[variant][row["id"]])
                        item = assess(series["ab"], series["aa"], .10 if cfg["comparison"] == "parameter" else .20)
                        item["verdicts"] = agreement(labels["baseline"], labels["candidate"])
                        if cfg["mode"] == "pilot":
                            item["status"] = "not_evaluated_pilot"
                            item["baseline_drift_p95"] = None
                            for metric in ("p50", "p95"):
                                item[metric]["ratio_interval_95"] = None
                    else:
                        selected = [(j, o) for j, o in zip(jobs, outcomes) if j["bucket"] == bucket and j["phase"] == "baseline"]
                        values, blocks, labels = [], [], defaultdict(list)
                        for job, out in selected:
                            if out["job"] != job or [r["id"] for r in out["rows"]] != job["ids"]:
                                raise ValueError("Incomplete baseline series")
                            expected = {r["id"]: r["index"] for r in rows}
                            block_values = []
                            for record in out["rows"]:
                                if record["input_sha256"] != expected[record["id"]]["input_sha256"]:
                                    raise ValueError("Baseline input changed")
                                values.append(record["latency_ms"])
                                block_values.append(record["latency_ms"])
                                labels[record["id"]].append(record["score"])
                            blocks.append(block_values)
                        aa_jobs = [j for j in jobs if j["phase"] == "aa"]
                        aa_out = [o for j, o in zip(jobs, outcomes) if j["phase"] == "aa"]
                        series, aa_labels = paired_rows(aa_jobs, aa_out, bucket)
                        for key, values_for_id in aa_labels["baseline"].items():
                            labels[key].extend(values_for_id)
                        for row in rows:
                            labels[row["id"]].append(preflight["baseline"][row["id"]])
                        item = {"status": "measured_baseline", **baseline_diagnostics(blocks, series["aa"]),
                                "verdicts": agreement(labels, labels)}
                    result["buckets"][bucket] = item
                if cfg["mode"] == "screening":
                    for variant in variants:
                        for repeat in range(3):
                            mem = checked(root / "memory" / f"{variant}_{repeat}", variant)
                            result["memory"].append({k: mem[k] for k in ("variant", "pid", "memory", "torch_setup", "torch_inference", "size", "setup_seconds")})
                        checked(root / "profiles" / variant, variant)
                        for label in ("short", "long"):
                            path = root / "profiles" / variant / (label + ".trace.json.gz")
                            if not path.exists() or not path.stat().st_size:
                                raise ValueError("Missing profile trace")
                            with gzip.open(path, "rt") as stream:
                                events = json.load(stream)["traceEvents"]
                            if not any(e.get("cat") == "kernel" for e in events):
                                raise ValueError("Profile has no GPU kernels")
                    result["memory_status"] = "measured" if all(m["memory"]["windows"][w]["status"] == "measured" for m in result["memory"] for w in ("setup", "inference")) else "partially_unavailable"
                items = list(result["buckets"].values())
                result["macro_verdict_change_fraction"] = sum(i["verdicts"]["changed_fraction"] for i in items) / len(items)
                unstable = any(i["verdicts"]["unstable_ids"] for i in items)
                flips = any(i["verdicts"]["changed_count"] for i in items)
                result["verdict_status"] = "unstable" if unstable else "changed" if flips else "unchanged"
                if cfg["mode"] == "pilot":
                    result.update(status="pilot_complete", speed_status="not_evaluated_pilot")
                elif not cfg.get("candidate"):
                    result.update(status="baseline_complete" if not unstable else "baseline_unstable", speed_status="measured_baseline")
                else:
                    statuses = [i["status"] for i in items]
                    speed = "screening_pass" if all(s == "pass" for s in statuses) else "inconclusive" if "inconclusive" in statuses else "below_target"
                    result["speed_status"] = speed
                    if cfg["comparison"] == "quantization":
                        result.update(status="quantization_measured", acceptance="not_assessed_without_quality_rule",
                                      speed_status="inconclusive" if speed == "inconclusive" else "measured_no_acceptance_threshold")
                        for item in items:
                            item["status"] = "inconclusive" if item["status"] == "inconclusive" else "measured_no_acceptance_threshold"
                    else:
                        result["status"] = "rejected" if unstable or flips or speed == "below_target" else speed
        else:
            result["status"] = "quality_complete"
        if cfg.get("quality_data") and not state.get("preflight_only"):
            result["quality"] = {}
            for variant in variants:
                value = checked(root / "quality" / variant, variant)
                from .quality import metrics
                quality = metrics(read(root / "quality_inputs.json"), lines(root / "quality" / variant / "predictions.jsonl"), cfg["label_mapping"])
                if quality != value["quality"]:
                    raise ValueError("Quality summary does not match raw predictions")
                result["quality"][variant] = quality
            result.update(quality_evaluated=True, quality_status="evaluated")
        if len(identities) != 1:
            raise ValueError("Workers used different runtime/GPU/package identities")
        result.update(runtime_identity=next(iter(identities)), dataset_identity=read(root / "dataset_identity.json"))
        if fixtures:
            result.update(status="fixture_complete", gpu_acceptance=False, acceptance="interface_test_only")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, EOFError) as error:
        result.update(status="incomplete")
        result["errors"].append(str(error))
        baseline = root / "preflight/baseline/result.json"
        if state["status"] == "failed" and state.get("variant") == "candidate" and baseline.exists() and read(baseline).get("status") == "passed":
            result["status"] = "rejected_candidate_execution"
    write(root / "summary.json", result)
    text = ["# Измерения У1", "", f"{cfg['model']}: {result['status']} ({cfg['mode']}).",
            "Качество и численное согласие разделены; независимое подтверждение не заявляется.", "",
            "Бакет | p95 baseline / candidate, мс | Изменённые ID", "--- | --- | ---"]
    for bucket, item in result["buckets"].items():
        p = item["p95"]
        text.append(f"{bucket} | {p['baseline_ms']:.3f} / {p.get('candidate_ms', '—')} | {item['verdicts']['changed_count']}/{item['verdicts']['n']}")
        if not cfg.get("candidate"):
            text.append(f"\n{bucket}: A/A envelope={item['aa_envelope_p95']:.2%}; drift={item['baseline_drift_p95']:.2%}; "
                        f"series p95={item['baseline_series_p95_ms']}; p95 CI={p['baseline_interval_95_ms']}. "
                        "Диагностика, не заключение о стабильности.\n")
    text += ["", *result["errors"]]
    (root / "REPORT.md").write_text("\n".join(text) + "\n")
    return result


@contextlib.contextmanager
def unpack(path):
    path = Path(path)
    if path.is_dir():
        yield path
        return
    with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(path) as archive:
        root = Path(temp).resolve()
        if sum(i.file_size for i in archive.infolist()) > 5 * 1024**3:
            raise ValueError("Archive exceeds 5 GiB extraction budget")
        for info in archive.infolist():
            if not (root / info.filename).resolve().is_relative_to(root) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive path")
        archive.extractall(root)
        yield root


def table(paths, output):
    rows, omitted = [], []
    for path in paths:
        with unpack(path) as root:
            cfg = read(root / "config.json")
            if cfg.get("schema_version") in (2, 3):
                summary = build(root)
            else:
                summary = read(root / "summary.json")
            if summary["mode"] != "screening" or summary["status"] not in ("baseline_complete", "baseline_unstable", "screening_pass", "rejected", "inconclusive", "quantization_measured"):
                omitted.append({"path": str(path), "status": summary["status"]})
                continue
            pf = read(root / "preflight/baseline/result.json")
            dataset = summary.get("dataset_identity")
            if dataset is None:
                dataset = {"manifest_sha256": sha(root / "dataset/manifest.json")}
            runtime = read(root / "preflight/baseline/runtime.json")
            gpu = read(root / "preflight/baseline/telemetry_before.json")
            environment = {k: runtime.get(k) for k in ("python", "torch", "cuda", "cudnn", "gpu", "cpu_threads", "tf32", "hub_kernels")}
            environment.update(driver=gpu.get("driver"), packages=read(root / "preflight/baseline/packages.json"))
            group = digest({"lock": read(root / "lock.json"), "data": dataset["manifest_sha256"], "environment": environment})
            memory = [m for m in summary["memory"] if m["variant"] == "baseline"]
            for bucket, item in summary["buckets"].items():
                rows.append({"source": str(path), "comparison_group": group, "model": summary.get("model", cfg.get("preset")),
                    "bucket": bucket, "p95_ms": item["p95"]["baseline_ms"], "p50_ms": item["p50"]["baseline_ms"],
                    "baseline_drift_p95": item.get("baseline_drift_p95"), "aa_envelope_p95": item.get("aa_envelope_p95"),
                    "baseline_series_p95_ms": item.get("baseline_series_p95_ms"),
                    "p95_interval_95_ms": item["p95"].get("baseline_interval_95_ms"),
                    "aa": item.get("aa"), "stability_assessed": item.get("stability_assessed", False),
                    "memory": memory, "size": pf["size"], "contract": pf["contract"], "metadata": pf["metadata"],
                    "status": summary["status"], "environment": environment})
    result = {"rows": rows, "omitted": omitted, "note": "Different comparison_group values are not silently pooled; baseline performance does not rank safety quality."}
    write(output, result)
    return {"status": "completed", "rows": len(rows), "omitted": len(omitted)}
