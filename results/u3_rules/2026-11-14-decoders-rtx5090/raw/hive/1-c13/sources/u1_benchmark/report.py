"""Offline report. No Torch, GPU, model loading or network access."""
from pathlib import Path

from .core import PRESETS, load_data, read, schedule, write, lines, digest
from .statistics import assess, compare_scores, paired_rows


def build(directory):
    directory = Path(directory)
    cfg = read(directory / "config.json")
    if cfg.get("schema_version") in (2, 3):
        from .report_v2 import build as build_v2
        return build_v2(directory)
    state = read(directory / "status.json")
    result = {"schema_version": 1, "status": "incomplete", "mode": cfg["mode"],
              "quality_evaluated": False, "independently_confirmed": False,
              "speed_status": "incomplete", "verdict_status": "incomplete", "quality_status": "not_evaluated",
              "buckets": {}, "memory": [], "profiles": [], "errors": []}
    if state.get("error"):
        result["errors"].append(state["error"])
    try:
        model = PRESETS[cfg["preset"]][0]
        data, _ = load_data(directory / "dataset", model)
        jobs = schedule(cfg, data)
        if read(directory / "schedule.json") != jobs:
            raise ValueError("Schedule differs from configuration and corpus")
        outcomes = []
        preflight = {}
        for variant in ("baseline", "candidate"):
            path = directory / "preflight" / variant
            pf = read(path / "result.json")
            if pf["status"] != "passed":
                raise ValueError("Preflight failed: " + variant)
            if pf.get("config_sha256") != digest(cfg) or pf.get("variant") != variant:
                raise ValueError("Preflight configuration mismatch")
            scores = lines(path / "scores.jsonl")
            expected_ids = [r["id"] for rows in data.values() for r in rows]
            if [r["id"] for r in scores] != expected_ids:
                raise ValueError("Incomplete preflight score set")
            preflight[variant] = {r["id"]: r["score"] for r in scores}
        contracts = []
        runtimes = set()
        for variant in ("baseline", "candidate"):
            contracts.append(read(directory / "preflight" / variant / "result.json")["contract"])
        if contracts[0] != contracts[1]:
            raise ValueError("Baseline and candidate changed inference contract")
        for i, job in enumerate(jobs):
            path = directory / "latency" / f"{i:03d}"
            outcome = read(path / "result.json")
            if outcome["contract"] != contracts[0]:
                raise ValueError("Inference contract changed between workers")
            if outcome.get("config_sha256") != digest(cfg) or outcome.get("variant") != job["variant"]:
                raise ValueError("Worker configuration/variant mismatch")
            runtime = read(path / "runtime.json")
            gpu = read(path / "telemetry_before.json")
            # Executable and NVRTC absolute paths are not environment identities.
            identity = {k: runtime[k] for k in ("python", "torch", "cuda", "cudnn", "gpu",
                                                "seed", "cpu_threads", "tf32", "hub_kernels")}
            identity.update(uuid=gpu["uuid"], driver=gpu["driver"])
            runtimes.add(digest(identity))
            outcomes.append(outcome)
        if len(runtimes) != 1:
            raise ValueError("Runtime identity changed inside comparison")
        for bucket, rows in data.items():
            series, labels = paired_rows(jobs, outcomes, bucket)
            for variant in labels:
                for row in rows:
                    labels[variant][row["id"]].append(preflight[variant][row["id"]])
            item = assess(series["ab"], series["aa"])
            item["speed_status"] = item["status"]
            item["verdicts"] = compare_scores(labels["baseline"], labels["candidate"])
            if item["verdicts"]["status"] != "ok":
                item["status"] = "rejected_verdicts"
            result["buckets"][bucket] = item
        result["macro_verdict_change_fraction"] = sum(
            b["verdicts"]["changed_fraction"] for b in result["buckets"].values()) / len(data)
        statuses = [r["status"] for r in result["buckets"].values()]
        speeds = [r["speed_status"] for r in result["buckets"].values()]
        result["speed_status"] = ("screening_pass" if all(s == "pass" for s in speeds)
            else "inconclusive" if "inconclusive" in speeds else "below_target")
        result["verdict_status"] = "ok" if all(
            b["verdicts"]["status"] == "ok" for b in result["buckets"].values()) else "not_ok"
        result["speed_and_verdict_status"] = ("screening_pass" if all(s == "pass" for s in statuses)
            else "rejected" if result["verdict_status"] != "ok"
            else "inconclusive" if "inconclusive" in statuses else "rejected")
        if cfg["mode"] == "pilot":
            result.update(status="pilot_complete", speed_and_verdict_status="not_evaluated_pilot",
                          speed_status="not_evaluated_pilot")
            for item in result["buckets"].values():
                item.update(status="pilot_complete", speed_status="not_evaluated_pilot",
                            uncertainty_note="Pilot: one paired block; uncertainty and drift are not estimable")
                item["baseline_drift_p95"] = None
                for metric in ("p50", "p95"):
                    item[metric]["ratio_interval_95"] = None
        else:
            for variant in ("baseline", "candidate"):
                for repeat in range(3):
                    memory = read(directory / "memory" / f"{variant}_{repeat}" / "result.json")
                    if memory["status"] != "passed":
                        raise ValueError("Memory run failed")
                    if memory.get("config_sha256") != digest(cfg) or memory.get("variant") != variant:
                        raise ValueError("Memory configuration mismatch")
                    result["memory"].append({k: memory[k] for k in (
                        "variant", "pid", "memory", "torch_setup", "torch_at_inference_start", "torch_inference", "size")})
                profile = read(directory / "profiles" / variant / "result.json")
                if profile["status"] != "passed":
                    raise ValueError("Profile missing")
                result["profiles"].append({"variant": variant, "status": "measured"})
            result["memory_status"] = "measured" if all(
                r["memory"]["windows"][w]["status"] == "measured"
                for r in result["memory"] for w in ("setup", "inference")) else "partially_unavailable"
            result["status"] = result["speed_and_verdict_status"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        result["errors"].append(str(e))
    if state["status"] != "completed":
        baseline_path = directory / "preflight/baseline/result.json"
        baseline_valid = baseline_path.exists() and read(baseline_path).get("status") == "passed"
        if state.get("variant") == "candidate" and baseline_valid and state.get("phase") in ("preflight", "latency"):
            result["candidate_execution_status"] = "failed"
            result["status"] = "rejected_candidate_execution" if state["status"] != "timeout" else "incomplete"
        else:
            result["status"] = "incomplete"
            result["baseline_contract_status"] = "preflight_passed" if baseline_valid else "unconfirmed"
    write(directory / "summary.json", result)
    table = ["# Измерения У1", "", f"Статус: {result['status']}. Режим: {cfg['mode']}.",
             "Качество по разметке не оценивалось; независимое подтверждение не выполнено.", "",
             "Бакет | p95 baseline/candidate, мс | Снижение | Изменения вердиктов | Статус",
             "--- | ---: | ---: | ---: | ---"]
    for name, item in result["buckets"].items():
        p = item["p95"]
        table.append(f"{name} | {p['baseline_ms']:.3f} / {p['candidate_ms']:.3f} | "
                     f"{item['p95_reduction']:.1%} | {item['verdicts']['changed_count']} | {item['status']}")
    table += ["", "Память: " + result.get("memory_status", "неполный/не выполненный замер"), "",
              "Полные интервалы, A/A, дрейф, память и статусы: summary.json."]
    table += ["", *result["errors"]]
    (directory / "REPORT.md").write_text("\n".join(table) + "\n")
    return result
