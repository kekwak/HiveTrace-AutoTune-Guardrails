import csv
import json
import sys
from pathlib import Path

root, dest = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in (root / "results.jsonl").read_text().splitlines()]
rows = list({r["output"]: r for r in rows}.values())
full, memory = [], []
for r in rows:
    run = root / r["model"] / Path(r["output"]).name
    if not (run / "summary.json").exists():
        full.append({"model": r["model"], "step": r["step"], "rules": "+".join(r["rules"]), "run_status": r["run_status"]})
        continue
    s = json.loads((run / "summary.json").read_text())
    pre = json.loads((run / "preflight/candidate/result.json").read_text())
    meta = pre.get("metadata", {})
    base = {"model": r["model"], "step": r["step"], "rules": "+".join(r["rules"]),
            "rules_applied": ",".join(meta.get("rules_applied", [])),
            "rules_skipped": json.dumps(meta.get("rules_skipped", {}), ensure_ascii=False),
            "run_status": s["status"], "speed_status": s["speed_status"], "verdict_status": s["verdict_status"],
            "memory_status": s["memory_status"], "macro_verdict_change_fraction": s["macro_verdict_change_fraction"],
            "load_s": pre.get("load_seconds"), "setup_s": pre.get("setup_seconds"),
            "prep_s": pre.get("backend_preparation_seconds"), "compile_count": pre.get("prepared_compile_count"),
            "run_s": r["seconds"], "output": f"{r['model']}/{run.name}"}
    for bucket, b in s["buckets"].items():
        row = {**base, "bucket": bucket, "bucket_status": b["status"], "p95_reduction": b["p95_reduction"],
               "aa_envelope_p95": b["aa_envelope_p95"], "baseline_drift_p95": b["baseline_drift_p95"],
               "reasons": "; ".join(b["reasons"]), "uncertainty_note": b["uncertainty_note"]}
        for q in ("p95", "p50"):
            row.update({f"{q}_baseline_ms": b[q]["baseline_ms"], f"{q}_candidate_ms": b[q]["candidate_ms"],
                        f"{q}_ratio": b[q]["ratio"], f"{q}_ratio_ci_low": b[q]["ratio_interval_95"][0],
                        f"{q}_ratio_ci_high": b[q]["ratio_interval_95"][1], f"{q}_block_ratios": json.dumps(b[q]["block_ratios"])})
        v = b["verdicts"]
        row.update({"verdicts_n": v["n"], "verdicts_changed": v["changed_count"], "verdicts_changed_fraction": v["changed_fraction"],
                    "verdicts_status": v["status"], "changed_ids": " ".join(v["changed_ids"]), "unstable_ids": " ".join(v["unstable_ids"]),
                    "max_abs_score_delta": json.dumps(v["max_abs_score_delta"]), "directions": json.dumps(v["directions"])})
        full.append(row)
    for i, m in enumerate(s["memory"]):
        rec = m["memory"]
        row = {"model": r["model"], "step": r["step"], "rules": "+".join(r["rules"]), "variant": m["variant"], "process": i,
               "setup_s": m.get("setup_seconds"), "torch_setup_allocated": m["torch_setup"]["allocated"],
               "torch_setup_reserved": m["torch_setup"]["reserved"], "torch_inference_allocated": m["torch_inference"]["allocated"],
               "torch_inference_reserved": m["torch_inference"]["reserved"],
               "parameter_bytes": m["size"]["unique_parameter_buffer_storage_bytes"]}
        for w in ("setup", "inference"):
            row.update({f"{w}_gpu_peak_mib": rec["windows"][w]["sampled_gpu_peak_mib"],
                        f"{w}_host_rss_peak_bytes": rec["windows"][w]["host_rss_peak_bytes"],
                        f"{w}_samples": rec["windows"][w]["samples"]})
        memory.append(row)

for name, table in (("results_full.csv", full), ("memory.csv", memory)):
    fields = list(dict.fromkeys(k for row in table for k in row))
    with open(dest / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(table)
    print(dest / name, len(table))
