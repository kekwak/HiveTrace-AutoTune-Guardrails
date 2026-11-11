import csv
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/tune")
rows = [json.loads(l) for l in (root / "results.jsonl").read_text().splitlines()]
rows = list({r["output"]: r for r in rows}.values())
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = (root / "machine.txt").read_text().split(",")[0]
out = [f"# Последовательное добавление правил, {gpu}, screening", "",
       "p95/p50 в мс, Δ — снижение p95 относительно baseline того же прогона, A/A — шумовой конверт p95.", ""]
fields = ["model", "step", "rules_applied", "rules_skipped", "bucket", "p95_base_ms", "p95_cand_ms", "p95_reduction_pct",
          "p50_base_ms", "p50_cand_ms", "aa_envelope_pct", "verdicts_changed", "verdicts_n", "bucket_status",
          "run_verdict", "prep_s", "run_s", "gpu", "output"]
table = []
for model in dict.fromkeys(r["model"] for r in rows):
    out += [f"## {model}", "", "| Шаг | Правила | Бакет | p95 base | p95 cand | Δp95 | p50 base | p50 cand | A/A | Вердикты изм. | Бакет / прогон | Подготовка | Время |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in (r for r in rows if r["model"] == model):
        run = root / r["model"] / Path(r["output"]).name
        path = run / "summary.json"
        s = json.loads(path.read_text()) if path.exists() else {"buckets": {}}
        pre = run / "preflight/candidate/result.json"
        res = json.loads(pre.read_text()) if pre.exists() else {}
        meta = res.get("metadata", {})
        rules = ",".join(meta.get("rules_applied", r["rules"]))
        skipped = "; ".join(f"{k}: {v}" for k, v in meta.get("rules_skipped", {}).items())
        applied = rules + (f" (пропущено {skipped})" if skipped else "")
        prep = res.get("backend_preparation_seconds")
        base = {"model": model, "step": r["step"], "rules_applied": rules, "rules_skipped": skipped,
                "run_verdict": r["verdict"] or r["run_status"], "prep_s": prep if prep is None else round(prep, 1),
                "run_s": r["seconds"], "gpu": gpu, "output": r["output"]}
        if not s["buckets"]:
            out.append(f"| {r['step']} | {applied} | — | | | | | | | | {r['run_status']} / {r['verdict']} | | {r['seconds']}s |")
            table.append(base)
        for b, v in s["buckets"].items():
            p95, p50, ver = v["p95"], v["p50"], v["verdicts"]
            out.append(f"| {r['step']} | {applied} | {b} | {p95['baseline_ms']:.2f} | {p95['candidate_ms']:.2f} | "
                       f"{v['p95_reduction'] * 100:+.1f}% | {p50['baseline_ms']:.2f} | {p50['candidate_ms']:.2f} | "
                       f"{v['aa_envelope_p95'] * 100:.1f}% | {ver['changed_count']}/{ver['n']} | {v['status']} / {s['status']} | {base['prep_s']}s | {r['seconds']}s |")
            table.append({**base, "bucket": b, "p95_base_ms": round(p95["baseline_ms"], 3), "p95_cand_ms": round(p95["candidate_ms"], 3),
                          "p95_reduction_pct": round(v["p95_reduction"] * 100, 2), "p50_base_ms": round(p50["baseline_ms"], 3),
                          "p50_cand_ms": round(p50["candidate_ms"], 3), "aa_envelope_pct": round(v["aa_envelope_p95"] * 100, 2),
                          "verdicts_changed": ver["changed_count"], "verdicts_n": ver["n"], "bucket_status": v["status"]})
    out.append("")
(root / "RESULTS.md").write_text("\n".join(out))
with open(root / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(table)
print("\n".join(out))
