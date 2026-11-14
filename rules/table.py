import csv
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/tune-v4")
rows = [json.loads(l) for l in (root / "results.jsonl").read_text().splitlines()]
rows = list({r["output"]: r for r in rows}.values())
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = (root / "machine.txt").read_text().split(",")[0]


def job(run, stage, variant):
    for path in sorted((run / "jobs").glob("*/job.json")):
        spec = json.loads(path.read_text())
        result = path.parent / "result.json"
        if spec.get("stage") == stage and spec.get("variant") == variant and result.exists():
            return json.loads(result.read_text())
    return {}


def fmt(x):
    return "" if x is None else f"{x:.2f}"


def rnd(x, n=3):
    return None if x is None else round(x, n)


out = [f"# Последовательное добавление правил, {gpu}, measurer v4", "", "Бакет — бакет/размер батча.",
       "Шаг 0 — baseline, шаги 1+ — search по сохранённому baseline (без парных интервалов).",
       "p95/p50 в мс, Δ — снижение p95 относительно baseline.", ""]
fields = ["model", "step", "rules_applied", "rules_skipped", "bucket", "p95_base_ms", "p95_cand_ms", "p95_reduction_pct",
          "p50_base_ms", "p50_cand_ms", "verdicts_changed", "verdicts_n", "cell_status", "run_status",
          "prep_s", "run_s", "gpu", "output"]
table = []
for model in dict.fromkeys(r["model"] for r in rows):
    out += [f"## {model}", "", "| Шаг | Правила | Бакет | p95 base | p95 cand | Δp95 | p50 base | p50 cand | Вердикты изм. | Статус | Подготовка | Время |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in (r for r in rows if r["model"] == model):
        run = root / r["model"] / Path(r["output"]).name
        path = run / "summary.json"
        s = json.loads(path.read_text()) if path.exists() else {"cells": {}}
        variant = "candidate" if r["rules"] else "baseline"
        proof = job(run, "preflight", variant).get("backend_proof") or {}
        rules = ",".join(proof.get("rules_applied", r["rules"])) or "—"
        skipped = "; ".join(f"{k}: {v}" for k, v in proof.get("rules_skipped", {}).items())
        applied = rules + (f" (пропущено {skipped})" if skipped else "")
        prep = job(run, "latency", variant).get("backend_preparation_seconds")
        base = {"model": model, "step": r["step"], "rules_applied": rules, "rules_skipped": skipped,
                "run_status": s.get("status", r["run_status"]), "prep_s": rnd(prep, 1),
                "run_s": r["seconds"], "gpu": gpu, "output": r["output"]}
        if not s["cells"]:
            out.append(f"| {r['step']} | {applied} | — | | | | | | | {r['run_status']} | | {r['seconds']}s |")
            table.append(base)
        for cell, v in s["cells"].items():
            p95, p50, ver = v.get("p95", {}), v.get("p50", {}), v.get("verdicts", {})
            red = v.get("p95_reduction")
            out.append(f"| {r['step']} | {applied} | {cell} | {fmt(p95.get('baseline_ms'))} | {fmt(p95.get('candidate_ms'))} | "
                       f"{'' if red is None else f'{red * 100:+.1f}%'} | {fmt(p50.get('baseline_ms'))} | {fmt(p50.get('candidate_ms'))} | "
                       f"{ver.get('changed_count', '')}/{ver.get('n', '')} | {v.get('status')} | {base['prep_s']}s | {r['seconds']}s |")
            table.append({**base, "bucket": cell, "p95_base_ms": rnd(p95.get("baseline_ms")), "p95_cand_ms": rnd(p95.get("candidate_ms")),
                          "p95_reduction_pct": rnd(None if red is None else red * 100, 2),
                          "p50_base_ms": rnd(p50.get("baseline_ms")), "p50_cand_ms": rnd(p50.get("candidate_ms")),
                          "verdicts_changed": ver.get("changed_count"), "verdicts_n": ver.get("n"), "cell_status": v.get("status")})
    out.append("")
(root / "RESULTS.md").write_text("\n".join(out))
with open(root / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(table)
print("\n".join(out))
