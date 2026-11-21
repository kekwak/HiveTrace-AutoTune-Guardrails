import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(l) for path in sorted((root / "raw").glob("*.jsonl")) for l in path.read_text().splitlines()]
rows = [r for r in rows if "bucket" in r]
gpu = (root / "raw" / "machine.txt").read_text().split(",")[0]

out = [f"# Правила из плана CLI на новых моделях, {gpu}", "", "Бакет — бакет/размер батча.",
       "Шаг 0 — без правил, шаг 1 — правила из `guardrail-tuner plan` (один проход после прогрева, без парных интервалов).",
       "p95/p50 в мс, Δ — снижение p95 относительно шага 0.", ""]
fields = ["model", "step", "rules_applied", "rules_skipped", "bucket", "p95_base_ms", "p95_cand_ms", "p95_reduction_pct",
          "p50_base_ms", "p50_cand_ms", "verdicts_changed", "verdicts_n", "max_abs_score_delta", "prep_s", "run_s", "gpu"]
table = []
for model in dict.fromkeys(r["model"] for r in rows):
    out += [f"## {model}", "", "| Шаг | Правила | Бакет | p95 base | p95 cand | Δp95 | p50 base | p50 cand | Вердикты изм. | Статус | Подготовка | Время |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = {r["bucket"]: r for r in rows if r["model"] == model and r["step"] == 0}
    for r in (r for r in rows if r["model"] == model):
        b = base[r["bucket"]]
        rules = ",".join(r["rules"]) or "—"
        skipped = "; ".join(f"{k}: {v}" for k, v in r.get("skipped", {}).items())
        changed = sum(x != y for x, y in zip(b["verdicts"], r["verdicts"]))
        n = len(r["verdicts"])
        delta = max((abs(x - y) for x, y in zip(b.get("scores", []), r.get("scores", []))), default=None)
        red = (1 - r["p95"] / b["p95"]) * 100 if r["step"] else None
        cand = lambda key: f"{r[key]:.2f}" if r["step"] else ""
        out.append(f"| {r['step']} | {rules + (f' (пропущено {skipped})' if skipped else '')} | {r['bucket']}/1 | {b['p95']:.2f} | {cand('p95')} | "
                   f"{'' if red is None else f'{red:+.1f}%'} | {b['p50']:.2f} | {cand('p50')} | {changed}/{n} | "
                   f"{'exploratory' if r['step'] else 'baseline'} | {r['warmup_s']}s | {r['seconds']}s |")
        table.append({"model": model, "step": r["step"], "rules_applied": rules, "rules_skipped": skipped, "bucket": f"{r['bucket']}/1",
                      "p95_base_ms": round(b["p95"], 3), "p95_cand_ms": round(r["p95"], 3) if r["step"] else None,
                      "p95_reduction_pct": None if red is None else round(red, 2), "p50_base_ms": round(b["p50"], 3),
                      "p50_cand_ms": round(r["p50"], 3) if r["step"] else None, "verdicts_changed": changed, "verdicts_n": n,
                      "max_abs_score_delta": None if delta is None or not r["step"] else round(delta, 4),
                      "prep_s": r["warmup_s"], "run_s": r["seconds"], "gpu": gpu})
    out.append("")
(root / "RESULTS.md").write_text("\n".join(out))
with open(root / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(table)
print("\n".join(out))
