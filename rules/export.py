import csv
import json
import sys
from pathlib import Path

root, dest = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in (root / "results.jsonl").read_text().splitlines()]
rows = list({r["output"]: r for r in rows}.values())
full, memory = [], []
SKIP = ("series_evidence", "worker_evidence", "profiles", "reference_diagnostics", "throughput")


def flat(prefix, value, row):
    if isinstance(value, dict):
        for k, v in value.items():
            flat(f"{prefix}{k}_", v, row)
    elif isinstance(value, list):
        row[prefix[:-1]] = json.dumps(value, ensure_ascii=False)
    else:
        row[prefix[:-1]] = value


for r in rows:
    run = root / r["model"] / Path(r["output"]).name
    base = {"model": r["model"], "step": r["step"], "rules": "+".join(r["rules"]), "run_s": r["seconds"],
            "output": f"{r['model']}/{run.name}"}
    if not (run / "summary.json").exists():
        full.append({**base, "run_status": r["run_status"]})
        continue
    s = json.loads((run / "summary.json").read_text())
    base.update(run_status=s["status"], speed_status=s["speed_status"], verdict_status=s["verdict_status"],
                memory_status=s["memory_status"])
    for cell, v in s["cells"].items():
        row = {**base, "cell": cell}
        flat("", {k: x for k, x in v.items() if k not in SKIP}, row)
        full.append(row)
    for i, w in enumerate(s["worker_evidence"]):
        row = {**base, "worker": i}
        flat("", w, row)
        memory.append(row)

for name, table in (("results_full.csv", full), ("memory.csv", memory)):
    fields = list(dict.fromkeys(k for row in table for k in row))
    with open(dest / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(table)
    print(dest / name, len(table))
