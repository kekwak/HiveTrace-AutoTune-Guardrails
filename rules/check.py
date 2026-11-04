import json
import os
import sys
from pathlib import Path

os.environ["USE_HUB_KERNELS"] = "NO"

import torch as th

from u1_benchmark.adapters import load
from u1_benchmark.core import sha
from u1_benchmark.datasets import load_speed
from u1_benchmark.messages import payload
from u1_benchmark.worker_v2 import current_rows

model, rules = sys.argv[1], sys.argv[2].split(",")
cfg = json.loads(Path(f"configs/benchmark-v3/{model}_baseline.json").read_text())
data, _ = load_speed(cfg["data"]["path"], model, "smoke")
rows = current_rows("smoke", data)
out = Path("runs/rules/check")
out.mkdir(parents=True, exist_ok=True)


def answers(spec):
    adapter = load(model, spec, out, input_version=3)
    adapter.prepare_backend([payload(rows[0])])
    values = [adapter.verify(payload(r), r["index"]) for r in rows]
    meta = adapter.metadata
    adapter.close()
    return values, meta


with th.inference_mode():
    cached = out / f"{model}-baseline.json"
    if not cached.exists():
        cached.write_text(json.dumps(answers(cfg["baseline"])[0]))
    base = json.loads(cached.read_text())
    spec = {"module": "rules/adapter.py", "source_sha256": sha("rules/adapter.py"), "dependencies": {},
            "options": {"rules": rules}}
    cand, meta = answers(spec)

same = all(a["label"] == b["label"] and a.get("category") == b.get("category") for a, b in zip(base, cand))
diff = max((abs(a["scores"][k] - b["scores"][k]) for a, b in zip(base, cand) for k in a.get("scores", {})), default=0)
print(model, ",".join(rules), "OK" if same else "DIFF", f"{diff:.1e}", meta["rules_applied"], meta["rules_skipped"])
