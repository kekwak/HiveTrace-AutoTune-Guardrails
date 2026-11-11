import json
import subprocess
import sys
import time
from pathlib import Path

PLAN = {
    "llama": ["u1", "c8", "u4", "c12", "u3"],
    "laya": ["u1", "c9", "u4"],
    "open": ["c4", "c1", "c12", "u3"],
    "hive": ["c13", "c2", "c1", "u1", "u4", "c3", "c12", "u3"],
    "halo": ["c2", "c1", "u1", "u4", "c3", "c12", "u3", "c7"],
    "shield": ["c2", "u1", "u4", "c3", "c12", "u3"],
    "koala": ["u1", "c8", "u4", "c12", "u3"],
    "gliner": ["u1", "c8", "u4", "c12"],
}
root = Path("runs/tune")
python = "./u1-rtx-env/venv/bin/python"


def log(*a):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, a))
    print(line, flush=True)
    with open(root / "progress.log", "a") as f:
        f.write(line + "\n")


for model in sys.argv[1:] or PLAN:
    steps = PLAN[model]
    for i in range(len(steps)):
        rules = steps[: i + 1]
        name = f"{i + 1}-" + "+".join(rules)
        out = root / model / name
        status = out / "status.json"
        if status.exists() and json.loads(status.read_text()).get("status") == "completed":
            log(model, name, "skip, done")
            continue
        cfg = root / model / f"{name}.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(subprocess.check_output([python, "rules/make_config.py", model,",".join(rules), "screening"], text=True))
        log(model, name, "start")
        start = time.time()
        with open(root / model / f"{name}.log", "w") as f:
            code = subprocess.call([python, "tools/benchmark.py", "run", "--config", str(cfg), "--environment", "./u1-rtx-env",
                                    "--output", str(out)], stdout=f, stderr=subprocess.STDOUT)
        state = json.loads(status.read_text()) if status.exists() else {}
        summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        record = {"model": model, "step": i + 1, "rules": rules, "exit": code, "seconds": round(time.time() - start),
                  "run_status": state.get("status"), "verdict": summary.get("status"), "output": str(out)}
        with open(root / "results.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log(model, name, "end", code, record["run_status"], record["verdict"], f"{record['seconds']}s")
log("ALL DONE")
