import json
import subprocess
import sys
import time
from pathlib import Path

PLAN = {
    "llama": ["u1", "c8", "u4", "c12", "u3"],
    "laya": ["u1", "c9", "u4"],
    "open": ["c4", "u1+u4"],
    "hive": ["c13", "c2", "u1+u4"],
    "halo": ["c2", "u1+u4", "c7"],
    "shield": ["u1+u4", "c3"],
    "koala": ["u1", "c8", "u4", "c12", "u3"],
    "gliner": ["u1", "c8", "u4", "c12"],
}
args = sys.argv[1:]
batch = int(args.pop(args.index("--batch") + 1)) if "--batch" in args else 1
args = [a for a in args if a != "--batch"]
root = Path("runs/tune-v4" + (f"-b{batch}" if batch > 1 else ""))
DONE = ("baseline_complete", "exploratory")
python = "./u1-rtx-env/venv/bin/python"
root.mkdir(parents=True, exist_ok=True)


def done(out):
    status = out / "status.json"
    return status.exists() and json.loads(status.read_text()).get("status") in DONE


def log(*a):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, a))
    print(line, flush=True)
    with open(root / "progress.log", "a") as f:
        f.write(line + "\n")


for model in args or PLAN:
    steps = PLAN[model]
    base = root / model / "0-baseline"
    for i in range(len(steps) + 1):
        rules = "+".join(steps[:i]).split("+") if i else []
        name = f"{i}-" + "+".join(rules) if rules else base.name
        out = root / model / name
        status = out / "status.json"
        if done(out):
            log(model, name, "skip, done")
            continue
        if rules and not done(base):
            log(model, name, "skip, no baseline")
            break
        for old in (out, out.with_suffix(".zip")):
            if old.exists():
                old.rename(old.with_name(f"{old.name}.failed-{int(time.time())}"))
        cfg = root / model / f"{name}.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(subprocess.check_output([python, "rules/make_config.py", model, ",".join(rules), str(batch)], text=True))
        log(model, name, "start")
        start = time.time()
        ref = ["--baseline-ref", str(base)] if rules else []
        with open(root / model / f"{name}.log", "w") as f:
            code = subprocess.call([python, "tools/benchmark.py", "run", "--config", str(cfg), "--environment", "./u1-rtx-env",
                                    *ref, "--output", str(out)], stdout=f, stderr=subprocess.STDOUT)
        state = json.loads(status.read_text()) if status.exists() else {}
        summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        record = {"model": model, "step": i, "rules": rules, "exit": code, "seconds": round(time.time() - start),
                  "run_status": state.get("status"), "verdict": summary.get("status"), "output": str(out)}
        with open(root / "results.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log(model, name, "end", code, record["run_status"], record["verdict"], f"{record['seconds']}s")
log("ALL DONE")
