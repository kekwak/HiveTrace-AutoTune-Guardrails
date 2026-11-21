import json
import statistics
import subprocess
import sys
import time
import urllib.request

CLI = [sys.executable, "-m", "guardrail_tuner.cli"]
PORT = 8000
CHUNK = 300_000
BUCKETS = json.load(open(sys.argv[1]))
MODEL = sys.argv[2]
LIMIT = int(sys.argv[3])


def cli(*args):
    return json.loads(subprocess.run([*CLI, *args], check=True, capture_output=True).stdout)


def ask(text):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/infer",
        json.dumps({"inputs": [text]}).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(request))["outputs"][0]["label"]


def served(config):
    server = subprocess.Popen(
        [*CLI, "serve", "--config", config, "--port", str(PORT)], stdout=subprocess.PIPE
    )
    try:
        server.stdout.readline()
        for bucket, texts in BUCKETS.items():
            start = time.perf_counter()
            for text in texts:
                ask(text)
            warm = time.perf_counter() - start
            labels, lat = [], []
            for text in texts:
                t = time.perf_counter()
                labels.append(ask(text))
                lat.append((time.perf_counter() - t) * 1000)
            lat.sort()
            yield {
                "bucket": bucket,
                "p50": statistics.median(lat),
                "p95": lat[int(len(lat) * 0.95)],
                "warmup_s": round(warm, 1),
                "seconds": round(time.perf_counter() - start),
                "verdicts": labels,
            }
    finally:
        server.terminate()
        server.wait()


def chunks(texts):
    part, size = [], 0
    for text in texts:
        if part and size + len(text.encode()) > CHUNK:
            yield part
            part, size = [], 0
        part.append(text)
        size += len(text.encode())
    yield part


plan = cli("plan", MODEL, "--device", "cuda")
name = MODEL.split("/")[-1]
json.dump(plan, open(f"{name}.plan.json", "w"), indent=2)
planned = plan["execution"]["rules"]
for step, rules in enumerate(([], planned)):
    config = f"{name}.{step}.json"
    execution = {**plan["execution"], "rules": rules, "max_length": LIMIT, "batch_size": 1}
    json.dump({**plan, "execution": execution}, open(config, "w"))
    for row in served(config):
        print(json.dumps({"model": MODEL, "step": step, "rules": rules, **row}), flush=True)

report = {"applied": planned, "skipped": {}, "label_changes": 0, "max_abs_logit_error": 0.0}
for part in chunks([text for texts in BUCKETS.values() for text in texts]):
    result = cli("run", "--config", config, *(f"--text={text}" for text in part))["rules"]
    report["applied"] = [rule for rule in report["applied"] if rule in result["applied"]]
    report["skipped"].update(result["skipped"])
    report["label_changes"] += result["parity"]["label_changes"]
    report["max_abs_logit_error"] = max(
        report["max_abs_logit_error"], result["parity"]["max_abs_logit_error"]
    )
print(json.dumps({"model": MODEL, "run": report}), flush=True)
