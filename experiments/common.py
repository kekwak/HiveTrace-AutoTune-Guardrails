"""Artifact and statistical helpers. No Torch dependency."""

import csv
import hashlib
import json
import math
import random
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()


def append_csv(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as file:
            if next(csv.reader(file)) != list(row):
                raise ValueError("CSV schema changed")
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def archive(directory):
    directory = Path(directory)
    destination = directory.with_suffix(".zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as output:
        for file in sorted(directory.rglob("*")):
            if (file.is_file() and not file.is_symlink() and file.suffix not in (".tmp", ".onnx", ".data", ".zip", ".pyc")
                    and ".onnx." not in file.name):
                output.write(file, file.relative_to(directory))
    return destination


def quantile(values, q):
    values = sorted(float(x) for x in values)
    if not values or not 0 <= q <= 1 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected nonempty finite values and q in [0,1]")
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def latency_summary(values):
    if any(x <= 0 for x in values):
        raise ValueError("Latencies must be positive")
    return {"n": len(values), "p50_ms": quantile(values, .5),
            "p95_ms": quantile(values, .95), "min_ms": min(values),
            "max_ms": max(values), "mean_ms": statistics.mean(values)}


def check_score(score):
    for key in ("p_safe", "p_unsafe", "logit_margin"):
        if not math.isfinite(score[key]):
            raise ValueError("Nonfinite score: " + key)
    if not (0 <= score["p_safe"] <= 1 and 0 <= score["p_unsafe"] <= 1):
        raise ValueError("Invalid probability")
    if abs(score["p_safe"] + score["p_unsafe"] - 1) > 1e-5:
        raise ValueError("Probabilities do not sum to one")
    if score["label"] not in ("safe", "unsafe"):
        raise ValueError("Invalid verdict")


def regression(reference, candidate, dtype, threshold):
    if dtype not in ("float32", "float16"):
        raise ValueError("Unsupported regression precision")
    def index(rows):
        result = {row["id"]: row for row in rows}
        if len(result) != len(rows) or not rows:
            raise ValueError("Empty or duplicate regression IDs")
        return result
    ref, cand = index(reference), index(candidate)
    if ref.keys() != cand.keys():
        raise ValueError("Regression input IDs differ")
    deltas, flips, margins = [], [], []
    for key, row in ref.items():
        other = cand[key]
        if row["input_hash"] != other["input_hash"]:
            raise ValueError("Different inputs for " + key)
        check_score(row)
        check_score(other)
        delta = abs(row["p_unsafe"] - other["p_unsafe"])
        margin = abs(row["p_unsafe"] - threshold)
        deltas.append(delta)
        margins.append(margin)
        if row["label"] != other["label"]:
            flips.append({"id": key, "from": row["label"], "to": other["label"],
                          "baseline_margin": margin, "delta": delta,
                          "outside_band": margin > .02})
    p99_limit, max_limit = (1e-4, 1e-3) if dtype == "float32" else (.01, .05)
    p99, maximum = quantile(deltas, .99), max(deltas)
    passed = (p99 <= p99_limit and maximum <= max_limit and
              len(flips) / len(ref) <= .01 and not any(x["outside_band"] for x in flips))
    return {"passed": passed, "n": len(ref), "delta_p99": p99, "delta_max": maximum,
            "flips": flips, "flip_fraction": len(flips) / len(ref),
            "near_threshold_count": sum(x <= .02 for x in margins),
            "limits": {"p99": p99_limit, "max": max_limit, "band": .02,
                       "flip_fraction": .01}, "quality_evaluated": False}


def paired_schedule(blocks, seed):
    rng = random.Random(seed)
    first = rng.choice(("AB", "BA"))
    return [first if i % 2 == 0 else first[::-1] for i in range(blocks)]


def compare_blocks(rows, baseline, candidate, bootstrap=2000, seed=42):
    """Bootstrap sessions then paired blocks, never individual requests."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["variant"] in (baseline, candidate):
            grouped[(row["session_id"], row["block"])][row["variant"]].append(float(row["latency_ms"]))
    sessions = defaultdict(list)
    for (session, block), groups in grouped.items():
        if set(groups) != {baseline, candidate} or len(groups[baseline]) != len(groups[candidate]):
            raise ValueError("Incomplete paired block")
        a, b = latency_summary(groups[baseline]), latency_summary(groups[candidate])
        sessions[session].append({"block": block, "p50_ratio": b["p50_ms"] / a["p50_ms"],
                                  "p95_ratio": b["p95_ms"] / a["p95_ms"]})
    if not sessions:
        raise ValueError("No paired measurements")
    estimates = {s: {"blocks": values,
                     "p50_ratio": statistics.median(x["p50_ratio"] for x in values),
                     "p95_ratio": statistics.median(x["p95_ratio"] for x in values)}
                 for s, values in sessions.items()}
    rng, samples, keys = random.Random(seed), [], list(sessions)
    for _ in range(bootstrap):
        ratios = []
        for key in rng.choices(keys, k=len(keys)):
            values = sessions[key]
            ratios.append(statistics.median(x["p50_ratio"] for x in rng.choices(values, k=len(values))))
        samples.append(statistics.median(ratios))
    complete = len(sessions) >= 3 and all(len(v) >= 5 for v in sessions.values())
    useful = complete and all(v["p50_ratio"] <= .9 and v["p95_ratio"] <= 1.05 for v in estimates.values())
    return {"sessions": estimates, "confirmation_complete": complete,
            "performance_gate_passed": useful,
            "p50_ratio_interval_95": [quantile(samples, .025), quantile(samples, .975)],
            "interval_scope": "session/paired-block resampling; not production SLA"}


def break_even(setup_ms, baseline_ms, candidate_ms):
    if any(not math.isfinite(x) or x < 0 for x in (setup_ms, baseline_ms, candidate_ms)):
        raise ValueError("Invalid time")
    return math.ceil(setup_ms / (baseline_ms - candidate_ms)) if baseline_ms > candidate_ms else None
