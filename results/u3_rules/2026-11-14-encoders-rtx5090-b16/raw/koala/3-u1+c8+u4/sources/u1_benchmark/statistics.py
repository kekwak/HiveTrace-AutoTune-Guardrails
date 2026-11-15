"""Paired pooled-quantile analysis, without treating requests as independent runs."""
import math
import random
from collections import defaultdict
from functools import lru_cache

from .core import score_check


def quantile(values, q):
    values = sorted(values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Empty/nonfinite measurement series")
    x = (len(values) - 1) * q
    a, b = math.floor(x), math.ceil(x)
    return values[a] + (values[b] - values[a]) * (x - a)


def paired(blocks, q=.95):
    if not blocks or any(not a or not b for a, b in blocks):
        raise ValueError("Missing paired blocks")
    @lru_cache(None)
    def estimate(draw):
        a = [v for i in draw for v in blocks[i][0]]
        b = [v for i in draw for v in blocks[i][1]]
        return quantile(b, q) / quantile(a, q)
    rng = random.Random(42)
    values = [estimate(tuple(sorted(rng.randrange(len(blocks)) for _ in blocks))) for _ in range(2000)]
    a, b = ([v for block in blocks for v in block[side]] for side in (0, 1))
    return {"baseline_ms": quantile(a, q), "candidate_ms": quantile(b, q),
            "ratio": estimate(tuple(range(len(blocks)))),
            "ratio_interval_95": [quantile(values, .025), quantile(values, .975)],
            "block_ratios": [quantile(b, q) / quantile(a, q) for a, b in blocks]}


def baseline_diagnostics(blocks, aa):
    if not blocks or any(not block or any(v <= 0 or not math.isfinite(v) for v in block) for block in blocks):
        raise ValueError("Missing/invalid baseline blocks")
    p95s = [quantile(block, .95) for block in blocks]
    control = paired(aa)
    result = {"baseline_series_p95_ms": p95s, "baseline_drift_p95": max(p95s) / min(p95s) - 1,
              "aa": control, "aa_envelope_p95": max(max(r, 1 / r) - 1 for r in control["block_ratios"]),
              "stability_assessed": False, "noise_flags": None,
              "uncertainty_note": f"Descriptive block bootstrap over {len(blocks)} series; no baseline stability threshold or independent confirmation"}
    for name, q in (("p50", .5), ("p95", .95)):
        @lru_cache(None)
        def estimate(draw):
            return quantile([v for i in draw for v in blocks[i]], q)
        rng = random.Random(42)
        draws = [estimate(tuple(sorted(rng.randrange(len(blocks)) for _ in blocks))) for _ in range(2000)]
        result[name] = {"baseline_ms": estimate(tuple(range(len(blocks)))),
                        "baseline_interval_95_ms": [quantile(draws, .025), quantile(draws, .975)]}
    return result


def compare_scores(baseline, candidate):
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("Missing scores or unequal ID sets")
    changed, deltas, margins, unstable = [], [], [], []
    for i in baseline:
        a, b = baseline[i], candidate[i]
        if not a or not b:
            raise ValueError("Empty score series")
        for s in a + b:
            score_check(s)
        if len({s["label"] for s in a}) > 1 or len({s["label"] for s in b}) > 1:
            unstable.append(i)
        if a[0]["label"] != b[0]["label"]:
            changed.append({"id": i, "baseline": a[0]["label"], "candidate": b[0]["label"]})
        deltas.append(max(abs(x["p_unsafe"] - y["p_unsafe"]) for x in a for y in b))
        margins.append(max(abs(x["logit_margin"] - y["logit_margin"]) for x in a for y in b))
    return {"status": "ok" if not changed and not unstable else "not_ok", "n": len(baseline),
            "changed": changed, "changed_count": len(changed), "changed_fraction": len(changed) / len(baseline),
            "unstable_ids": unstable, "max_abs_delta_p": max(deltas),
            "p99_abs_delta_p": quantile(deltas, .99), "max_abs_delta_logit_margin": max(margins)}


def assess(ab, aa, threshold=.20):
    p95, p50 = paired(ab), paired(ab, .5)
    control = [quantile(a, .95) for a, _ in ab]
    drift = max(control) / min(control) - 1
    aa_ratios = paired(aa)["block_ratios"]
    envelope = max(max(r, 1 / r) - 1 for r in aa_ratios)
    gain = 1 - p95["ratio"]
    reasons = []
    if gain <= envelope:
        reasons.append("effect_not_above_aa")
    if gain <= drift:
        reasons.append("effect_not_above_drift")
    if any(r >= 1 for r in p95["block_ratios"]):
        reasons.append("inconsistent_direction")
    if p95["ratio_interval_95"][1] >= 1:
        reasons.append("interval_includes_no_improvement")
    status = "inconclusive" if reasons else ("pass" if gain > threshold else "below_target")
    # A consistently slower candidate is a negative result, not uncertain speedup.
    if (-gain > max(envelope, drift)
            and all(r > 1 for r in p95["block_ratios"])
            and p95["ratio_interval_95"][0] > 1):
        status, reasons = "below_target", ["consistent_slowdown"]
    return {"status": status, "p95": p95, "p50": p50, "p95_reduction": gain,
            "aa_envelope_p95": envelope, "baseline_drift_p95": drift, "reasons": reasons,
            "uncertainty_note": f"One VM, {len(ab)} paired blocks; bootstrap is not independent confirmation"}


def paired_rows(jobs, outcomes, bucket):
    grouped = {"aa": defaultdict(dict), "ab": defaultdict(dict)}
    labels = {"baseline": defaultdict(list), "candidate": defaultdict(list)}
    for job, result in zip(jobs, outcomes, strict=True):
        if job["bucket"] != bucket:
            continue
        if result["job"] != job or result["status"] != "passed":
            raise ValueError("Incomplete or mismatched side")
        rows = result["rows"]
        if [r["id"] for r in rows] != job["ids"]:
            raise ValueError("Paired input order changed")
        if any(not math.isfinite(r["latency_ms"]) or r["latency_ms"] <= 0 for r in rows):
            raise ValueError("Invalid raw latency")
        grouped[job["phase"]][job["block"]][job["side"]] = rows
        for r in rows:
            labels[job["variant"]][r["id"]].append(r["score"])
    output = {}
    for phase, blocks in grouped.items():
        output[phase] = []
        for sides in blocks.values():
            if set(sides) != {"a", "b"}:
                raise ValueError("Missing side")
            identity = lambda rs: [(r["id"], r["input_sha256"], r["total_tokens"]) for r in rs]
            if identity(sides["a"]) != identity(sides["b"]):
                raise ValueError("Pair identity mismatch")
            output[phase].append(tuple([r["latency_ms"] for r in sides[s]] for s in ("a", "b")))
    return output, labels
