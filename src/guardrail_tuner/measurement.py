"""Shared measurement and conservative selection for synchronous runners."""

import math
import statistics
import time
from typing import Any, Protocol

from .benchmark import percentile

MIN_REQUESTS = 100
ROUNDS = 3
MIN_IMPROVEMENT = 0.05
MAX_REFERENCE_SPREAD = 0.05


class Runner(Protocol):
    """Internal interface for a single request; concrete runners own their lifecycle."""

    def run(self, item: Any) -> Any: ...


def _measure(runner: Runner, items: Any) -> dict[str, Any]:
    durations = []
    for item in items:
        start = time.perf_counter()
        runner.run(item)
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "requests": len(durations),
        "items": sum(len(item) if isinstance(item, list) else 1 for item in items),
        "items_per_second": (
            sum(len(item) if isinstance(item, list) else 1 for item in items)
            / (sum(durations) / 1000)
            if sum(durations) > 0
            else None
        ),
        "p50_ms": percentile(durations, 0.5),
        "p95_ms": percentile(durations, 0.95),
        "p99_ms": percentile(durations, 0.99),
    }


def measure_pair(reference: Runner, candidate: Runner, items: Any, warmup: int) -> dict[str, Any]:
    """Alternate order over rounds; small datasets produce a preliminary single round."""
    for runner in (reference, candidate):
        for index in range(warmup):
            runner.run(items[index % len(items)])
    rounds = ROUNDS if len(items) >= MIN_REQUESTS else 1
    measurements: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
    for index in range(rounds):
        order = (("reference", reference), ("candidate", candidate))
        for name, runner in order if index % 2 == 0 else reversed(order):
            measurements[name].append(_measure(runner, items))
    return measurements


def measure_aa_ab(
    reference_a: Runner, reference_b: Runner, candidate: Runner, items: Any, warmup: int
) -> dict[str, list[dict[str, Any]]]:
    """Compare two independent references and a candidate with rotating order."""
    runners = (("reference_a", reference_a), ("reference_b", reference_b), ("candidate", candidate))
    for _, runner in runners:
        for index in range(warmup):
            runner.run(items[index % len(items)])
    measurements: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in runners}
    for index in range(ROUNDS if len(items) >= MIN_REQUESTS else 1):
        for name, runner in runners[index % 3 :] + runners[: index % 3]:
            measurements[name].append(_measure(runner, items))
    return measurements


def aa_selection(
    measurements: dict[str, Any], aa_quality: bool, ab_quality: bool
) -> dict[str, Any]:
    """Gate AB selection on independent AA output parity and per-round p99 agreement."""
    paired = {"reference": measurements["reference_a"], "candidate": measurements["candidate"]}
    decision = selection(paired, ab_quality)
    decision["aa_quality_passed"] = aa_quality
    decision["aa_max_p99_difference_fraction"] = None
    if not aa_quality:
        decision.update(selected="reference", status="rejected", reason="aa_quality_failed")
        return decision
    if not ab_quality:
        return decision
    a, b = measurements["reference_a"], measurements["reference_b"]
    if (
        len(a) < ROUNDS
        or len(b) < ROUNDS
        or any(run["requests"] < MIN_REQUESTS for run in [*a, *b])
    ):
        if ab_quality:
            decision.update(
                selected="reference", status="preliminary", reason="insufficient_aa_measurements"
            )
        return decision
    differences = [
        abs(x["p99_ms"] - y["p99_ms"]) / max(x["p99_ms"], y["p99_ms"])
        if max(x["p99_ms"], y["p99_ms"]) > 0
        else float("inf")
        for x, y in zip(a, b)
    ]
    maximum = max(differences)
    decision["aa_max_p99_difference_fraction"] = maximum if math.isfinite(maximum) else None
    if maximum > MAX_REFERENCE_SPREAD:
        decision.update(selected="reference", status="preliminary", reason="aa_latency_unstable")
    return decision


def selection(measurements: dict[str, Any], quality_passed: bool) -> dict[str, Any]:
    """Apply a conservative heuristic, not a statistical confidence guarantee."""
    ref, cand = measurements["reference"], measurements["candidate"]
    result: dict[str, Any] = {
        "selected": "reference",
        "status": "preliminary",
        "reason": "insufficient_measurements",
        "min_requests_per_round": MIN_REQUESTS,
        "required_rounds": ROUNDS,
        "min_improvement_fraction": MIN_IMPROVEMENT,
        "max_reference_spread_fraction": MAX_REFERENCE_SPREAD,
        "reference_p99_spread_fraction": None,
    }
    if not quality_passed:
        result.update(status="rejected", reason="quality_failed")
        return result
    if (
        len(ref) < ROUNDS
        or len(cand) < ROUNDS
        or any(run["requests"] < MIN_REQUESTS for run in [*ref, *cand])
    ):
        return result
    values = [run["p99_ms"] for run in ref]
    median = statistics.median(values)
    spread = (max(values) - min(values)) / median if median > 0 else float("inf")
    result["reference_p99_spread_fraction"] = spread if spread != float("inf") else None
    if spread > MAX_REFERENCE_SPREAD:
        result.update(reason="reference_unstable")
    elif not all(c["p99_ms"] <= r["p99_ms"] * (1 - MIN_IMPROVEMENT) for r, c in zip(ref, cand)):
        result.update(status="selected", reason="insufficient_improvement")
    else:
        result.update(status="selected", selected="candidate", reason="criteria_passed")
    return result
