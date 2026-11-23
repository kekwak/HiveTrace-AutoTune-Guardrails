"""Paired quality evaluation for caller-owned inference runtimes."""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path
from statistics import NormalDist
from typing import Any

LABELS = {"safe", "unsafe"}
STATUSES = {"ok", "over_limit", "invalid_output", "inference_error"}


@dataclass(frozen=True)
class QualityInput:
    id: str
    text: str


@dataclass(frozen=True)
class QualityRow:
    id: str
    text: str
    target: str
    slices: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    id: str
    status: str
    verdict: str | None = None
    unsafe_score: float | None = None
    truncated: bool | None = None


Predictor = Callable[[list[QualityInput]], Iterable[Prediction]]


def _rows(dataset: Iterable[QualityRow]) -> list[QualityRow]:
    rows = list(dataset)
    seen = set()
    for row in rows:
        if not isinstance(row.id, str) or not row.id.strip() or row.id in seen:
            raise ValueError(f"Invalid or duplicate dataset ID: {row.id!r}")
        if row.target not in LABELS or not isinstance(row.text, str):
            raise ValueError(f"Invalid target or text for {row.id!r}")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in row.slices.items()):
            raise ValueError(f"Invalid slices for {row.id!r}")
        seen.add(row.id)
    if not rows:
        raise ValueError("Quality dataset is empty")
    return rows


def _index(values: Iterable[Prediction], expected: set[str]) -> dict[str, Prediction]:
    indexed: dict[str, Prediction] = {}
    for value in values:
        if value.id not in expected or value.id in indexed:
            raise ValueError(f"Unexpected or duplicate prediction ID: {value.id!r}")
        if value.status not in STATUSES or (
            value.truncated is not None and type(value.truncated) is not bool
        ):
            raise ValueError(f"Invalid prediction status or truncation flag for {value.id!r}")
        if value.status == "ok":
            if value.verdict not in LABELS:
                raise ValueError(f"Invalid verdict for {value.id!r}")
            score = value.unsafe_score
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(f"Invalid unsafe score for {value.id!r}")
        elif value.verdict is not None or value.unsafe_score is not None:
            raise ValueError(f"Failed prediction carries a verdict or score for {value.id!r}")
        indexed[value.id] = value
    if expected - indexed.keys():
        raise ValueError(f"Missing predictions for {len(expected - indexed.keys())} IDs")
    return indexed


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def _auc(pairs: list[tuple[QualityRow, Prediction]]) -> float | None:
    if any(p.unsafe_score is None for _, p in pairs):
        return None
    ranked = sorted(
        (p.unsafe_score, r.target == "unsafe") for r, p in pairs if p.unsafe_score is not None
    )
    positives = sum(label for _, label in ranked)
    negatives = len(ranked) - positives
    if not positives or not negatives:
        return None
    credit = 0.0
    negatives_before = 0
    for _, group in groupby(ranked, key=lambda item: item[0]):
        labels = [label for _, label in group]
        positive = sum(labels)
        negative = len(labels) - positive
        credit += positive * (negatives_before + negative / 2)
        negatives_before += negative
    return credit / (positives * negatives)


def _metrics(rows: Sequence[QualityRow], values: Mapping[str, Prediction]) -> dict[str, Any]:
    pairs = [(r, values[r.id]) for r in rows if values[r.id].status == "ok"]
    counts = Counter((r.target, p.verdict) for r, p in pairs)
    tp, fp = counts["unsafe", "unsafe"], counts["safe", "unsafe"]
    tn, fn = counts["safe", "safe"], counts["unsafe", "safe"]
    f1 = [_ratio(2 * tp, 2 * tp + fp + fn), _ratio(2 * tn, 2 * tn + fp + fn)]
    defined = [v for v in f1 if v is not None]
    return {
        "n_total": len(rows),
        "n_evaluated": len(pairs),
        "coverage": _ratio(len(pairs), len(rows)),
        "statuses": dict(Counter(values[r.id].status for r in rows)),
        "n_truncated": sum(values[r.id].truncated is True for r in rows),
        "n_truncation_unknown": sum(p.truncated is None for _, p in pairs),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "fnr": _ratio(fn, fn + tp),
        "fpr": _ratio(fp, fp + tn),
        "macro_f1": sum(defined) / len(defined) if defined else None,
        "roc_auc": _auc(pairs),
        "n_scored": sum(p.unsafe_score is not None for _, p in pairs),
    }


def _mcnemar(worse: int, better: int) -> float:
    n = worse + better
    return min(
        1.0,
        math.fsum(
            math.exp(
                math.lgamma(n + 1)
                - math.lgamma(i + 1)
                - math.lgamma(n - i + 1)
                - (n - 1) * math.log(2)
            )
            for i in range(min(worse, better) + 1)
        ),
    )


def _mde(n: int) -> float | None:
    # Normal approximation, two-sided alpha .05, power .8, better flip rate .005.
    z_alpha, z_power = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(0.8)
    for step in range(1, 996):
        delta = step / 1000
        discordance = 0.01 + delta
        if discordance > 1:
            break
        bound = z_alpha * math.sqrt(discordance)
        bound += z_power * math.sqrt(discordance - delta * delta)
        if n * delta * delta >= bound * bound:
            return delta
    return None


def _group(
    rows: Sequence[QualityRow], base: Mapping[str, Prediction], cand: Mapping[str, Prediction]
) -> dict[str, Any]:
    common = [r for r in rows if base[r.id].status == cand[r.id].status == "ok"]
    result = {
        "targets": dict(Counter(r.target for r in rows)),
        "baseline": _metrics(rows, base),
        "candidate": _metrics(rows, cand),
    }
    comparison: dict[str, Any] = {
        "n_common": len(common),
        "coverage": _ratio(len(common), len(rows)),
        "baseline": _metrics(common, base),
        "candidate": _metrics(common, cand),
    }
    flips = 0
    for key, target in (("fnr", "unsafe"), ("fpr", "safe")):
        subset = [r for r in common if r.target == target]
        worse = sum(base[r.id].verdict == target != cand[r.id].verdict for r in subset)
        better = sum(cand[r.id].verdict == target != base[r.id].verdict for r in subset)
        flips += worse + better
        comparison[key] = {
            "n": len(subset),
            "worse": worse,
            "better": better,
            "delta": _ratio(worse - better, len(subset)),
            "p": _mcnemar(worse, better) if subset else None,
            "mde": _mde(len(subset)) if subset else None,
        }
    comparison.update(flips=flips, flip_rate=_ratio(flips, len(common)))
    result["comparison"] = comparison
    return result


def _holm(groups: Mapping[str, Any], key: str) -> None:
    entries = [g["comparison"][key] for g in groups.values()]
    valid = sorted((e for e in entries if e["p"] is not None), key=lambda e: e["p"])
    running = 0.0
    for i, entry in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - i) * entry["p"]))
        entry["p_holm"] = running
    for entry in entries:
        entry.setdefault("p_holm", None)


def compare_predictions(
    dataset: Iterable[QualityRow],
    baseline_predictions: Iterable[Prediction],
    candidate_predictions: Iterable[Prediction],
    *,
    dataset_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a versioned report from two complete prediction streams."""
    rows = _rows(dataset)
    expected = {r.id for r in rows}
    base = _index(baseline_predictions, expected)
    cand = _index(candidate_predictions, expected)
    partitions: dict[str, dict[str, list[QualityRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.slices.items():
            partitions[key][value].append(row)
    slices = {}
    for key, values in sorted(partitions.items()):
        groups = {value: _group(items, base, cand) for value, items in sorted(values.items())}
        for metric in ("fnr", "fpr"):
            _holm(groups, metric)
        slices[key] = groups
    return {
        "schema_version": 1,
        "dataset": {
            **(dataset_metadata or {}),
            "n_total": len(rows),
            "targets": dict(Counter(r.target for r in rows)),
        },
        "statistics": {"alpha": 0.05, "power": 0.8, "assumed_better_flip_rate": 0.005},
        **_group(rows, base, cand),
        "slices": slices,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    pending = path.with_suffix(".tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def evaluate_quality(
    dataset: Iterable[QualityRow],
    baseline_predict: Predictor,
    candidate_predict: Predictor,
    *,
    batch_size: int,
    output_dir: Path,
    dataset_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate both variants sequentially and persist text-free predictions and report."""
    rows = _rows(dataset)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    metadata = dict(dataset_metadata or {})
    json.dumps(metadata, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_dir / "dataset.json",
        {
            "schema_version": 1,
            "metadata": metadata,
            "rows": [{"id": r.id, "target": r.target, "slices": dict(r.slices)} for r in rows],
        },
    )
    status = output_dir / "status.json"
    _write_json(status, {"status": "running", "batch_size": batch_size})
    runs = []
    try:
        for name, predict in (("baseline", baseline_predict), ("candidate", candidate_predict)):
            predictions: list[Prediction] = []
            with (output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as stream:
                for start in range(0, len(rows), batch_size):
                    inputs = [QualityInput(r.id, r.text) for r in rows[start : start + batch_size]]
                    indexed = _index(predict(inputs), {item.id for item in inputs})
                    for item in inputs:
                        value = indexed[item.id]
                        stream.write(json.dumps(asdict(value), allow_nan=False) + "\n")
                        predictions.append(value)
                    stream.flush()
            runs.append(predictions)
        report = compare_predictions(rows, runs[0], runs[1], dataset_metadata=metadata)
        report["batch_size"] = batch_size
        _write_json(output_dir / "quality_report.json", report)
    except BaseException:
        _write_json(status, {"status": "incomplete", "batch_size": batch_size})
        raise
    _write_json(status, {"status": "complete", "batch_size": batch_size})
    return report
