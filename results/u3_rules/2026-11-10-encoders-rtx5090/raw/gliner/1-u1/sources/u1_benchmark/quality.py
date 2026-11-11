"""Native verdict agreement and labeled quality are different measurements."""
from collections import Counter

from .contracts import validate_response


def agreement(baseline, candidate):
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("Missing or mismatched unique response IDs")
    changed, unstable, directions = [], [], Counter()
    deltas = {}
    for key in baseline:
        a, b = baseline[key], candidate[key]
        if not a or not b:
            raise ValueError("Missing response")
        for value in a + b:
            validate_response(value)
        if len({s["label"] for s in a}) != 1 or len({s["label"] for s in b}) != 1:
            unstable.append(key)
        if a[0]["label"] != b[0]["label"]:
            changed.append(key)
            directions[a[0]["label"], b[0]["label"]] += 1
        if a[0].get("score_type") == b[0].get("score_type") and a[0].get("scores") and set(a[0]["scores"]) == set(b[0].get("scores", {})):
            for label in a[0]["scores"]:
                deltas[label] = max(deltas.get(label, 0), max(abs(x["scores"][label] - y["scores"][label]) for x in a for y in b))
    return {"n": len(baseline), "changed_count": len(changed), "changed_fraction": len(changed) / len(baseline),
            "changed_ids": changed, "unstable_ids": unstable, "max_abs_score_delta": deltas or None,
            "directions": [{"baseline": a, "candidate": b, "count": n} for (a, b), n in sorted(directions.items())],
            "status": "ok" if not changed and not unstable else "not_ok"}


def validate_mapping(rows, mapping):
    if not isinstance(mapping, dict) or set(mapping) - {"targets", "verdicts", "labels", "positive_label"}:
        raise ValueError("Explicit targets/verdicts mapping required")
    labels = mapping["labels"]
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError("Invalid quality label universe")
    for kind in ("targets", "verdicts"):
        if not isinstance(mapping[kind], dict) or not mapping[kind] or not set(mapping[kind].values()) <= set(labels):
            raise ValueError("Invalid label mapping")
    if not {r["target"] for r in rows} <= mapping["targets"].keys():
        raise ValueError("Unmapped dataset target")
    if "positive_label" in mapping and (len(labels) != 2 or mapping["positive_label"] not in labels):
        raise ValueError("FPR/FNR require an explicit binary positive label")


def metrics(rows, predictions, mapping):
    validate_mapping(rows, mapping)
    if len({p["id"] for p in predictions}) != len(predictions) or {r["id"] for r in rows} != {p["id"] for p in predictions}:
        raise ValueError("Every quality row needs a prediction or explicit failure")
    by_id = {p["id"]: p for p in predictions}
    def compute(subset):
        labels = mapping["labels"]
        matrix = {a: {b: 0 for b in labels} for a in labels}
        failures = Counter()
        for row in subset:
            p = by_id[row["id"]]
            if p.get("status") != "passed":
                failures[p.get("status", "missing")] += 1
                continue
            native = validate_response(p["score"])["label"]
            if native not in mapping["verdicts"]:
                raise ValueError("Native verdict is outside explicit quality mapping")
            matrix[mapping["targets"][row["target"]]][mapping["verdicts"][native]] += 1
        n = sum(sum(row.values()) for row in matrix.values())
        per_class = {}
        for label in labels:
            tp = matrix[label][label]
            fp = sum(matrix[x][label] for x in labels if x != label)
            fn = sum(matrix[label][x] for x in labels if x != label)
            denom = 2 * tp + fp + fn
            per_class[label] = {"support": sum(matrix[label].values()), "f1": 2 * tp / denom if denom else None}
        values = [v["f1"] for v in per_class.values()]
        result = {"total": len(subset), "evaluated": n, "coverage": n / len(subset) if subset else None,
                  "failures": dict(failures), "confusion_matrix": matrix, "per_class": per_class,
                  "accuracy": sum(matrix[l][l] for l in labels) / n if n else None,
                  "macro_f1": sum(values) / len(values) if all(v is not None for v in values) else None,
                  "undefined_f1_labels": [l for l in labels if per_class[l]["f1"] is None]}
        if "positive_label" in mapping:
            pos = mapping["positive_label"]
            neg = next(l for l in labels if l != pos)
            negatives, positives = sum(matrix[neg].values()), sum(matrix[pos].values())
            result.update(fpr=matrix[neg][pos] / negatives if negatives else None,
                          fnr=matrix[pos][neg] / positives if positives else None,
                          negative_n=negatives, positive_n=positives)
        return result
    slices = {}
    keys = set(rows[0].get("slices", {}))
    if any(set(r.get("slices", {})) != keys for r in rows):
        raise ValueError("Incomplete slice fields")
    for name in sorted(keys):
        slices[name] = {value: compute([r for r in rows if r["slices"][name] == value])
                        for value in sorted({r["slices"][name] for r in rows})}
    return {"status": "evaluated", "quality_preserved": None, "overall": compute(rows), "slices": slices}
