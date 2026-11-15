"""Paired offline analysis of two U1 quality prediction files (reference vs candidate) on tails_v2.

python -m u2_quant.analyze --corpus <quality/corpus.jsonl> --baseline <predictions.jsonl> \
    --candidate <predictions.jsonl> --out <dir> [--slice technique]
"""
import argparse
import json
from math import ceil, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from guard_eval.metrics import mcnemar, roc_auc

BOOTSTRAP = 1000
POSITIVE = "unsafe"


def mde(n, better=0.005, alpha=0.05, power=0.8):
    """Minimal net share of worse flips that a two-sided McNemar detects on n pairs (as in tails_v2 README)."""
    z = NormalDist().inv_cdf
    for d in (x / 1000 for x in range(1, 1000)):
        p = 2 * better + d
        if ceil((z(1 - alpha / 2) * sqrt(p) + z(power) * sqrt(p - d * d)) ** 2 / d ** 2) <= n:
            return d
    return float("nan")


def _passed(prediction):
    return prediction is not None and prediction.get("status") == "passed"


def _rate(mask, among):
    return mask[among].mean() if among.any() else float("nan")


def _auroc(y, predictions):
    scores = [p["score"].get("scores", {}).get(POSITIVE) for p in predictions]
    if any(s is None for s in scores) or y.all() or not y.any():
        return float("nan")
    return float(roc_auc(y, np.asarray(scores, float)))


def _group(items, base, cand, rng):
    common = [r for r in items if _passed(base.get(r["id"])) and _passed(cand.get(r["id"]))]
    n = len(common)
    row = {"n_total": len(items), "n_common": n, "coverage": n / len(items) if items else float("nan"),
           "failures_base": sum(not _passed(base.get(r["id"])) for r in items),
           "failures_cand": sum(not _passed(cand.get(r["id"])) for r in items)}
    y = np.array([r["target"] == POSITIVE for r in common], bool)
    a = np.array([base[r["id"]]["score"]["label"] == POSITIVE for r in common], bool)
    b = np.array([cand[r["id"]]["score"]["label"] == POSITIVE for r in common], bool)
    flips = a != b
    worse, better = int((flips & (b != y)).sum()), int((flips & (b == y)).sum())
    if n:
        samples = flips[rng.integers(0, n, size=(BOOTSTRAP, n))].mean(-1)
        lo, hi = np.percentile(samples, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    row.update(flips=int(flips.sum()), worse=worse, better=better, flip_rate=flips.mean() if n else float("nan"),
               flip_lo=float(lo), flip_hi=float(hi), p_mcnemar=mcnemar(worse, better),
               d_fnr=_rate(~b, y) - _rate(~a, y), d_fpr=_rate(b, ~y) - _rate(a, ~y),
               mde=mde(n) if n else float("nan"), mde_fnr=mde(int(y.sum())) if y.any() else float("nan"),
               mde_fpr=mde(int((~y).sum())) if (~y).any() else float("nan"), zero_flip_bound=3 / n if n else float("nan"),
               auroc_base=_auroc(y, [base[r["id"]] for r in common]),
               auroc_cand=_auroc(y, [cand[r["id"]] for r in common]))
    return row


def _holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def compare(rows, base, cand, slice_key="technique", seed=17):
    rng = np.random.default_rng(seed)
    groups = {}
    for value in sorted({r["slices"][slice_key] for r in rows}):
        groups[value] = _group([r for r in rows if r["slices"][slice_key] == value], base, cand, rng)
    holm = _holm({k: v["p_mcnemar"] for k, v in groups.items()})
    for key, value in groups.items():
        value["p_holm"] = holm[key]
    pooled = _group(rows, base, cand, rng)
    pooled["p_holm"] = pooled["p_mcnemar"]
    return pd.DataFrame({**groups, "pooled": pooled}).T


def _lines(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("--corpus", "--baseline", "--candidate", "--out"):
        parser.add_argument(name, required=True)
    parser.add_argument("--slice", default="technique")
    args = parser.parse_args(argv)
    rows = _lines(args.corpus)
    base = {p["id"]: p for p in _lines(args.baseline)}
    cand = {p["id"]: p for p in _lines(args.candidate)}
    table = compare(rows, base, cand, args.slice)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"analysis_{args.slice}.csv")
    (out / f"analysis_{args.slice}.md").write_text(table.to_string() + "\n")   # tabulate нет в lock
    print(table[["n_common", "flips", "worse", "better", "p_mcnemar", "p_holm", "d_fnr", "d_fpr", "mde"]].to_string())


if __name__ == "__main__":
    main()
