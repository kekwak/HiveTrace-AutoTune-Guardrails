"""Deterministic, JSON-safe batch plans; no model loading or input rendering.

Indices and effective-input uniqueness are verified by the dataset loader. This
module also checks the structural invariants needed to avoid partial batches.
Occurrences are zero-based per ID within a series (not across paired sides).
``padded_tokens`` counts all input positions, not just added padding.
"""
import copy
import math
import random
from collections import Counter
from collections.abc import Mapping

from .contracts import required_buckets, MODEL_SPECS
from .core import digest


SIZES = (1, 4, 8, 16, 32, 64, 128)


def _buckets(model):
    if not isinstance(model, str) or model not in MODEL_SPECS:
        raise ValueError("Unknown batch model")
    return sorted(required_buckets(model))


def _size(size):
    if type(size) is not int or size not in SIZES or size > 400:
        raise ValueError("Unsupported batch size")
    count = max(400, 100 * size)
    if count % 400 or count % size:
        raise ValueError("Batch size cannot evenly replay the screening corpus")
    return size


def default_grid(model):
    """Return a fresh agreed class preset; custom grids may use any SIZES."""
    buckets = _buckets(model)
    if len(buckets) == 1:
        return {"B1": [1, 8, 16, 32, 64]}
    return {"B1": [1, 4, 8, 16, 32], "B2": [1, 4, 8],
            "B3": [1, 4], "B4": [1]}


def validate_grid(model, grid):
    """Canonicalize explicit grids, requiring exactly the model's buckets.

    Nonempty subsets of SIZES are accepted for custom smoke or limited runs;
    missing buckets, duplicate sizes, booleans and coerced strings are rejected.
    """
    buckets = _buckets(model)
    if not isinstance(grid, Mapping) or set(grid) != set(buckets):
        raise ValueError("Batch grid must contain exactly the model buckets")
    result = {}
    for bucket in buckets:
        sizes = grid[bucket]
        if not isinstance(sizes, (list, tuple)) or not sizes:
            raise ValueError("Each bucket needs a nonempty sequence of sizes")
        checked = [_size(size) for size in sizes]
        if len(set(checked)) != len(checked):
            raise ValueError("Duplicate batch size")
        result[bucket] = sorted(checked)
    return result


def cells(cfg):
    """Return canonical (bucket, size) pairs; absent grid means singleton."""
    model = cfg["model"]
    grid = cfg.get("batch_sizes_by_bucket", {b: [1] for b in _buckets(model)})
    return [(bucket, size) for bucket, sizes in validate_grid(model, grid).items()
            for size in sizes]


def plan_hash(value):
    """SHA256 of canonical JSON, matching the repository digest convention."""
    return digest(value)


def _rows(rows):
    rows = list(rows)
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Invalid batch input row")
        key, index = row.get("id"), row.get("index")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing or duplicate input ID")
        if not isinstance(index, Mapping):
            raise ValueError("Missing prepared input index")
        length, checksum = index.get("total_tokens"), index.get("input_sha256")
        if type(length) is not int or length <= 0:
            raise ValueError("Invalid full input length")
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)):
            raise ValueError("Missing or malformed prepared input SHA256")
        seen.add(key)
    if not rows:
        raise ValueError("No batch input rows")
    return sorted(rows, key=lambda row: row["id"])


def _seed(seed):
    if type(seed) is not int:
        raise ValueError("Batch seed must be an integer")
    return seed


def _permutation(rows, seed):
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def _batch(rows, occurrences):
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ID within a batch")
    lengths = [row["index"]["total_tokens"] for row in rows]
    padded = len(rows) * max(lengths)
    result = {"ids": ids, "occurrences": list(occurrences),
              "total_tokens": lengths,
              "input_sha256": [row["index"]["input_sha256"] for row in rows],
              "padded_length": max(lengths), "useful_tokens": sum(lengths),
              "padded_tokens": padded, "padding_tokens": padded - sum(lengths)}
    return {"id": plan_hash(result), **result}


def _plans(order, size, count):
    if len(order) < size:
        raise ValueError("Not enough unique IDs for a full batch")
    if count <= 0 or count % size:
        raise ValueError("Plan would contain a partial batch")
    seen = Counter()
    plans = []
    # Replay one permutation, including across a cycle boundary. With N >= B,
    # a contiguous window of B positions cannot repeat an ID within a batch.
    for offset in range(0, count, size):
        members = [order[(offset + i) % len(order)] for i in range(size)]
        occurrences = [seen[row["id"]] for row in members]
        plans.append(_batch(members, occurrences))
        seen.update(row["id"] for row in members)
    return plans


def smoke_plans(rows, size, seed=17):
    """Three full batches: shortest, longest, mixed; overlaps across them allowed."""
    size = _size(size)
    rows = _rows(rows)
    if len(rows) < size:
        raise ValueError("Not enough unique IDs for smoke batch")
    by_length = sorted(rows, key=lambda row: (row["index"]["total_tokens"], row["id"]))
    groups = [by_length[:size], by_length[-size:],
              _permutation(rows, _seed(seed))[:size]]
    seen, plans = Counter(), []
    for group in groups:
        plans.append(_batch(group, [seen[row["id"]] for row in group]))
        seen.update(row["id"] for row in group)
    return plans


def profile_plans(rows, size, seed=17):
    """Return the short and long batches from smoke planning."""
    return smoke_plans(rows, size, seed)[:2]


def memory_plans(rows, size, seed=17):
    """Minimal complete corpus replays forming full batches (LCM(N, B))."""
    size = _size(size)
    rows = _rows(rows)
    order = _permutation(rows, _seed(seed))
    return _plans(order, size, math.lcm(len(rows), size))


def materialize(plan, rows):
    """Resolve an authenticated batch to original rows in order, without copies.

    The returned rows may contain private text; unlike plans they must not be
    archived by default. Reject stale token lengths/hashes or altered metadata.
    """
    by_id = {row["id"]: row for row in _rows(rows)}
    if not isinstance(plan, Mapping):
        raise ValueError("Invalid batch plan")
    ids, occurrences = plan.get("ids"), plan.get("occurrences")
    if (not isinstance(ids, list) or not ids or not isinstance(occurrences, list)
            or len(ids) != len(occurrences)
            or any(type(n) is not int or n < 0 for n in occurrences)):
        raise ValueError("Invalid batch membership/occurrences")
    if any(not isinstance(key, str) or key not in by_id for key in ids):
        raise ValueError("Batch references missing input IDs")
    result = [by_id[key] for key in ids]
    if _batch(result, occurrences) != plan:
        raise ValueError("Batch plan or input index changed")
    return result


def schedule(cfg, data):
    """Cell-first latency jobs with independent, identical plans for paired sides.

    Smoke/quality have no latency schedule. Dataset validation remains upstream;
    screening additionally requires exactly 400 rows here. Plan hashes identify
    batches, not timing labels: A and B therefore share a plan hash.
    """
    mode = cfg.get("mode", "screening")
    if mode not in ("screening", "pilot", "smoke", "quality"):
        raise ValueError("Unknown batch run mode")
    grid_cells = cells(cfg)
    if mode in ("smoke", "quality"):
        return []
    if not isinstance(data, Mapping) or set(data) != set(_buckets(cfg["model"])):
        raise ValueError("Missing or unexpected model buckets")
    seed = _seed(cfg.get("seed", 17))
    has_candidate = cfg.get("candidate") is not None
    phase = "ab" if has_candidate else "baseline"
    phases = [(phase, 1)] if mode == "pilot" else [("aa", 2), (phase, 3)]
    jobs = []
    for bucket, size in grid_cells:
        rows = _rows(data[bucket])
        if mode == "screening" and len(rows) != 400:
            raise ValueError("Screening requires exactly 400 unique input IDs")
        for phase, blocks in phases:
            for block in range(blocks):
                order = _permutation(rows, f"{seed}:{phase}:{block}:{bucket}")
                count = (12 * size if mode == "pilot" else
                         30 * size if phase == "aa" else max(400, 100 * size))
                batches = _plans(order, size, count)
                sides = (("a",) if phase == "baseline" else
                         ("a", "b") if block % 2 == 0 else ("b", "a"))
                for side in sides:
                    jobs.append({"phase": phase, "block": block, "bucket": bucket,
                                 "side": side, "variant": "candidate" if phase == "ab" and side == "b" else "baseline",
                                 "ids": [key for batch in batches for key in batch["ids"]],
                                 "batch_size": size, "batches": copy.deepcopy(batches),
                                 "plan_sha256": plan_hash(batches)})
    return jobs
