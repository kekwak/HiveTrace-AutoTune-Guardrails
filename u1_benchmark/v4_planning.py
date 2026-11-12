"""Model-independent v4 plans over prepared, bucketed rows.

Variants are opaque configuration dictionaries. Plans contain only identifiers
and input-index metadata, never messages. Search uses the baseline seed phase
so it can replay a baseline even without loading its archived schedule.
"""
import copy
import random
from collections import Counter
from collections.abc import Mapping

from .core import digest as plan_hash


def _integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}")
    return value


def cells(cfg):
    """Canonical (bucket, B) cells; B's corpus-dependent upper bound is checked later."""
    buckets = cfg.get("buckets")
    if not isinstance(buckets, Mapping) or not buckets:
        raise ValueError("Nonempty configured buckets are required")
    for bucket, bounds in buckets.items():
        if not isinstance(bucket, str) or not bucket or not isinstance(bounds, Mapping):
            raise ValueError("Invalid bucket configuration")
        low = _integer(bounds.get("min"), "bucket minimum", 1)
        high = _integer(bounds.get("max"), "bucket maximum", 1)
        if low > high:
            raise ValueError("Reversed bucket bounds")
    grid = cfg.get("batch_sizes_by_bucket")
    if not isinstance(grid, Mapping) or set(grid) != set(buckets):
        raise ValueError("Batch grid must contain exactly the configured buckets")
    result = []
    for bucket in sorted(buckets):
        sizes = grid[bucket]
        if not isinstance(sizes, (list, tuple)) or not sizes:
            raise ValueError("Each bucket needs batch sizes")
        for size in sizes:
            _integer(size, "batch size", 1)
        if len(set(sizes)) != len(sizes):
            raise ValueError("Duplicate batch size")
        result.extend((bucket, size) for size in sorted(sizes))
    return result


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
        _integer(index.get("total_tokens"), "full input length", 1)
        checksum = index.get("input_sha256")
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)):
            raise ValueError("Malformed prepared input SHA256")
        seen.add(key)
    if not rows:
        raise ValueError("No batch input rows")
    return sorted(rows, key=lambda row: row["id"])


def _size(size, rows):
    _integer(size, "batch size", 1)
    if size > len(rows):
        raise ValueError("Not enough unique IDs for a full batch")
    return size


def _permutation(rows, seed):
    order = list(rows)
    random.Random(seed).shuffle(order)
    return order


def _batch(rows, occurrences):
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ID within a batch")
    lengths = [row["index"]["total_tokens"] for row in rows]
    padded = len(rows) * max(lengths)
    value = {"ids": ids, "occurrences": list(occurrences),
             "total_tokens": lengths,
             "input_sha256": [row["index"]["input_sha256"] for row in rows],
             "padded_length": max(lengths), "useful_tokens": sum(lengths),
             "padded_tokens": padded, "padding_tokens": padded - sum(lengths)}
    return {"id": plan_hash(value), **value}


def _plans(order, size, count):
    seen, plans = Counter(), []
    # A contiguous window of B <= N in one cyclic permutation has no duplicates.
    for offset in range(0, count * size, size):
        members = [order[(offset + i) % len(order)] for i in range(size)]
        plans.append(_batch(members, [seen[row["id"]] for row in members]))
        seen.update(row["id"] for row in members)
    return plans


def smoke_plans(rows, B, seed=17):
    """Three full batches: shortest, longest, deterministic mixed."""
    rows = _rows(rows)
    _size(B, rows)
    _integer(seed, "seed")
    ordered = sorted(rows, key=lambda row: (row["index"]["total_tokens"], row["id"]))
    seen, plans = Counter(), []
    for group in (ordered[:B], ordered[-B:], _permutation(rows, seed)[:B]):
        plans.append(_batch(group, [seen[row["id"]] for row in group]))
        seen.update(row["id"] for row in group)
    return plans


def profile_plans(rows, B, seed=17):
    return smoke_plans(rows, B, seed)[:2]


def materialize(plan, rows):
    """Authenticate a legacy-shaped batch and resolve original row objects."""
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
    expected = _batch(result, occurrences)
    if expected != plan or plan_hash(expected) != plan_hash(plan):
        raise ValueError("Batch plan or input index changed")
    return result


def _reference(jobs, reference_schedule, pilot=False):
    if not isinstance(reference_schedule, (list, tuple)):
        raise ValueError("Reference schedule must be a sequence of jobs")
    reference = {}
    for job in reference_schedule:
        if not isinstance(job, Mapping):
            raise ValueError("Invalid reference job")
        if job.get("phase") != "baseline":
            continue
        block = _integer(job.get("block"), "reference block", 0)
        size = _integer(job.get("batch_size"), "reference batch size", 1)
        bucket = job.get("bucket")
        if not isinstance(bucket, str):
            raise ValueError("Invalid reference bucket")
        key = (bucket, size, block)
        if key in reference or job.get("variant") != "baseline" or job.get("side") != "a":
            raise ValueError("Duplicate or invalid baseline reference")
        reference[key] = job
    expected = ({(j["bucket"], j["batch_size"], block) for j in jobs for block in range(3)}
                if pilot else {(j["bucket"], j["batch_size"], j["block"]) for j in jobs})
    if set(reference) != expected:
        raise ValueError("Reference main cells/blocks differ")
    for original in reference.values():
        batches = original.get("batches")
        if (not isinstance(batches, list) or (pilot and len(batches) < 100)
                or any(not isinstance(batch, Mapping) or not isinstance(batch.get("ids"), list)
                       for batch in batches)
                or original.get("plan_sha256") != plan_hash(batches)
                or original.get("ids") != [key for batch in batches for key in batch["ids"]]):
            raise ValueError("Invalid full baseline reference plan")
    for job in jobs:
        original = reference[(job["bucket"], job["batch_size"], job["block"])]
        batches = original["batches"][:12] if pilot else original["batches"]
        if (batches != job["batches"] or plan_hash(batches) != job["plan_sha256"]
                or [key for batch in batches for key in batch["ids"]] != job["ids"]):
            raise ValueError("Reference main plan differs from prepared inputs/seed")


def schedule(cfg, data, reference_schedule=None):
    """Cell-first jobs: baseline AA + main, search main only, confirm AA + AB.

    AA has two 30-batch pairs; main/AB has three blocks of max(100, ceil(N/B))
    batches. Pilot replaces these with one 12-batch main block (one AB pair in
    confirm). Search always uses the baseline permutation, but labels jobs
    ``search``/``candidate``/``b``. Pilot search replays the first 12 batches of
    block 0 per cell from a full reference (all three main blocks are required).
    Full baseline evidence/status validation belongs to the reference loader.
    Paired sides have independent plan objects.
    """
    mode = cfg.get("mode", "baseline")
    if mode not in ("baseline", "search", "confirm", "smoke", "quality"):
        raise ValueError("Unknown v4 run mode")
    pilot = cfg.get("pilot", False)
    if type(pilot) is not bool:
        raise ValueError("Pilot must be boolean")
    seed = _integer(cfg.get("seed", 17), "seed")
    if mode in ("smoke", "quality"):
        return []
    grid = cells(cfg)
    if mode in ("search", "confirm") and not isinstance(cfg.get("candidate"), Mapping):
        raise ValueError("Candidate variant is required")
    if not isinstance(data, Mapping) or set(data) != set(cfg["buckets"]):
        raise ValueError("Data must contain exactly the configured buckets")
    prepared, seen = {}, set()
    for bucket, source in data.items():
        rows = prepared[bucket] = _rows(source)
        bounds = cfg["buckets"][bucket]
        for row in rows:
            if row["id"] in seen:
                raise ValueError("Duplicate input ID across buckets")
            if not bounds["min"] <= row["index"]["total_tokens"] <= bounds["max"]:
                raise ValueError("Input length outside configured bucket")
            seen.add(row["id"])
    main = {"baseline": "baseline", "search": "search", "confirm": "ab"}[mode]
    phases = ([] if pilot or mode == "search" else [("aa", 2)])
    phases += [(main, 1 if pilot else 3)]
    jobs = []
    for bucket, size in grid:
        rows = prepared[bucket]
        _size(size, rows)
        for phase, blocks in phases:
            for block in range(blocks):
                seed_phase = "baseline" if phase == "search" else phase
                order = _permutation(rows, f"{seed}:{seed_phase}:{block}:{bucket}")
                count = (12 if pilot else 30 if phase == "aa" else
                         max(100, (len(rows) + size - 1) // size))
                batches = _plans(order, size, count)
                sides = (("a",) if phase == "baseline" else ("b",) if phase == "search"
                         else ("a", "b") if block % 2 == 0 else ("b", "a"))
                for side in sides:
                    variant = "candidate" if phase in ("search", "ab") and side == "b" else "baseline"
                    jobs.append({"phase": phase, "block": block, "bucket": bucket,
                                 "batch_size": size, "variant": variant, "side": side,
                                 "ids": [key for batch in batches for key in batch["ids"]],
                                 "batches": copy.deepcopy(batches),
                                 "plan_sha256": plan_hash(batches)})
    if reference_schedule is not None:
        if mode != "search":
            raise ValueError("Reference schedule is only supported for search")
        _reference(jobs, reference_schedule, pilot=pilot)
    return jobs


def workload(cfg, data, reference_schedule=None):
    """Print-safe timed-job counts, excluding parent warmup/preflight work.

    processed_rows counts row occurrences, not batches. Smoke/quality have no
    timed latency jobs, so their inference workload is not represented here.
    """
    jobs = schedule(cfg, data, reference_schedule)
    counts = {bucket: len(rows) for bucket, rows in data.items()}
    return {"scope": "timed_jobs_only", "rows": sum(counts.values()), "rows_by_bucket": counts,
            "jobs": len(jobs), "batches": sum(len(job["batches"]) for job in jobs),
            "processed_rows": sum(len(job["ids"]) for job in jobs)}
