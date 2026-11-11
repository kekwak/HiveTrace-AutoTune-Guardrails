"""Labeled evaluation plans, independent of speed buckets and timing protocols."""
import random
import time
from copy import deepcopy

from .core import digest, write
from .datasets import jsonl
from .messages import UnsupportedInput, payload, private_strings
from .precision import PrecisionError, validate_precision
from .quality import metrics


def batch_sizes(cfg):
    sizes = cfg.get("quality_batch_sizes", [1])
    if (not isinstance(sizes, list) or not sizes
            or any(type(n) is not int or n <= 0 for n in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("quality_batch_sizes requires unique positive integers")
    return sizes


def ordered_ids(rows, seed):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate quality IDs")
    random.Random(seed).shuffle(ids)
    return ids


def make_plan(rows, indices, errors, size, seed):
    ids = ordered_ids(rows, seed)
    eligible = [key for key in ids if key not in errors]
    batches = []
    for start in range(0, len(eligible), size):
        members = eligible[start:start + size]
        lengths = [indices[key]["total_tokens"] for key in members]
        batches.append({"ids": members, "actual_batch_size": len(members),
                        "lengths": lengths, "padded_length": max(lengths),
                        "inputs_sha256": digest([indices[key] for key in members])})
    return {"quality_protocol_version": 1, "batch_size": size, "seed": seed,
            "ordered_ids": ids, "indices": indices, "excluded": errors, "batches": batches}


def prepare_plan(adapter, rows, size, seed, precision):
    indices, errors = {}, {}
    for row in rows:
        key = row["id"]
        try:
            index = precision.call(adapter.index, payload(row))
            # Keep only index evidence, never tokenized payloads or rendered prompts.
            fields = ("user_tokens", "total_tokens", "input_sha256", "ids_sha256", "mask_sha256",
                      "target_role", "target_index", "context_indices", "source_mode", "source_sha256")
            index = {k: index[k] for k in fields if k in index}
            indices[key] = index
            if "target_role" in row and (index.get("target_role"), index.get("target_index")) != (
                    row["target_role"], row["target_index"]):
                errors[key] = "target_mismatch"
            elif not 0 < index["total_tokens"] <= adapter.limit:
                errors[key] = "over_limit"
        except PrecisionError:
            raise
        except UnsupportedInput:
            errors[key] = "not_applicable"
        except Exception as error:
            if type(error).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError", "UnsupportedBatch", "UnsupportedExperiment"):
                raise
            errors[key] = "error"
    return make_plan(rows, indices, errors, size, seed)


def public_checkpoint(result, rows, private=True):
    """Sanitize a copy before writing, even if the worker is killed without cleanup."""
    snapshot = deepcopy(result)
    if not private:
        return snapshot
    texts = sorted({text for row in rows for text in private_strings(payload(row)) if text},
                   key=len, reverse=True)

    def clean(value, key=None):
        if key == "precision":
            try:
                return validate_precision(value)
            except PrecisionError:
                pass
        enums = {"padding": ("longest", "none"), "padding_side": ("left", "right"),
                 "implementation": ("native", "single_only")}
        if key in enums and value in enums[key]:
            return value
        if isinstance(value, dict):
            return {clean(k): clean(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, str):
            for text in texts:
                if value == text:
                    value = "[redacted-input]"
                elif len(text) >= 4:
                    value = value.replace(text, "[redacted-input]")
        return value

    if "metadata" in snapshot:
        snapshot["metadata"] = clean(snapshot["metadata"])
    return snapshot


def evaluate(adapter, rows, cfg, size, directory, result, precision, *, private=True):
    """One pass including its partial tail. The caller owns the fresh process."""
    from .batch_worker import responses
    plan = prepare_plan(adapter, rows, size, cfg["seed"], precision)
    write(directory / "quality_plan.json", plan)
    result.update(quality_protocol_version=1, quality_batch_size=size,
                  quality_plan_sha256=digest(plan), quality_preserved=None)
    by_id = {r["id"]: r for r in rows}
    predictions = {r["id"]: {**{k: r[k] for k in ("id", "text_sha256", "content_sha256") if k in r},
                             "status": plan["excluded"].get(r["id"], "not_run")} for r in rows}
    groups = [[payload(by_id[key]) for key in batch["ids"]] for batch in plan["batches"]]

    def save():
        values = [predictions[key] for key in plan["ordered_ids"]]
        jsonl(directory / "predictions.jsonl", values)
        result["quality"] = metrics(rows, values, cfg["label_mapping"])
        result["quality_evaluated"] = result["quality"]["overall"]["evaluated"] > 0
        result["metadata"] = adapter.metadata
        write(directory / "result.json", public_checkpoint(result, rows, private))

    save()
    active = None
    try:
        if size > 1:
            from .batch_adapters import BatchAdapter
            if isinstance(adapter, BatchAdapter):
                adapter.require_native_batch()
        if groups:
            start = time.monotonic()
            if size == 1:
                precision.call(adapter.prepare_backend, [group[0] for group in groups])
                proof = getattr(adapter, "prove_backend", None)
                if not callable(proof):
                    raise ValueError("Quality adapter requires backend proof")
                returned = precision.call(proof, groups[0][0])
                if isinstance(returned, dict) and "backend_proof" not in adapter.metadata:
                    adapter.metadata["backend_proof"] = deepcopy(returned.get("backend_proof", returned))
            else:
                precision.call(adapter.prepare_batch_backend, groups)
                precision.call(adapter.prove_batch_backend, groups[0])
            result["backend_preparation_seconds"] = time.monotonic() - start
            result["prepared_shapes"] = sorted({(len(g), p["padded_length"])
                                                for g, p in zip(groups, plan["batches"], strict=True)})
        for batch, group in zip(plan["batches"], groups, strict=True):
            active = batch["ids"]
            expected = [plan["indices"][key] for key in active]
            values = ([precision.call(adapter.verify, group[0], expected[0])] if size == 1 else
                      precision.call(adapter.verify_batch, group, expected))
            values = responses(values, len(group))
            if any(value["label"] not in cfg["label_mapping"]["verdicts"] for value in values):
                raise ValueError("Native verdict is outside explicit quality mapping")
            for key, value in zip(active, values, strict=True):
                predictions[key].update(status="passed", score=value)
            active = None
            save()
    except BaseException as error:
        kind = type(error).__name__
        status = ("oom" if kind in ("OutOfMemoryError", "CUDAOutOfMemoryError") else
                  "unsupported" if kind in ("UnsupportedBatch", "UnsupportedExperiment") else "error")
        for key in active or []:
            predictions[key].update(status=status, error_type=kind)
        raise
    finally:
        save()


def execute_matrix(cfg, output, launch):
    """Keep failures local to one size/variant; global deadline still applies."""
    from .core import read
    variants = ["baseline"] + (["candidate"] if cfg.get("candidate") else [])
    state = {}
    for size in batch_sizes(cfg):
        for variant in variants:
            key = f"{size}/{variant}"
            directory = output / "quality" / str(size) / variant
            try:
                launch("quality", variant, directory, {"quality_batch_size": size})
                state[key] = {"status": read(directory / "result.json")["status"]}
            except Exception as error:
                state[key] = {"status": "timeout" if isinstance(error, TimeoutError) else "failed",
                              "error_type": type(error).__name__}
                if (directory / "result.json").exists() and not isinstance(error, TimeoutError):
                    try:
                        saved_status = read(directory / "result.json").get("status")
                    except (OSError, ValueError, TypeError, AttributeError):
                        saved_status = None
                    if saved_status in ("oom", "unsupported"):
                        state[key]["status"] = saved_status
                write(output / "quality_status.json", state)
                if isinstance(error, TimeoutError):
                    raise
            write(output / "quality_status.json", state)
    return state
