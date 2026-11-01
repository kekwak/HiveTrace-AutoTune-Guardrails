"""Role corpus v4: one source pool, independently stratified native-input buckets."""
from collections import Counter, defaultdict
from pathlib import Path
import random

from .core import digest, lines, read, sha, write
from .contracts import BUCKETS, MODEL_SPECS, model_contract, required_buckets
from .messages import content_hash, messages, payload
from .statistics import quantile


def import_rows(source, fields, kind):
    if not {"id"} <= fields.keys() or ("text" in fields) == ("messages" in fields):
        raise ValueError("Map id and exactly one of text/messages")
    if set(fields) - {"id", "text", "messages", "target", "target_role", "target_index", "slices", "provenance", "language", "source"}:
        raise ValueError("Unknown role corpus fields")
    if kind == "quality" and not {"target", "target_role"} <= fields.keys():
        raise ValueError("Role quality data requires explicit target and target_role")
    result, seen, hashes, slice_keys = [], set(), set(), None
    for raw in lines(source):
        row = {key: raw[value] for key, value in fields.items()}
        key = row["id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Duplicate/invalid ID")
        row["messages"] = messages(payload(row))
        row.pop("text", None)
        checksum = content_hash(row["messages"])
        if kind == "speed" and checksum in hashes:
            raise ValueError("Duplicate source messages")
        row["content_sha256"] = checksum
        if kind == "quality":
            if not isinstance(row["target"], str) or not row["target"] or row["target_role"] not in ("user", "assistant"):
                raise ValueError("Invalid quality target/role")
            indices = [i for i, m in enumerate(row["messages"]) if m["role"] == row["target_role"]]
            if len(indices) != 1 or row.get("target_index", indices[0]) != indices[0]:
                raise ValueError("Quality target does not identify a source message")
            row["target_index"] = indices[0]
            slices = row.setdefault("slices", {})
            if not isinstance(slices, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in slices.items()):
                raise ValueError("Invalid quality slices")
            if slice_keys is not None and slice_keys != set(slices):
                raise ValueError("Incomplete slice columns")
            slice_keys = set(slices)
        seen.add(key)
        hashes.add(checksum)
        result.append(row)
    if not result:
        raise ValueError("Empty role corpus")
    return result


def stratum(length, bounds):
    if not bounds["min"] <= length <= bounds["max"]:
        raise ValueError("Length outside bucket")
    return min(7, (length - bounds["min"]) * 8 // (bounds["max"] - bounds["min"] + 1))


def select_bucket(rows, index, bounds, seed):
    cells = defaultdict(lambda: defaultdict(list))
    seen = set()
    # Dedupe actual token IDs: a different ignored context is not a new workload.
    for row in sorted(rows, key=lambda r: r["id"]):
        ix = index[row["id"]]
        n = ix.get("total_tokens")
        if not ix["supported"] or n is None or not bounds["min"] <= n <= bounds["max"]:
            continue
        fingerprint = ix["input_sha256"]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        group = (row.get("language", "unknown"), ix["source_mode"])
        cells[stratum(n, bounds)][group].append(row["id"])
    for interval, groups in cells.items():
        for group, ids in groups.items():
            random.Random(f"42:{seed}:{interval}:{group}").shuffle(ids)
    queues = {}
    for interval, groups in cells.items():
        queue = []
        keys = sorted(groups)
        while any(groups.values()):
            for group in keys:
                if groups[group]:
                    queue.append(groups[group].pop())
        queues[interval] = queue
    eligible = sum(map(len, queues.values()))
    selected, counts = [], Counter()
    # First meet interval targets, then redistribute deficits round-robin.
    for interval in range(8):
        chunk = queues.get(interval, [])[:50]
        selected.extend(chunk)
        counts[interval] += len(chunk)
        queues[interval] = queues.get(interval, [])[50:]
    while len(selected) < 400 and any(queues.values()):
        for interval in range(8):
            if queues[interval] and len(selected) < 400:
                selected.append(queues[interval].pop(0))
                counts[interval] += 1
    return selected, eligible, [counts[i] for i in range(8)]


def prepare(spec, output, preparer_factory=None):
    from .datasets import jsonl
    allowed = {"schema_version", "source", "kind", "fields", "models", "version", "provenance", "include_texts"}
    if set(spec) - allowed or spec.get("schema_version") != 3 or spec.get("kind") not in ("speed", "quality"):
        raise ValueError("Invalid role import spec")
    if not spec.get("version") or "provenance" not in spec:
        raise ValueError("Version and provenance required")
    rows = import_rows(spec["source"], spec["fields"], spec["kind"])
    if len(rows) > 48000 and spec["kind"] == "speed":
        raise ValueError("Role pool exceeds 48000 records")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jsonl(output / "corpus.jsonl", rows)
    manifest = {"schema_version": 3, "kind": spec["kind"], "version": spec["version"], "seed": 42,
                "count": len(rows), "source_sha256": sha(spec["source"]), "provenance": spec["provenance"],
                "include_texts": bool(spec.get("include_texts", False)), "quality_evaluated": False,
                "indices": {}, "model_buckets": {}, "status": "not_ready", "blockers": {}}
    if spec["kind"] == "quality":
        manifest.update(status="ready", targets=sorted({r["target"] for r in rows}), slices=sorted(rows[0]["slices"]))
    else:
        models = spec.get("models", [])
        if not models or len(set(models)) != len(models) or not set(models) <= MODEL_SPECS.keys():
            raise ValueError("Explicit unique participating models required")
        if preparer_factory is None:
            from .role_preparation import Preparer
            preparer_factory = Preparer
        for model in models:
            try:
                p = preparer_factory(model)
                parity = p.parity()
                if parity["status"] != "passed":
                    raise ValueError("Native preparation parity failed")
                path = output / "indices" / (model + ".jsonl")
                path.parent.mkdir(exist_ok=True)
                values = []
                with path.open("w") as stream:
                    import json
                    for row in rows:
                        ix = p.index(row)
                        values.append(ix)
                        stream.write(json.dumps(ix, ensure_ascii=False, allow_nan=False) + "\n")
                index = {r["id"]: r for r in values}
                manifest["indices"][model] = {"contract": model_contract(model, 3), "assets": p.assets,
                    "versions": p.versions, "parity": parity, "rejections": dict(Counter(
                        x["reason"] for x in values if not x["supported"]))}
                buckets = manifest["model_buckets"][model] = {}
                for name in required_buckets(model):
                    bounds = BUCKETS[name]
                    ids, eligible, histogram = select_bucket(rows, index, bounds, f"{model}:{name}")
                    write(output / "buckets" / model / (name + ".json"), ids)
                    lengths = [index[key]["total_tokens"] for key in ids]
                    selected_ids = set(ids)
                    selected = {r["id"]: r for r in rows if r["id"] in selected_ids}
                    buckets[name] = {"min": bounds["min"], "max": bounds["max"], "selected": len(ids),
                        "eligible_unique_inputs": eligible, "ids_file": f"buckets/{model}/{name}.json",
                        "status": "ready" if len(ids) == 400 else "not_ready", "histogram_8": histogram,
                        "lengths": {"min": min(lengths), "median": quantile(lengths, .5), "p95": quantile(lengths, .95),
                                    "max": max(lengths)} if lengths else None,
                        "languages": dict(Counter(selected[key].get("language", "unknown") for key in ids)),
                        "formats": dict(Counter(index[key]["source_mode"] for key in ids))}
                write(output / "manifest.partial.json", manifest)
            except Exception as error:
                # Pool is public; only a bounded exception class is published by default.
                manifest["blockers"][model] = {"error_type": type(error).__name__, "status": "not_ready"}
                private = output / ".private"
                private.mkdir(exist_ok=True)
                (private / (model + ".error.txt")).write_text(str(error))
        complete = len(manifest["model_buckets"]) == len(models)
        manifest["status"] = "ready" if complete and not manifest["blockers"] and all(
            b["status"] == "ready" for bs in manifest["model_buckets"].values() for b in bs.values()) else "not_ready"
    (output / "manifest.partial.json").unlink(missing_ok=True)
    report = ["# Ролевой корпус скорости", "", "CPU-подготовка; GPU и качество не проверены.",
              "Недостижимые интервалы не заполняются искусственно. Межмодельные нагрузки различаются.", "",
              "Модель | Бакет | N | min/median/p95/max | Восемь интервалов", "--- | --- | --- | --- | ---"]
    for model, buckets in manifest["model_buckets"].items():
        for bucket, b in buckets.items():
            report.append(f"{model} | {bucket} | {b['selected']} | {b['lengths']} | {b['histogram_8']}")
    report += ["", "Блокеры: " + str(manifest["blockers"])]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    manifest["files"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                         if p.is_file() and ".private" not in p.parts and p.name != "manifest.json"}
    write(output / "manifest.json", manifest)
    return {"status": manifest["status"], "path": str(output.resolve()), "manifest_sha256": sha(output / "manifest.json")}


def load_speed(root, model, mode, manifest, rows, *, expected_contract=None):
    expected = model_contract(model, 3) if expected_contract is None else expected_contract
    if manifest["kind"] != "speed" or manifest["indices"][model]["contract"] != expected:
        raise ValueError("Stale role preparation/checkpoint contract")
    indices = lines(Path(root) / f"indices/{model}.jsonl")
    index = {r["id"]: r for r in indices}
    by_id = {r["id"]: r for r in rows}
    if len(index) != len(indices) or set(index) != set(by_id):
        raise ValueError("Incomplete/duplicate role index")
    selected, seen_ids, seen_inputs = {}, set(), set()
    for name in required_buckets(model):
        bounds = manifest["model_buckets"][model][name]
        if (bounds["min"], bounds["max"]) != (BUCKETS[name]["min"], BUCKETS[name]["max"]):
            raise ValueError("Changed role bucket bounds")
        if bounds["ids_file"] != f"buckets/{model}/{name}.json":
            raise ValueError("Invalid model bucket path")
        ids = read(Path(root) / bounds["ids_file"])
        if not ids or len(ids) != len(set(ids)) or set(ids) & seen_ids or len(ids) > 400 or (mode == "screening" and len(ids) != 400):
            raise ValueError("Incomplete/duplicate role bucket")
        selected[name] = []
        for key in ids:
            ix = index[key]
            if (not ix["supported"] or not bounds["min"] <= ix["total_tokens"] <= bounds["max"]
                    or ix["input_sha256"] in seen_inputs or ix["source_sha256"] != by_id[key]["content_sha256"]):
                raise ValueError("Invalid/duplicate effective input")
            seen_inputs.add(ix["input_sha256"])
            selected[name].append({**by_id[key], "bucket": name, "index": ix})
        seen_ids.update(ids)
    return selected, manifest
