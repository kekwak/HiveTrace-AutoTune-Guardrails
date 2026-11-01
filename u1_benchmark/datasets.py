"""Explicit JSONL import and shared full-input intersections, without weights."""
import hashlib
import json
import random
from pathlib import Path

from .core import MANIFEST_SHA, digest, lines, read, sha, write
from .contracts import BUCKETS, MODEL_SPECS, model_contract, required_buckets


def jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def import_rows(source, fields, kind):
    required = {"id", "text"} | ({"target"} if kind == "quality" else set())
    if not required <= fields.keys() or set(fields) - {"id", "text", "target", "slices", "provenance"}:
        raise ValueError("Explicit id/text/target field mapping required")
    result, ids, texts = [], set(), set()
    slice_keys = None
    for raw in lines(source):
        row = {name: raw[key] for name, key in fields.items()}
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids or not isinstance(row["text"], str):
            raise ValueError("Duplicate/invalid ID or text")
        row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        if kind == "speed" and row["text_sha256"] in texts:
            raise ValueError("Speed corpus contains duplicate texts")
        if kind == "quality":
            if not isinstance(row["target"], str) or not row["target"]:
                raise ValueError("Quality targets must be nonempty strings")
            slices = row.setdefault("slices", {})
            if not isinstance(slices, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in slices.items()):
                raise ValueError("Slices must be named string values")
            if slice_keys is not None and slice_keys != set(slices):
                raise ValueError("Incomplete slice columns; fill an explicit unknown value")
            slice_keys = set(slices)
        ids.add(row["id"])
        texts.add(row["text_sha256"])
        result.append(row)
    if not result:
        raise ValueError("Empty corpus")
    return result


def prepare(spec, output, preparer_factory=None):
    """Import first, then index the same bytes with every selected tokenizer."""
    if spec.get("schema_version") == 3:
        from .role_data import prepare as role_prepare
        return role_prepare(spec, output, preparer_factory)
    if set(spec) - {"source", "kind", "fields", "models", "version", "provenance", "include_texts"}:
        raise ValueError("Unknown import fields")
    kind = spec["kind"]
    if kind not in ("speed", "quality") or not spec.get("version") or "provenance" not in spec:
        raise ValueError("Dataset kind, version and provenance required")
    rows = import_rows(spec["source"], spec["fields"], kind)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 2, "kind": kind, "version": spec["version"], "seed": 42,
                "include_texts": bool(spec.get("include_texts", False)), "provenance": spec["provenance"],
                "source_sha256": sha(spec["source"]), "files": {}, "indices": {}, "buckets": {}}
    jsonl(output / "corpus.jsonl", rows)
    if kind == "speed":
        models = spec.get("models", [])
        if not models or len(set(models)) != len(models) or not set(models) <= MODEL_SPECS.keys():
            raise ValueError("Explicit unique participating models required")
        if preparer_factory is None:
            from speed_dataset.preparation import Preparer
            preparer_factory = Preparer
        indices = {}
        for model in models:
            p = preparer_factory(model)
            parity = p.parity()
            if parity["status"] != "passed":
                raise ValueError("Input preparation parity failed")
            values = [p.index(row) for row in rows]
            indices[model] = {row["id"]: row for row in values}
            jsonl(output / "indices" / (model + ".jsonl"), values)
            manifest["indices"][model] = {"contract": model_contract(model), "assets": p.assets,
                                           "versions": p.versions, "parity": parity}
        used = set()
        for bucket, bounds in BUCKETS.items():
            participants = [m for m in models if m in bounds["models"]]
            if not participants:
                continue
            eligible = [r["id"] for r in rows if r["id"] not in used and all(
                indices[m][r["id"]]["supported"] and bounds["min"] <= indices[m][r["id"]]["total_tokens"] <= bounds["max"]
                for m in participants)]
            random.Random("42:" + bucket).shuffle(eligible)
            selected = eligible[:400]
            used.update(selected)
            manifest["buckets"][bucket] = {"min": bounds["min"], "max": bounds["max"], "models": participants,
                "ids_file": f"buckets/{bucket}.json", "intersection_size": len(eligible), "selected": len(selected)}
            write(output / f"buckets/{bucket}.json", selected)
        manifest["status"] = "ready" if all(b["selected"] == 400 for b in manifest["buckets"].values()) else "not_ready"
    else:
        manifest.update(status="ready", count=len(rows), targets=sorted({r["target"] for r in rows}),
                        slices=sorted(rows[0]["slices"]))
    manifest["files"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write(output / "manifest.json", manifest)
    return {"status": manifest["status"], "path": str(output.resolve()), "manifest_sha256": sha(output / "manifest.json")}


def verified(root, expected=None):
    root = Path(root).resolve()
    if expected is not None and sha(root / "manifest.json") != expected:
        raise ValueError("Wrong dataset manifest hash")
    manifest = read(root / "manifest.json")
    for name, checksum in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != checksum:
            raise ValueError("Dataset file mismatch")
    rows = lines(root / "corpus.jsonl")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    for row in rows:
        if manifest.get("schema_version") == 3:
            from .messages import content_hash
            if content_hash(row["messages"]) != row["content_sha256"]:
                raise ValueError("Message content hash mismatch")
            continue
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("Text hash mismatch")
    return rows, manifest


def load_speed(root, model, mode, expected=None):
    rows, manifest = verified(root, expected)
    if manifest.get("schema_version") == 3:
        from .role_data import load_speed as role_load
        return role_load(root, model, mode, manifest, rows)
    published = sha(Path(root) / "manifest.json") == MANIFEST_SHA
    if not published and manifest.get("kind") != "speed":
        raise ValueError("Expected speed dataset")
    if not published and manifest["indices"][model]["contract"] != model_contract(model):
        raise ValueError("Stale input preparation/checkpoint contract")
    if published and manifest["indices"][model]["contract"] != MODEL_SPECS[model]:
        raise ValueError("Published model contract mismatch")
    index_rows = lines(Path(root) / f"indices/{model}.jsonl")
    index = {r["id"]: r for r in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("Duplicate input index IDs")
    by_id = {r["id"]: r for r in rows}
    if not published and set(index) != set(by_id):
        raise ValueError("Incomplete pool index")
    selected, seen, hashes = {}, set(), set()
    for bucket in required_buckets(model):
        spec = manifest["buckets"][bucket]
        if model not in spec["models"] or (spec["min"], spec["max"]) != (BUCKETS[bucket]["min"], BUCKETS[bucket]["max"]):
            raise ValueError("Changed bucket contract")
        ids = read(Path(root) / spec["ids_file"])
        if len(set(ids)) != len(ids) or set(ids) & seen or not ids or len(ids) > 400 or (mode == "screening" and len(ids) != 400):
            raise ValueError("Incomplete/duplicate screening bucket")
        seen.update(ids)
        for participant in spec["models"]:
            ix_rows = lines(Path(root) / f"indices/{participant}.jsonl")
            partner = {r["id"]: r for r in ix_rows}
            if len(partner) != len(ix_rows):
                raise ValueError("Duplicate partner index IDs")
            for key in ids:
                ix = partner[key]
                if not ix["supported"] or ix["bucket"] != bucket or not spec["min"] <= ix["total_tokens"] <= spec["max"]:
                    raise ValueError("Shared intersection invalid for another model")
        selected[bucket] = []
        for key in ids:
            row, ix = by_id[key], index[key]
            if row["text_sha256"] in hashes or not ix["supported"] or ix["bucket"] != bucket or not spec["min"] <= ix["total_tokens"] <= spec["max"]:
                raise ValueError("Duplicate text or invalid input index")
            hashes.add(row["text_sha256"])
            selected[bucket].append({**row, "bucket": bucket, "index": ix})
    return selected, manifest
