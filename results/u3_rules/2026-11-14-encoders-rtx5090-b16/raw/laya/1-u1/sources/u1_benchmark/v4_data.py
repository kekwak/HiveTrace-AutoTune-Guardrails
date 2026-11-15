"""Explicit CPU-only corpus preparation and verified, preparation-keyed v4 data.

Import specs use schema_version=4, source, fields, kind, version, provenance,
and (for speed) preparations=[{config: path-or-dict, variant: baseline|candidate,
name: string}]. Each reference supplies its own config's bucket bounds. Direct
references {spec: full_variant, buckets: {...}, name: string} are also supported.
Reindex references may supply old_source={path, manifest_sha256, model}; Open
requires this to point to a verified task-template overlay, not the native index.
The prepare-data CLI can call prepare with {operation: "reindex", source: path
or {path, manifest_sha256}, configs: [references]}; schema_version=4 is optional
for this dispatch envelope. Ordinary import specs still require schema_version.

The corpus is private input even when include_texts is false: that flag permits
downstream reporting, not redaction of the data needed for execution. public_rows
is the text-free reporting surface. Nothing here publishes or loads weights.
"""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from .core import digest, sha, write
from .messages import content_hash
from .role_data import import_rows, select_bucket, stratum


class DataError(ValueError):
    """Invalid or unverified dataset; messages never contain source text."""


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise DataError("Nonfinite JSON number")


def _json(value):
    try:
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, TypeError) as error:
        raise DataError("Invalid JSON") from error


def _read(path):
    return _json(Path(path).read_text(encoding="utf-8"))


def _lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [_json(line) for line in stream if line.strip()]


def _jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _safe(root, name):
    if (not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") or part.startswith(".") for part in name.split("/"))):
        raise DataError("Unsafe dataset path")
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise DataError("Dataset symlinks are forbidden")
    if not path.resolve().is_relative_to(root):
        raise DataError("Dataset path escapes root")
    return path


def _listed(root, manifest, name):
    if name not in manifest["files"]:
        raise DataError("Referenced file is not checksummed")
    return _safe(root, name)


def _checked_files(path, expected=None):
    root = Path(path).resolve()
    manifest_path = _safe(root, "manifest.json")
    if expected is not None and (not _hash(expected) or sha(manifest_path) != expected):
        raise DataError("Wrong dataset manifest hash")
    manifest = _read(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise DataError("Missing dataset files")
    for name, checksum in manifest["files"].items():
        file = _safe(root, name)
        if name == "manifest.json" or not _hash(checksum) or not file.is_file() or sha(file) != checksum:
            raise DataError("Dataset file checksum mismatch")
        # Validate every listed JSON document, including unused preparation indices.
        if file.suffix == ".json":
            _read(file)
        elif file.suffix == ".jsonl":
            _lines(file)
    rows = _lines(_listed(root, manifest, "corpus.jsonl"))
    _validate_rows(rows, manifest.get("kind"))
    if manifest.get("count") != len(rows):
        raise DataError("Corpus count mismatch")
    return root, rows, manifest


def _validate_rows(rows, kind):
    if kind not in ("speed", "quality") or not rows:
        raise DataError("Invalid corpus kind or empty corpus")
    ids, hashes, slice_keys = set(), set(), None
    for row in rows:
        if not isinstance(row, dict):
            raise DataError("Invalid corpus row")
        key = row.get("id")
        if not isinstance(key, str) or not key or key in ids:
            raise DataError("Duplicate or invalid corpus ID")
        checksum = content_hash(row.get("messages"))
        if row.get("content_sha256") != checksum or (kind == "speed" and checksum in hashes):
            raise DataError("Invalid or duplicate content identity")
        for name in ("language", "source"):
            if name in row and not isinstance(row[name], str):
                raise DataError("Invalid language/source")
        ids.add(key)
        hashes.add(checksum)
        if kind == "quality":
            role, target = row.get("target_role"), row.get("target")
            positions = [i for i, m in enumerate(row["messages"]) if m["role"] == role]
            if (role not in ("user", "assistant") or not isinstance(target, str) or not target
                    or len(positions) != 1 or type(row.get("target_index")) is not int
                    or row["target_index"] != positions[0]):
                raise DataError("Invalid quality target/role")
            slices = row.get("slices")
            if not isinstance(slices, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                   for k, v in slices.items()):
                raise DataError("Invalid quality slices")
            if slice_keys is not None and slice_keys != set(slices):
                raise DataError("Incomplete quality slices")
            slice_keys = set(slices)


def _coverage(rows):
    return {"targets": sorted({r["target"] for r in rows}),
            "target_roles": dict(Counter(r["target_role"] for r in rows)),
            "slices": sorted(rows[0]["slices"]),
            "slice_coverage": {k: dict(Counter(r["slices"][k] for r in rows))
                               for k in rows[0]["slices"]},
            "target_role_coverage": {role: dict(Counter(r["target"] for r in rows
                                                        if r["target_role"] == role))
                                     for role in sorted({r["target_role"] for r in rows})}}


def public_rows(rows):
    """Return non-text evidence, including no raw IDs, provenance or slice values."""
    return [{"id_sha256": digest(r["id"]), "content_sha256": r["content_sha256"],
             **({k: r["index"][k] for k in ("input_sha256", "ids_sha256", "mask_sha256",
                                            "total_tokens")} if "index" in r else {})} for r in rows]


def _identity(spec):
    # Config owns identity semantics; there is deliberately no legacy fallback.
    from .v4_config import preparation_identity
    identity = preparation_identity(spec)
    if not _hash(identity):
        raise DataError("Preparation identity must be a SHA256 hex digest")
    return identity


def _bounds(buckets):
    if not isinstance(buckets, dict) or not buckets:
        raise DataError("Explicit buckets required")
    if any(not isinstance(b, dict) or set(b) != {"min", "max"}
           or type(b["min"]) is not int or type(b["max"]) is not int for b in buckets.values()):
        raise DataError("Invalid bucket bounds")
    previous = 0
    for name, b in sorted(buckets.items(), key=lambda item: item[1].get("min", 0)):
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None
                or not isinstance(b, dict) or set(b) != {"min", "max"}
                or type(b["min"]) is not int or type(b["max"]) is not int
                or b["min"] <= previous or b["max"] < b["min"]):
            raise DataError("Invalid or overlapping buckets")
        previous = b["max"]
    return buckets


def _references(configs):
    if not isinstance(configs, list) or not configs:
        raise DataError("Explicit preparation references required")
    result, names = {}, set()
    for reference in configs:
        if not isinstance(reference, dict) or set(reference) - {"config", "variant", "name", "spec", "buckets", "old_source"}:
            raise DataError("Invalid preparation reference")
        name = reference.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise DataError("Unique preparation names required")
        names.add(name)
        if "config" in reference:
            if "spec" in reference or "buckets" in reference:
                raise DataError("Ambiguous preparation reference")
            from .v4_config import validate_config
            config = reference["config"]
            config = validate_config(_read(config) if isinstance(config, (str, Path)) else config)
            variant = reference.get("variant", "baseline")
            if variant not in ("baseline", "candidate"):
                raise DataError("Invalid variant reference")
            spec, buckets = config[variant], config["buckets"]
        else:
            spec, buckets = reference["spec"], reference["buckets"]
        key = _identity(spec)
        entry = {"spec": spec, "buckets": _bounds(buckets), "names": [name],
                 "old_source": reference.get("old_source")}
        if key in result:
            if (result[key]["buckets"] != buckets or result[key]["old_source"] != entry["old_source"]):
                raise DataError("Conflicting bounds or old source for one preparation identity")
            result[key]["names"].append(name)
        else:
            result[key] = entry
    return result


_INDEX_FIELDS = {"id", "source_sha256", "supported", "reason", "total_tokens", "input_sha256",
                 "ids_sha256", "mask_sha256", "source_mode", "target_index", "target_role",
                 "context_indices", "user_tokens", "content_tokens", "input_changed", "bucket"}


def _index(values, rows):
    by_id = {row["id"]: row for row in rows}
    result = {}
    for ix in values:
        if not isinstance(ix, dict) or set(ix) - _INDEX_FIELDS:
            raise DataError("Unexpected index fields (possible private payload)")
        key = ix.get("id")
        if not isinstance(key, str) or key in result or key not in by_id:
            raise DataError("Duplicate or unknown index ID")
        if ix.get("source_sha256") != by_id[key]["content_sha256"] or type(ix.get("supported")) is not bool:
            raise DataError("Index source identity mismatch")
        if ix["supported"]:
            source = by_id[key]["messages"]
            target, context = ix.get("target_index"), ix.get("context_indices")
            source_mode = {("user",): "user-only", ("assistant",): "assistant-only",
                           ("user", "assistant"): "user+assistant"}[tuple(m["role"] for m in source)]
            if (type(ix.get("total_tokens")) is not int or ix["total_tokens"] <= 0
                    or not all(_hash(ix.get(k)) for k in ("input_sha256", "ids_sha256", "mask_sha256"))
                    or ix.get("source_mode") != source_mode
                    or ix.get("input_changed", False)):
                raise DataError("Invalid native input evidence")
            if (type(target) is not int or not 0 <= target < len(source)
                    or ix.get("target_role") != source[target]["role"]
                    or not isinstance(context, list) or any(type(i) is not int or not 0 <= i < len(source)
                                                           or i == target for i in context)
                    or len(context) != len(set(context))):
                raise DataError("Invalid native role evidence")
        elif not isinstance(ix.get("reason"), str) or not ix["reason"]:
            raise DataError("Unsupported input needs explicit reason")
        result[key] = ix
    if set(result) != set(by_id):
        raise DataError("Incomplete pool index")
    return result


def _membership(ids, index, bounds, seen_ids, seen_inputs):
    if (not isinstance(ids, list) or any(not isinstance(k, str) for k in ids)
            or len(ids) != len(set(ids)) or len(ids) > 400 or set(ids) & seen_ids):
        raise DataError("Invalid or duplicate bucket membership")
    for key in ids:
        ix = index.get(key)
        if (ix is None or not ix["supported"] or not bounds["min"] <= ix["total_tokens"] <= bounds["max"]
                or ix["input_sha256"] in seen_inputs):
            raise DataError("Invalid or duplicate effective input")
        seen_inputs.add(ix["input_sha256"])
    seen_ids.update(ids)


def verified(path, sha=None):
    """Return (rows, manifest); every used file must belong to the checksum ledger."""
    root, rows, manifest = _checked_files(path, sha)
    if manifest.get("format_version") != 4:
        raise DataError("Expected data format_version=4")
    if manifest.get("kind") == "quality":
        if manifest.get("indices") or manifest.get("model_buckets"):
            raise DataError("Quality data must not have index quotas")
        if any(manifest.get(k) != v for k, v in _coverage(rows).items()):
            raise DataError("Quality coverage mismatch")
    else:
        indices, buckets = manifest.get("indices"), manifest.get("model_buckets")
        if not isinstance(indices, dict) or not isinstance(buckets, dict) or set(indices) != set(buckets):
            raise DataError("Incomplete preparation manifest")
        for key, meta in indices.items():
            if not _hash(key) or meta.get("preparation_identity") != key or meta.get("parity", {}).get("status") != "passed":
                raise DataError("Invalid preparation identity or parity")
            index = _index(_lines(_listed(root, manifest, meta["index_file"])), rows)
            _bounds({k: {f: b[f] for f in ("min", "max")} for k, b in buckets[key].items()})
            seen_ids, seen_inputs = set(), set()
            for b in buckets[key].values():
                ids = _read(_listed(root, manifest, b["ids_file"]))
                _membership(ids, index, b, seen_ids, seen_inputs)
                if b.get("selected") != len(ids) or b.get("histogram_8") != [sum(
                        stratum(index[i]["total_tokens"], b) == n for i in ids) for n in range(8)]:
                    raise DataError("Bucket evidence mismatch")
    return rows, manifest


def load_speed(path, spec, buckets, mode, pilot=False, manifest_sha256=None):
    """Load configured indexed buckets; require N=400 except for smoke/pilot."""
    if mode not in ("baseline", "search", "confirm", "smoke", "quality") or type(pilot) is not bool:
        raise DataError("Invalid mode/pilot")
    if mode == "quality":
        raise DataError("Quality mode requires quality data")
    rows, manifest = verified(path, manifest_sha256)
    key = _identity(spec)
    if manifest["kind"] != "speed" or key not in manifest["indices"] or key in manifest.get("blockers", {}):
        raise DataError("Missing or blocked preparation identity")
    expected = _bounds(buckets)
    actual = manifest["model_buckets"][key]
    if not expected.keys() <= actual.keys():
        raise DataError("Unknown configured bucket")
    if any(bounds != {k: actual[name][k] for k in ("min", "max")} for name, bounds in expected.items()):
        raise DataError("Changed bucket bounds")
    root = Path(path).resolve()
    index = _index(_lines(_listed(root, manifest, manifest["indices"][key]["index_file"])), rows)
    by_id, result = {r["id"]: r for r in rows}, {}
    for name in expected:
        b = actual[name]
        ids = _read(_listed(root, manifest, b["ids_file"]))
        if not ids or (not (pilot or mode == "smoke") and len(ids) != 400):
            raise DataError("Full run requires N=400 per bucket; empty buckets cannot run")
        result[name] = [{**by_id[i], "bucket": name, "index": index[i]} for i in ids]
    return result, manifest


def _factory(spec):
    from .v4_adapters import create_preparer
    return create_preparer(spec)


def _native(entry, rows, factory):
    preparer = factory(entry["spec"])
    try:
        if not callable(getattr(preparer, "parity", None)):
            raise DataError("CPU preparer must expose parity()")
        parity = preparer.parity()
        if not isinstance(parity, dict) or parity.get("status") != "passed":
            raise DataError("Native preparation parity failed")
        values = [dict(preparer.index(row)) for row in rows]
        index = _index(values, rows)
        for value in values:
            if value["supported"] and value["total_tokens"] > entry["spec"]["preparation"]["limit"]:
                raise DataError("Native supported input exceeds preparation limit")
            # Unsupported reasons from external adapters may contain source text.
            reason = value.get("reason")
            value["reason"] = (None if value["supported"] else reason if reason in
                               ("processor_changed_text", "outside_input_limit") else "unsupported_input")
    finally:
        close = getattr(preparer, "close", None)
        if callable(close):
            close()
    # Do not copy arbitrary adapter metadata or exception messages into public files.
    return values, index, {"status": "passed", "scope": "CPU native input preparation"}


def _blocker(error, stage):
    public = {"Explicit old_source required; Open must use task overlay",
              "Open old source is not a task-template overlay", "Old source corpus differs",
              "Old bucket bounds differ", "Native preparation parity failed",
              "Old native preparation parity is not passed", "Wrong dataset manifest hash",
              "CPU preparer must expose parity()"}
    return {"status": "not_ready", "stage": stage, "error_type": type(error).__name__,
            "reason": str(error) if type(error) is DataError and str(error) in public else stage + "_failed"}


def _bucket_record(output, key, name, ids, eligible, index, rows, bounds):
    file = f"buckets/{key}/{name}.json"
    write(output / file, ids)
    by_id = {r["id"]: r for r in rows}
    lengths = sorted(index[i]["total_tokens"] for i in ids)
    return {**bounds, "ids_file": file, "selected": len(ids), "eligible_unique_inputs": eligible,
            "status": "ready" if len(ids) == 400 else "not_ready",
            "histogram_8": [sum(stratum(n, bounds) == i for n in lengths) for i in range(8)],
            "languages": dict(Counter(by_id[i].get("language", "unknown") for i in ids)),
            "sources": dict(Counter(by_id[i].get("source", "unknown") for i in ids)),
            "formats": dict(Counter(index[i]["source_mode"] for i in ids))}


def _finish(output, manifest):
    ready = not manifest["blockers"] and (manifest["kind"] == "quality" or bool(manifest["indices"]) and all(
        b["selected"] == 400 for bs in manifest["model_buckets"].values() for b in bs.values()))
    manifest["status"] = "ready" if ready else "not_ready"
    manifest["files"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write(output / "manifest.json", manifest)
    return {"status": manifest["status"], "path": str(output.resolve()),
            "manifest_sha256": sha(output / "manifest.json"), "blockers": manifest["blockers"]}


def _manifest(rows, kind, version, provenance, include_texts=False, seed=42):
    return {"format_version": 4, "kind": kind, "version": version, "provenance": provenance,
            "count": len(rows), "seed": seed, "include_texts": include_texts, "quality_evaluated": False,
            "indices": {}, "model_buckets": {}, "blockers": {}}


def prepare(spec, out, preparer_factory=None):
    """Import mapped JSONL or dispatch an explicit operation='reindex' envelope."""
    if isinstance(spec, dict) and "operation" in spec:
        if (spec["operation"] != "reindex"
                or set(spec) - {"operation", "schema_version", "source", "configs"}
                or spec.get("schema_version", 4) != 4
                or not {"source", "configs"} <= spec.keys()):
            raise DataError("Invalid reindex operation spec")
        return reindex(spec["source"], out, spec["configs"], preparer_factory)
    allowed = {"schema_version", "source", "fields", "kind", "version", "provenance", "include_texts",
               "preparations", "configs", "seed"}
    if (not isinstance(spec, dict) or set(spec) - allowed or spec.get("schema_version") != 4
            or spec.get("kind") not in ("speed", "quality") or not spec.get("version")
            or "provenance" not in spec or type(spec.get("seed", 42)) is not int
            or type(spec.get("include_texts", False)) is not bool):
        raise DataError("Invalid v4 import spec")
    if "preparations" in spec and "configs" in spec:
        raise DataError("Ambiguous preparation references")
    # Strictly parse before calling the reusable, field-mapping-only legacy helper.
    _lines(spec["source"])
    rows = import_rows(spec["source"], spec["fields"], spec["kind"])
    _validate_rows(rows, spec["kind"])
    if spec["kind"] == "speed" and len(rows) > 48000:
        raise DataError("Speed pool exceeds 48000 records")
    refs = spec.get("preparations", spec.get("configs"))
    if spec["kind"] == "quality" and refs:
        raise DataError("Quality data has no preparation quotas")
    entries = _references(refs) if spec["kind"] == "speed" else {}
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    _jsonl(output / "corpus.jsonl", rows)
    manifest = _manifest(rows, spec["kind"], spec["version"], spec["provenance"],
                         spec.get("include_texts", False), spec.get("seed", 42))
    manifest["source_sha256"] = sha(spec["source"])
    if spec["kind"] == "quality":
        manifest.update(_coverage(rows))
    for key, entry in entries.items():
        try:
            values, index, parity = _native(entry, rows, preparer_factory or _factory)
            # Reuse eight-stratum allocation, balancing language, source and role format.
            balanced = {i: {**ix, "source_mode": digest([row.get("source", "unknown"), ix.get("source_mode")])}
                        for row in rows for i, ix in [(row["id"], index[row["id"]])]}
            selected = {}
            for name, bounds in entry["buckets"].items():
                ids, eligible, _ = select_bucket(rows, balanced, bounds, f"{manifest['seed']}:{key}:{name}")
                selected[name] = _bucket_record(output, key, name, ids, eligible, index, rows, bounds)
            file = f"indices/{key}.jsonl"
            _jsonl(output / file, values)
            manifest["indices"][key] = {"preparation_identity": key, "names": entry["names"],
                                        "index_file": file, "parity": parity}
            manifest["model_buckets"][key] = selected
        except Exception as error:
            manifest["blockers"][key] = _blocker(error, "native_preparation")
    return _finish(output, manifest)


_PARITY_FIELDS = ("source_sha256", "supported", "total_tokens", "input_sha256", "ids_sha256",
                  "mask_sha256", "user_tokens", "content_tokens", "source_mode", "target_index",
                  "target_role", "context_indices", "input_changed")


def _old(entry, source, rows):
    mapping = entry.get("old_source")
    adapter = entry["spec"]["adapter"]
    if mapping is None:
        if not isinstance(adapter, str) or adapter == "open":
            raise DataError("Explicit old_source required; Open must use task overlay")
        mapping = {"path": str(source), "model": adapter}
    if not isinstance(mapping, dict) or set(mapping) - {"path", "manifest_sha256", "model"}:
        raise DataError("Invalid old_source mapping")
    root, old_rows, manifest = _checked_files(mapping["path"], mapping.get("manifest_sha256"))
    if manifest.get("schema_version") != 3 or manifest["kind"] != "speed":
        raise DataError("Reindex requires a v3 role corpus")
    if old_rows != rows:
        raise DataError("Old source corpus differs")
    model = mapping["model"]
    if not isinstance(model, str) or re.fullmatch(r"[A-Za-z0-9_-]+", model) is None:
        raise DataError("Invalid old preparation name")
    meta = manifest["indices"][model]
    if meta.get("parity", {}).get("status") != "passed":
        raise DataError("Old native preparation parity is not passed")
    if adapter == "open" and "task_role_preparation" not in meta.get("assets", {}):
        raise DataError("Open old source is not a task-template overlay")
    values = _lines(_listed(root, manifest, f"indices/{model}.jsonl"))
    index = _index(values, rows)
    buckets = manifest["model_buckets"][model]
    if entry["buckets"] != {name: {k: b[k] for k in ("min", "max")} for name, b in buckets.items()}:
        raise DataError("Old bucket bounds differ")
    ids, seen_ids, seen_inputs = {}, set(), set()
    for name, b in buckets.items():
        ids[name] = _read(_listed(root, manifest, b["ids_file"]))
        _membership(ids[name], index, b, seen_ids, seen_inputs)
        if not ids[name] or len(ids[name]) != b["selected"]:
            raise DataError("Old bucket cardinality differs")
    return index, ids, {"manifest_sha256": sha(root / "manifest.json"), "model": model}


def reindex(source, out, configs, preparer_factory=None):
    """Reindex every old row on CPU; retain corpus bytes and IDs only after parity.

    A mismatch produces a not_ready manifest with an explicit parity blocker,
    never a fresh selection that conceals the mismatch. Existing output fails.
    """
    if isinstance(source, dict) and (set(source) - {"path", "manifest_sha256"} or "path" not in source):
        raise DataError("Reindex source requires path and optional manifest_sha256")
    source_path = source["path"] if isinstance(source, dict) else source
    expected = source.get("manifest_sha256") if isinstance(source, dict) else None
    root, rows, old_manifest = _checked_files(source_path, expected)
    if old_manifest.get("schema_version") != 3 or old_manifest["kind"] != "speed":
        raise DataError("Reindex requires a v3 speed corpus")
    entries = _references(configs)
    output = Path(out).resolve()
    if output.is_relative_to(root):
        raise DataError("Reindex output must be outside the old dataset")
    for entry in entries.values():
        mapping = entry.get("old_source")
        if isinstance(mapping, dict) and "path" in mapping and output.is_relative_to(Path(mapping["path"]).resolve()):
            raise DataError("Output must be outside every old source")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(_listed(root, old_manifest, "corpus.jsonl"), output / "corpus.jsonl")
    manifest = _manifest(rows, "speed", old_manifest["version"], old_manifest.get("provenance"),
                         old_manifest.get("include_texts", False), old_manifest.get("seed", 42))
    manifest["reindex"] = {"source_manifest_sha256": sha(root / "manifest.json"),
                           "source_corpus_sha256": sha(root / "corpus.jsonl"), "comparisons": {}}
    for key, entry in entries.items():
        stage = "old_source"
        try:
            old_index, bucket_ids, provenance = _old(entry, root, rows)
            stage = "native_preparation"
            values, index, parity = _native(entry, rows, preparer_factory or _factory)
            mismatches = [i for i in index if any(index[i].get(k) != old_index[i].get(k) for k in _PARITY_FIELDS)]
            if mismatches:
                manifest["blockers"][key] = {"status": "not_ready", "stage": "reindex_parity",
                                              "mismatch_count": len(mismatches),
                                              "row_id_hashes": [digest(i) for i in mismatches]}
                continue
            stage = "bucket_parity"
            seen_ids, seen_inputs = set(), set()
            for name, ids in bucket_ids.items():
                _membership(ids, index, entry["buckets"][name], seen_ids, seen_inputs)
            file = f"indices/{key}.jsonl"
            _jsonl(output / file, values)
            manifest["indices"][key] = {"preparation_identity": key, "names": entry["names"],
                                        "index_file": file, "parity": parity}
            manifest["model_buckets"][key] = {name: _bucket_record(
                output, key, name, ids, len({ix["input_sha256"] for ix in index.values()
                    if ix["supported"] and entry["buckets"][name]["min"] <= ix["total_tokens"]
                    <= entry["buckets"][name]["max"]}), index, rows, entry["buckets"][name])
                for name, ids in bucket_ids.items()}
            manifest["reindex"]["comparisons"][key] = {**provenance, "status": "passed",
                                                       "compared_rows": len(rows), "bucket_ids_preserved": True}
        except Exception as error:
            manifest["blockers"][key] = _blocker(error, stage)
    return _finish(output, manifest)
