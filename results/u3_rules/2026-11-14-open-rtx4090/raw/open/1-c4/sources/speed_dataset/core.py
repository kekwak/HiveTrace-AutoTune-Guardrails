"""Deterministic files and corpus/index validation (standard library only)."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
CONTRACTS = json.loads((ROOT / "contracts.json").read_text())
SEED = 42
QUOTAS = {(lang, source): n for lang in ("en", "ru")
          for source, n in (("wikipedia", 150), ("synthetic", 50))}


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    data = value if isinstance(value, bytes) else serialized(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(serialized(row) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bucket(n):
    for name, spec in CONTRACTS["buckets"].items():
        if spec["min"] <= n <= spec["max"]:
            return name
    return None


def implementation_hash():
    return digest({p.name: file_hash(p) for p in sorted(ROOT.glob("*.py"))})


def preparation_hash():
    return file_hash(ROOT / "preparation.py")


def validate_pool(directory):
    directory = Path(directory)
    meta = json.loads((directory / "pool_manifest.json").read_text())
    if file_hash(directory / "pool.jsonl") != meta["pool_sha256"]:
        raise ValueError("Pool checksum mismatch")
    rows = read_jsonl(directory / "pool.jsonl")
    seen = set()
    for row in rows:
        h = hashlib.sha256(row["text"].encode()).hexdigest()
        if row["id"] != h or row["text_sha256"] != h or h in seen or not row["text"].strip():
            raise ValueError("Duplicate, empty or altered corpus text")
        seen.add(h)
    if len(rows) != meta["count"]:
        raise ValueError("Pool count mismatch")
    return rows, meta


def validate_index(directory, name, pool, manifest):
    directory = Path(directory)
    meta = json.loads((directory / f"{name}.meta.json").read_text())
    expected = {"model": name, "pool_sha256": manifest["pool_sha256"],
                "contract_sha256": digest(CONTRACTS["models"][name]),
                "preparation_sha256": preparation_hash(), "status": "complete"}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{name}: incomplete/stale index or mismatched contract")
    versions = {**CONTRACTS["dependencies"], **CONTRACTS["models"][name].get("dependencies", {})}
    if any(meta.get("versions", {}).get(k) != v for k, v in versions.items()):
        raise ValueError(f"{name}: dependency mismatch")
    if meta.get("parity", {}).get("status") != "passed" or not meta.get("assets"):
        raise ValueError(f"{name}: missing preparation evidence")
    path = directory / f"{name}.jsonl"
    if file_hash(path) != meta["index_sha256"]:
        raise ValueError(f"{name}: index checksum mismatch")
    rows = read_jsonl(path)
    mapping = {row["id"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != {r["id"] for r in pool}:
        raise ValueError(f"{name}: missing, duplicate or extra IDs")
    for row in rows:
        n = row["total_tokens"]
        if type(n) is not int or n < 0 or type(row["user_tokens"]) is not int or row["user_tokens"] < 0:
            raise ValueError(f"{name}: invalid token count")
        if row["bucket"] != bucket(n) or row["supported"] != (0 < n <= CONTRACTS["models"][name]["limit"] and not row.get("input_changed", False)):
            raise ValueError(f"{name}: inconsistent bucket/support")
        if any(len(row.get(key, "")) != 64 for key in ("input_sha256", "ids_sha256", "mask_sha256")):
            raise ValueError(f"{name}: missing input hashes")
    return mapping, meta
