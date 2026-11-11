import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "vXofi/guardrails-speed-en-ru"
REVISION = "49c2c0b069290c29d45c9c42e41cac4bea344649"
MANIFEST_SHA = "6c0f94e66162a6fa6e4e84074431a79bf65dd7861234e4892c23cb515e96c4dd"
PRESETS = {"halo_fla": ("halo", "fla"), "gliner_compile": ("gliner", "compile"),
           "gliner_ort": ("gliner", "ort")}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(s) for s in stream if s.strip()]


def config(value):
    value = read(value) if isinstance(value, (str, Path)) else dict(value)
    if set(value) - {"preset", "environment", "seed", "mode", "timeout_seconds"}:
        raise ValueError("Unknown configuration fields")
    if value.get("preset") not in PRESETS:
        raise ValueError("Unknown preset")
    if not value.get("environment"):
        raise ValueError("An explicit setup environment directory is required")
    result = {"seed": 17, "mode": "screening", "timeout_seconds": 36000, **value}
    if result["seed"] not in (17, 42, 123) or result["mode"] not in ("screening", "pilot"):
        raise ValueError("Invalid seed/mode")
    if not 0 < result["timeout_seconds"] <= 36000:
        raise ValueError("Maximum runtime is 10 hours")
    result["environment"] = str(Path(result["environment"]).resolve())
    return result


def buckets(model):
    return ["B1", "B2", "B3", "B4"] if model == "halo" else ["B1"]


def score_check(score):
    if score.get("label") not in ("safe", "unsafe"):
        raise ValueError("Invalid native verdict")
    for key in ("p_safe", "p_unsafe", "logit_margin"):
        if not isinstance(score.get(key), (int, float)) or not math.isfinite(score[key]):
            raise ValueError("Nonfinite or missing score: " + key)
    if not all(0 <= score[k] <= 1 for k in ("p_safe", "p_unsafe")):
        raise ValueError("Invalid probabilities")
    if abs(score["p_safe"] + score["p_unsafe"] - 1) > 1e-5:
        raise ValueError("Probabilities do not sum to one")


def load_data(root, model):
    from collections import Counter
    root = Path(root)
    if sha(root / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Wrong pinned dataset manifest")
    manifest = read(root / "manifest.json")
    for name, expected in manifest["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p) != expected:
            raise ValueError("Dataset file checksum mismatch: " + name)
    rows = lines(root / "corpus.jsonl")
    by_id = {r["id"]: r for r in rows}
    if len(rows) != len(by_id) or len(rows) != 1600:
        raise ValueError("Invalid corpus cardinality")
    for r in rows:
        if hashlib.sha256(r["text"].encode()).hexdigest() != r["id"] or r["text_sha256"] != r["id"]:
            raise ValueError("Changed text")
    index_rows = lines(root / "indices" / f"{model}.jsonl")
    index = {r["id"]: r for r in index_rows}
    selected, seen = {}, set()
    for b, spec in manifest["buckets"].items():
        ids = read(root / spec["ids_file"])
        if len(set(ids)) != 400 or set(ids) & seen or not set(ids) <= by_id.keys():
            raise ValueError("Invalid bucket membership")
        seen.update(ids)
        counts = Counter((by_id[i]["language"], by_id[i]["source"]) for i in ids)
        if counts != Counter({(l, s): n for l in ("en", "ru")
                              for s, n in (("wikipedia", 150), ("synthetic", 50))}):
            raise ValueError("Invalid bucket quotas")
        if b not in buckets(model):
            continue
        selected[b] = []
        for i in ids:
            ix = index[i]
            if not ix["supported"] or ix["bucket"] != b or not spec["min"] <= ix["total_tokens"] <= spec["max"]:
                raise ValueError("Invalid model index")
            selected[b].append({**by_id[i], "bucket": b, "index": ix})
    if len(index) != len(index_rows) or set(index) != {r["id"] for rs in selected.values() for r in rs}:
        raise ValueError("Compact index contains missing/extra/duplicate IDs")
    return selected, manifest


def schedule(cfg, data):
    jobs = []
    pilot = cfg["mode"] == "pilot"
    for phase, count, n in (("aa", 2, 30), ("ab", 1 if pilot else 3, 12 if pilot else 400)):
        for block in range(count):
            for bucket, rows in data.items():
                ids = [r["id"] for r in rows]
                random.Random(f"{cfg['seed']}:{phase}:{block}:{bucket}").shuffle(ids)
                ids = ids[:n]
                for side in (("a", "b") if block % 2 == 0 else ("b", "a")):
                    jobs.append({"phase": phase, "block": block, "bucket": bucket,
                                 "side": side, "variant": "baseline" if phase == "aa" or side == "a" else "candidate",
                                 "ids": ids})
    return jobs
