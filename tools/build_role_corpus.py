"""Build a common role pool and native CPU indices, never model weights or GPU runs."""
import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.core import digest, lines, read, sha, write
from u1_benchmark.datasets import jsonl, prepare
from u1_benchmark.messages import content_hash


def role_row(source, ordinal, original=False):
    text = source["text"]
    mode = 0 if original else ordinal % 3
    # Model-independent word-aligned prefixes spread the candidate lengths.
    fraction = 32 if original else (ordinal // 3) % 32 + 1
    end = len(text) * fraction // 32
    while end < len(text) and not text[end].isspace():
        end += 1
    body = text[:end].strip()
    if not body:
        body = text.strip()
    if mode == 0:
        messages = [{"role": "user", "content": body}]
    elif mode == 1:
        messages = [{"role": "assistant", "content": body}]
    else:
        # An explicitly synthetic reference exchange, not an alleged real LLM answer.
        prompt = "Приведи справочный фрагмент по теме: " if source["language"] == "ru" else "Provide a reference excerpt about: "
        topic = source.get("provenance", {}).get("title", "reference " + str(ordinal))
        messages = [{"role": "user", "content": prompt + topic}, {"role": "assistant", "content": body}]
    checksum = content_hash(messages)
    return {"id": checksum, "messages": messages, "language": source["language"], "source": source["source"],
            "provenance": {"parent_id": source["id"], "parent_text_sha256": source["text_sha256"],
                "original": source["provenance"], "role_assignment": "original text as user" if mode == 0 else "synthetic role assignment; not an observed assistant response",
                "slice_end_char": end, "synthetic_dialogue": mode == 2, "seed": 42},
            "content_sha256": checksum}


def pool(base, source, size):
    if size > 48000 or size < len(base):
        raise ValueError("Pool size outside budget")
    result = [role_row(row, i, original=True) for i, row in enumerate(base)]
    seen = {r["id"] for r in result}
    original_ids = {r["id"] for r in base}
    candidates = [r for r in source if r["id"] not in original_ids]
    random.Random(42).shuffle(candidates)
    for i, row in enumerate(candidates):
        if len(result) >= size:
            break
        candidate = role_row(row, i)
        if candidate["id"] not in seen:
            seen.add(candidate["id"])
            result.append(candidate)
    return result


def build(base_path, pool_path, output, sizes=(12000, 24000, 36000, 48000)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base, source = lines(base_path), lines(pool_path)
    if len(base) != 1600:
        raise ValueError("Expected the historical 1600-record seed corpus")
    parent = read(Path(pool_path).parent / "pool_manifest.json")
    if sha(pool_path) != parent["pool_sha256"]:
        raise ValueError("Historical source pool checksum mismatch")
    from u1_benchmark.contracts import MODEL_SPECS
    attempts = []
    for size in sizes:
        directory = output / str(size)
        directory.mkdir()
        source_file = directory / "pool.jsonl"
        candidates = pool(base, source, size)
        jsonl(source_file, candidates)
        write(directory / "pool_status.json", {"requested": size, "actual": len(candidates),
              "source_exhausted": len(candidates) < size})
        spec = {"schema_version": 3, "kind": "speed", "source": str(source_file),
                "fields": {k: k for k in ("id", "messages", "provenance", "language", "source")},
                "models": list(MODEL_SPECS), "version": "u1-role-speed-v4-" + str(size), "include_texts": True,
                "provenance": {"base_corpus_sha256": sha(base_path), "parent_pool_sha256": sha(pool_path),
                    "source_shards": parent["source_shards"], "builder_sha256": sha(__file__),
                    "note": "No model-specific source texts; no safety ground truth"}}
        write(directory / "import.json", {**spec, "source": str(source_file.resolve())})
        result = prepare(spec, directory / "dataset")
        attempts.append(result)
        manifest = read(Path(result["path"]) / "manifest.json")
        write(output / "build.json", {"status": result["status"], "attempts": attempts,
              "gpu_checked": False, "quality_evaluated": False})
        print(f"Pool {size}: {result['status']}; blockers={list(manifest['blockers'])}", flush=True)
        if result["status"] == "ready" or manifest["blockers"]:
            break
    write(output / "latest.json", attempts[-1])
    return attempts[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.base, args.pool, args.output))
