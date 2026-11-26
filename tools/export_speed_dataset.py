"""Build an allowlisted public dataset package; never uploads anything."""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speed_dataset.core import (CONTRACTS, QUOTAS, digest, file_hash, read_jsonl,
                                validate_index, validate_pool, write_json, write_jsonl)


def export(directory, destination, repo_id):
    directory, destination = Path(directory), Path(destination)
    if destination.exists():
        raise ValueError("Destination already exists; use a fresh export directory")
    latest = json.loads((directory / "latest.json").read_text())
    release = (directory / latest["release"]).resolve()
    if not release.is_relative_to(directory.resolve()):
        raise ValueError("Release must be inside dataset directory")
    source = json.loads((release / "manifest.json").read_text())
    if latest["status"] != "ready" or source["status"] != "ready":
        raise ValueError("Only a complete release can be published")
    source_hash = file_hash(release / "manifest.json")
    if source_hash != latest["manifest_sha256"] or source["contracts"] != CONTRACTS:
        raise ValueError("Release checksum or current contracts differ")
    for name, expected in source["files"].items():
        path = (release / name).resolve()
        if not path.is_relative_to(release) or file_hash(path) != expected:
            raise ValueError("Release file checksum mismatch")
    pool, pool_meta = validate_pool(directory)
    if source["pool_sha256"] != pool_meta["pool_sha256"]:
        raise ValueError("Release refers to another pool")
    corpus = read_jsonl(release / "corpus.jsonl")
    by_id = {r["id"]: r for r in corpus}
    if len(by_id) != len(corpus) or len(corpus) != 1600:
        raise ValueError("Expected 1600 distinct selected texts")
    pool_by_id = {r["id"]: r for r in pool}
    if any(pool_by_id.get(r["id"]) != r for r in corpus):
        raise ValueError("Selected text or provenance differs from the verified pool")
    indices, metas = {}, {}
    for model in CONTRACTS["models"]:
        indices[model], metas[model] = validate_index(directory / "indices", model, pool, pool_meta)
        if metas[model] != source["indices"][model]:
            raise ValueError("Index metadata differs from selected release")
    membership, buckets = {}, {}
    for name, spec in CONTRACTS["buckets"].items():
        ids = json.loads((release / f"{name}.ids.json").read_text())
        if len(set(ids)) != 400 or any(i not in by_id or i in membership for i in ids):
            raise ValueError("Invalid bucket IDs")
        if Counter((by_id[i]["language"], by_id[i]["source"]) for i in ids) != Counter(QUOTAS):
            raise ValueError("Bucket composition differs")
        for model in spec["models"]:
            if not all(indices[model][i]["supported"] and
                       spec["min"] <= indices[model][i]["total_tokens"] <= spec["max"] for i in ids):
                raise ValueError("Bucket does not fit every required model")
        membership.update({i: name for i in ids})
        buckets[name] = {**spec, "count": len(ids), "ids_file": f"buckets/{name}.ids.json",
                         "lengths": source["buckets"][name]["lengths"],
                         "composition": source["buckets"][name]["composition"]}
    if set(membership) != set(by_id):
        raise ValueError("Unassigned corpus rows")

    # Export only selected data and explicit public metadata, never the working folder.
    destination.mkdir(parents=True)
    shutil.copyfile(release / "corpus.jsonl", destination / "corpus.jsonl")
    for name in buckets:
        write_json(destination / buckets[name]["ids_file"],
                   json.loads((release / f"{name}.ids.json").read_text()))
    published_indices = {}
    for model, mapping in indices.items():
        applicable = [b for b, s in CONTRACTS["buckets"].items() if model in s["models"]]
        rows = [mapping[r["id"]] for r in corpus if membership[r["id"]] in applicable]
        path = destination / "indices" / f"{model}.jsonl"
        write_jsonl(path, rows)
        meta = metas[model]
        public = {"model": model, "buckets": applicable, "count": len(rows),
                  "index_file": f"indices/{model}.jsonl", "index_sha256": file_hash(path),
                  "source_full_index_sha256": meta["index_sha256"],
                  "source_index_meta_sha256": digest(meta), "contract": meta["contract"],
                  "versions": meta["versions"], "assets": meta["assets"],
                  "preparation_sha256": meta["preparation_sha256"],
                  "input_contract_sha256": digest(meta["input_contract"]),
                  "preparation_parity_status": meta["parity"]["status"]}
        write_json(destination / "indices" / f"{model}.meta.json", public)
        published_indices[model] = public
    write_json(destination / "contracts.json", CONTRACTS)
    shutil.copyfile(release / "REPORT.md", destination / "REPORT.md")
    for name in ("README.md", "LICENSE.md"):
        template = ROOT / "docs" / "hf_dataset" / name
        (destination / name).write_text(template.read_text().replace("__REPO_ID__", repo_id))
    manifest = {"schema_version": 1, "dataset_version": source["version"], "status": "ready",
                "source_release_fingerprint": source["fingerprint"],
                "source_release_manifest_sha256": source_hash,
                "source_pool_sha256": source["pool_sha256"], "source_pool_count": len(pool),
                "selected_count": len(corpus), "seed": source["seed"], "buckets": buckets,
                "indices": published_indices, "gpu_verified": False, "quality_evaluated": False,
                "files": {str(p.relative_to(destination)): file_hash(p)
                          for p in sorted(destination.rglob("*")) if p.is_file()}}
    write_json(destination / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("destination")
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    result = export(args.directory, args.destination, args.repo_id)
    print(json.dumps({"status": result["status"], "selected_count": result["selected_count"],
                      "files": len(result["files"]) + 1, "destination": args.destination}))
