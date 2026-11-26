"""Verify and rebuild baseline smoke archives offline, without loading GPU libraries."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.core import ROOT, lines, read, sha, write
from u1_benchmark.report_v2 import build


def extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:
            raise ValueError("ZIP checksum failed")
        for item in source.infolist():
            path = (destination / item.filename).resolve()
            if not path.is_relative_to(destination.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive path/symlink")
        source.extractall(destination)


def verify(archive, output):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    extract(archive, output / "outer")
    source_manifest = read(output / "outer/bundle_manifest.json")
    models = {}
    for row in read(output / "outer/smoke-matrix.json"):
        model = row["model"]
        if model not in ("open", "shield"):
            raise ValueError("This acceptance expects only OpenGuardrails and ShieldGemma")
        root = output / model
        extract(output / "outer" / (model + ".zip"), root)
        before = read(root / "summary.json")
        rebuilt = build(root)
        if before != rebuilt:
            raise ValueError("Offline summary differs: " + model)
        if rebuilt["status"] != "smoke_passed" or row["status"] != rebuilt["status"]:
            raise ValueError("Smoke did not pass: " + model)
        for name, checksum in read(root / "source_files.json").items():
            relative = Path(name).relative_to("sources")
            if sha(ROOT / relative) != checksum or source_manifest[str(relative)] != checksum:
                raise ValueError("Current/source bundle mismatch: " + str(relative))
        cfg, worker = read(root / "config.json"), read(root / "preflight/baseline/result.json")
        if cfg["mode"] != "smoke" or cfg.get("candidate") or worker["status"] != "passed":
            raise ValueError("Expected a passed baseline-only smoke")
        scores = lines(root / "preflight/baseline/scores.jsonl")
        if len(scores) != 8 or len(worker["timings"]) != 8:
            raise ValueError("Expected two inputs for each of four buckets")
        models[model] = {"status": rebuilt["status"], "contract": worker["contract"],
            "metadata": worker["metadata"], "timed_inputs": len(scores),
            "boundary": worker["boundary"], "worker_seconds": worker["seconds"],
            "gpu": read(root / "preflight/baseline/runtime.json")["gpu"],
            "offline_rebuild_matches": True, "source_hashes_match": True}
    if set(models) != {"open", "shield"}:
        raise ValueError("Both models are required")
    result = {"status": "passed", "archive_sha256": sha(archive), "models": models,
              "quality_evaluated": False, "screening_run": False, "rtx_checked": False,
              "published": False}
    write(output / "validation.json", result)
    print(json.dumps({"status": "passed", "models": {k: v["status"] for k, v in models.items()}}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.archive, args.output)
