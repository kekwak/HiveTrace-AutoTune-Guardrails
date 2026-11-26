"""CPU-only acceptance: repeat hashes and inputs from the current explicit contracts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.core import read, sha, write
from u1_benchmark.contracts import MODEL_SPECS
from u1_benchmark.datasets import load_speed, verified
from u1_benchmark.adapters import check_index
from u1_benchmark.messages import payload
from u1_benchmark.role_preparation import Preparer


def validate(dataset, repeat, output):
    dataset, repeat = Path(dataset), Path(repeat)
    rows, manifest = verified(dataset)
    _, duplicate = verified(repeat)
    if manifest != duplicate or sha(dataset / "manifest.json") != sha(repeat / "manifest.json"):
        raise ValueError("Repeat build is not byte-reproducible")
    if manifest["status"] != "ready" or any("target" in row for row in rows):
        raise ValueError("Incomplete speed corpus or unexpected quality targets")
    result = {"status": "running", "gpu_checked": False, "quality_evaluated": False,
        "manifest_sha256": sha(dataset / "manifest.json"), "repeat_identical": True, "pool_count": len(rows),
        "cpu_environment": {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p.metadata["Name"]: p.version for p in importlib.metadata.distributions() if p.metadata.get("Name")}},
        "models": {}}
    assignments = {}
    for model in MODEL_SPECS:
        data, _ = load_speed(dataset, model, "screening", result["manifest_sha256"])
        p = Preparer(model)
        parity = p.parity()
        if parity["status"] != "passed":
            raise ValueError("CPU preparation parity failed")
        for bucket, values in data.items():
            if len(values) != 400 or len({row["index"]["input_sha256"] for row in values}) != 400:
                raise ValueError("Incomplete or duplicate effective inputs")
            for row in values:
                check_index(p.index(row), row["index"])
                assignments.setdefault(row["id"], set()).add(bucket)
        result["models"][model] = {"status": "cpu_preparation_passed", "parity": parity,
            "bucket_counts": {k: len(v) for k, v in data.items()}, "gpu": "not_checked"}
        write(output, result)
        print(model + ": selected prepared inputs verified", flush=True)
    result.update(status="passed", ids_with_different_model_buckets=sum(len(v) > 1 for v in assignments.values()))
    write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--repeat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate(args.dataset, args.repeat, args.output)
