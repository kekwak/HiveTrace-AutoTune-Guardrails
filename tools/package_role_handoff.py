"""Prepare a local console-only handoff ZIP; no notebook, weights, tokens or publication."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.core import ROOT, read, sha
from u1_benchmark.contracts import MODEL_SPECS
from u1_benchmark.datasets import load_speed, verified


def build(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new bundle name")
    _, manifest = verified(dataset)
    if manifest.get("schema_version") != 3 or manifest.get("status") != "ready":
        raise ValueError("A complete native role corpus is required")
    for model in MODEL_SPECS:
        load_speed(dataset, model, "screening", sha(dataset / "manifest.json"))
    files = []
    for package in ("u1_benchmark", "experiments", "followup", "speed_dataset"):
        files += list((ROOT / package).glob("*.py"))
    files += [ROOT / n for n in ("tools/benchmark.py", "tools/build_role_corpus.py", "tools/package_role_handoff.py",
        "tools/build_rtx_lock.py", "speed_dataset/contracts.json", "examples/u1_external_adapter.py",
        "env/team_environment.lock.json", "env/team_environment.lock.txt", "env/team_rtx.lock.json", "env/team_rtx.lock.txt",
        "docs/u1_handoff.md", "docs/u1_handoff_readiness.md")]
    files += list((ROOT / "configs/benchmark-v3").glob("*.json"))
    files += [ROOT / "configs/data/role_speed_v4.json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        def add(name, raw):
            checksums[name] = hashlib.sha256(raw).hexdigest()
            archive.writestr(name, raw)
        for path in sorted(files):
            raw = path.read_bytes()
            if path.suffix == ".py":
                ast.parse(raw, filename=str(path))
            add(str(path.relative_to(ROOT)), raw)
        for name in ("manifest.json", *manifest["files"]):
            path = (dataset / name).resolve()
            if not path.is_relative_to(dataset) or ".private" in path.parts:
                raise ValueError("Unexpected dataset path")
            add("data/speed/role-corpus/" + name, path.read_bytes())
        add("START.md", ("# У1: консольный комплект\n\nРаспакуйте архив и запускайте из его корня. "
            "Ничего не запускается автоматически. Linux x86_64 / Python 3.13 / RTX 4090 или 5090, только GPU 0.\n\n"
            "```bash\npython tools/benchmark.py setup --profile team-rtx --environment ./u1-rtx-env\n"
            "./u1-rtx-env/venv/bin/python tools/benchmark.py auth-check --model shield\n"
            "python tools/benchmark.py preflight --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/preflight-1\n"
            "python tools/benchmark.py run --pilot --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/pilot-1\n```\n\n"
            "Полные screening запускают владельцы после проверки пилота. Baseline-конфигурации есть для восьми моделей. "
            "Инструкция: docs/u1_handoff.md. RTX GPU и качество не проверены; корпус подготовлен на CPU.\n").encode())
        archive.writestr("bundle_manifest.json", json.dumps(checksums, sort_keys=True, indent=2))
    return {"status": "prepared", "path": str(output), "sha256": sha(output), "files": len(checksums),
            "gpu_checked": False, "published": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.output), ensure_ascii=False))
