"""Reindex selected models without rewriting the shared source corpus or other models."""
import argparse
import copy
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.core import read, sha, write
from u1_benchmark.contracts import model_contract
from u1_benchmark.datasets import load_speed, verified
from u1_benchmark.role_data import prepare


def rebuild(source, output, models, version, preparer_factory=None):
    source, output = Path(source), Path(output)
    rows, previous = verified(source)
    if previous["schema_version"] != 3 or previous["kind"] != "speed" or previous["status"] != "ready":
        raise ValueError("A verified ready role corpus is required")
    if not models or len(set(models)) != len(models) or not set(models) <= previous["indices"].keys():
        raise ValueError("Select existing unique model names")
    preserved = set(previous["indices"]) - set(models)
    for model in preserved:
        if previous["indices"][model]["contract"] != model_contract(model, 3):
            raise ValueError("Another model also needs reindexing: " + model)
    fields = {key: key for key in rows[0] if key != "content_sha256"}
    spec = {"schema_version": 3, "kind": "speed", "source": str(source / "corpus.jsonl"),
            "fields": fields, "models": models, "version": version,
            "include_texts": previous["include_texts"], "provenance": {
                "parent_manifest_sha256": sha(source / "manifest.json"),
                "parent_provenance": previous["provenance"], "reindexed_models": models,
                "builder_sha256": sha(__file__), "note": "Same messages; new explicit preparation contract"}}
    prepare(spec, output, preparer_factory)
    if sha(output / "corpus.jsonl") != sha(source / "corpus.jsonl"):
        raise ValueError("Reindexing must preserve exact shared corpus bytes")
    current = read(output / "manifest.json")
    for model in sorted(preserved):
        current["indices"][model] = copy.deepcopy(previous["indices"][model])
        current["model_buckets"][model] = copy.deepcopy(previous["model_buckets"][model])
        path = output / "indices" / (model + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(source / path.relative_to(output), path)
        for bucket in current["model_buckets"][model].values():
            path = output / bucket["ids_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / bucket["ids_file"], path)
    current["status"] = "ready" if not current["blockers"] and all(
        b["status"] == "ready" for bs in current["model_buckets"].values() for b in bs.values()) else "not_ready"
    report = ["# Ролевой корпус: обновление подготовки", "", "Те же исходные сообщения; качество не оценивалось.",
              "Повторно индексированы: " + ", ".join(models) + ". GPU-совместимость проверяется отдельно.", "",
              "Модель | Бакет | N | min/median/p95/max | Восемь интервалов", "--- | --- | --- | --- | ---"]
    for model, buckets in sorted(current["model_buckets"].items()):
        for name, bucket in buckets.items():
            report.append(f"{model} | {name} | {bucket['selected']} | {bucket['lengths']} | {bucket['histogram_8']}")
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    current["files"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                        if p.is_file() and ".private" not in p.parts and p.name != "manifest.json"}
    write(output / "manifest.json", current)
    verified(output)
    if current["status"] == "ready":
        for model in current["indices"]:
            load_speed(output, model, "screening")
    return {"status": current["status"], "path": str(output.resolve()),
            "manifest_sha256": sha(output / "manifest.json"), "gpu_checked": False,
            "quality_evaluated": False, "reindexed_models": models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(rebuild(args.source, args.output, args.models, args.version))
