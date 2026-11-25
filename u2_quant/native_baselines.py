"""Prepare and collect the seven native U1 eager baselines."""

import argparse
import json
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from u1_benchmark.core import ROOT, read, sha, write
from u1_benchmark.v4_config import PROTOCOL_VERSION, RESULT_VERSION, validate_config
from u1_benchmark.v4_data import prepare, verified
from u1_benchmark.v4_evidence import source_inventory
from u1_benchmark.v4_privacy import public_config

MODELS = ("laya", "gliner", "koala", "llama", "halo", "shield", "hive")
BINARY = {"safe": "safe", "unsafe": "unsafe"}
VERDICTS = {
    "laya": {"A": "unsafe", "B": "safe"},
    "gliner": BINARY,
    "koala": {**dict.fromkeys(("H", "H2", "HR", "S", "S3", "SH", "V", "V2"), "unsafe"), "OK": "safe"},
    "llama": {"LABEL_0": "safe", "LABEL_1": "unsafe"},
    "halo": BINARY,
    "shield": {"Yes": "unsafe", "No": "safe"},
    "hive": BINARY,
}
STAGES = {"preflight": ("preflight", "preflight_passed"),
          "baseline": ("run", "baseline_complete"), "quality": ("quality", "quality_complete"),
          "quality-smoke": ("quality", "quality_complete")}


def speed_configs(*, environment, speed_data):
    return {
        model: {**read(ROOT / f"configs/v4/{model}.json"), "environment": str(environment),
                "data": speed_data}
        for model in MODELS
    }


def quality_configs(*, environment, quality_data, gliner_data, label_mappings=VERDICTS):
    configs = {}
    for model, cfg in speed_configs(environment=environment, speed_data=None).items():
        cfg.pop("data")
        configs[model] = {
            **cfg, "mode": "quality", "quality_batch_sizes": [1],
            "quality_data": gliner_data if model == "gliner" else quality_data,
            "label_mapping": {"labels": ["safe", "unsafe"], "targets": BINARY,
                              "verdicts": label_mappings[model], "positive_label": "unsafe"},
        }
    return configs


def reindex_spec(source):
    return {"operation": "reindex", "source": source,
            "configs": [{"config": str(ROOT / f"configs/v4/{m}.json"), "variant": "baseline",
                         "name": m} for m in MODELS]}


def data_ref(directory):
    path = Path(directory).resolve()
    if read(path / "manifest.json").get("status") != "ready":
        raise ValueError(f"Dataset is not ready: {path}")
    return {"path": str(path), "manifest_sha256": sha(path / "manifest.json")}


def validate_quality_pair(raw, normalized):
    def targets(ref):
        rows, manifest = verified(ref["path"], ref["manifest_sha256"])
        if manifest["kind"] != "quality":
            raise ValueError("Expected a quality corpus")
        return {r["id"]: (r["target"], r.get("slices", {})) for r in rows}

    base, gliner = targets(raw), targets(normalized)
    if len(base) != 7121 or base != gliner:
        raise ValueError("Expected matching tails_v2 IDs, labels and slices in both corpora")


def smoke_corpus(ref, directory):
    """Keep the same input IDs for the raw and normalized smoke corpora."""
    rows, _ = verified(ref["path"], ref["manifest_sha256"])
    selected = []
    for target in BINARY:
        selected.extend(sorted((r for r in rows if r["target"] == target), key=lambda r: r["id"])[:4])
    source = directory.with_suffix(".jsonl")
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    if not directory.exists():
        source.write_text(content, encoding="utf-8")
        prepare({"schema_version": 4, "kind": "quality", "source": str(source),
                 "fields": {k: k for k in ("id", "messages", "target", "target_role",
                                          "target_index", "slices")},
                 "version": "tails_v2-smoke", "provenance": {"parent": ref},
                 "include_texts": False}, directory)
    result = data_ref(directory)
    _, manifest = verified(result["path"], result["manifest_sha256"])
    if manifest["provenance"]["parent"] != ref:
        raise ValueError(f"Stale smoke corpus: {directory}")
    return result


def generate(args):
    speed, quality, gliner = map(data_ref, (args.speed, args.quality, args.gliner_quality))
    validate_quality_pair(quality, gliner)
    args.out.mkdir(parents=True, exist_ok=True)
    smoke = smoke_corpus(quality, args.out / "quality-smoke")
    gliner_smoke = smoke_corpus(gliner, args.out / "gliner-quality-smoke")
    configs = {
        "baseline": speed_configs(environment=args.environment, speed_data=speed),
        "quality": quality_configs(environment=args.environment, quality_data=quality, gliner_data=gliner),
        "quality-smoke": quality_configs(environment=args.environment, quality_data=smoke,
                                         gliner_data=gliner_smoke),
    }
    for stage, models in configs.items():
        for model, cfg in models.items():
            validate_config(cfg)
            write(args.out / f"{model}-{stage}.json", cfg)


def archive_report(archive, stage):
    with ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt entry: {bad}")
        summary = json.loads(bundle.read("summary.json"))
        resolved = json.loads(bundle.read("resolved_run.json"))
    if summary["status"] != STAGES[stage][1] or not summary["complete"]:
        raise ValueError(f"Incomplete {stage}: {summary['status']}")
    if stage in ("quality", "quality-smoke"):
        quality = summary["quality"]["quality_by_batch_size"]["1"]["variants"]["baseline"]["quality"]
        overall = quality["overall"]
        if not overall["positive_n"] or not overall["negative_n"]:
            raise ValueError(f"Missing evaluated labels in {stage}")
    if (resolved["protocol_version"], resolved["result_version"]) != (PROTOCOL_VERSION, RESULT_VERSION):
        raise ValueError("Archive protocol differs from the requested U1 protocol")
    return summary, resolved


def collect(args):
    args.out.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for stage in STAGES:
            filename = f"{model}-{stage}.zip"
            path = args.out / filename
            if path.exists():
                archive_report(path, stage)
                continue
            pending = path.with_suffix(".zip.part")
            command = ["scp", "-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                       "-o", "ConnectTimeout=10",
                       "-P", str(args.port), "-i", str(args.identity),
                       f"{args.host}:{args.remote_dir}/{filename}", str(pending)]
            if subprocess.run(command, check=False).returncode:
                continue
            try:
                archive_report(pending, stage)
            except (BadZipFile, ValueError, KeyError) as exc:
                print(f"Skipped {filename}: {exc}", flush=True)
                continue
            pending.replace(path)
            print(f"Saved {path}", flush=True)
    return all((args.out / f"{model}-{stage}.zip").is_file()
               for model in args.models for stage in ("baseline", "quality"))


def summarize(args):
    runs = {}
    for model in args.models:
        stages = {}
        for stage in ("baseline", "quality"):
            archive = args.directory / f"{model}-{stage}.zip"
            summary, resolved = archive_report(archive, stage)
            cfg = resolved["config"]
            if cfg["baseline"]["adapter"] != model:
                raise ValueError(f"Unexpected adapter in {archive}")
            stages[stage] = {
                "archive": archive.name, "archive_sha256": sha(archive),
                "status": summary["status"], "protocol": resolved["protocol_version"],
                "data_manifest_sha256": resolved["data_manifest_sha256"],
                "source_inventory": resolved["source_inventory"],
                "environment": resolved["environment"], "baseline": cfg["baseline"],
                "precision": summary.get("precision"), "cells": summary.get("cells"),
                "quality": summary.get("quality"),
            }
        if stages["baseline"]["baseline"] != stages["quality"]["baseline"]:
            raise ValueError(f"Baseline and quality variants differ for {model}")
        runs[model] = stages
    write(args.out, {"schema_version": 1, "models": runs})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("reindex-spec")
    index.add_argument("--source", type=Path, required=True)
    index.add_argument("--out", type=Path, required=True)
    gen = commands.add_parser("configs")
    for option in ("environment", "speed", "quality", "gliner-quality", "out"):
        gen.add_argument(f"--{option}", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("archive", type=Path)
    verify.add_argument("stage", choices=STAGES)
    verify.add_argument("config", type=Path)
    get = commands.add_parser("collect")
    get.add_argument("--host", required=True)
    get.add_argument("--port", type=int, required=True)
    get.add_argument("--identity", type=Path, required=True)
    get.add_argument("--remote-dir", required=True)
    get.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    get.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("summary")
    report.add_argument("--directory", type=Path, required=True)
    report.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "reindex-spec":
        write(args.out, reindex_spec(data_ref(args.source)))
    elif args.command == "configs":
        generate(args)
    elif args.command == "verify":
        _, resolved = archive_report(args.archive, args.stage)
        config = validate_config(args.config)
        if args.stage == "preflight":
            config["mode"] = "smoke"
        if resolved["source_inventory"] != source_inventory():
            raise ValueError("Saved measurement source inventory differs")
        if resolved["config"] != public_config(config):
            raise ValueError("Saved config differs from the requested run")
    elif args.command == "collect":
        return 0 if collect(args) else 1
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
