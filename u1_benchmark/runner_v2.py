"""Version two orchestration. Model work stays in fresh, sequential processes."""
import random
import shutil
import time
import traceback
from pathlib import Path

from .contracts import parse_config
from .core import ROOT, digest, read, sha, write
from .datasets import load_speed, verified
from .process import check_archive_target, finish_archive, invoke
from .precision import PrecisionTracker

ENTRY = ROOT / "tools/benchmark.py"


def schedule(cfg, data):
    if cfg["mode"] in ("smoke", "quality"):
        return []
    jobs = []
    candidate = cfg.get("candidate") is not None
    for phase, count, size in (("aa", 2, 30), ("ab" if candidate else "baseline", 1 if cfg["mode"] == "pilot" else 3, 12 if cfg["mode"] == "pilot" else 400)):
        for block in range(count):
            for bucket, rows in data.items():
                ids = [r["id"] for r in rows]
                random.Random(f"{cfg['seed']}:{phase}:{block}:{bucket}").shuffle(ids)
                ids = ids[:size]
                if phase == "baseline":
                    sides = ("a",)
                else:
                    sides = ("a", "b") if block % 2 == 0 else ("b", "a")
                for side in sides:
                    jobs.append({"phase": phase, "block": block, "bucket": bucket, "side": side,
                                 "variant": "candidate" if phase == "ab" and side == "b" else "baseline", "ids": ids})
    return jobs


def public_rows(rows):
    return [{k: r[k] for k in ("id", "text_sha256", "content_sha256", "bucket", "index", "target",
                              "target_role", "target_index", "slices") if k in r} for r in rows]


def run(value, output_dir, preflight_only=False):
    cfg = parse_config(value)
    output = Path(output_dir).resolve()
    check_archive_target(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", cfg)
    state = {"schema_version": cfg["schema_version"], "status": "running", "phase": "environment", "quality_evaluated": False}
    write(output / "status.json", state)
    deadline = time.monotonic() + cfg["timeout_seconds"]
    env = Path(cfg["environment"])
    python = env / "venv/bin/python"
    sources = {}
    variants = ["baseline"] + (["candidate"] if cfg.get("candidate") else [])
    precision = PrecisionTracker(cfg["comparison"])
    def launch(phase, variant, directory, measurement=None):
        for name, checksum in sources.items():
            if sha(name) != checksum:
                raise ValueError("Source changed during run")
        state.update(phase=phase, variant=variant)
        write(output / "status.json", state)
        job = {"schema_version": cfg["schema_version"], "config": cfg, "phase": phase, "variant": variant,
               "directory": str(directory), "dataset": str(dataset) if dataset else None,
               "quality_dataset": cfg.get("quality_data"), "measurement": measurement,
               "private": private, "inputs_sha256": sha(output / "inputs.json")}
        write(directory / "job.json", job)
        invoke([python, ENTRY, "_worker_v2", "--job", directory / "job.json"],
               directory / "stdout.log", deadline - time.monotonic())
        precision.add(variant, read(directory / "result.json"))
    try:
        from .environment import expected_lock
        if read(env / "setup/status.json")["status"] != "passed" or read(env / "setup/lock.json") != expected_lock(env):
            raise ValueError("Run successful setup with unchanged lock first")
        write(output / "lock.json", read(env / "setup/lock.json"))
        for package in ("u1_benchmark", "experiments", "followup", "speed_dataset"):
            for path in (ROOT / package).glob("*.py"):
                target = output / "sources" / package / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                sources[str(path)] = sha(path)
        for path in (ENTRY, ROOT / "speed_dataset/contracts.json"):
            target = output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            sources[str(path)] = sha(path)
        for variant in variants:
            if "module" in cfg[variant]:
                path = Path(cfg[variant]["module"])
                shutil.copyfile(path, output / "sources" / (variant + "_external.py"))
                sources[str(path)] = sha(path)
        write(output / "sources.json", sources)
        write(output / "source_files.json", {str(p.relative_to(output)): sha(p)
              for p in (output / "sources").rglob("*") if p.is_file()})
        dataset, data, manifest = None, {}, {}
        private = True
        if cfg["mode"] != "quality":
            if cfg["data"] == {"profile": "published"}:
                dataset = output / ".private/dataset"
                invoke([python, ENTRY, "_dataset", "--output", dataset], output / "dataset.log", min(600, deadline - time.monotonic()))
                private = False
            else:
                dataset = Path(cfg["data"]["path"])
                private = not read(dataset / "manifest.json").get("include_texts", False)
            data, manifest = load_speed(dataset, cfg["model"], cfg["mode"], cfg["data"].get("manifest_sha256"))
            if (manifest.get("schema_version") == 3) != (cfg["schema_version"] == 3):
                raise ValueError("Configuration and speed input format versions disagree")
        rows = [r for rs in data.values() for r in rs]
        if cfg.get("quality_data"):
            qrows, qmanifest = verified(cfg["quality_data"]["path"], cfg["quality_data"]["manifest_sha256"])
            if qmanifest.get("kind") != "quality":
                raise ValueError("Separate labeled quality dataset required")
            if (qmanifest.get("schema_version") == 3) != (cfg["schema_version"] == 3):
                raise ValueError("Configuration and quality input format versions disagree")
            from .quality import validate_mapping
            validate_mapping(qrows, cfg.get("label_mapping"))
            write(output / "quality_inputs.json", public_rows(qrows))
            write(output / "quality_manifest.json", qmanifest)
        write(output / "inputs.json", public_rows(rows))
        write(output / "dataset_manifest.json", manifest)
        write(output / "dataset_identity.json", {"manifest_sha256": sha(dataset / "manifest.json") if dataset else None,
                                                 "model": cfg["model"]})
        if dataset and not private:
            from .datasets import jsonl
            jsonl(output / "corpus.jsonl", rows)
        jobs = schedule(cfg, data)
        write(output / "schedule.json", jobs)
        if cfg["mode"] != "quality":
            for variant in variants:
                launch("smoke" if cfg["mode"] == "smoke" else "preflight", variant, output / "preflight" / variant)
            if not preflight_only and cfg["mode"] != "smoke":
                for i, job in enumerate(jobs):
                    launch("latency", job["variant"], output / "latency" / f"{i:03d}", job)
                if cfg["mode"] == "screening":
                    for variant in variants:
                        for repeat in range(3):
                            launch("memory", variant, output / "memory" / f"{variant}_{repeat}")
                        launch("profile", variant, output / "profiles" / variant)
        if cfg.get("quality_data") and not preflight_only:
            private = not qmanifest.get("include_texts", False)
            for variant in variants:
                launch("quality", variant, output / "quality" / variant)
        state.update(status="completed", preflight_only=preflight_only)
    except BaseException as error:
        state.update(status="timeout" if isinstance(error, TimeoutError) else "failed", error_type=type(error).__name__)
        # Do not archive arbitrary data-bearing exception strings from custom code.
        hidden = output / ".private"
        hidden.mkdir(exist_ok=True)
        (hidden / "error.txt").write_text(traceback.format_exc())
    finally:
        write(output / "status.json", state)
        from .report_v2 import build
        summary = build(output)
        if not finish_archive(output, state, output / "status.json"):
            summary = build(output)
    return summary
