"""Baseline once, exploratory reuse, explicit paired confirmation."""
from copy import deepcopy
from pathlib import Path
import shutil
import time
import uuid

from .core import ROOT, digest, read, sha, write
from .process import check_archive_target, finish_archive, invoke
from .v4_config import PROTOCOL_VERSION, RESULT_VERSION, validate_config
from .v4_data import DataError, load_speed, verified
from .v4_evidence import check_reference, copy_reference, immutable, read_reference, source_inventory
from .v4_planning import cells, schedule
from .v4_privacy import public_config


def input_indices(data):
    from .v4_worker import public_index
    return {b: [{"id": r["id"], **public_index(r["index"])} for r in rows] for b, rows in data.items()}


def inputs(cfg):
    if cfg["mode"] == "quality":
        source = cfg["quality_data"]
        rows, manifest = verified(source["path"], source["manifest_sha256"])
        if manifest["kind"] != "quality":
            raise ValueError("Expected labeled quality data")
        from .quality import validate_mapping
        validate_mapping(rows, cfg["label_mapping"])
        return rows, manifest, {}
    source = cfg["data"]
    data, manifest = load_speed(source["path"], cfg["baseline"], cfg["buckets"], cfg["mode"],
                                cfg["pilot"], source["manifest_sha256"])
    if any(size > len(data[bucket]) for bucket, size in cells(cfg)):
        raise DataError("Requested batch exceeds unique prepared input count")
    return data, manifest, input_indices(data)


def plan(config):
    cfg = validate_config(config)
    try:
        data, manifest, indices = inputs(cfg)
    except (DataError, OSError) as error:
        return {"status": "not_ready", "phase": "data", "error_type": type(error).__name__,
                "schema_version": 4, "data_ready": False, "gpu_checked": False}
    sources = source_inventory()
    try:
        reference = read_reference(cfg["baseline_ref"]) if cfg["mode"] == "search" else None
        if reference:
            check_reference(cfg, reference, sources, indices)
    except (ValueError, OSError) as error:
        return {"status": "not_ready", "phase": "reference", "error_type": type(error).__name__,
                "schema_version": 4, "data_ready": True, "reference_ready": False, "gpu_checked": False}
    jobs = [] if cfg["mode"] == "quality" else schedule(cfg, data, reference["resolved"]["schedule"] if reference else None)
    variants = ["candidate"] if cfg["mode"] == "search" else ["baseline"] + (["candidate"] if "candidate" in cfg else [])
    grid = [] if cfg["mode"] == "quality" else cells(cfg)
    result = {"status": "ready", "schema_version": 4, "mode": cfg["mode"], "pilot": cfg["pilot"],
              "effective_settings": cfg, "data_ready": True, "reference_ready": bool(reference),
              "quality_workers": len(variants) * len(cfg["quality_batch_sizes"]) if cfg["mode"] == "quality" else 0,
              "preflight_workers": len(grid) * len(variants), "latency_workers": len(jobs),
              "warmup_batches": 20 * len(jobs), "timed_batches": sum(len(j["batches"]) for j in jobs),
              "timed_input_occurrences": sum(len(j["ids"]) for j in jobs),
              "memory_workers": 0, "profile_workers": len(grid) * len(variants) if cfg["profile_trace"] else 0,
              "environment_verified": False, "gpu_checked": False}
    if cfg["mode"] == "quality":
        result["quality_input_occurrences"] = len(data) * len(variants) * len(cfg["quality_batch_sizes"])
    return result


def run(config, output_dir, *, preflight_only=False, launcher=invoke):
    cfg = validate_config(config)
    if preflight_only:
        cfg["mode"] = "smoke"
        cfg.pop("baseline_ref", None)
    directory = Path(output_dir).resolve()
    check_archive_target(directory)
    directory.mkdir(parents=True, exist_ok=False)
    private = directory / ".private"
    private.mkdir(mode=0o700)
    start = time.monotonic()
    deadline = start + cfg["timeout_seconds"]
    state = {"status": "running", "schema_version": 4, "quality_evaluated": False, "stages": []}
    write(directory / "status.json", state)

    def launch_command(args, log):
        launcher(args, log, deadline - time.monotonic())

    try:
        data, manifest, indices = inputs(cfg)
        sources = source_inventory()
        reference = read_reference(cfg["baseline_ref"]) if cfg["mode"] == "search" else None
        if reference:
            check_reference(cfg, reference, sources, indices)
        python = Path(cfg["environment"]) / "venv/bin/python"
        if not python.is_file():
            raise FileNotFoundError("Pinned environment Python missing")
        launch_command([python, ROOT / "tools/benchmark.py", "_runtime_v4", "--environment", cfg["environment"],
                        "--output", private / "runtime.json"], private / "runtime.log")
        environment = read(private / "runtime.json")
        if reference:
            check_reference(cfg, reference, sources, indices, environment)
            copy_reference(reference, directory)
        jobs = [] if cfg["mode"] == "quality" else schedule(cfg, data, reference["resolved"]["schedule"] if reference else None)
        private_config = private / "config.json"
        private_config_sha = immutable(private_config, cfg)
        resolved = {"schema_version": 4, "result_version": RESULT_VERSION, "protocol_version": PROTOCOL_VERSION,
                    "config": public_config(cfg), "config_sha256": digest(cfg),
                    "variant_identities": {k: digest(cfg[k]) for k in ("baseline", "candidate") if k in cfg},
                    "source_inventory": sources, "environment": environment,
                    "session_id": str(uuid.uuid4()), "created_at": time.time(), "input_indices": indices,
                    "data_manifest_sha256": digest(manifest), "schedule": jobs,
                    "monitoring_validation": None}
        if cfg["mode"] == "quality":
            fields = ("id", "target", "target_role", "target_index", "slices", "content_sha256", "text_sha256")
            resolved["quality_rows"] = [{k: row[k] for k in fields if k in row} for row in data]
        if cfg.get("monitor_validation"):
            check = read(cfg["monitor_validation"])
            if check.get("monitor_method") != environment["monitor_method"] or check.get("environment") != {
                    k: v for k, v in environment.items() if k not in {"gpu_uuid", "host_sha256", "pid", "timestamp", "session_id"}}:
                raise ValueError("Monitoring validation environment mismatch")
            resolved["monitoring_validation"] = check
        state["resolved_sha256"] = immutable(directory / "resolved_run.json", resolved)
        write(directory / "status.json", state)
        for name in sources:
            target = directory / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        for name in ("baseline", "candidate"):
            spec = cfg.get(name, {})
            adapter = spec.get("adapter")
            if isinstance(adapter, dict):
                path = Path(adapter["module"])
                if sha(path) != adapter["source_sha256"]:
                    raise ValueError("External adapter source changed")
                target = directory / "sources" / "external" / (adapter["source_sha256"] + ".py")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        variants = ["candidate"] if cfg["mode"] == "search" else ["baseline"] + (["candidate"] if "candidate" in cfg else [])
        counter = 0

        def launch(stage, variant, rows, **extra):
            nonlocal counter
            target = directory / "jobs" / f"{counter:05d}"
            counter += 1
            target.mkdir(parents=True)
            input_path = private / (target.name + ".inputs.json")
            write(input_path, rows)
            job = {**extra, "stage": stage, "variant": variant, "seed": cfg["seed"],
                   "resolved_path": str(directory / "resolved_run.json"), "resolved_sha256": state["resolved_sha256"],
                   "private_config_path": str(private_config), "private_config_sha256": private_config_sha,
                   "input_path": str(input_path), "input_file_sha256": sha(input_path)}
            write(target / "job.json", job)
            interruption = None
            try:
                launch_command([python, ROOT / "tools/benchmark.py", "_worker_v4", "--job", target / "job.json"],
                               private / (target.name + ".log"))
                result = read(target / "result.json")
            except (Exception, KeyboardInterrupt) as error:
                if (target / "result.json").exists():
                    result = read(target / "result.json")
                else:
                    result = {"result_version": 4, "job": job, "resolved_sha256": state["resolved_sha256"]}
                if result.get("status") in (None, "running", "passed"):
                    result.update(status="timeout" if isinstance(error, TimeoutError) else "failed", error_type=type(error).__name__)
                write(target / "result.json", result)
                if isinstance(error, (TimeoutError, KeyboardInterrupt)):
                    interruption = error
            state["stages"].append({"path": str(target.relative_to(directory)), "stage": stage,
                                     "variant": variant, "status": result["status"]})
            write(directory / "status.json", state)
            if interruption is not None:
                raise interruption
            return target, result

        if cfg["mode"] == "quality":
            for size in cfg["quality_batch_sizes"]:
                for variant in variants:
                    launch("quality", variant, data, quality_batch_size=size, label_mapping=cfg["label_mapping"])
        else:
            for bucket, size in cells(cfg):
                proofs = {}
                for variant in variants:
                    path, result = launch("preflight", variant, data[bucket], bucket=bucket,
                                          batch_size=size, bucket_max=cfg["buckets"][bucket]["max"])
                    if result["status"] == "passed":
                        proofs[variant] = path / "result.json"
                for scheduled in jobs:
                    if (scheduled["bucket"], scheduled["batch_size"]) != (bucket, size):
                        continue
                    variant = scheduled["variant"]
                    if variant not in proofs:
                        continue
                    metadata = {k: v for k, v in scheduled.items() if k != "variant"}
                    launch("latency", variant, data[bucket], **metadata, preflight_path=str(proofs[variant]),
                           preflight_sha256=sha(proofs[variant]))
                if cfg["profile_trace"]:
                    for variant in proofs:
                        launch("profile", variant, data[bucket], bucket=bucket, batch_size=size)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="timeout" if isinstance(error, TimeoutError) else "failed", error_type=type(error).__name__)
    finally:
        state["seconds"] = time.monotonic() - start
        write(directory / "status.json", state)
        if (directory / "resolved_run.json").exists():
            try:
                from .v4_report import build
                summary = build(directory)
                if state["status"] == "completed":
                    state["status"] = summary["status"]
                write(directory / "status.json", state)
            except Exception as error:
                state["report_error_type"] = type(error).__name__
                if state["status"] == "completed":
                    state["status"] = "failed"
                write(directory / "status.json", state)
        finish_archive(directory, state, directory / "status.json")
    return state
