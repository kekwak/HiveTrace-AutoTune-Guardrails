import shutil
import sys
import time
import traceback
from pathlib import Path

from .core import ROOT, PRESETS, DATASET, REVISION, config as parse_config, load_data, read, schedule, sha, write
from .process import archive, invoke

ENTRY = ROOT / "tools/benchmark.py"


def setup(directory, profile="team-t4"):
    if profile == "team-rtx":
        from .environment import setup_rtx
        return setup_rtx(directory)
    if profile != "team-t4":
        raise ValueError("Unknown environment profile")
    from followup.team_skeleton import precheck
    from followup.team_lock import requirements
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    folder = directory / "setup"
    folder.mkdir()
    state = {"status": "running", "gpu_verified_for_models": False}
    write(folder / "status.json", state)
    try:
        write(folder / "host.json", precheck())
        lock = read(ROOT / "env/team_environment.lock.json")
        write(folder / "lock.json", lock)
        text = requirements(lock)
        if text != (ROOT / "env/team_environment.lock.txt").read_text():
            raise ValueError("Lock text differs from manifest")
        (folder / "requirements.txt").write_text(text)
        python = directory / "venv/bin/python"
        invoke([sys.executable, "-m", "venv", "--without-pip", directory / "venv"], folder / "venv.log", 120)
        invoke([sys.executable, "-m", "pip", "--python", python, "install", "--no-deps", "--require-hashes",
                "--only-binary=:all:", "-r", folder / "requirements.txt", "--report", folder / "install.json"],
               folder / "install.log", 3600)
        invoke([sys.executable, "-m", "pip", "--python", python, "check"], folder / "pip-check.log", 120)
        invoke([python, ENTRY, "_probe", "--environment", directory, "--output", folder], folder / "probe.log", 180)
        state.update(status="passed", lock_sha256=sha(folder / "lock.json"), cuda_primitive_passed=True)
    except BaseException as e:
        state.update(status="failed", error=str(e))
        (folder / "error.txt").write_text(traceback.format_exc())
    finally:
        write(folder / "status.json", state)
        archive(directory)
    return state


def download(directory):
    from huggingface_hub import snapshot_download
    destination = Path(directory)
    snapshot = Path(snapshot_download(DATASET, repo_type="dataset", revision=REVISION, token=False))
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read(snapshot / "manifest.json")
    from .core import MANIFEST_SHA
    if sha(snapshot / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Pinned manifest changed")
    for name in ["manifest.json", *manifest["files"]]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid dataset path")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / name, target)
    load_data(destination, "halo")
    load_data(destination, "gliner")


def run(config, output_dir, *, preflight_only=False):
    raw = read(config) if isinstance(config, (str, Path)) else config
    if raw.get("schema_version") in (2, 3):
        from .runner_v2 import run as run_v2
        return run_v2(raw, output_dir, preflight_only=preflight_only)
    cfg = parse_config(config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = Path(cfg["environment"])
    python = env / "venv/bin/python"
    deadline = time.monotonic() + cfg["timeout_seconds"]
    state = {"status": "running", "quality_evaluated": False, "gpu_acceptance": "not_completed",
             "phase": "environment", "mode": cfg["mode"]}
    write(output / "config.json", cfg)
    write(output / "status.json", state)
    def launch(phase, variant, target, measurement=None):
        if any(sha(ROOT / name) != checksum for name, checksum in sources.items()):
            raise ValueError("Sources changed during execution")
        state.update(phase=phase, variant=variant, active_directory=str(target.relative_to(output)))
        write(output / "status.json", state)
        job = {"config": cfg, "environment": str(env), "dataset": str(output / "dataset"),
               "directory": str(target), "phase": phase, "variant": variant, "measurement": measurement}
        write(target / "job.json", job)
        print(f"{cfg['preset']}: {phase} {variant} {target.name}", flush=True)
        invoke([python, ENTRY, "_worker", "--job", target / "job.json"], target / "stdout.log",
               deadline - time.monotonic())
    try:
        if read(env / "setup/status.json")["status"] != "passed":
            raise ValueError("Run setup successfully first")
        if read(env / "setup/lock.json") != read(ROOT / "env/team_environment.lock.json"):
            raise ValueError("Setup lock does not match repository lock")
        sources = {}
        for package in ("u1_benchmark", "experiments", "followup"):
            for path in (ROOT / package).glob("*.py"):
                target = output / "sources" / package / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                sources[str(path.relative_to(ROOT))] = sha(path)
        sources["tools/benchmark.py"] = sha(ENTRY)
        shutil.copyfile(ENTRY, output / "sources/benchmark.py")
        write(output / "sources.json", sources)
        write(output / "lock.json", read(env / "setup/lock.json"))
        invoke([python, ENTRY, "_dataset", "--output", output / "dataset"], output / "dataset.log",
               min(600, deadline - time.monotonic()))
        model, _ = PRESETS[cfg["preset"]]
        data, _ = load_data(output / "dataset", model)
        jobs = schedule(cfg, data)
        write(output / "schedule.json", jobs)
        for variant in ("baseline", "candidate"):
            launch("preflight", variant, output / "preflight" / variant)
        if not preflight_only:
            for i, job in enumerate(jobs):
                if any(sha(ROOT / name) != checksum for name, checksum in sources.items()):
                    raise ValueError("Sources changed during measurement")
                launch("latency", job["variant"], output / "latency" / f"{i:03d}", job)
            if cfg["mode"] != "pilot":
                for variant in ("baseline", "candidate"):
                    for repeat in range(3):
                        launch("memory", variant, output / "memory" / f"{variant}_{repeat}")
                    launch("profile", variant, output / "profiles" / variant)
        state.update(status="completed", gpu_acceptance="preflight_only" if preflight_only else "execution_completed")
        if preflight_only:
            from .statistics import compare_scores
            from .core import lines
            scores = {}
            for variant in ("baseline", "candidate"):
                scores[variant] = {r["id"]: [r["score"]] for r in lines(output / "preflight" / variant / "scores.jsonl")}
            checks = {b: compare_scores({r["id"]: scores["baseline"][r["id"]] for r in rows},
                                       {r["id"]: scores["candidate"][r["id"]] for r in rows})
                      for b, rows in data.items()}
            write(output / "preflight_summary.json", {"status": "execution_passed", "buckets": checks,
                  "speed_measured": False, "quality_evaluated": False})
    except BaseException as e:
        state.update(status="timeout" if isinstance(e, TimeoutError) else "failed", error=str(e))
        (output / "error.txt").write_text(traceback.format_exc())
    finally:
        write(output / "status.json", state)
        try:
            if not preflight_only:
                from .report import build
                summary = build(output)
                state["execution_status"] = state["status"]
                state["status"] = summary["status"]
        finally:
            archive(output)
    return state
