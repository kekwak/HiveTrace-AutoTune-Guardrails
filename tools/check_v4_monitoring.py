"""One-time GPU sampler check: GLiNER/Open B1 x 1, no benchmark series."""
import argparse
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from u1_benchmark.core import ROOT, digest, read, sha, write
from u1_benchmark.process import check_archive_target, finish_archive, invoke
from u1_benchmark.v4_config import MONITOR_VERSION, PROTOCOL_VERSION, resolve_options
from u1_benchmark.v4_evidence import environment_identity, immutable, source_inventory
from u1_benchmark.v4_planning import schedule
from u1_benchmark.v4_runner import inputs, input_indices
from u1_benchmark.v4_privacy import public_config


def run(configs, environment, output):
    root = Path(output).resolve()
    check_archive_target(root)
    root.mkdir(parents=True, exist_ok=False)
    private = root / ".private"
    private.mkdir(mode=0o700)
    state = {"status": "unvalidated", "monitor_method": MONITOR_VERSION, "checks": {}}
    deadline = time.monotonic() + 36000
    python = Path(environment) / "venv/bin/python"
    try:
        invoke([python, ROOT / "tools/benchmark.py", "_runtime_v4", "--environment", environment,
                "--output", private / "runtime.json"], private / "probe.log", deadline - time.monotonic())
        runtime = read(private / "runtime.json")
        state["environment"] = environment_identity(runtime)
        state["source_inventory_sha256"] = digest(source_inventory())
        for name in source_inventory():
            target = root / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        for path in configs:
            cfg = resolve_options(path, mode="baseline", environment=environment)
            key = cfg["baseline"]["adapter"]
            if key not in ("gliner", "open") or key in state["checks"]:
                raise ValueError("Exactly one GLiNER and one Open baseline required")
            data, manifest, _ = inputs(cfg)
            cfg["buckets"] = {"B1": cfg["buckets"]["B1"]}
            cfg["batch_sizes_by_bucket"] = {"B1": [1]}
            data = {"B1": data["B1"]}
            resolved = {"schema_version": 4, "result_version": 4, "protocol_version": PROTOCOL_VERSION,
                        "config": public_config(cfg), "config_sha256": digest(cfg),
                        "variant_identities": {"baseline": digest(cfg["baseline"])},
                        "source_inventory": source_inventory(), "environment": runtime,
                        "input_indices": input_indices(data), "data_manifest_sha256": digest(manifest),
                        "schedule": [], "diagnostic_only": True}
            directory = root / key
            private_config = private / (key + ".config.json")
            config_sha = immutable(private_config, cfg)
            checksum = immutable(directory / "resolved_run.json", resolved)
            input_path = private / (key + ".inputs.json")
            write(input_path, data["B1"])
            batches = next(j for j in schedule(cfg, data) if j["phase"] == "baseline")["batches"][:30]
            job = {"stage": "monitor", "variant": "baseline", "seed": cfg["seed"], "batch_size": 1,
                   "bucket": "B1", "batches": batches, "resolved_path": str(directory / "resolved_run.json"),
                   "resolved_sha256": checksum, "input_path": str(input_path), "input_file_sha256": sha(input_path)}
            job.update(private_config_path=str(private_config), private_config_sha256=config_sha)
            write(directory / "job.json", job)
            try:
                invoke([python, ROOT / "tools/benchmark.py", "_worker_v4", "--job", directory / "job.json"],
                        private / (key + ".log"), deadline - time.monotonic())
            except Exception as error:
                state["checks"][key] = {"status": "unvalidated", "error_type": type(error).__name__}
            if (directory / "result.json").exists():
                worker = read(directory / "result.json")
                state["checks"][key] = worker.get("monitoring", {"status": "unvalidated", "worker_status": worker["status"]})
            write(root / "monitor_validation.json", state)
        state["status"] = "passed" if set(state["checks"]) == {"gliner", "open"} and all(
            check["status"] == "passed" for check in state["checks"].values()) else "unvalidated"
    except BaseException as error:
        state["error_type"] = type(error).__name__
    finally:
        write(root / "monitor_validation.json", state)
        write(root / "status.json", state)
        finish_archive(root, state, root / "status.json")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs=2, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.configs, args.environment, args.output)
    print({"status": result["status"], "output": str(Path(args.output).resolve())})
    raise SystemExit(0 if result["status"] == "passed" else 1)
