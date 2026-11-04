"""U1 console entry point; no notebook required."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def public_result(command, result, args):
    fields = {"status", "error_type", "phase", "archive_error_type", "archive_status"}
    if command == "auth-check":
        fields |= {"model", "revision", "credential_available", "access", "weights_downloaded", "gpu_checked"}
    elif command in ("fetch-data", "prepare-data"):
        fields |= {"path", "manifest_sha256", "scope", "revision", "origin", "gpu_checked"}
    elif command == "table":
        fields |= {"rows", "omitted"}
    elif command == "setup":
        fields |= {"profile", "lock_sha256", "cuda_primitive_passed", "gpu_verified_for_models"}
    summary = {key: result[key] for key in sorted(fields) if key in result}
    if command == "fetch-data" and "model_contract_status" in result:
        from u1_benchmark.contracts import MODEL_SPECS
        summary["model_contract_status"] = {
            model: status for model, status in result["model_contract_status"].items()
            if model in MODEL_SPECS and status in ("current", "requires_reindex")}
    if command in ("run", "preflight", "quality", "report", "setup"):
        from u1_benchmark.core import read
        from u1_benchmark.process import archive_path
        root = Path(args.environment if command == "setup" else args.output).resolve()
        summary["output_path"] = str(root)
        status = root / ("setup/status.json" if command == "setup" else "status.json")
        if status.is_file():
            try:
                saved = read(status)
            except (OSError, ValueError):
                saved = {}
            for key in ("error_type", "phase", "archive_error_type", "archive_status"):
                if key in saved:
                    summary[key] = saved[key]
        if command != "setup" and (root / "summary.json").is_file():
            summary["report_path"] = str(root / "summary.json")
        if archive_path(root).is_file():
            summary["archive_path"] = str(archive_path(root))
    elif command == "table":
        summary["output_path"] = str(Path(args.output).resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("setup")
    p.add_argument("--environment", required=True)
    p.add_argument("--profile", choices=("team-t4", "team-rtx", "team-t4-laya"), default="team-t4")
    p = commands.add_parser("auth-check")
    p.add_argument("--model", default="shield")
    p = commands.add_parser("fetch-data")
    p.add_argument("--source")
    p.add_argument("--revision")
    p.add_argument("--output")
    for name in ("preflight", "run"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--environment", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--pilot", action="store_true")
    p = commands.add_parser("report")
    p.add_argument("--output", required=True)
    p = commands.add_parser("prepare-data")
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("table")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("quality")
    p.add_argument("--config", required=True)
    p.add_argument("--environment", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("_worker_v2")
    p.add_argument("--job", required=True)
    p = commands.add_parser("_worker")
    p.add_argument("--job", required=True)
    p = commands.add_parser("_probe")
    p.add_argument("--environment", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("_dataset")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = execute(args, parser)
    except Exception as error:
        # Exception messages may contain HF credentials or private model inputs.
        result = {"status": "failed", "error_type": type(error).__name__}
    print(json.dumps(public_result(args.command, result, args), ensure_ascii=False))
    return 0 if result["status"] in ("passed", "completed", "screening_pass", "pilot_complete", "ready",
                                    "smoke_passed", "preflight_passed", "baseline_complete", "quality_complete",
                                    "quantization_measured") else 1


def execute(args, parser):
    if args.command == "setup":
        from u1_benchmark.orchestrator import setup
        result = setup(args.environment, args.profile)
    elif args.command == "auth-check":
        from u1_benchmark.environment import auth_check
        result = auth_check(args.model)
    elif args.command == "fetch-data":
        from u1_benchmark.distribution import fetch
        result = fetch(args.output, args.source, args.revision)
    elif args.command in ("run", "preflight", "quality"):
        from u1_benchmark.core import read
        from u1_benchmark.orchestrator import run
        cfg = read(args.config)
        cfg["environment"] = args.environment
        if getattr(args, "pilot", False):
            cfg["mode"] = "pilot"
        if args.command == "quality":
            if cfg.get("schema_version") not in (2, 3):
                parser.error("quality requires schema_version=2 or 3")
            cfg["mode"] = "quality"
        result = run(cfg, args.output, preflight_only=args.command == "preflight")
    elif args.command == "prepare-data":
        from u1_benchmark.core import read
        from u1_benchmark.datasets import prepare
        result = prepare(read(args.spec), args.output)
    elif args.command == "table":
        from u1_benchmark.report_v2 import table
        result = table(args.inputs, args.output)
    elif args.command == "_worker_v2":
        from u1_benchmark.worker_v2 import execute
        result = execute(args.job)
    elif args.command == "report":
        from u1_benchmark.report import build
        result = build(args.output)
    elif args.command == "_probe":
        from u1_benchmark.environment import probe
        probe(Path(args.environment), Path(args.output))
        return {"status": "passed"}
    elif args.command == "_dataset":
        from u1_benchmark.orchestrator import download
        download(args.output)
        return {"status": "completed"}
    else:
        from u1_benchmark.worker import execute
        result = execute(args.job)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
