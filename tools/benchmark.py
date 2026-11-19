"""U1 console entry point; no notebook required."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def public_result(command, result, args):
    fields = {"status", "error_type", "phase", "archive_error_type", "archive_status",
              "batch_protocol_version", "completeness", "all_cells_faster", "all_cells_meet_threshold", "quality_batch_size",
              "quality_protocol_version", "quality_batch_sizes", "quality_status", "quality_complete"}
    fields |= {"diagnostic_code", "reference_compatibility", "comparison_design", "speed_status"}
    if command == "auth-check":
        fields |= {"model", "revision", "credential_available", "access", "weights_downloaded", "gpu_checked"}
    elif command in ("fetch-data", "prepare-data"):
        fields |= {"path", "manifest_sha256", "scope", "revision", "origin", "gpu_checked"}
    elif command == "plan":
        fields |= {"schema_version", "mode", "pilot", "effective_settings", "data_ready", "reference_ready",
                   "quality_workers", "preflight_workers", "latency_workers", "warmup_batches", "timed_batches",
                   "timed_input_occurrences", "memory_workers", "profile_workers", "environment_verified",
                   "gpu_checked", "quality_input_occurrences"}
        fields |= {"model_loads", "model_worker_processes", "preflight_stages", "latency_stages"}
    elif command == "table":
        fields |= {"rows", "omitted"}
    elif command == "setup":
        fields |= {"profile", "lock_sha256", "cuda_primitive_passed", "gpu_verified_for_models"}
    summary = {key: result[key] for key in sorted(fields) if key in result}
    if "effective_settings" in summary:
        from u1_benchmark.v4_privacy import public_config
        summary["effective_settings"] = public_config(summary["effective_settings"])
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


def explain(result):
    """Only fixed messages and validated numeric fields may reach the console."""
    messages = {
        "reference_configuration_differs": "Baseline несовместим: модель, precision/backend, сетка или seed отличаются. Выберите соответствующий reference или исправьте конфиг.",
        "reference_inputs_differ": "Baseline несовместим: подготовленные входы или их порядок отличаются. Используйте одинаковый корпус и подготовку.",
        "reference_corpus_differs": "Baseline несовместим: отличается версия корпуса. Укажите baseline для этого корпуса.",
        "reference_gpu_differs": "Baseline снят на другой GPU. Укажите baseline для текущей модели GPU.",
        "reference_environment_differs": "Baseline несовместим: среда или мониторинг отличаются. Используйте совпадающее окружение и настройки.",
        "reference_format_unsupported": "Неизвестный формат baseline. Поддерживаются полный v4 и текущий формат.",
        "reference_invalid": "Baseline не прочитан или не прошёл проверку целостности/полноты. Укажите полный неповреждённый архив.",
    }
    if result.get("diagnostic_code") in messages:
        print(messages[result["diagnostic_code"]], file=sys.stderr)
    compatibility = result.get("reference_compatibility")
    if isinstance(compatibility, dict) and compatibility.get("status") == "compatible":
        version = compatibility.get("reference_result_version")
        if type(version) is int and version in (4, 5):
            print(f"Baseline v{version}: совместим. Повторный запуск baseline не требуется.", file=sys.stderr)
        if compatibility.get("lifecycle_differs"):
            print("Отличие: baseline использовал свежие процессы для серий; candidate сохраняет модель. Эффект включает различие lifecycle.", file=sys.stderr)
    if type(result.get("model_loads")) is int and type(result.get("latency_stages", 0)) is int:
        print(f"План: загрузок моделей {result['model_loads']}; измерительных серий {result.get('latency_stages', 0)}.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("setup")
    p.add_argument("--environment", required=True)
    p.add_argument("--profile", choices=("team-t4", "team-rtx", "team-t4-laya", "team-rtx-quant"), default="team-t4")
    p = commands.add_parser("auth-check")
    p.add_argument("--model", default="shield")
    p.add_argument("--config", help="Check the explicitly configured HF sources without loading weights")
    p = commands.add_parser("fetch-data")
    p.add_argument("--source")
    p.add_argument("--revision")
    p.add_argument("--output")
    for name in ("preflight", "run", "plan"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--environment", required=True)
        if name != "plan":
            p.add_argument("--output", required=True)
        p.add_argument("--pilot", action="store_true")
        p.add_argument("--mode", choices=("baseline", "search", "confirm", "smoke"))
        p.add_argument("--baseline-ref", help="Complete v4 or current baseline directory/ZIP; detected automatically, search only")
        p.add_argument("--profile-trace", action="store_true", help="Optional separate short/long profiles")
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
    p = commands.add_parser("_worker_v4")
    p.add_argument("--job", required=True)
    p = commands.add_parser("_runtime_v4")
    p.add_argument("--environment", required=True)
    p.add_argument("--output", required=True)
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
    if args.command in ("plan", "run", "preflight", "quality", "report"):
        explain(result)
    print(json.dumps(public_result(args.command, result, args), ensure_ascii=False))
    return 0 if result["status"] in ("passed", "completed", "screening_pass", "pilot_complete", "ready",
                                    "smoke_passed", "preflight_passed", "baseline_complete", "quality_complete",
                                    "quantization_measured", "exploratory", "confirm_pass", "reference_pass") else 1


def execute(args, parser):
    if args.command == "setup":
        from u1_benchmark.orchestrator import setup
        result = setup(args.environment, args.profile)
    elif args.command == "auth-check":
        if args.config:
            from u1_benchmark.v4_evidence import auth_check
            result = auth_check(args.config)
        else:
            from u1_benchmark.environment import auth_check
            result = auth_check(args.model)
    elif args.command == "fetch-data":
        from u1_benchmark.distribution import fetch
        result = fetch(args.output, args.source, args.revision)
    elif args.command in ("run", "preflight", "quality", "plan"):
        from u1_benchmark.v4_config import resolve_options
        from u1_benchmark.v4_runner import run, plan
        if args.command == "preflight" and getattr(args, "mode", None) not in (None, "smoke"):
            raise ValueError("Preflight cannot select a measurement mode")
        mode = "quality" if args.command == "quality" else "smoke" if args.command == "preflight" else getattr(args, "mode", None)
        cfg = resolve_options(args.config, environment=args.environment, mode=mode,
                              baseline_ref=getattr(args, "baseline_ref", None),
                              pilot=getattr(args, "pilot", False), profile_trace=getattr(args, "profile_trace", False))
        result = plan(cfg) if args.command == "plan" else run(cfg, args.output, progress=explain)
    elif args.command == "prepare-data":
        from u1_benchmark.core import read
        from u1_benchmark.v4_data import prepare
        result = prepare(read(args.spec), args.output)
    elif args.command == "table":
        from u1_benchmark.v4_report import table
        result = table(args.inputs, args.output)
    elif args.command == "_worker_v4":
        from u1_benchmark.v4_worker import execute
        result = execute(args.job)
    elif args.command == "_runtime_v4":
        from u1_benchmark.v4_evidence import configure_gpu
        from u1_benchmark.core import write
        write(args.output, configure_gpu(args.environment))
        result = {"status": "passed"}
    elif args.command == "report":
        from u1_benchmark.v4_report import build
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
        raise ValueError("Unsupported command")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
