"""Inspect model metadata and report unverified runtime candidates."""

import argparse
import json
from pathlib import Path
from typing import Any

from .application import compare_applications
from .benchmark import run_benchmark
from .datasets import DatasetSource, load_source
from .generation import check_generation
from .model_tuning import tune_model
from .optimize import optimize
from .vllm_probe import probe_vllm
from .workload import DEFAULT_BATCH_SIZES, DEFAULT_BUCKETS, Workload


def read_model(model: str, revision: str | None) -> tuple[Path, str | None]:
    path = Path(model).expanduser()
    if path.is_dir():
        if revision is not None:
            raise ValueError("--revision applies only to Hugging Face models")
        return path / "config.json", None

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from huggingface_hub.utils import HFValidationError
    except ImportError as exc:
        raise ValueError("Install guardrail-tuner[hub] to inspect a Hugging Face model") from exc

    try:
        config = Path(hf_hub_download(model, "config.json", revision=revision))
    except (HfHubHTTPError, LocalEntryNotFoundError, HFValidationError) as exc:
        raise ValueError(f"Could not read config.json for {model}: {exc}") from exc
    return config, config.parent.name


def inspect_model(model: str, revision: str | None) -> dict[str, Any]:
    path, resolved_revision = read_model(model, revision)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    architectures = config.get("architectures")
    if architectures is None:
        architectures = []
    if not isinstance(architectures, list) or not all(
        isinstance(name, str) for name in architectures
    ):
        raise ValueError("architectures must be a list of strings")

    return {
        "model": model,
        "resolved_revision": resolved_revision,
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "id2label": config.get("id2label"),
        "problem_type": config.get("problem_type"),
        "custom_code_declared": bool(config.get("auto_map")),
    }


def plan_checks(metadata: dict[str, Any]) -> dict[str, Any]:
    architectures = metadata["architectures"]
    candidates = []
    mode = None

    if any(name.endswith("ForSequenceClassification") for name in architectures):
        mode = "sequence_classification"
        candidates.append(
            {
                "runtime": "onnxruntime",
                "mode": "sequence_classification",
                "next_check": "Export or load ONNX and compare logits against Transformers.",
            }
        )
        candidates.append(
            {
                "runtime": "vllm",
                "mode": "pooling_classify",
                "next_check": "Verify checkpoint support and compare the required outputs.",
            }
        )
    elif any(name.endswith(("ForCausalLM", "ForConditionalGeneration")) for name in architectures):
        mode = "generation"
        candidates.append(
            {
                "runtime": "vllm",
                "mode": "generation",
                "next_check": (
                    "Probe the model with the application's prompt and validate its output."
                ),
            }
        )

    if mode is not None:
        candidates.extend(
            [
                {
                    "runtime": "torch_compile",
                    "mode": mode,
                    "next_check": (
                        "Compile the model forward and compare outputs against eager PyTorch."
                    ),
                },
                {
                    "runtime": "tensorrt",
                    "mode": mode,
                    "next_check": (
                        "Verify ONNX export and CUDA support, then compare TensorRT outputs "
                        "against eager PyTorch."
                    ),
                },
            ]
        )

    return {
        **metadata,
        "status": "unverified" if candidates else "needs_inference_contract",
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="guardrail-tuner", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("inspect", "Read model metadata"),
        ("plan", "Suggest runtime checks based on model metadata"),
    ):
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument("model", help="Hugging Face model ID or local model directory")
        subparser.add_argument("--revision", help="Hugging Face branch, tag or commit")
    datasets_parser = commands.add_parser(
        "datasets", help="Load and inspect quality and performance datasets"
    )
    model_parser = commands.add_parser("tune-model", help="Tune text-to-logits model forward")
    model_parser.add_argument("model")
    model_parser.add_argument("--revision")
    model_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    model_parser.add_argument("--text-column", default="text")
    model_parser.add_argument("--max-length", type=int, default=512)
    model_parser.add_argument("--buckets", nargs="+", type=int, help="Token-length upper bounds")
    model_parser.add_argument("--batch-sizes", nargs="+", type=int, help="Batch sizes to test")
    model_parser.add_argument("--warmup", type=int, default=5)
    model_parser.add_argument("--max-logit-error", type=float, default=0.001)
    model_parser.add_argument("--output-dir", required=True)
    generation_parser = commands.add_parser(
        "check-generation", help="Compare cached and uncached greedy generation"
    )
    generation_parser.add_argument("model")
    generation_parser.add_argument("--revision")
    generation_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    generation_parser.add_argument("--text-column", default="text")
    generation_parser.add_argument("--max-length", type=int, default=512)
    generation_parser.add_argument("--max-new-tokens", type=int, default=8)
    generation_parser.add_argument("--max-score-error", type=float, default=0.001)
    generation_parser.add_argument("--compile", action="store_true")
    add_dataset_options(generation_parser, "quality")
    vllm_parser = commands.add_parser("probe-vllm", help="Launch vLLM and compare greedy tokens")
    vllm_parser.add_argument("model")
    vllm_parser.add_argument("--revision")
    vllm_parser.add_argument("--text-column", default="text")
    vllm_parser.add_argument("--max-length", type=int, default=512)
    vllm_parser.add_argument("--max-new-tokens", type=int, default=8)
    vllm_parser.add_argument(
        "--task", choices=("generation", "classification"), default="generation"
    )
    vllm_parser.add_argument(
        "--output-mode", choices=("multiclass", "multilabel"), default="multiclass"
    )
    vllm_parser.add_argument("--threshold", type=float, default=0.5)
    vllm_parser.add_argument("--max-score-error", type=float, default=0.001)
    vllm_parser.add_argument("--timeout", type=float, default=600)
    add_dataset_options(vllm_parser, "quality")
    application_parser = commands.add_parser(
        "compare-app", help="Compare JSONL application commands"
    )
    application_parser.add_argument("--reference-command", required=True, help="JSON argv array")
    application_parser.add_argument("--candidate-command", required=True, help="JSON argv array")
    application_parser.add_argument(
        "--model", required=True, help="Model identity reported by user"
    )
    application_parser.add_argument(
        "--model-revision", required=True, help="Model revision or digest"
    )
    application_parser.add_argument("--reference-runtime", required=True)
    application_parser.add_argument("--candidate-runtime", required=True)
    application_parser.add_argument("--exact-field", action="append", default=[])
    application_parser.add_argument("--numeric-field", action="append", default=[])
    application_parser.add_argument("--warmup", type=int, default=1)
    application_parser.add_argument("--timeout", type=float, default=60)
    application_parser.add_argument(
        "--batch-tokenizer", help="Enable inputs/outputs JSONL batch protocol"
    )
    application_parser.add_argument("--batch-tokenizer-revision")
    application_parser.add_argument("--batch-text-column", default="text")
    application_parser.add_argument("--buckets", nargs="+", type=int)
    application_parser.add_argument("--batch-sizes", nargs="+", type=int)
    application_parser.add_argument(
        "--process-mode", choices=("persistent", "fresh"), default="persistent"
    )
    application_parser.add_argument("--output-dir", required=True)
    for role in ("quality", "performance"):
        add_dataset_options(model_parser, role)
        add_dataset_options(application_parser, role)
    for name in ("benchmark", "optimize"):
        run_parser = commands.add_parser(
            name, help="Compare PyTorch and ONNX Runtime sequence classification"
        )
        run_parser.add_argument("model", help="Hugging Face model ID or local model directory")
        run_parser.add_argument("--revision", help="Hugging Face model revision")
        run_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
        run_parser.add_argument("--ort-provider", choices=("auto", "tensorrt"), default="auto")
        run_parser.add_argument(
            "--output-mode", choices=("multiclass", "multilabel", "logits"), required=True
        )
        run_parser.add_argument("--threshold", type=float, default=0.5)
        run_parser.add_argument("--text-column", default="text")
        run_parser.add_argument("--quality-text-column")
        run_parser.add_argument("--quality-label-column")
        run_parser.add_argument("--performance-text-column")
        run_parser.add_argument("--max-length", type=int, default=512)
        run_parser.add_argument("--buckets", nargs="+", type=int)
        run_parser.add_argument("--batch-sizes", nargs="+", type=int)
        run_parser.add_argument("--warmup", type=int, default=5)
        run_parser.add_argument("--max-logit-error", type=float, default=0.001)
        if name == "optimize":
            run_parser.add_argument("--output-dir", required=True)
        for role in ("quality", "performance"):
            add_dataset_options(run_parser, role)
    for role in ("quality", "performance"):
        add_dataset_options(datasets_parser, role)
    args = parser.parse_args()

    try:
        if args.command == "probe-vllm":
            result = probe_vllm(
                args.model,
                args.revision,
                dataset_source(args, "quality"),
                args.text_column,
                args.max_length,
                args.max_new_tokens,
                args.timeout,
                args.task,
                args.output_mode,
                args.threshold,
                args.max_score_error,
            )
        elif args.command == "check-generation":
            result = check_generation(
                args.model,
                args.revision,
                args.device,
                dataset_source(args, "quality"),
                args.text_column,
                args.max_length,
                args.max_new_tokens,
                args.max_score_error,
                args.compile,
            )
        elif args.command in ("datasets", "benchmark", "optimize", "tune-model", "compare-app"):
            sources = {role: dataset_source(args, role) for role in ("quality", "performance")}
            if args.command == "datasets":
                result = {role: load_source(source)[1] for role, source in sources.items()}
            elif args.command == "tune-model":
                result = tune_model(
                    args.model,
                    args.revision,
                    args.device,
                    sources["quality"],
                    sources["performance"],
                    args.text_column,
                    args.max_length,
                    args.warmup,
                    args.max_logit_error,
                    args.output_dir,
                    Workload(
                        tuple(args.buckets or DEFAULT_BUCKETS),
                        tuple(args.batch_sizes or DEFAULT_BATCH_SIZES),
                    ),
                )
            elif args.command == "compare-app":
                if not args.batch_tokenizer and (
                    args.buckets or args.batch_sizes or args.batch_tokenizer_revision
                ):
                    raise ValueError("--batch-tokenizer is required for JSONL buckets and batches")
                result = compare_applications(
                    args.reference_command,
                    args.candidate_command,
                    sources["quality"],
                    sources["performance"],
                    args.exact_field,
                    args.numeric_field,
                    args.warmup,
                    args.timeout,
                    args.output_dir,
                    {
                        "model": args.model,
                        "model_revision": args.model_revision,
                        "reference_runtime": args.reference_runtime,
                        "candidate_runtime": args.candidate_runtime,
                    },
                    args.process_mode,
                    Workload(
                        tuple(args.buckets or DEFAULT_BUCKETS),
                        tuple(args.batch_sizes or DEFAULT_BATCH_SIZES),
                    )
                    if args.batch_tokenizer
                    else None,
                    args.batch_tokenizer,
                    args.batch_tokenizer_revision,
                    args.batch_text_column,
                )
            else:
                import math

                if args.max_length < 1 or args.warmup < 0:
                    raise ValueError("--max-length must be positive; --warmup must be non-negative")
                if args.ort_provider == "tensorrt" and args.device != "cuda":
                    raise ValueError("TensorRT execution provider requires CUDA")
                if not math.isfinite(args.max_logit_error) or args.max_logit_error < 0:
                    raise ValueError("--max-logit-error must be finite and non-negative")
                if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
                    raise ValueError("--threshold must be finite and between 0 and 1")
                if args.quality_label_column and args.output_mode == "logits":
                    raise ValueError(
                        "--quality-label-column requires multiclass or multilabel output"
                    )
                runner = optimize if args.command == "optimize" else run_benchmark
                options = {
                    "label_column": args.quality_label_column,
                    "ort_provider": args.ort_provider,
                    "workload": Workload(
                        tuple(args.buckets or DEFAULT_BUCKETS),
                        tuple(args.batch_sizes or DEFAULT_BATCH_SIZES),
                    ),
                }
                if args.command == "optimize":
                    options["output_dir"] = args.output_dir
                result = runner(
                    args.model,
                    args.revision,
                    args.device,
                    sources["quality"],
                    sources["performance"],
                    args.quality_text_column or args.text_column,
                    args.performance_text_column or args.text_column,
                    args.max_length,
                    args.warmup,
                    args.max_logit_error,
                    args.output_mode,
                    args.threshold,
                    **options,
                )
        else:
            result = inspect_model(args.model, args.revision)
            if args.command == "plan":
                result = plan_checks(result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"guardrail-tuner: {exc}\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.command == "optimize" and result["status"] != "selected":
        return 1
    if args.command == "benchmark" and not result["quality"]["passed"]:
        return 1
    if args.command == "compare-app" and not result["quality"]["passed"]:
        return 1
    if args.command == "check-generation" and (
        not result["cache"]["passed"] or (args.compile and not result["candidate"]["passed"])
    ):
        return 1
    if args.command == "probe-vllm" and result["status"] not in ("tokens_match", "scores_match"):
        return 1
    return 0


def add_dataset_options(parser: argparse.ArgumentParser, role: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(f"--{role}-hf", help="Hugging Face dataset ID")
    group.add_argument(f"--{role}-file", help="Local JSON, JSONL, CSV or Parquet file")
    parser.add_argument(f"--{role}-split", help="Hugging Face split (default: train)")
    parser.add_argument(f"--{role}-revision", help="Hugging Face revision")
    parser.add_argument(f"--{role}-config", help="Hugging Face dataset configuration")


def dataset_source(args: argparse.Namespace, role: str) -> DatasetSource:
    hub_id = getattr(args, f"{role}_hf")
    file_path = getattr(args, f"{role}_file")
    split = getattr(args, f"{role}_split")
    revision = getattr(args, f"{role}_revision")
    config = getattr(args, f"{role}_config")
    if file_path and any(value is not None for value in (split, revision, config)):
        raise ValueError(
            f"--{role}-split, --{role}-revision and --{role}-config apply only to --{role}-hf"
        )
    return DatasetSource(
        kind="hub" if hub_id else "file",
        path=hub_id or file_path,
        split=split,
        revision=revision,
        config=config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
