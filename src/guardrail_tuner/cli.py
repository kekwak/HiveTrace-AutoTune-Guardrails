"""Inspect model metadata and report unverified runtime candidates."""

import argparse
import json
from pathlib import Path
from typing import Any

from .datasets import DatasetSource, load_source


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
    elif any(name.endswith("ForCausalLM") for name in architectures):
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
    for role in ("quality", "performance"):
        group = datasets_parser.add_mutually_exclusive_group(required=True)
        group.add_argument(f"--{role}-hf", help="Hugging Face dataset ID")
        group.add_argument(f"--{role}-file", help="Local JSON, JSONL, CSV or Parquet file")
        datasets_parser.add_argument(f"--{role}-split", help="Hugging Face split (default: train)")
        datasets_parser.add_argument(f"--{role}-revision", help="Hugging Face revision")
        datasets_parser.add_argument(f"--{role}-config", help="Hugging Face dataset configuration")
    args = parser.parse_args()

    try:
        if args.command == "datasets":
            result = {}
            for role in ("quality", "performance"):
                hub_id = getattr(args, f"{role}_hf")
                file_path = getattr(args, f"{role}_file")
                split = getattr(args, f"{role}_split")
                revision = getattr(args, f"{role}_revision")
                config = getattr(args, f"{role}_config")
                if file_path and any(value is not None for value in (split, revision, config)):
                    raise ValueError(
                        f"--{role}-split, --{role}-revision and --{role}-config "
                        f"apply only to --{role}-hf"
                    )
                source = DatasetSource(
                    kind="hub" if hub_id else "file",
                    path=hub_id or file_path,
                    split=split,
                    revision=revision,
                    config=config,
                )
                _, result[role] = load_source(source)
        else:
            result = inspect_model(args.model, args.revision)
            if args.command == "plan":
                result = plan_checks(result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"guardrail-tuner: {exc}\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
