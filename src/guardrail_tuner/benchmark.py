"""Compare sequence-classification inference using PyTorch and ONNX Runtime."""

import math
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .datasets import DatasetSource, load_source


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def texts(dataset: Any, column: str, name: str) -> list[str]:
    if column not in dataset.column_names:
        raise ValueError(f"{name} dataset is missing text column {column!r}")
    values = list(dataset[column])
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{name} text column must contain non-empty rows of strings")
    return values


def run_benchmark(
    model_id: str,
    revision: str | None,
    device: str,
    quality_source: DatasetSource,
    performance_source: DatasetSource,
    quality_text_column: str,
    performance_text_column: str,
    max_length: int,
    warmup: int,
    tolerance: float,
    output_mode: str,
    threshold: float,
) -> dict[str, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from huggingface_hub import snapshot_download
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "Install inference and the appropriate inference-cpu or inference-cuda extra"
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable to PyTorch")
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        raise ValueError(f"ONNX Runtime provider {provider} is unavailable")

    quality, quality_meta = load_source(quality_source)
    performance, performance_meta = load_source(performance_source)
    quality_text = texts(quality, quality_text_column, "quality")
    performance_text = texts(performance, performance_text_column, "performance")

    local_model = Path(model_id).expanduser()
    if local_model.is_dir():
        if revision is not None:
            raise ValueError("--revision applies only to Hugging Face models")
        model_path = str(local_model.resolve())
        resolved_revision = None
    else:
        model_path = snapshot_download(repo_id=model_id, revision=revision)
        resolved_revision = Path(model_path).name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path, export=True, provider=provider
    )
    actual_providers = ort_model.model.get_providers()
    if actual_providers[0] != provider:
        raise ValueError(f"ONNX Runtime did not activate {provider}: {actual_providers}")

    def predict(text: str, backend: str) -> Any:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        if backend == "pytorch":
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
        else:
            with torch.inference_mode():
                logits = ort_model(**inputs).logits
        if device == "cuda":
            torch.cuda.synchronize()
        return logits.detach().cpu().numpy()

    max_error = 0.0
    max_probability_error = 0.0
    disagreements = 0
    for text in quality_text:
        reference = predict(text, "pytorch")
        candidate = predict(text, "onnxruntime")
        if reference.shape != candidate.shape or not (
            np.isfinite(reference).all() and np.isfinite(candidate).all()
        ):
            raise ValueError("Backend outputs have different shapes or contain non-finite values")
        max_error = max(max_error, float(np.max(np.abs(reference - candidate))))
        if output_mode == "multiclass":
            if reference.shape[-1] < 2:
                raise ValueError("multiclass mode requires at least two logits")
            ref_exp = np.exp(reference - np.max(reference, axis=-1, keepdims=True))
            ort_exp = np.exp(candidate - np.max(candidate, axis=-1, keepdims=True))
            ref_probs = ref_exp / np.sum(ref_exp, axis=-1, keepdims=True)
            ort_probs = ort_exp / np.sum(ort_exp, axis=-1, keepdims=True)
            max_probability_error = max(
                max_probability_error, float(np.max(np.abs(ref_probs - ort_probs)))
            )
            disagreements += int(np.argmax(reference) != np.argmax(candidate))
        elif output_mode == "multilabel":
            ref_probs = 1 / (1 + np.exp(-np.clip(reference, -80, 80)))
            ort_probs = 1 / (1 + np.exp(-np.clip(candidate, -80, 80)))
            max_probability_error = max(
                max_probability_error, float(np.max(np.abs(ref_probs - ort_probs)))
            )
            disagreements += int(np.any((ref_probs >= threshold) != (ort_probs >= threshold)))

    def measure(backend: str) -> dict[str, Any]:
        for i in range(warmup):
            predict(performance_text[i % len(performance_text)], backend)
        durations = []
        for text in performance_text:
            start = time.perf_counter()
            predict(text, backend)
            durations.append((time.perf_counter() - start) * 1000)
        return {
            "requests": len(durations),
            "p50_ms": statistics.median(durations),
            "p95_ms": percentile(durations, 0.95),
            "p99_ms": percentile(durations, 0.99),
        }

    pytorch_latency = measure("pytorch")
    onnx_latency = measure("onnxruntime")
    parity_passed = max_error <= tolerance and disagreements == 0
    return {
        "model": model_id,
        "model_revision": resolved_revision,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "providers": actual_providers,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "transformers": version("transformers"),
            "optimum-onnx": version("optimum-onnx"),
            "cuda": torch.version.cuda,
        },
        "datasets": {"quality": quality_meta, "performance": performance_meta},
        "settings": {
            "quality_text_column": quality_text_column,
            "performance_text_column": performance_text_column,
            "max_length": max_length,
            "warmup": warmup,
            "max_logit_error": tolerance,
            "output_mode": output_mode,
            "threshold": threshold if output_mode == "multilabel" else None,
            "batch_size": 1,
            "measurement": "sequential, includes tokenization and output transfer",
        },
        "quality": {
            "samples": len(quality_text),
            "max_abs_logit_error": max_error,
            "max_abs_probability_error": max_probability_error if output_mode != "logits" else None,
            "decision_disagreements": disagreements if output_mode != "logits" else None,
            "passed": parity_passed,
        },
        "latency": {"pytorch": pytorch_latency, "onnxruntime": onnx_latency},
    }
