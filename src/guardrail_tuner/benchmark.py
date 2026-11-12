"""Compare sequence-classification inference using PyTorch and ONNX Runtime."""

import math
import platform
import statistics
import time
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from typing import Any

from .datasets import DatasetSource, load_source
from .workload import Workload


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
    ort_optimization: str = "all",
    ort_threads: int = 0,
    label_column: str | None = None,
    rigorous: bool = False,
    ort_provider: str = "auto",
    workload: Workload | None = None,
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
    if ort_provider not in ("auto", "tensorrt"):
        raise ValueError("Unknown ONNX Runtime provider")
    if ort_provider == "tensorrt" and device != "cuda":
        raise ValueError("TensorRT execution provider requires CUDA")
    provider = (
        "TensorrtExecutionProvider"
        if ort_provider == "tensorrt"
        else "CUDAExecutionProvider"
        if device == "cuda"
        else "CPUExecutionProvider"
    )
    if provider not in ort.get_available_providers():
        raise ValueError(f"ONNX Runtime provider {provider} is unavailable")

    quality, quality_meta = load_source(quality_source)
    performance, performance_meta = load_source(performance_source)
    quality_text = texts(quality, quality_text_column, "quality")
    performance_text = texts(performance, performance_text_column, "performance")
    labels = None
    if label_column is not None:
        if output_mode == "logits":
            raise ValueError("--quality-label-column requires multiclass or multilabel output")
        if label_column not in quality.column_names:
            raise ValueError(f"quality dataset is missing label column {label_column!r}")
        labels = list(quality[label_column])

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
    second_model = (
        AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
        if rigorous or workload is not None
        else None
    )
    session_options = ort.SessionOptions()
    levels = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    }
    session_options.graph_optimization_level = levels[ort_optimization]
    session_options.intra_op_num_threads = ort_threads
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path, export=True, provider=provider, session_options=session_options
    )
    actual_providers = ort_model.model.get_providers()
    if actual_providers[0] != provider:
        raise ValueError(f"ONNX Runtime did not activate {provider}: {actual_providers}")

    def predict(text: Any, backend: str) -> Any:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=isinstance(text, list) and len(text) > 1,
        )
        if backend in ("pytorch", "pytorch2"):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                active_model = second_model if backend == "pytorch2" else model
                if active_model is None:
                    raise ValueError("Second reference was not initialized")
                logits = active_model(**inputs).logits
        else:
            with torch.inference_mode():
                logits = ort_model(**inputs).logits
        if device == "cuda":
            torch.cuda.synchronize()
        return logits.detach().cpu().numpy()

    max_error = 0.0
    max_probability_error = 0.0
    aa_error = 0.0
    aa_passed = True
    disagreements = 0
    label_metrics: dict[str, Any] | None = None
    for row_index, text in enumerate(quality_text):
        reference = predict(text, "pytorch")
        if rigorous:
            aa_reference = predict(text, "pytorch2")
            if reference.shape != aa_reference.shape or not np.isfinite(aa_reference).all():
                aa_passed = False
            else:
                aa_error = max(aa_error, float(np.max(np.abs(reference - aa_reference))))
                if output_mode == "multiclass":
                    aa_passed = aa_passed and bool(
                        np.array_equal(
                            np.argmax(reference, axis=-1), np.argmax(aa_reference, axis=-1)
                        )
                    )
                elif output_mode == "multilabel":
                    ref_probs = 1 / (1 + np.exp(-np.clip(reference, -80, 80)))
                    aa_probs = 1 / (1 + np.exp(-np.clip(aa_reference, -80, 80)))
                    aa_passed = aa_passed and bool(
                        np.array_equal(ref_probs >= threshold, aa_probs >= threshold)
                    )
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
            ref_decisions = [int(np.argmax(reference))]
            ort_decisions = [int(np.argmax(candidate))]
            disagreements += int(ref_decisions != ort_decisions)
        elif output_mode == "multilabel":
            ref_probs = 1 / (1 + np.exp(-np.clip(reference, -80, 80)))
            ort_probs = 1 / (1 + np.exp(-np.clip(candidate, -80, 80)))
            max_probability_error = max(
                max_probability_error, float(np.max(np.abs(ref_probs - ort_probs)))
            )
            ref_decisions = (ref_probs >= threshold).astype(int).flatten().tolist()
            ort_decisions = (ort_probs >= threshold).astype(int).flatten().tolist()
            disagreements += int(ref_decisions != ort_decisions)

        if labels is not None:
            label = labels[row_index]
            if output_mode == "multiclass":
                if isinstance(label, bool) or not isinstance(label, Integral):
                    raise ValueError("Multiclass labels must be integer class indices")
                if label < 0 or label >= reference.shape[-1]:
                    raise ValueError("Multiclass label is outside the model output range")
                expected = [int(label)]
            else:
                if not isinstance(label, (list, tuple)) or len(label) != len(ref_decisions):
                    raise ValueError("Multilabel labels must have one binary value per output")
                if any(not isinstance(value, Integral) or value not in (0, 1) for value in label):
                    raise ValueError("Multilabel labels must contain only 0 or 1")
                expected = [int(value) for value in label]
            if label_metrics is None:
                label_metrics = {
                    "samples": 0,
                    "pytorch_correct": 0,
                    "onnxruntime_correct": 0,
                    "per_class": [
                        {
                            "class_index": i,
                            "pytorch": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                            "onnxruntime": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                        }
                        for i in range(reference.shape[-1])
                    ],
                }
            label_metrics["samples"] += 1
            label_metrics["pytorch_correct"] += int(ref_decisions == expected)
            label_metrics["onnxruntime_correct"] += int(ort_decisions == expected)
            for index, counts in enumerate(label_metrics["per_class"]):
                actual = (
                    expected[0] == index if output_mode == "multiclass" else bool(expected[index])
                )
                for backend, decisions in (
                    ("pytorch", ref_decisions),
                    ("onnxruntime", ort_decisions),
                ):
                    predicted = (
                        decisions[0] == index
                        if output_mode == "multiclass"
                        else bool(decisions[index])
                    )
                    if actual:
                        key = "tp" if predicted else "fn"
                    else:
                        key = "fp" if predicted else "tn"
                    counts[backend][key] += 1

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
    aa_passed = aa_passed and aa_error <= tolerance
    rigorous_result = None
    if rigorous and parity_passed and aa_passed:
        from .measurement import aa_selection, measure_aa_ab

        class PredictionRunner:
            def __init__(self, backend: str) -> None:
                self.backend = backend

            def run(self, text: str) -> Any:
                return predict(text, self.backend)

        latency_rounds = measure_aa_ab(
            PredictionRunner("pytorch"),
            PredictionRunner("pytorch2"),
            PredictionRunner("onnxruntime"),
            performance_text,
            warmup,
        )
        rigorous_result = {
            "latency": latency_rounds,
            "selection": aa_selection(latency_rounds, aa_passed, parity_passed),
        }
    workload_result = None
    if workload is not None and parity_passed and aa_passed:
        from .classification_workload import profile_classification

        workload_result = profile_classification(
            predict,
            tokenizer,
            quality_text,
            performance_text,
            workload,
            max_length,
            warmup,
            tolerance,
            output_mode,
            threshold,
        )
    return {
        "model": model_id,
        "model_revision": resolved_revision,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "providers": actual_providers,
        "provider_note": (
            "TensorRT EP is enabled; node-level TensorRT coverage is not verified"
            if ort_provider == "tensorrt"
            else None
        ),
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
            "quality_label_column": label_column,
            "performance_text_column": performance_text_column,
            "max_length": max_length,
            "warmup": warmup,
            "ort_optimization": ort_optimization,
            "ort_intra_op_threads": ort_threads,
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
            "aa_passed": aa_passed if rigorous else None,
            "aa_max_abs_logit_error": aa_error if rigorous else None,
            "labels": label_metrics,
        },
        "latency": {"pytorch": pytorch_latency, "onnxruntime": onnx_latency},
        "rigorous": rigorous_result,
        "workload": workload_result,
    }
