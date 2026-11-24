"""Tune the standardized text-to-logits forward call of a Transformers model."""

import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from .benchmark import texts
from .datasets import DatasetSource, load_source
from .manifest import record_model, record_token_inputs, record_workload
from .measurement import MIN_IMPROVEMENT, aa_selection, measure_aa_ab
from .workload import Workload, batches, recommend_batches


class TorchForwardRunner:
    """Text-to-logits runner; this capability is intentionally narrower than all models."""

    def __init__(self, model: Any, tokenizer: Any, device: str, max_length: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def run(self, text: Any) -> Any:
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=isinstance(text, list) and len(text) > 1,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor) or not torch.isfinite(logits).all():
            raise ValueError("Model must return finite logits")
        if self.device == "cuda":
            torch.cuda.synchronize()
        return logits.detach().cpu()


class CandidateRuntimeFailure(Exception):
    """A candidate runtime failed without exposing inputs or exception messages."""


class CandidateRunner:
    def __init__(self, runner: TorchForwardRunner) -> None:
        self.runner = runner

    def run(self, text: Any) -> Any:
        try:
            return self.runner.run(text)
        except (RuntimeError, OSError, ValueError) as exc:
            raise CandidateRuntimeFailure(type(exc).__name__) from exc


def tune_model(
    model_id: str,
    revision: str | None,
    device: str,
    quality: DatasetSource,
    performance: DatasetSource,
    text_column: str,
    max_length: int,
    warmup: int,
    tolerance: float,
    output_dir: str,
    workload: Workload | None = None,
    quality_column: str | None = None,
    performance_column: str | None = None,
) -> dict[str, Any]:
    import torch
    import transformers
    import yaml
    from huggingface_hub import snapshot_download

    if max_length < 1 or warmup < 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid max length, warmup or tolerance")
    workload = workload or Workload()
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable to PyTorch")
    destination = Path(output_dir).expanduser()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory must be empty or not exist")
    qdata, qmeta = load_source(quality)
    pdata, pmeta = load_source(performance)
    qtexts = texts(qdata, quality_column or text_column, "quality")
    ptexts = texts(pdata, performance_column or text_column, "performance")
    local = Path(model_id).expanduser()
    if local.is_dir():
        if revision:
            raise ValueError("--revision applies only to Hugging Face models")
        model_path, resolved = str(local.resolve()), None
    else:
        model_path = snapshot_download(repo_id=model_id, revision=revision)
        resolved = Path(model_path).name
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    architectures = getattr(config, "architectures", None) or []
    if len(architectures) != 1:
        raise ValueError("Model must declare exactly one Transformers architecture")
    cls = getattr(transformers, architectures[0], None)
    if (
        cls is None
        or not isinstance(cls, type)
        or not issubclass(cls, transformers.PreTrainedModel)
    ):
        raise ValueError("Declared architecture is not available in installed Transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    record_token_inputs("quality", tokenizer, qtexts, max_length)
    record_token_inputs("performance", tokenizer, ptexts, max_length)
    performance_buckets, truncated = workload.partition(ptexts, tokenizer, max_length)
    quality_buckets, quality_truncated = workload.partition(qtexts, tokenizer, max_length)
    record_workload("quality", quality_buckets, workload.batch_sizes)
    record_workload("performance", performance_buckets, workload.batch_sizes)
    model = cls.from_pretrained(model_path, trust_remote_code=False).to(device).eval()
    record_model(model_id, resolved, model, tokenizer)
    second_model = cls.from_pretrained(model_path, trust_remote_code=False).to(device).eval()

    # Fail early for unsupported model interfaces rather than silently optimizing a base model.
    reference = TorchForwardRunner(model, tokenizer, device, max_length)
    reference_b = TorchForwardRunner(second_model, tokenizer, device, max_length)
    reference.run(qtexts[0])
    aa_error = 0.0
    aa_shape_mismatch = False
    for text in qtexts:
        left, right = reference.run(text), reference_b.run(text)
        if left.shape != right.shape:
            aa_shape_mismatch = True
            break
        aa_error = max(aa_error, float((left.float() - right.float()).abs().max()))
    aa_passed = not aa_shape_mismatch and aa_error <= tolerance
    max_error = 0.0
    compile_error = None
    latency: dict[str, Any] = {"reference_a": [], "reference_b": [], "candidate": []}
    compile_seconds = None
    passed = False
    start = time.perf_counter()
    compiled_model = None
    if aa_passed:
        try:
            compiled_model = torch.compile(model)
        except (RuntimeError, OSError) as exc:
            compile_error = type(exc).__name__
    if compiled_model is not None:
        compiled = CandidateRunner(
            TorchForwardRunner(compiled_model, tokenizer, device, max_length)
        )
        try:
            for text in qtexts:
                ref = reference.run(text)
                candidate = compiled.run(text)
                if ref.shape != candidate.shape:
                    compile_error = "output_shape_mismatch"
                    break
                max_error = max(max_error, float((ref.float() - candidate.float()).abs().max()))
            compile_seconds = time.perf_counter() - start
            passed = compile_error is None and max_error <= tolerance
            if passed:
                latency = measure_aa_ab(reference, reference_b, compiled, ptexts, warmup)
        except CandidateRuntimeFailure as exc:
            compile_error = str(exc)
            passed = False
    decision = aa_selection(latency, aa_passed, passed)
    selected = "torch_compile" if decision["selected"] == "candidate" else "torch_eager"
    workload_results: list[dict[str, Any]] = []
    if aa_passed and passed:
        for bucket, bucket_texts in performance_buckets.items():
            for batch_size in workload.batch_sizes:
                bucket_quality = quality_buckets.get(bucket, [])
                if not bucket_quality:
                    workload_results.append(
                        {
                            "bucket": bucket,
                            "batch_size": batch_size,
                            "status": "unverified",
                            "reason": "no_quality_samples_in_bucket",
                        }
                    )
                    continue
                if len(bucket_quality) < batch_size:
                    workload_results.append(
                        {
                            "bucket": bucket,
                            "batch_size": batch_size,
                            "status": "unverified",
                            "reason": "insufficient_quality_samples_for_batch",
                        }
                    )
                    continue
                if batch_size > 1 and tokenizer.pad_token_id is None:
                    workload_results.append(
                        {
                            "bucket": bucket,
                            "batch_size": batch_size,
                            "status": "unsupported",
                            "reason": "no_padding_token",
                        }
                    )
                    continue
                requests = batches(bucket_texts, batch_size)
                quality_metrics = {
                    "aa_max_logit_error": 0.0,
                    "ab_max_logit_error": 0.0,
                    "batch_max_logit_error": 0.0,
                    "samples": len(bucket_quality),
                }
                try:
                    for qbatch in batches(bucket_quality, batch_size):
                        left, other, optimized = (
                            reference.run(qbatch),
                            reference_b.run(qbatch),
                            compiled.run(qbatch),
                        )
                        if left.shape != other.shape or left.shape != optimized.shape:
                            raise ValueError("output_shape_mismatch")
                        aa_diff = float((left.float() - other.float()).abs().max())
                        ab_diff = float((left.float() - optimized.float()).abs().max())
                        quality_metrics["aa_max_logit_error"] = max(
                            quality_metrics["aa_max_logit_error"], aa_diff
                        )
                        quality_metrics["ab_max_logit_error"] = max(
                            quality_metrics["ab_max_logit_error"], ab_diff
                        )
                        if aa_diff > tolerance or ab_diff > tolerance:
                            raise ValueError("output_parity_failed")
                        if len(qbatch) > 1:
                            encoded = tokenizer(
                                qbatch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=max_length,
                            )
                            for i, text in enumerate(qbatch):
                                single = reference.run(text)[0]
                                row = left[i]
                                if (
                                    row.ndim == 2
                                    and row.shape[0] == encoded["attention_mask"].shape[1]
                                ):
                                    row = row[encoded["attention_mask"][i].bool()]
                                if row.shape != single.shape:
                                    raise ValueError("batch_changes_output")
                                batch_diff = float((row.float() - single.float()).abs().max())
                                quality_metrics["batch_max_logit_error"] = max(
                                    quality_metrics["batch_max_logit_error"], batch_diff
                                )
                                if batch_diff > tolerance:
                                    raise ValueError("batch_changes_output")
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    batch_latency = measure_aa_ab(
                        reference, reference_b, compiled, requests, warmup
                    )
                    peak_memory = torch.cuda.max_memory_allocated() if device == "cuda" else None
                    batch_decision = aa_selection(batch_latency, True, True)
                    workload_results.append(
                        {
                            "bucket": bucket,
                            "batch_size": batch_size,
                            "samples": len(bucket_texts),
                            "quality": {**quality_metrics, "passed": True},
                            "latency": batch_latency,
                            "process_peak_cuda_bytes": peak_memory,
                            "selection": batch_decision,
                            "status": batch_decision["status"],
                        }
                    )
                except (CandidateRuntimeFailure, RuntimeError, ValueError) as exc:
                    reason = (
                        str(exc)
                        if isinstance(exc, ValueError)
                        and str(exc)
                        in {
                            "output_shape_mismatch",
                            "output_parity_failed",
                            "batch_changes_output",
                        }
                        else type(exc).__name__
                    )
                    workload_results.append(
                        {
                            "bucket": bucket,
                            "batch_size": batch_size,
                            "status": "failed",
                            "reason": reason,
                            "quality": {**quality_metrics, "passed": False},
                        }
                    )
                    if device == "cuda":
                        torch.cuda.empty_cache()
    batch_one = [row for row in workload_results if row["batch_size"] == 1]
    recommendations = recommend_batches(workload_results, MIN_IMPROVEMENT)
    for recommendation in recommendations:
        recommendation["runtime"] = (
            "torch_compile" if recommendation["selected"] == "candidate" else "torch_eager"
        )
    if selected == "torch_compile" and (
        len(batch_one) != len(performance_buckets)
        or any(row.get("selection", {}).get("selected") != "candidate" for row in batch_one)
    ):
        selected = "torch_eager"
        decision = {
            **decision,
            "selected": "reference",
            "status": "preliminary",
            "reason": "per_bucket_validation_incomplete",
        }
    result = {
        "status": decision["status"],
        "selected": selected,
        "selection": decision,
        "scope": "model_forward_logits",
        "model": model_id,
        "model_revision": resolved,
        "architecture": architectures[0],
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "datasets": {"quality": qmeta, "performance": pmeta},
        "settings": {
            "text_column": text_column,
            "quality_text_column": quality_column or text_column,
            "performance_text_column": performance_column or text_column,
            "max_length": max_length,
            "warmup": warmup,
            "max_abs_logit_error": tolerance,
            "batch_size": 1,
            "measurement": "sequential, includes tokenization and output transfer",
        },
        "quality": {
            "samples": len(qtexts),
            "passed": passed and aa_passed,
            "max_abs_logit_error": max_error if compile_error is None else None,
            "aa": {
                "passed": aa_passed,
                "max_abs_logit_error": aa_error if not aa_shape_mismatch else None,
            },
        },
        "latency": latency,
        "workload": {
            "bucket_boundaries_tokens": workload.buckets,
            "batch_sizes": workload.batch_sizes,
            "truncated_inputs": truncated,
            "quality_truncated_inputs": quality_truncated,
            "results": workload_results,
            "recommendations": recommendations,
            "selection": "global selection requires every batch-1 bucket to pass",
        },
        "compile_seconds_including_quality_checks": compile_seconds,
        "candidate_error": compile_error,
        "limitations": [
            "Text input and logits only",
            "No generation or cache validation",
            "Application outputs and quality not validated",
            "ONNX and TensorRT not tested",
            "AA uses two model instances in one process and requires additional memory",
            "Selection rules are heuristics and do not establish statistical significance",
            "CUDA peak memory is process-wide across loaded model instances",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    config_out = {
        key: result[key]
        for key in (
            "model",
            "model_revision",
            "architecture",
            "device",
            "selected",
            "selection",
            "scope",
            "settings",
            "workload",
        )
    }
    (destination / "config.yaml").write_text(
        yaml.safe_dump(config_out, sort_keys=False), encoding="utf-8"
    )
    (destination / "report.md").write_text(
        "# Model forward tuning\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return result
