"""Measure classification workloads without assuming a particular runtime implementation."""

from typing import Any, Callable

from .measurement import MIN_IMPROVEMENT, aa_selection, measure_aa_ab
from .workload import Workload, batches, recommend_batches


def profile_classification(
    predict: Callable[[Any, str], Any],
    tokenizer: Any,
    quality: list[str],
    performance: list[str],
    workload: Workload,
    max_length: int,
    warmup: int,
    tolerance: float,
    output_mode: str,
    threshold: float,
) -> dict[str, Any]:
    import numpy as np

    quality_buckets, quality_truncated = workload.partition(quality, tokenizer, max_length)
    performance_buckets, performance_truncated = workload.partition(
        performance, tokenizer, max_length
    )
    results = []

    class Runner:
        def __init__(self, backend: str) -> None:
            self.backend = backend

        def run(self, texts: Any) -> Any:
            return predict(texts, self.backend)

    def decisions(values: Any) -> Any:
        if output_mode == "multiclass":
            return np.argmax(values, axis=-1)
        if output_mode == "multilabel":
            return 1 / (1 + np.exp(-np.clip(values, -80, 80))) >= threshold
        return None

    for bucket, samples in performance_buckets.items():
        for size in workload.batch_sizes:
            row: dict[str, Any] = {"bucket": bucket, "batch_size": size, "samples": len(samples)}
            qsamples = quality_buckets.get(bucket, [])
            if len(qsamples) < size:
                row.update(status="unverified", reason="insufficient_quality_samples_for_batch")
            elif size > 1 and tokenizer.pad_token_id is None:
                row.update(status="unsupported", reason="no_padding_token")
            else:
                try:
                    max_error = 0.0
                    for qbatch in batches(qsamples, size):
                        a = predict(qbatch, "pytorch")
                        b = predict(qbatch, "pytorch2")
                        candidate = predict(qbatch, "onnxruntime")
                        individual = np.concatenate([predict(text, "pytorch") for text in qbatch])
                        if not all(
                            v.shape == a.shape and np.isfinite(v).all()
                            for v in (a, b, candidate, individual)
                        ):
                            raise ValueError("invalid_outputs")
                        for other in (b, candidate, individual):
                            max_error = max(max_error, float(np.max(np.abs(a - other))))
                            if output_mode != "logits" and not np.array_equal(
                                decisions(a), decisions(other)
                            ):
                                raise ValueError("decision_mismatch")
                        if max_error > tolerance:
                            raise ValueError("output_parity_failed")
                    latency = measure_aa_ab(
                        Runner("pytorch"),
                        Runner("pytorch2"),
                        Runner("onnxruntime"),
                        batches(samples, size),
                        warmup,
                    )
                    decision = aa_selection(latency, True, True)
                    row.update(
                        status=decision["status"],
                        quality={
                            "passed": True,
                            "max_abs_logit_error": max_error,
                            "samples": len(qsamples),
                        },
                        latency=latency,
                        selection=decision,
                    )
                except (ValueError, RuntimeError) as exc:
                    allowed = {"invalid_outputs", "decision_mismatch", "output_parity_failed"}
                    row.update(
                        status="failed",
                        reason=str(exc) if str(exc) in allowed else type(exc).__name__,
                    )
            results.append(row)
    return {
        "bucket_boundaries_tokens": workload.buckets,
        "batch_sizes": workload.batch_sizes,
        "quality_truncated_inputs": quality_truncated,
        "truncated_inputs": performance_truncated,
        "results": results,
        "recommendations": recommend_batches(results, MIN_IMPROVEMENT),
    }
