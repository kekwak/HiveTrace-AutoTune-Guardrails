"""Profile opt-in JSONL batch protocol with user-supplied tokenizer."""

from typing import Any, Callable

from .measurement import MIN_IMPROVEMENT, aa_selection, measure_aa_ab
from .workload import Workload, recommend_batches


def profile_application(
    reference_a: Any,
    reference_b: Any,
    candidate: Any,
    quality: Any,
    performance: Any,
    tokenizer: Any,
    text_column: str,
    workload: Workload,
    warmup: int,
    matches: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    def group(records: Any) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            text = record.get(text_column)
            if not isinstance(text, str):
                raise ValueError("Workload text column must contain strings")
            length = len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
            ceiling = next((x for x in workload.buckets if length <= x), None)
            name = str(ceiling) if ceiling is not None else f">{workload.buckets[-1]}"
            grouped.setdefault(name, []).append(record)
        return grouped

    def batches(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
        return [rows[i : i + size] for i in range(0, len(rows), size)]

    class BatchRunner:
        def __init__(self, runner: Any) -> None:
            self.runner = runner

        def run(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            response = self.runner.run({"inputs": rows})
            outputs = response.get("outputs")
            if (
                not isinstance(outputs, list)
                or len(outputs) != len(rows)
                or not all(isinstance(v, dict) for v in outputs)
            ):
                raise ValueError("Invalid JSONL batch response")
            return outputs

    a, b, c = BatchRunner(reference_a), BatchRunner(reference_b), BatchRunner(candidate)
    qgroups, pgroups = group(quality), group(performance)
    results = []
    for bucket, records in pgroups.items():
        for size in workload.batch_sizes:
            row: dict[str, Any] = {"bucket": bucket, "batch_size": size, "samples": len(records)}
            qsamples = qgroups.get(bucket, [])
            if len(qsamples) < size:
                row.update(status="unverified", reason="insufficient_quality_samples_for_batch")
            else:
                try:
                    for chunk in batches(qsamples, size):
                        baseline, other, optimized = a.run(chunk), b.run(chunk), c.run(chunk)
                        for item, left, aa, ab in zip(chunk, baseline, other, optimized):
                            single = reference_a.run(item)
                            if not (
                                matches(left, aa) and matches(left, ab) and matches(left, single)
                            ):
                                raise ValueError("batch_output_mismatch")
                    latency = measure_aa_ab(a, b, c, batches(records, size), warmup)
                    decision = aa_selection(latency, True, True)
                    row.update(
                        status=decision["status"],
                        quality={"passed": True, "samples": len(qsamples)},
                        latency=latency,
                        selection=decision,
                    )
                except (RuntimeError, ValueError) as exc:
                    row.update(
                        status="failed",
                        reason="batch_output_mismatch"
                        if str(exc) == "batch_output_mismatch"
                        else type(exc).__name__,
                    )
            results.append(row)
    return {
        "bucket_boundaries_tokens": workload.buckets,
        "batch_sizes": workload.batch_sizes,
        "length_boundary": (
            "tokenizer on input text only; application prompt and truncation not included"
        ),
        "batch_protocol": "inputs-outputs",
        "results": results,
        "recommendations": recommend_batches(results, MIN_IMPROVEMENT),
    }
