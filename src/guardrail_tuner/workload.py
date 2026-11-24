"""Build deterministic token-length workload buckets and batch requests."""

from dataclasses import dataclass
from typing import Any

DEFAULT_BUCKETS = (128, 256, 512, 1024, 2048)
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)


@dataclass(frozen=True)
class Workload:
    buckets: tuple[int, ...] = DEFAULT_BUCKETS
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES

    def __post_init__(self) -> None:
        if (
            not self.buckets
            or any(v <= 0 for v in self.buckets)
            or tuple(sorted(set(self.buckets))) != self.buckets
        ):
            raise ValueError(
                "Bucket boundaries must be unique positive integers in ascending order"
            )
        if (
            not self.batch_sizes
            or any(v <= 0 for v in self.batch_sizes)
            or tuple(sorted(set(self.batch_sizes))) != self.batch_sizes
        ):
            raise ValueError("Batch sizes must be unique positive integers in ascending order")
        if self.batch_sizes[0] != 1:
            raise ValueError("Batch sizes must include 1 as the reference workload")

    def partition(
        self, texts: list[str], tokenizer: Any, max_length: int
    ) -> tuple[dict[str, list[str]], int]:
        result: dict[str, list[str]] = {}
        truncated = 0
        for text in texts:
            encoded = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            length = len(encoded)
            truncated += int(length > max_length)
            effective_length = min(length, max_length)
            ceiling = next((x for x in self.buckets if effective_length <= x), None)
            key = str(ceiling) if ceiling is not None else f">{self.buckets[-1]}"
            result.setdefault(key, []).append(text)
        return result, truncated


def batches(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def recommend_batches(
    results: list[dict[str, Any]], minimum_improvement: float
) -> list[dict[str, Any]]:
    """Choose only candidates that beat the same bucket's single-item reference in every round."""
    recommendations = []
    for bucket in dict.fromkeys(row["bucket"] for row in results):
        rows = [row for row in results if row["bucket"] == bucket]
        baseline = next((row for row in rows if row["batch_size"] == 1), None)
        qualified = []
        if baseline and baseline.get("selection", {}).get("status") == "selected":
            reference = baseline["latency"]["reference_a"]
            for row in rows:
                if row.get("selection", {}).get("selected") != "candidate":
                    continue
                candidate = row["latency"]["candidate"]
                if len(candidate) == len(reference) and all(
                    c["p99_ms"] <= r["p99_ms"] * (1 - minimum_improvement)
                    for r, c in zip(reference, candidate)
                ):
                    qualified.append(row)
        best = (
            min(
                qualified, key=lambda row: sum(run["p99_ms"] for run in row["latency"]["candidate"])
            )
            if qualified
            else None
        )
        recommendations.append(
            {
                "bucket": bucket,
                "selected": "candidate" if best else "reference",
                "batch_size": best["batch_size"] if best else 1,
                "status": "selected" if best else "reference_or_preliminary",
            }
        )
    return recommendations
