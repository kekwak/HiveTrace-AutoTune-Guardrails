"""Select a verified inference configuration and write a reproducible report."""

import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark
from .datasets import DatasetSource
from .workload import Workload


def optimize(
    model: str,
    revision: str | None,
    device: str,
    quality: DatasetSource,
    performance: DatasetSource,
    quality_column: str,
    performance_column: str,
    max_length: int,
    warmup: int,
    tolerance: float,
    output_mode: str,
    threshold: float,
    output_dir: str,
    label_column: str | None = None,
    ort_provider: str = "auto",
    workload: Workload | None = None,
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("Install the inference extra to write YAML results") from exc

    destination = Path(output_dir).expanduser()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory must be empty or not exist")
    destination.mkdir(parents=True, exist_ok=True)

    settings = [
        (level, threads)
        for level in ("all", "extended")
        for threads in ((0, 1) if device == "cpu" else (0,))
    ]
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for level, threads in settings:
        try:
            result = run_benchmark(
                model,
                revision,
                device,
                quality,
                performance,
                quality_column,
                performance_column,
                max_length,
                warmup,
                tolerance,
                output_mode,
                threshold,
                ort_optimization=level,
                ort_threads=threads,
                label_column=label_column,
                rigorous=True,
                ort_provider=ort_provider,
                workload=workload,
            )
        except Exception as exc:
            failures.append(
                {"optimization": level, "threads": threads, "error": type(exc).__name__}
            )
            continue

        if first is None:
            first = result
            if result["model_revision"]:
                revision = result["model_revision"]
            for role in ("quality", "performance"):
                source = quality if role == "quality" else performance
                if source.kind == "hub":
                    source = replace(source, revision=result["datasets"][role]["resolved_revision"])
                    if role == "quality":
                        quality = source
                    else:
                        performance = source
        else:
            for role in ("quality", "performance"):
                key = (
                    "sha256"
                    if result["datasets"][role]["source"] == "file"
                    else "resolved_revision"
                )
                if result["datasets"][role][key] != first["datasets"][role][key]:
                    raise ValueError(f"{role} dataset changed during optimization")
        runs.append(result)

    passing = [
        run
        for run in runs
        if run["quality"]["passed"]
        and run.get("rigorous")
        and run["rigorous"]["selection"]["selected"] == "candidate"
        and (
            workload is None
            or (
                run.get("workload")
                and all(
                    row.get("selection", {}).get("selected") == "candidate"
                    for row in run["workload"]["results"]
                    if row["batch_size"] == 1
                )
            )
        )
    ]
    status = "no_valid_candidate"
    chosen: dict[str, Any] | None = None
    if first is not None:
        baseline = first["latency"]["pytorch"]["p95_ms"]
        best = (
            min(
                passing,
                key=lambda run: sum(x["p99_ms"] for x in run["rigorous"]["latency"]["candidate"]),
            )
            if passing
            else None
        )
        if best is None:
            chosen = {"runtime": "pytorch", "p95_ms": baseline}
        else:
            chosen = {
                "runtime": "onnxruntime",
                "p95_ms": best["latency"]["onnxruntime"]["p95_ms"],
                "p99_ms_median": statistics.median(
                    run["p99_ms"] for run in best["rigorous"]["latency"]["candidate"]
                ),
                "graph_optimization": best["settings"]["ort_optimization"],
                "intra_op_threads": best["settings"]["ort_intra_op_threads"],
                "providers": best["providers"],
            }
        status = (
            "selected"
            if any(
                run.get("rigorous") and run["rigorous"]["selection"]["status"] == "selected"
                for run in runs
            )
            else "preliminary"
        )

    lines = [
        "# Inference optimization report",
        "",
        f"Status: {status}",
        "",
        "Measurements: sequential requests, batch size 1; p95 includes tokenization "
        "and output transfer.",
        "Quality: output agreement with PyTorch on the supplied dataset."
        + (" Label-based exact-match counts are included in details." if label_column else ""),
        "Selection requires AA and AB checks and repeated p99 improvement.",
        "",
        "## Candidates",
        "",
        "| Runtime | Graph optimization | Intra-op threads | p95 (ms) | Quality |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    if first is not None:
        lines.append(
            f"| PyTorch | — | — | {first['latency']['pytorch']['p95_ms']:.3f} | reference |"
        )
    for run in runs:
        lines.append(
            f"| ONNX Runtime | {run['settings']['ort_optimization']} | "
            f"{run['settings']['ort_intra_op_threads']} | "
            f"{run['latency']['onnxruntime']['p95_ms']:.3f} | "
            f"{'pass' if run['quality']['passed'] else 'fail'} |"
        )
    for failure in failures:
        lines.append(
            f"| ONNX Runtime | {failure['optimization']} | {failure['threads']} | — | "
            f"error: {failure['error']} |"
        )
    lines.extend(
        [
            "",
            "## Details",
            "",
            "```json",
            json.dumps(
                {"selected": chosen, "runs": runs, "failures": failures},
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
        ]
    )
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")

    if chosen is not None and first is not None:
        config = {
            "model": first["model"],
            "model_revision": first["model_revision"],
            "device": device,
            "selected": chosen,
            "settings": first["settings"],
            "datasets": first["datasets"],
            "versions": first["versions"],
        }
        profiled = best if best is not None else first
        if profiled.get("workload"):
            config["workload"] = {
                key: profiled["workload"][key]
                for key in ("bucket_boundaries_tokens", "batch_sizes", "recommendations")
            }
        if chosen["runtime"] == "onnxruntime":
            config["settings"] = {
                **first["settings"],
                "ort_optimization": chosen["graph_optimization"],
                "ort_intra_op_threads": chosen["intra_op_threads"],
            }
        else:
            config["settings"] = {
                key: value
                for key, value in first["settings"].items()
                if key not in ("ort_optimization", "ort_intra_op_threads")
            }
        config = json.loads(json.dumps(config))
        (destination / "config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    return {"status": status, "selected": chosen, "output_dir": str(destination)}
