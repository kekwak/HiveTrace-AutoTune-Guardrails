"""One CLI evaluation delegates every measurement to the existing U1 runner."""

import json
import time
from copy import deepcopy
from pathlib import Path

from .core import digest, read, write
from .process import check_archive_target, finish_archive
from .v4_privacy import public_config


def report(directory, *, save=True):
    from .v4_report import build

    root = Path(directory)
    state = read(root / "status.json")
    result = {
        "report_version": 1,
        "status": state["status"],
        "quality_evaluated": False,
        "comparison_design": "unpaired_reference",
        "complete": False,
        "baseline_reused": state["baseline_reused"],
        "stages": state["stages"],
        "requested_config_sha256": state["requested_config_sha256"],
    }
    for name in ("baseline", "candidate", "quality"):
        target = root / name
        if (target / "status.json").is_file():
            result[name] = build(target, save=False)
    candidate = result.get("candidate", {})
    if candidate:
        result.update(
            status=candidate["status"],
            complete=candidate["complete"],
            cells=candidate["cells"],
            verdict_status=candidate["verdict_status"],
            speed_status=candidate["speed_status"],
            memory_status=candidate["memory_status"],
        )
    result["rule_evidence"] = []
    for path in sorted(root.glob("*/jobs/*/result.json")):
        saved = read(path)
        backend = saved.get("backend_proof")
        proof = backend.get("team_rules") if isinstance(backend, dict) else None
        if not isinstance(proof, dict):
            continue
        job = read(path.parent / "job.json")
        result["rule_evidence"].append(
            {
                **{
                    key: job[key]
                    for key in ("variant", "bucket", "batch_size", "stage")
                    if key in job
                },
                **{
                    key: proof[key]
                    for key in (
                        "requested",
                        "applied",
                        "already_active",
                        "skipped",
                        "compile_count",
                    )
                    if key in proof
                },
            }
        )
    quality = result.get("quality", {}).get("quality", {})
    if result.get("quality") and not result["quality"].get("complete"):
        result.update(complete=False, status="incomplete")
    result["quality_evaluated"] = quality.get("quality_evaluated", False)
    if quality:
        result["team_quality"] = team_quality(root / "quality", quality)
    if state["status"] in ("failed", "timeout", "interrupted"):
        result.update(status=state["status"], complete=False, error_type=state.get("error_type"))
    if save:
        write(root / "report.json", result)
        lines = [
            "# CLI evaluation",
            "",
            "Status: " + result["status"],
            "Comparison: unpaired reference; no independent replication.",
            "Quality evaluated: " + str(result["quality_evaluated"]),
            "",
            "Cell | Status | Baseline p95 ms | Candidate p95 ms",
            "--- | --- | --- | ---",
        ]
        for key, value in result.get("cells", {}).items():
            p95 = value.get("p95", {})
            lines.append(
                f"{key} | {value['status']} | {p95.get('baseline_ms', '-')} | "
                f"{p95.get('candidate_ms', '-')}"
            )
        (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


def team_quality(directory, summary):
    """Adapt verified public predictions, never reopen the labeled source texts."""
    from guardrail_tuner.quality import Prediction, QualityRow, compare_predictions

    resolved = read(directory / "resolved_run.json")
    mapping = resolved["config"]["label_mapping"]
    if set(mapping["labels"]) != {"safe", "unsafe"}:
        return {"status": "not_applicable", "reason": "explicit_safe_unsafe_mapping_required"}
    rows = [
        QualityRow(r["id"], "", mapping["targets"][r["target"]], r.get("slices", {}))
        for r in resolved["quality_rows"]
    ]
    pairs = {}
    for path in sorted((directory / "jobs").glob("*/result.json")):
        value = read(path)
        job = value.get("job", read(path.parent / "job.json"))
        if job["stage"] != "quality" or value.get("status") != "passed":
            continue
        predictions = [
            json.loads(line)
            for line in (path.parent / "predictions.jsonl").read_text().splitlines()
        ]
        converted = []
        for item in predictions:
            passed = item["status"] == "passed"
            converted.append(
                Prediction(
                    item["id"],
                    "ok"
                    if passed
                    else "over_limit"
                    if item["status"] == "over_limit"
                    else "inference_error",
                    mapping["verdicts"][item["score"]["label"]] if passed else None,
                )
            )
        pairs.setdefault(str(job["quality_batch_size"]), {})[job["variant"]] = converted
    results = {}
    for size, sides in pairs.items():
        comparison = summary["quality_by_batch_size"][size].get("baseline_to_candidate", {})
        if set(sides) != {"baseline", "candidate"} or comparison.get("status", "").startswith(
            "blocked_"
        ):
            results[size] = {"status": "unavailable"}
        else:
            results[size] = compare_predictions(rows, sides["baseline"], sides["candidate"])
    return {"status": "evaluated", "by_batch_size": results, "quality_preserved": None}


def evaluate(document, output, baseline_ref=None, *, runner=None):
    from .v4_config import validate_config
    from .v4_evidence import immutable
    from .v4_runner import run

    runner = runner or run
    root = Path(output).resolve()
    check_archive_target(root)
    root.mkdir(parents=True, exist_ok=False)
    cfg = deepcopy(document["config"])
    cfg.pop("baseline_ref", None)
    cfg["pilot"] = False
    state = {
        "status": "running",
        "baseline_reused": bool(baseline_ref),
        "stages": [],
        "requested_config_sha256": digest(cfg),
    }
    write(root / "status.json", state)
    immutable(root / "requested_plan.json", {**document, "config": public_config(cfg)})
    deadline = time.monotonic() + cfg["timeout_seconds"]

    def stage(name, config):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Evaluation deadline exhausted")
        config["timeout_seconds"] = remaining
        value = runner(validate_config(config), root / name)
        state["stages"].append({"name": name, "status": value["status"]})
        write(root / "status.json", state)
        return value

    try:
        if baseline_ref is None:
            baseline = deepcopy(cfg)
            baseline.pop("candidate", None)
            baseline["mode"] = "baseline"
            if stage("baseline", baseline)["status"] != "baseline_complete":
                raise ValueError("Baseline did not complete; candidate was not started")
            baseline_ref = root / "baseline"
        candidate = {**cfg, "mode": "search", "baseline_ref": str(baseline_ref)}
        stage("candidate", candidate)
        if cfg.get("quality_data"):
            quality = {**cfg, "mode": "quality"}
            quality.pop("data", None)
            stage("quality", quality)
        state["status"] = "completed"
    except BaseException as error:
        state.update(
            status="timeout"
            if isinstance(error, TimeoutError)
            else "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "failed",
            error_type=type(error).__name__,
        )
    finally:
        write(root / "status.json", state)
        try:
            result = report(root)
        except Exception as error:
            result = {**state, "report_error_type": type(error).__name__, "complete": False}
            write(root / "report.json", result)
        private = root / ".private"
        private.mkdir(mode=0o700, exist_ok=True)
        write(
            private / "evaluated_plan.json",
            {
                **document,
                "evaluation": {
                    "verdict_status": result.get("verdict_status", "incomplete"),
                    "status": result["status"],
                },
            },
        )
        finish_archive(root, state, root / "status.json")
    return result


def infer_session(config, stream, output, directory):
    import json

    from .precision import PrecisionGuard
    from .v4_adapters import load_adapter
    from .v4_evidence import configure_gpu
    from .v4_worker import responses, safe_metadata

    adapter = None
    proven_shapes = set()
    try:
        for line in stream:
            request = json.loads(line)
            if request.get("operation") == "load":
                if adapter is not None:
                    raise ValueError("Runtime already loaded")
                configure_gpu(config["environment"])
                adapter = load_adapter(config["variant"], directory)
                guard = PrecisionGuard(adapter, required=config["variant"]["adapter"] == "laya")
                check_device = getattr(adapter, "check_device", guard.check)
                guard.call(check_device)
                value = {"status": "ready", "metadata": safe_metadata(adapter)}
            elif request.get("operation") == "predict" and adapter is not None:
                inputs = request.get("inputs")
                if not isinstance(inputs, list) or not inputs:
                    raise ValueError("Expected nonempty input list")
                batches = [
                    inputs[i : i + config["batch_size"]]
                    for i in range(0, len(inputs), config["batch_size"])
                ]
                guard.call(adapter.prepare_batch_backend, batches)
                answers = []
                for batch in batches:
                    guard.call(check_device)
                    indices = [guard.call(adapter.index, item) for item in batch]
                    shape = (len(batch), max(item["total_tokens"] for item in indices))
                    if shape not in proven_shapes:
                        guard.call(adapter.prove_batch_backend, batch)
                        proven_shapes.add(shape)
                    answers.extend(
                        responses(guard.call(adapter.verify_batch, batch, indices), len(batch))
                    )
                    guard.call(check_device)
                value = {"status": "passed", "outputs": answers, "metadata": safe_metadata(adapter)}
            else:
                raise ValueError("Unexpected native worker operation")
            output.write(json.dumps(value, allow_nan=False) + "\n")
            output.flush()
    except BaseException as error:
        output.write(json.dumps({"status": "failed", "error_type": type(error).__name__}) + "\n")
        output.flush()
    finally:
        if adapter is not None:
            adapter.close()
