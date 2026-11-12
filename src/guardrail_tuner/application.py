"""Compare application outputs through a deliberately small JSONL protocol."""

import json
import math
import platform
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .datasets import DatasetSource, load_source
from .jsonl_process import PersistentJSONLRunner
from .measurement import aa_selection, measure_aa_ab
from .workload import Workload


def _command(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Commands must be JSON arrays of strings") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ValueError("Commands must be non-empty JSON arrays of non-empty strings")
    return command


def _run(command: list[str], row: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            input=json.dumps(row, allow_nan=False) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Runner failed: {type(exc).__name__}") from exc
    if proc.returncode:
        raise ValueError(f"Runner exited with status {proc.returncode}")
    try:
        lines = proc.stdout.splitlines()
        if len(lines) != 1:
            raise ValueError("Runner must return exactly one JSON line")
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Runner returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Runner response must be a JSON object")
    return value


def _field(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Runner response is missing field {path!r}")
        current = current[part]
    return current


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON recursively; booleans are distinct from JSON numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


@dataclass
class CommandRunner:
    command: list[str]
    timeout: float

    def run(self, row: dict[str, Any]) -> dict[str, Any]:
        return _run(self.command, row, self.timeout)


def compare_applications(
    reference_command: str,
    candidate_command: str,
    quality: DatasetSource,
    performance: DatasetSource,
    exact_fields: list[str],
    numeric_fields: list[str],
    warmup: int,
    timeout: float,
    output_dir: str,
    provenance: dict[str, str] | None = None,
    process_mode: str = "fresh",
    workload: Workload | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    text_column: str = "text",
) -> dict[str, Any]:
    """Check two independent references and compare the candidate to the reference."""
    import yaml

    reference_argv, candidate_argv = _command(reference_command), _command(candidate_command)
    if process_mode not in ("persistent", "fresh"):
        raise ValueError("Unknown process mode")
    if workload is not None and not tokenizer_id:
        raise ValueError("A tokenizer is required for JSONL batch workload")
    if warmup < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Warmup must be non-negative and timeout must be positive and finite")
    numeric: dict[str, float] = {}
    for spec in numeric_fields:
        try:
            field, tolerance = spec.rsplit(":", 1)
            number = float(tolerance)
        except (ValueError, TypeError) as exc:
            raise ValueError("Numeric fields must have the form field.path:tolerance") from exc
        if not field or not math.isfinite(number) or number < 0 or field in numeric:
            raise ValueError("Numeric fields must be unique with finite non-negative tolerances")
        numeric[field] = number
    if not exact_fields and not numeric:
        raise ValueError("Specify at least one --exact-field or --numeric-field")
    if any(
        not path or any(not part for part in path.split(".")) for path in [*exact_fields, *numeric]
    ):
        raise ValueError("Field paths must contain non-empty dot-separated names")
    destination = Path(output_dir).expanduser()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory must be empty or not exist")
    qdata, qmeta = load_source(quality)
    pdata, pmeta = load_source(performance)
    if not len(qdata) or not len(pdata):
        raise ValueError("Both datasets must contain at least one row")
    comparisons: dict[str, Any] = {
        name: {
            "exact_disagreements": dict.fromkeys(exact_fields, 0),
            "max_absolute_errors": dict.fromkeys(numeric, 0.0),
        }
        for name in ("aa", "ab")
    }
    with ExitStack() as stack:

        def runner(command: list[str]) -> Any:
            if process_mode == "fresh":
                return CommandRunner(command, timeout)
            return stack.enter_context(PersistentJSONLRunner(command, timeout))

        reference_a, reference_b, candidate = (
            runner(reference_argv),
            runner(reference_argv),
            runner(candidate_argv),
        )
        for row in qdata:
            ref_a, ref_b, cand = reference_a.run(row), reference_b.run(row), candidate.run(row)
            for name, other in (("aa", ref_b), ("ab", cand)):
                differences = comparisons[name]["exact_disagreements"]
                errors = comparisons[name]["max_absolute_errors"]
                for field in exact_fields:
                    differences[field] += int(
                        not _json_equal(_field(ref_a, field), _field(other, field))
                    )
                for field in numeric:
                    left, right = _field(ref_a, field), _field(other, field)
                    if (
                        isinstance(left, bool)
                        or isinstance(right, bool)
                        or not isinstance(left, (float, int))
                        or not isinstance(right, (float, int))
                        or not math.isfinite(left)
                        or not math.isfinite(right)
                    ):
                        raise ValueError(f"Numeric field {field!r} must contain finite numbers")
                    errors[field] = max(errors[field], abs(left - right))
        for comparison in comparisons.values():
            comparison["passed"] = not any(comparison["exact_disagreements"].values()) and all(
                comparison["max_absolute_errors"][k] <= numeric[k] for k in numeric
            )
        aa_passed, passed = comparisons["aa"]["passed"], comparisons["ab"]["passed"]
        latency = (
            measure_aa_ab(reference_a, reference_b, candidate, pdata, warmup)
            if aa_passed and passed
            else {"reference_a": [], "reference_b": [], "candidate": []}
        )
        workload_result = None
        if workload is not None and aa_passed and passed:
            from transformers import AutoTokenizer

            from .application_workload import profile_application

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id, revision=tokenizer_revision, trust_remote_code=False
            )

            def matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
                if any(
                    not _json_equal(_field(left, key), _field(right, key)) for key in exact_fields
                ):
                    return False
                for key, tolerance in numeric.items():
                    lv, rv = _field(left, key), _field(right, key)
                    if (
                        isinstance(lv, bool)
                        or isinstance(rv, bool)
                        or not isinstance(lv, (float, int))
                        or not isinstance(rv, (float, int))
                        or not math.isfinite(lv)
                        or not math.isfinite(rv)
                        or abs(lv - rv) > tolerance
                    ):
                        return False
                return True

            workload_result = profile_application(
                reference_a,
                reference_b,
                candidate,
                qdata,
                pdata,
                tokenizer,
                text_column,
                workload,
                warmup,
                matches,
            )
            workload_result["tokenizer"] = {
                "id": tokenizer_id,
                "requested_revision": tokenizer_revision,
                "resolved_revision": tokenizer.init_kwargs.get("_commit_hash"),
            }
    decision = aa_selection(latency, aa_passed, passed)
    if (
        workload is not None
        and decision["selected"] == "candidate"
        and (
            not workload_result
            or any(
                row.get("selection", {}).get("selected") != "candidate"
                for row in workload_result["results"]
                if row["batch_size"] == 1
            )
        )
    ):
        decision.update(
            selected="reference", status="preliminary", reason="per_bucket_validation_incomplete"
        )
    selected = decision["selected"]
    result = {
        "status": decision["status"],
        "selected": selected,
        "selection": decision,
        "scope": "application_jsonl",
        "provenance_supplied_by_user": provenance,
        "quality": {
            "passed": passed and aa_passed,
            "samples": len(qdata),
            "exact_disagreements": comparisons["ab"]["exact_disagreements"],
            "max_absolute_errors": comparisons["ab"]["max_absolute_errors"],
            "numeric_tolerances": numeric,
            "aa": comparisons["aa"],
        },
        "latency": latency,
        "workload": workload_result,
        "datasets": {"quality": qmeta, "performance": pmeta},
        "settings": {
            "warmup": warmup,
            "timeout_s": timeout,
            "batch_size": 1,
            "process_mode": process_mode,
            "measurement": (
                "sequential, persistent process, includes JSONL IO, excludes startup"
                if process_mode == "persistent"
                else "sequential, fresh process per request, includes startup and JSONL IO"
            ),
        },
        "environment": {"cli_python": platform.python_version(), "platform": platform.platform()},
        "limitations": [
            "Only selected output fields were compared",
            "Model and backend identity are supplied by the caller and not verified",
            "Runner dependency versions and device settings are not detected",
            "Selection rules are heuristics and do not establish statistical significance",
        ]
        + (
            ["Warmup uses separate processes and does not warm measured processes"]
            if process_mode == "fresh"
            else []
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "selected": selected,
                "scope": "application_jsonl",
                "selection": decision,
                "settings": result["settings"],
                "provenance_supplied_by_user": provenance,
                "workload": workload_result,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (destination / "report.md").write_text(
        "# Application comparison\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return result
