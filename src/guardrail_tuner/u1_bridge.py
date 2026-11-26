"""Checkout-only orchestration; model runtimes stay in their pinned environments."""

import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .jsonl_process import PersistentJSONLRunner
from .rules import select_rules


class CheckoutRequired(ValueError):
    pass


class PinnedEnvironmentMissing(ValueError):
    pass


def checkout() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "tools/benchmark.py").is_file() or not (root / "u1_benchmark").is_dir():
        raise CheckoutRequired(
            "U1 commands require a repository checkout; use its documented entry point"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def module(name: str) -> Any:
    checkout()
    return importlib.import_module("u1_benchmark." + name)


def make_plan(raw: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    if raw.get("candidate") is not None or raw.get("mode", "baseline") != "baseline":
        raise ValueError("plan --config requires a baseline-only U1 configuration")
    config = module("v4_config").validate_config(raw)
    family = config["baseline"]["adapter"]
    task = (
        {"hf_sequence_classification": "sequence_classification", "d3_decision": "system_one"}.get(
            family
        )
        if isinstance(family, str)
        else None
    )
    if task is None:
        return {
            "plan_version": 2,
            "status": "needs_inference_contract",
            "missing": ["team_rule_capabilities"],
            "config": config,
        }
    proposal = select_rules(
        metadata["rule_features"], {"task": task, "runtime": "transformers", "device": "cuda"}
    )
    candidate = deepcopy(config["baseline"])
    candidate["execution"]["rules"] = proposal["rules"]
    checked = module("v4_config").validate_config(
        {**config, "mode": "smoke", "candidate": candidate}
    )
    checked["mode"] = "search"
    return {
        "plan_version": 2,
        "status": "unverified",
        "config": checked,
        "rules": proposal,
        "comparison_design": "unpaired_reference",
        "quality_evaluated": False,
    }


def validate_plan(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("plan_version") != 2 or not isinstance(document.get("config"), dict):
        raise ValueError("eval requires a version 2 plan from plan --config")
    if document.get("status") == "needs_inference_contract":
        raise ValueError("Complete the native inference contract before eval")
    config = deepcopy(document["config"])
    if "candidate" not in config:
        raise ValueError("Plan must contain a complete candidate")
    config.pop("baseline_ref", None)
    config["mode"] = "smoke"
    config["pilot"] = False
    return module("v4_config").validate_config(config)


def read_document(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def write_document(path: str | Path, document: dict[str, Any]) -> None:
    with Path(path).expanduser().open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def execution(document: dict[str, Any], variant: str, base: Path | None = None) -> dict[str, Any]:
    if document.get("execution", {}).get("version") == 2:
        result = deepcopy(document["execution"])
    else:
        config = validate_plan(document)
        result = {
            "version": 2,
            "variant": config[variant],
            "environment": config["environment"],
            "batch_size": 1,
            "seed": config["seed"],
            "timeout_seconds": config["timeout_seconds"],
            "comparison": config["comparison"],
        }
    selects_baseline = document.get("plan_version") == 2 and variant == "baseline"
    if not selects_baseline and (
        document.get("status") == "parity_failed"
        or document.get("rules", {}).get("parity", {}).get("passed") is False
        or (
            document.get("evaluation", {}).get("verdict_status") in ("changed", "unstable")
            and result.get("comparison", "engineering") != "quantization"
        )
    ):
        raise ValueError("The saved configuration failed output agreement")
    if type(result.get("batch_size")) is not int or result["batch_size"] < 1:
        raise ValueError("Execution batch_size must be positive")
    if result.get("variant", {}).get("adapter") == "exported_sequence_classification":
        if base is None:
            raise ValueError("Artifact document location is required")
        for field in ("weights", "tokenizer"):
            source = result["variant"][field]
            path = (base / source["path"]).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError("Export artifact must stay inside its output directory")
            source["path"] = str(path)
        return result
    module("v4_config").variant(result["variant"])
    return result


def _python(environment: str) -> str:
    path = Path(environment).expanduser() / "venv/bin/python"
    if not path.is_file():
        raise PinnedEnvironmentMissing("Run benchmark setup for the configured environment profile")
    return str(path)


def offline_report(source: str | Path) -> dict[str, Any]:
    with module("v4_evidence").reference_folder(source) as root:
        if (root / "requested_plan.json").is_file():
            return module("cli_integration").report(root, save=False)
        return module("v4_report").build(root, save=False)


def evaluate(path: str, output: str, baseline_ref: str | None) -> dict[str, Any]:
    document = read_document(path)
    cfg = validate_plan(document)
    root = checkout()
    destination = Path(output).expanduser().resolve()
    module("process").check_archive_target(destination)
    if destination.exists():
        raise ValueError("Output directory already exists")
    command = [
        _python(cfg["environment"]),
        str(root / "tools/u1_cli_worker.py"),
        "eval",
        "--config",
        str(Path(path).expanduser().resolve()),
        "--output",
        str(destination),
    ]
    if baseline_ref:
        command += ["--baseline-ref", str(Path(baseline_ref).expanduser().resolve())]
    with subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=module("process").child_environment(),
    ) as process:
        try:
            process.wait(timeout=cfg["timeout_seconds"])
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    report = destination / "report.json"
    if not report.is_file():
        raise ValueError("Evaluation stopped before report; inspect the partial result directory")
    return read_document(str(report))


class NativeInference:
    def __init__(self, config: dict[str, Any]) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="u1-inference-")
        self.runner: PersistentJSONLRunner | None = None
        try:
            path = Path(self.temp.name) / "execution.json"
            write_document(path, config)
            os.chmod(path, 0o600)
            self.runner = PersistentJSONLRunner(
                [
                    _python(config["environment"]),
                    str(checkout() / "tools/u1_cli_worker.py"),
                    "infer",
                    "--config",
                    str(path),
                ],
                config["timeout_seconds"],
                env=module("process").child_environment(),
            )
            ready = self.runner.run({"operation": "load"})
            if ready.get("status") != "ready":
                raise ValueError(
                    "Native runtime initialization failed; verify its pinned environment"
                )
            self.metadata = ready.get("metadata", {})
        except BaseException:
            self.close()
            raise

    def predict(self, inputs: list[Any]) -> list[dict[str, Any]]:
        assert self.runner is not None
        result = self.runner.run({"operation": "predict", "inputs": inputs})
        if result.get("status") != "passed":
            self.close()
            raise ValueError("Native inference failed; no fallback was used")
        values = result.get("outputs")
        if not isinstance(values, list) or len(values) != len(inputs):
            self.close()
            raise ValueError("Native inference returned an invalid response count")
        self.metadata = result.get("metadata", self.metadata)
        return values

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
        self.temp.cleanup()


def run_native(config: dict[str, Any], inputs: list[Any]) -> dict[str, Any]:
    runtime = load_native(config)
    try:
        return {
            "outputs": runtime.predict(inputs),
            "metadata": runtime.metadata,
            "quality_evaluated": False,
        }
    finally:
        runtime.close()


def serve_native(config: dict[str, Any], port: int) -> None:
    runtime = load_native(config)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/infer" or not 0 < length <= 1_000_000:
                    raise ValueError
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) not in ({"inputs"}, {"messages"}):
                    raise ValueError
                inputs = body.get("inputs", [body.get("messages")])
                if not isinstance(inputs, list) or not inputs:
                    raise ValueError
                answer = {"outputs": runtime.predict(inputs)}
                status = 200
            except (ValueError, OSError, TypeError):
                status, answer = 400, {"error": "Invalid input or native inference failed"}
            payload = json.dumps(answer, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    try:
        with HTTPServer(("127.0.0.1", port), Handler) as server:
            print(f"Listening on http://127.0.0.1:{server.server_port}/infer", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()


def load_native(config: dict[str, Any]) -> Any:
    if config["variant"]["adapter"] == "exported_sequence_classification":
        from .exported_classifier import ExportedClassifier

        return ExportedClassifier(config)
    return NativeInference(config)
