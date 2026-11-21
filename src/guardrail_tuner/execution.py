"""Execute fixed, architecture-based optimization plans."""

import json
import math
import platform
import statistics
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .benchmark import texts
from .datasets import DatasetSource, load_source
from .manifest import ordered_digest, tree_digest
from .model_tuning import TorchForwardRunner

OPTIMIZATIONS = {"sequence_classification": "batching", "generation": "kv_cache"}


def execution_plan(metadata: dict[str, Any], mode: str | None) -> dict[str, Any] | None:
    if mode is None or metadata["custom_code_declared"] or len(metadata["architectures"]) != 1:
        return None
    model = Path(metadata["model"]).expanduser()
    return {
        "version": 1,
        "model": str(model.resolve()) if model.is_dir() else metadata["model"],
        "revision": metadata["resolved_revision"],
        "architecture": metadata["architectures"][0],
        "runtime": "transformers",
        "task": mode,
        "optimization": OPTIMIZATIONS[mode],
        "device": "cpu",
        "max_length": 128,
        "max_new_tokens": 8,
        "batch_size": 4 if mode == "sequence_classification" else 1,
        "warmup": 2,
        "tolerance": 0.001,
    }


def read_execution(path: str) -> dict[str, Any]:
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON plan")
    if document.get("status") == "parity_failed":
        raise ValueError("This run failed output parity")
    config = document.get("execution")
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("No executable plan")
    task = config.get("task")
    if (
        not isinstance(task, str)
        or task not in OPTIMIZATIONS
        or config.get("optimization") != OPTIMIZATIONS[task]
        or config.get("runtime") != "transformers"
        or config.get("device") not in ("cpu", "cuda")
        or not isinstance(config.get("model"), str)
        or not isinstance(config.get("architecture"), str)
        or (config.get("revision") is not None and not isinstance(config["revision"], str))
    ):
        raise ValueError("Invalid model or runtime configuration")
    for key in ("batch_size", "max_length", "max_new_tokens", "warmup"):
        minimum = 0 if key == "warmup" else 1
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f"Invalid {key}")
    tolerance = config.get("tolerance")
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid tolerance")
    return {**config, "revision": config.get("revision")}


class Inference:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ValueError("Install the inference extra") from exc

        self.config = config
        device = config["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        options = {"revision": config["revision"], "trust_remote_code": False}
        metadata = transformers.AutoConfig.from_pretrained(config["model"], **options)
        self.revision = getattr(metadata, "_commit_hash", None) or config["revision"]
        if self.revision is not None:
            options["revision"] = self.revision
        architecture = config["architecture"]
        if metadata.architectures != [architecture]:
            raise ValueError("Checkpoint architecture differs from plan")
        cls = getattr(transformers, architecture, None)
        if not isinstance(cls, type) or not issubclass(cls, transformers.PreTrainedModel):
            raise ValueError("Architecture unavailable in installed Transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"], **options)
        self.model = (
            cls.from_pretrained(config["model"], use_safetensors=True, **options)
            .float()
            .to(device)
            .eval()
        )
        self.runner = TorchForwardRunner(self.model, self.tokenizer, device, config["max_length"])
        self.labels = metadata.id2label
        self.environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "threads": torch.get_num_threads(),
        }

    def predict(self, inputs: list[str], optimized: bool) -> list[dict[str, Any]]:
        import torch

        config = self.config
        outputs = []
        if config["task"] == "sequence_classification":
            size = config["batch_size"] if optimized else 1
            for start in range(0, len(inputs), size):
                logits = self.runner.run(inputs[start : start + size])
                for row in logits.tolist():
                    index = max(range(len(row)), key=row.__getitem__)
                    outputs.append({"label": self.labels.get(index, str(index)), "logits": row})
            return outputs

        for text in inputs:
            encoded = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=config["max_length"]
            ).to(config["device"])
            with torch.inference_mode():
                tokens = self.model.generate(
                    **encoded,
                    use_cache=optimized,
                    do_sample=False,
                    max_new_tokens=config["max_new_tokens"],
                    return_dict_in_generate=False,
                )
            ids = tokens[0, encoded.input_ids.shape[1] :].tolist()
            outputs.append({"token_ids": ids, "text": self.tokenizer.decode(ids)})
        return outputs


def compare(
    config: dict[str, Any], quality: DatasetSource, performance: DatasetSource, column: str
) -> dict[str, Any]:
    qdata, qmeta = load_source(quality)
    pdata, pmeta = load_source(performance)
    qinputs = texts(qdata, column, "quality")
    pinputs = texts(pdata, column, "performance")
    local = Path(config["model"])
    model_digest = tree_digest(local) if local.is_dir() else None
    inference = Inference(config)
    reference = inference.predict(qinputs, False)
    candidate = inference.predict(qinputs, True)
    mismatches = 0
    error = 0.0
    for left, right in zip(reference, candidate, strict=True):
        if config["task"] == "generation":
            mismatches += left != right
        else:
            mismatches += left["label"] != right["label"]
            error = max(
                error,
                max(abs(a - b) for a, b in zip(left["logits"], right["logits"], strict=True)),
            )
    passed = mismatches == 0 and error <= config["tolerance"]
    for _ in range(config["warmup"]):
        inference.predict(pinputs, False)
        inference.predict(pinputs, True)
    timings: dict[str, list[float]] = {"baseline": [], "optimized": []}
    for order in ((False, True), (True, False), (False, True), (True, False), (False, True)):
        for optimized in order:
            start = time.perf_counter()
            inference.predict(pinputs, optimized)
            timings["optimized" if optimized else "baseline"].append(time.perf_counter() - start)
    baseline, optimized_time = (statistics.median(timings[key]) for key in timings)
    return {
        "execution": {**config, "revision": inference.revision},
        "status": "validated" if passed else "parity_failed",
        "environment": inference.environment,
        "local_model_digest": model_digest,
        "datasets": {
            "quality": {**qmeta, "ordered_rows": ordered_digest(qdata)},
            "performance": {**pmeta, "ordered_rows": ordered_digest(pdata)},
        },
        "text_column": column,
        "quality": {
            "passed": passed,
            "examples": len(qinputs),
            "mismatches": mismatches,
            "max_abs_logit_error": error if config["task"] == "sequence_classification" else None,
        },
        "performance": {
            "examples_per_round": len(pinputs),
            "boundary": "tokenization, model inference and output conversion",
            "samples_seconds": timings,
            "baseline_median_seconds": baseline,
            "optimized_median_seconds": optimized_time,
            "speedup": baseline / optimized_time,
        },
    }


def serve(config: dict[str, Any], port: int) -> None:
    inference = Inference(config)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/infer" or not 0 < length <= 1_000_000:
                    raise ValueError
                body = json.loads(self.rfile.read(length))
                inputs = body.get("inputs") if isinstance(body, dict) else None
                if (
                    not isinstance(inputs, list)
                    or not 0 < len(inputs) <= 128
                    or not all(isinstance(item, str) and item.strip() for item in inputs)
                ):
                    raise ValueError
            except (ValueError, UnicodeError):
                self.respond(400, {"error": "Expected POST /infer with a list of text inputs"})
                return
            try:
                self.respond(200, {"outputs": inference.predict(inputs, True)})
            except (RuntimeError, ValueError):
                self.respond(500, {"error": "Inference failed"})

        def respond(self, status: int, result: dict[str, Any]) -> None:
            payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Listening on http://127.0.0.1:{server.server_port}/infer", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
