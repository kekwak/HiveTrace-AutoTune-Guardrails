"""Execute fixed, architecture-based optimization plans."""

import json
import platform
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .model_tuning import TorchForwardRunner
from .rules import RULES, RuledModel

OPTIMIZATIONS = {"sequence_classification": "batching", "generation": "kv_cache"}


def execution_plan(
    metadata: dict[str, Any], mode: str | None, device: str = "cpu"
) -> dict[str, Any] | None:
    if mode is None or metadata["custom_code_declared"] or len(metadata["architectures"]) != 1:
        return None
    model = Path(metadata["model"]).expanduser()
    architecture = metadata["architectures"][0]
    return {
        "version": 1,
        "model": str(model.resolve()) if model.is_dir() else metadata["model"],
        "revision": metadata["resolved_revision"],
        "architecture": architecture,
        "runtime": "transformers",
        "task": mode,
        "optimization": OPTIMIZATIONS[mode],
        "device": device,
        "max_length": 128,
        "max_new_tokens": 8,
        "batch_size": 4 if mode == "sequence_classification" else 1,
    }


def _require_supported_runtime(config: dict[str, Any]) -> None:
    if config.get("runtime") != "transformers":
        raise ValueError("Unsupported runtime. Generate a new plan with guardrail-tuner plan.")


def read_execution(path: str) -> dict[str, Any]:
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON plan")
    if (
        document.get("status") == "parity_failed"
        or document.get("rules", {}).get("parity", {}).get("passed") is False
    ):
        raise ValueError("This run failed output parity")
    config = document.get("execution")
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("No executable plan")
    _require_supported_runtime(config)
    task = config.get("task")
    if (
        not isinstance(task, str)
        or task not in OPTIMIZATIONS
        or config.get("device") not in ("cpu", "cuda")
        or not isinstance(config.get("model"), str)
        or not isinstance(config.get("architecture"), str)
        or (config.get("revision") is not None and not isinstance(config["revision"], str))
    ):
        raise ValueError("Invalid model configuration")
    if config.get("optimization") != OPTIMIZATIONS[task]:
        raise ValueError("Invalid Transformers optimization")
    for key in ("batch_size", "max_length", "max_new_tokens"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    rules = config.get("rules", [])
    if not isinstance(rules, list) or any(rule not in RULES for rule in rules):
        raise ValueError("Invalid rules")
    return {**config, "revision": config.get("revision"), "rules": rules}


class Inference:
    def __init__(self, config: dict[str, Any]) -> None:
        _require_supported_runtime(config)
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
        from_pretrained = getattr(cls, "from_pretrained")
        self.model = (
            from_pretrained(config["model"], use_safetensors=True, **options)
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

    def enable_rules(self) -> RuledModel:
        ruled = RuledModel(self.model, self.config["rules"], self.tokenizer.pad_token_id)
        if self.config["task"] == "sequence_classification":
            self.runner.model = ruled
        return ruled

    def predict(self, inputs: list[str]) -> list[dict[str, Any]]:
        import torch

        config = self.config
        outputs = []
        if config["task"] == "sequence_classification":
            size = config["batch_size"]
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
                    use_cache=True,
                    do_sample=False,
                    max_new_tokens=config["max_new_tokens"],
                    return_dict_in_generate=False,
                )
            ids = tokens[0, encoded.input_ids.shape[1] :].tolist()
            outputs.append({"token_ids": ids, "text": self.tokenizer.decode(ids)})
        return outputs


def run_inference(config: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    if not inputs or any(not text.strip() for text in inputs):
        raise ValueError("Provide non-empty text with --text or stdin")
    inference = Inference(config)
    result = {
        "execution": {**config, "revision": inference.revision},
        "environment": inference.environment,
    }
    if not config["rules"]:
        return {**result, "outputs": inference.predict(inputs)}
    ruled = inference.enable_rules()
    outputs = inference.predict(inputs)
    return {
        **result,
        "rules": {"applied": ruled.applied, "skipped": ruled.skipped},
        "outputs": outputs,
    }


def serve(config: dict[str, Any], port: int) -> None:
    inference = Inference(config)
    if config["rules"]:
        inference.enable_rules()

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
                self.respond(200, {"outputs": inference.predict(inputs)})
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
