"""Launch an isolated vLLM generation probe and compare greedy tokens to Transformers."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
from typing import Any

from .benchmark import texts
from .datasets import DatasetSource, load_source


def probe_vllm(
    model: str,
    revision: str | None,
    quality: DatasetSource,
    text_column: str,
    max_length: int,
    max_new_tokens: int,
    timeout: float,
    task: str = "generation",
    output_mode: str = "multiclass",
    threshold: float = 0.5,
    tolerance: float = 0.001,
) -> dict[str, Any]:
    import math

    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            "Install the inference extra and vLLM in a compatible CUDA environment"
        ) from exc

    if (
        max_length < 1
        or max_new_tokens < 1
        or not math.isfinite(timeout)
        or timeout <= 0
        or task not in ("generation", "classification")
        or output_mode not in ("multiclass", "multilabel")
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Invalid generation length or timeout")
    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "cuda_unavailable", "model": model}
    if importlib.util.find_spec("vllm") is None:
        return {"status": "unavailable", "reason": "vllm_not_installed", "model": model}
    data, metadata = load_source(quality)
    payload = {
        "model": model,
        "revision": revision,
        "inputs": texts(data, text_column, "quality"),
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "task": task,
        "output_mode": output_mode,
        "threshold": threshold,
        "tolerance": tolerance,
    }
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "guardrail_tuner.vllm_worker"],
            stdin=subprocess.PIPE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
        output, _ = proc.communicate(input=json.dumps(payload), timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                proc.kill()
        else:
            proc.kill()
        proc.communicate()
        return {"status": "failed", "reason": "timeout", "model": model, "dataset": metadata}
    if proc.returncode:
        return {
            "status": "failed",
            "reason": "worker_failed",
            "exit_code": proc.returncode,
            "model": model,
            "dataset": metadata,
        }
    try:
        result = json.loads(output)
        if not isinstance(result, dict):
            raise ValueError("Invalid worker result")
    except (json.JSONDecodeError, ValueError):
        return {
            "status": "failed",
            "reason": "invalid_worker_output",
            "model": model,
            "dataset": metadata,
        }
    result.update(model=model, dataset=metadata)
    return result
