"""Privacy-conscious execution manifest for CLI runs."""

import base64
import contextvars
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

_ACTIVE: contextvars.ContextVar[Any] = contextvars.ContextVar("run_manifest", default=None)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("unsupported dataset value")


def ordered_digest(dataset: Any) -> dict[str, Any]:
    """Hash framed canonical rows in order; no per-row digest or contents are persisted."""
    digest = hashlib.sha256(b"guardrail-tuner-ordered-rows-v1\0")
    count = 0
    try:
        for row in dataset:
            encoded = json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
                default=_json_value,
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            count += 1
    except (TypeError, ValueError, OverflowError):
        return {
            "algorithm": "sha256-canonical-json-v1",
            "status": "unavailable",
            "reason": "unsupported_dataset_value",
        }
    return {
        "algorithm": "sha256-canonical-json-v1",
        "status": "captured",
        "rows": count,
        "sha256": digest.hexdigest(),
    }


def tree_digest(directory: Path, python_only: bool = False) -> dict[str, Any]:
    """Hash local model artifacts, including relative names, without retaining filenames."""
    tag = b"guardrail-tuner-python-tree-v1\0" if python_only else b"guardrail-tuner-model-tree-v1\0"
    digest = hashlib.sha256(tag)
    count = 0
    total = 0
    try:
        for path in sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and (not python_only or p.suffix == ".py")
        ):
            relative = path.relative_to(directory).as_posix().encode("utf-8")
            file_digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
                    size += len(chunk)
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            digest.update(file_digest.digest())
            total += size
            count += 1
    except OSError as exc:
        raise ValueError("Cannot hash the local model directory") from exc
    return {
        "algorithm": "sha256-python-tree-v1" if python_only else "sha256-model-tree-v1",
        "sha256": digest.hexdigest(),
        "files": count,
        "bytes": total,
    }


def set_seed(seed: int) -> dict[str, Any]:
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 4294967295")
    os.environ["GUARDRAIL_TUNER_SEED"] = str(seed)
    random.seed(seed)
    result: dict[str, Any] = {
        "value": seed,
        "python_random": True,
        "numpy": False,
        "torch": False,
        "external_process": "GUARDRAIL_TUNER_SEED provided; use not verified",
    }
    try:
        import numpy as np

        np.random.seed(seed)
        result["numpy"] = True
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result["torch"] = True
        result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        result["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        result["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    except ImportError:
        pass
    return result


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in (
        "guardrail-tuner",
        "torch",
        "transformers",
        "datasets",
        "huggingface-hub",
        "onnxruntime",
        "onnxruntime-gpu",
        "optimum-onnx",
        "vllm",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "gpu": None,
        "driver": None,
    }
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    result["cublas_workspace_config"] = (
        workspace_config
        if workspace_config in (":16:8", ":4096:8")
        else "other"
        if workspace_config is not None
        else None
    )
    hash_seed = os.environ.get("PYTHONHASHSEED")
    result["python_hash_seed"] = (
        hash_seed
        if hash_seed and (hash_seed.isdecimal() or hash_seed == "random")
        else "other"
        if hash_seed is not None
        else None
    )
    try:
        import torch

        result["torch_cuda_runtime"] = torch.version.cuda
        result["torch_threads"] = torch.get_num_threads()
        result["torch_interop_threads"] = torch.get_num_interop_threads()
        result["torch_default_dtype"] = str(torch.get_default_dtype())
        result["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        result["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        if torch.cuda.is_available():
            result["gpu"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            drivers = proc.stdout.strip().splitlines()
            result["driver"] = (
                drivers if all(set(s) <= set("0123456789.") for s in drivers) else None
            )
        except (OSError, subprocess.SubprocessError):
            pass
    project_root = Path(__file__).resolve().parents[2]
    lock = project_root / "uv.lock"
    if lock.is_file():
        try:
            digest = hashlib.sha256()
            with lock.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result["uv_lock_sha256"] = digest.hexdigest()
        except OSError:
            result["uv_lock_sha256"] = None
    if (project_root / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout
            if len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit):
                result["source_git_commit"] = commit
                result["source_git_dirty"] = bool(dirty)
        except (OSError, subprocess.SubprocessError):
            pass
    return result


class Recorder:
    def __init__(self, command: str, seed: int, sources: dict[str, Any], options: dict[str, Any]):
        self.command = command
        self.seed = set_seed(seed)
        self.sources = sources
        self.options = {key: value for key, value in options.items() if key != "model"}
        self.records: dict[str, list[dict[str, Any]]] = {role: [] for role in sources}
        self.model: dict[str, Any] = {}
        self.prepared: dict[str, list[dict[str, Any]]] = {}
        self.workloads: dict[str, list[dict[str, Any]]] = {}
        self.source_path = Path(__file__).resolve().parent
        self.source_before = tree_digest(self.source_path, python_only=True)
        model_path = Path(options["model"]).expanduser() if options.get("model") else None
        self.local_model_path = (
            model_path.resolve()
            if model_path is not None and model_path.is_dir() and command != "compare-app"
            else None
        )
        self.local_model_before = (
            tree_digest(self.local_model_path) if self.local_model_path else None
        )
        tokenizer_path = (
            Path(options["batch_tokenizer"]).expanduser()
            if options.get("batch_tokenizer")
            else None
        )
        self.local_tokenizer_path = (
            tokenizer_path.resolve()
            if tokenizer_path is not None and tokenizer_path.is_dir()
            else None
        )
        self.local_tokenizer_before = (
            tree_digest(self.local_tokenizer_path) if self.local_tokenizer_path else None
        )
        self.token: Any = None

    def __enter__(self) -> "Recorder":
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        _ACTIVE.reset(self.token)

    def dataset(self, source: Any, data: Any, metadata: dict[str, Any]) -> None:
        fingerprint = ordered_digest(data)
        allowed = (
            "source",
            "id",
            "requested_revision",
            "resolved_revision",
            "config",
            "split",
            "sha256",
            "preset",
            "manifest_sha256",
            "corpus_sha256",
            "rows",
            "text_projection",
            "model_specific_u1_buckets_applied",
        )
        record = {key: metadata[key] for key in allowed if key in metadata}
        record["ordered_rows"] = fingerprint
        for role, expected in self.sources.items():
            revision_matches = source.revision == expected.revision or any(
                previous.get("resolved_revision") == source.revision
                for previous in self.records[role]
            )
            if (
                source.kind == expected.kind
                and source.path == expected.path
                and source.split == expected.split
                and source.config == expected.config
                and revision_matches
            ):
                self.records[role].append(record)

    def set_model(
        self, model: str, revision: str | None, runtime: Any = None, tokenizer: Any = None
    ) -> None:
        self.model = {
            "id": None if self.local_model_path else model,
            "source": "local" if self.local_model_path else "huggingface",
            "resolved_revision": revision,
        }
        if runtime is not None:
            self.model["class"] = type(runtime).__name__
            self.model["parameter_dtypes"] = sorted(
                {str(parameter.dtype) for parameter in runtime.parameters()}
            )
        if tokenizer is not None:
            self.model["tokenizer"] = {
                "class": type(tokenizer).__name__,
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
            }

    def build(self, result: dict[str, Any]) -> dict[str, Any]:
        from .measurement import MAX_REFERENCE_SPREAD, MIN_IMPROVEMENT, MIN_REQUESTS, ROUNDS

        sources: dict[str, Any] = {}
        for role, source in self.sources.items():
            entries = self.records[role]
            distinct = {
                json.dumps(
                    {key: value for key, value in entry.items() if key != "requested_revision"},
                    sort_keys=True,
                )
                for entry in entries
            }
            sources[role] = {
                "loads": len(entries),
                "consistent": len(distinct) == 1 if entries else None,
                "identity": entries[0] if entries else None,
                "declared_kind": source.kind,
            }
        local_model = None
        if self.local_model_path is not None:
            after = tree_digest(self.local_model_path)
            local_model = {
                "before": self.local_model_before,
                "after": after,
                "unchanged": self.local_model_before == after,
            }
        local_tokenizer = None
        if self.local_tokenizer_path is not None:
            tokenizer_after = tree_digest(self.local_tokenizer_path)
            local_tokenizer = {
                "before": self.local_tokenizer_before,
                "after": tokenizer_after,
                "unchanged": self.local_tokenizer_before == tokenizer_after,
            }
        source_after = tree_digest(self.source_path, python_only=True)
        changed = (
            self.source_before != source_after
            or (local_model is not None and not local_model["unchanged"])
            or (local_tokenizer is not None and not local_tokenizer["unchanged"])
            or any(info["consistent"] is False for info in sources.values())
        )
        incomplete = any(
            info["identity"] is None or info["identity"]["ordered_rows"]["status"] != "captured"
            for info in sources.values()
        )
        if self.command in ("benchmark", "optimize", "tune-model", "check-generation"):
            incomplete = incomplete or any(
                role not in self.prepared
                or any(item["status"] != "captured" for item in self.prepared[role])
                for role in sources
            )
        boundary = {
            "benchmark": "sequential; includes tokenization, model execution and output transfer",
            "optimize": "sequential; includes tokenization, model execution and output transfer",
            "tune-model": "sequential; includes tokenization, model execution and output transfer",
            "check-generation": "no performance measurement",
            "probe-vllm": "no performance measurement",
        }.get(self.command)
        if self.command == "compare-app":
            boundary = (result.get("settings") or {}).get("measurement")
        raw_validation = result.get("quality")
        validation: dict[str, Any] = raw_validation if isinstance(raw_validation, dict) else {}
        validation_summary = {
            key: validation[key]
            for key in (
                "passed",
                "samples",
                "max_abs_logit_error",
                "max_abs_probability_error",
                "decision_disagreements",
                "aa_passed",
                "aa_max_abs_logit_error",
            )
            if key in validation
        }
        raw_settings = result.get("settings")
        runtime_settings = raw_settings if isinstance(raw_settings, dict) else {}
        effective_settings = {
            key: runtime_settings[key]
            for key in (
                "dtype",
                "reference_dtype",
                "decoding",
                "greedy",
                "ignore_eos",
                "max_length",
                "max_new_tokens",
                "max_score_error",
                "warmup",
                "timeout_s",
                "batch_size",
                "process_mode",
                "measurement",
                "ort_optimization",
                "ort_intra_op_threads",
                "max_logit_error",
                "output_mode",
                "threshold",
            )
            if key in runtime_settings
        }
        return {
            "schema_version": 1,
            "capture_status": "changed" if changed else "incomplete" if incomplete else "complete",
            "command": self.command,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cli_options": self.options,
            "seed": self.seed,
            "effective_runtime_settings": effective_settings or None,
            "environment": environment(),
            "model": self.model or None,
            "source_code": {
                "before": self.source_before,
                "after": source_after,
                "unchanged": self.source_before == source_after,
            },
            "local_model_tree": local_model,
            "local_tokenizer_tree": local_tokenizer,
            "datasets": sources,
            "prepared_token_inputs": self.prepared,
            "workload_order": self.workloads,
            "workload_tokenizer": (result.get("workload") or {}).get("tokenizer")
            if isinstance(result.get("workload"), dict)
            else None,
            "measurement": {
                "boundary": boundary,
                "warmup_requests": self.options.get("warmup"),
                "batching": "sequential static batches; no queue-wait simulation",
            },
            "decision_policy": {
                "applied": self.command in ("benchmark", "optimize", "tune-model", "compare-app"),
                "minimum_requests_per_round": MIN_REQUESTS,
                "rounds": ROUNDS,
                "minimum_improvement_fraction": MIN_IMPROVEMENT,
                "maximum_aa_p99_difference_fraction": MAX_REFERENCE_SPREAD,
                "qualification": "heuristic; statistical significance not established",
            },
            "validation_summary": validation_summary or None,
            "outcome": {
                key: result[key]
                for key in ("status", "selected", "selection", "scope")
                if key in result
            },
            "limitations": [
                "Dataset digest is aggregate and does not contain individual IDs or records",
                "Seed does not enable deterministic algorithms",
                "External JSONL runtime configuration and internal prompts are not verified",
                "Hashes of small or predictable datasets are not a confidentiality guarantee",
            ],
        }


def record_dataset(source: Any, data: Any, metadata: dict[str, Any]) -> None:
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder.dataset(source, data, metadata)


def record_model(
    model: str, revision: str | None, runtime: Any = None, tokenizer: Any = None
) -> None:
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder.set_model(model, revision, runtime, tokenizer)


def record_token_inputs(role: str, tokenizer: Any, texts: list[str], max_length: int) -> None:
    recorder = _ACTIVE.get()
    if recorder is not None:
        encoded = (dict(tokenizer(text, truncation=True, max_length=max_length)) for text in texts)
        recorder.prepared.setdefault(role, []).append(ordered_digest(encoded))


def record_workload(role: str, buckets: dict[str, Any], batch_sizes: tuple[int, ...]) -> None:
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder.workloads.setdefault(role, []).append(
            {
                "buckets": [
                    {"upper_bound": name, "rows": len(items), "ordered_rows": ordered_digest(items)}
                    for name, items in buckets.items()
                ],
                "batch_sizes": list(batch_sizes),
                "batching": "contiguous within each bucket; partial final batch retained",
            }
        )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ValueError("Manifest path already exists") from exc
