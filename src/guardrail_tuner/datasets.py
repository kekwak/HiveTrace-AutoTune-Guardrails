"""Load datasets for quality evaluation and performance benchmarking."""

import hashlib
from dataclasses import dataclass
from glob import escape
from pathlib import Path
from typing import Any

from .manifest import record_dataset

SPEED_PRESET = "role-v4"
SPEED_REPO = "vXofi/guardrails-speed-en-ru"
SPEED_REVISION = "7324b8823ded971c05076e0c7f4ace5d9b867863"
SPEED_MANIFEST_SHA256 = "760adad3e5946384de9fec98aebd9327229cc02c706af094322f2f37e95fc02c"


@dataclass(frozen=True)
class DatasetSource:
    kind: str
    path: str
    split: str | None = None
    revision: str | None = None
    config: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_speed_preset(source: DatasetSource, load_dataset: Any) -> tuple[Any, dict[str, Any]]:
    """Load the pinned role corpus; do not claim its model-specific U1 bucket semantics."""
    import json

    from huggingface_hub import hf_hub_download

    if source.path != SPEED_PRESET or any((source.revision, source.split, source.config)):
        raise ValueError("Invalid speed preset or unsupported dataset options")
    try:
        manifest_path = Path(
            hf_hub_download(
                SPEED_REPO, "manifest.json", repo_type="dataset", revision=SPEED_REVISION
            )
        )
        if _sha256(manifest_path) != SPEED_MANIFEST_SHA256:
            raise ValueError("Speed preset manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 3
            or manifest.get("status") != "ready"
            or manifest.get("count") != 12000
            or not isinstance(manifest.get("files"), dict)
        ):
            raise ValueError("Invalid speed preset manifest")
        expected = manifest["files"].get("corpus.jsonl")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Speed preset manifest is missing the corpus checksum")
        path = Path(
            hf_hub_download(
                SPEED_REPO, "corpus.jsonl", repo_type="dataset", revision=SPEED_REVISION
            )
        )
        if _sha256(path) != expected:
            raise ValueError("Speed preset corpus checksum mismatch")
        dataset = load_dataset("json", data_files=escape(str(path)), split="train")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Cannot load the role-v4 speed preset; check Hugging Face access") from exc
    if len(dataset) != 12000 or "messages" not in dataset.column_names:
        raise ValueError("Invalid speed preset corpus schema or row count")
    projection = []
    for messages in dataset["messages"]:
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(messages[-1], dict)
            or messages[-1].get("role") not in ("user", "assistant")
            or not isinstance(messages[-1].get("content"), str)
        ):
            raise ValueError("Invalid speed preset message structure")
        projection.append(messages[-1]["content"])
    if "text" in dataset.column_names:
        raise ValueError("Unexpected text column in speed preset")
    dataset = dataset.add_column("text", projection)
    return dataset, {
        "source": "preset",
        "preset": SPEED_PRESET,
        "id": SPEED_REPO,
        "requested_revision": SPEED_REVISION,
        "resolved_revision": SPEED_REVISION,
        "manifest_sha256": SPEED_MANIFEST_SHA256,
        "corpus_sha256": expected,
        "rows": len(dataset),
        "columns": dataset.column_names,
        "text_projection": "content of last message; original messages retained",
        "model_specific_u1_buckets_applied": False,
    }


def load_source(source: DatasetSource) -> tuple[Any, dict[str, Any]]:
    """Load a source and return the dataset with reproducibility metadata."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError("Install guardrail-tuner[datasets] to load datasets") from exc

    if source.kind == "preset":
        dataset, metadata = _load_speed_preset(source, load_dataset)
        record_dataset(source, dataset, metadata)
        return dataset, metadata
    if source.kind == "hub":
        try:
            info = HfApi().dataset_info(source.path, revision=source.revision)
            dataset = load_dataset(
                source.path,
                name=source.config,
                split=source.split or "train",
                revision=info.sha,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot load Hugging Face dataset {source.path!r}. "
                "Check access, authentication, revision, configuration and split."
            ) from exc
        metadata = {
            "source": "huggingface",
            "id": source.path,
            "requested_revision": source.revision or "main",
            "resolved_revision": info.sha,
            "config": source.config,
            "split": source.split or "train",
        }
    elif source.kind == "file":
        path = Path(source.path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Dataset file does not exist: {path}")
        formats = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}
        file_format = formats.get(path.suffix.lower())
        if file_format is None:
            raise ValueError("Local dataset must be a JSON, JSONL, CSV or Parquet file")
        before = _sha256(path)
        try:
            dataset = load_dataset(file_format, data_files=escape(str(path)), split="train")
        except Exception as exc:
            raise ValueError(f"Cannot load dataset file {path}: {exc}") from exc
        after = _sha256(path)
        if before != after:
            raise ValueError("Local dataset changed during loading")
        metadata = {"source": "file", "path": str(path), "sha256": after}
    else:
        raise ValueError(f"Unknown dataset source kind: {source.kind}")

    metadata["rows"] = len(dataset)
    metadata["columns"] = dataset.column_names
    record_dataset(source, dataset, metadata)
    return dataset, metadata
