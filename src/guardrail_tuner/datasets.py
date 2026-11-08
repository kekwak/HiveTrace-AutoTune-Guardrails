"""Load datasets for quality evaluation and performance benchmarking."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSource:
    kind: str
    path: str
    split: str | None = None
    revision: str | None = None
    config: str | None = None


def load_source(source: DatasetSource) -> tuple[Any, dict[str, Any]]:
    """Load a source and return the dataset with reproducibility metadata."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError("Install guardrail-tuner[datasets] to load datasets") from exc

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
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        try:
            dataset = load_dataset(file_format, data_files=str(path), split="train")
        except Exception as exc:
            raise ValueError(f"Cannot load dataset file {path}: {exc}") from exc
        metadata = {"source": "file", "path": str(path), "sha256": digest.hexdigest()}
    else:
        raise ValueError(f"Unknown dataset source kind: {source.kind}")

    metadata["rows"] = len(dataset)
    metadata["columns"] = dataset.column_names
    return dataset, metadata
