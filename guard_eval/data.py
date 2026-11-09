"""Загрузка датасета хвостов: локальный parquet или приватный HF-датасет."""

from pathlib import Path

import pandas as pd

DEFAULT_DATASET = "AutotuneGuardrails/tails"
REQUIRED = ["id", "text_hash", "text", "label", "benchmark", "split", "lang", "attack_type", "tail"]


def load_dataset(source=DEFAULT_DATASET, revision=None, benchmarks=None, limit_per_split=None, seed=42):
    path = Path(source)
    if path.is_dir():
        path = path / "data" / "tails.parquet"
    if path.is_file():
        df, origin = pd.read_parquet(path), {"path": str(path)}
    else:
        from huggingface_hub import HfApi, hf_hub_download
        sha = HfApi().dataset_info(source, revision=revision).sha
        df = pd.read_parquet(hf_hub_download(source, "data/tails.parquet", repo_type="dataset", revision=sha))
        origin = {"repo": source, "revision": sha}

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"В датасете нет колонок {sorted(missing)}")
    if benchmarks:
        df = df[df["benchmark"].isin(benchmarks)]
    if limit_per_split:
        df = df.sample(frac=1, random_state=seed).groupby(["benchmark", "split"]).head(limit_per_split)
    origin.update(rows=len(df), benchmarks=sorted(df["benchmark"].unique()))
    return df.reset_index(drop=True), origin
