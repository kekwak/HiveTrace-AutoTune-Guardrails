"""Кэш прогонов: results/quality/runs/<model>/<spec_hash>/.

predictions.parquet хранит только text_hash и скоры — без текстов и меток,
поэтому папку можно коммитить, а один прогон переиспользуется любой версией датасета.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .spec import ROOT

RUNS_DIR = ROOT / "results" / "quality" / "runs"
PRED_COLUMNS = ["text_hash", "p_unsafe", "n_tokens", "truncated", "error"]


class Run:
    def __init__(self, spec, key, runs_dir=RUNS_DIR):
        self.spec, self.key = spec, key
        self.dir = Path(runs_dir) / spec["name"] / key
        self.predictions_path = self.dir / "predictions.parquet"
        self.manifest_path = self.dir / "manifest.json"
        self.speed_path = self.dir / "speed.json"

    def predictions(self):
        if self.predictions_path.is_file():
            return pd.read_parquet(self.predictions_path)
        return pd.DataFrame(columns=PRED_COLUMNS)

    def missing(self, text_hashes):
        """Тексты без скора; строки с ошибкой пересчитываются при следующем запуске."""
        p = self.predictions()
        done = set(p.loc[p["error"].isna() & p["p_unsafe"].notna(), "text_hash"])
        return [h for h in dict.fromkeys(text_hashes) if h not in done]

    def append(self, rows):
        self.dir.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows, columns=PRED_COLUMNS)
        merged = pd.concat([self.predictions(), new], ignore_index=True).drop_duplicates("text_hash", keep="last")
        tmp = self.predictions_path.with_suffix(".tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(self.predictions_path)

    def manifest(self):
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.is_file() else {}

    def update_manifest(self, **fields):
        self.dir.mkdir(parents=True, exist_ok=True)
        data = self.manifest()
        data.update(fields, updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))

    def speed(self):
        return json.loads(self.speed_path.read_text()) if self.speed_path.is_file() else None

    def save_speed(self, result):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.speed_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
