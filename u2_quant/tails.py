"""tails_v2 (AutotuneGuardrails/tails, tag v2) -> U1 v4 labeled quality corpus.

python -m u2_quant.tails <tails.parquet> <out_dir>  # пишет quality.jsonl и import.json
"""
import json
from pathlib import Path
import sys

import pandas as pd

TARGETS = {"harm": "unsafe", "safe": "safe"}
SLICES = ("technique", "lang", "benchmark", "tail")


def to_rows(df):
    unknown = set(df["label"]) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown tails labels: {sorted(unknown)}")
    return [{"id": row.id, "messages": [{"role": "user", "content": row.text}], "target": TARGETS[row.label],
             "target_role": "user", "target_index": 0, "slices": {k: str(getattr(row, k)) for k in SLICES}}
            for row in df.itertuples(index=False)]


def write(parquet, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / "quality.jsonl").open("w") as stream:
        for row in to_rows(pd.read_parquet(parquet)):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    spec = {"schema_version": 4, "kind": "quality", "source": str((out_dir / "quality.jsonl").resolve()),
            "fields": {k: k for k in ("id", "messages", "target", "target_role", "target_index", "slices")},
            "version": "tails_v2",
            "provenance": {"owner": "U2", "dataset": "AutotuneGuardrails/tails", "revision": "v2",
                           "scope": "labeled evaluation"},
            "include_texts": False}
    path = out_dir / "import.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    return path


if __name__ == "__main__":
    print(write(sys.argv[1], sys.argv[2]))
