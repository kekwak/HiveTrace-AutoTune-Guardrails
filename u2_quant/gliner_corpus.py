"""GLiNER quality corpus from the U1 tails_v2 corpus, with texts pre-normalized by the GLiNER SDK processor.

The SDK appends a period to ~34% of tails_v2 texts; U1's native GLiNER path rejects any processor change.
The normalization is idempotent and yields identical input_ids, so the model sees exactly its native input.

python -m u2_quant.gliner_corpus <quality/corpus.jsonl> <out_dir>   # в u2-env (нужен gliner2)
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from u1_benchmark.core import ROOT, read


def sdk_normalizer():
    from huggingface_hub import snapshot_download
    from gliner2 import GLiNER2
    weights = read(ROOT / "configs/v4/gliner.json")["baseline"]["weights"]
    model = GLiNER2.from_pretrained(snapshot_download(weights["repo"], revision=weights["revision"]),
                                    attention_backend="standard", quantize=False, compile=False, map_location="cpu")
    schema = model.create_schema().classification(task="safety", labels=["safe", "unsafe"], multi_label=False)
    return lambda text: model.processor.collate_fn_inference(
        [(text, schema.build())], max_len=None, error_policy="raise", architecture="span").original_texts[0]


def normalize(rows, normalizer):
    out, changed = [], 0
    for row in rows:
        text = row["messages"][row["target_index"]]["content"]
        normalized = normalizer(text)
        if normalizer(normalized) != normalized:
            raise ValueError(f"GLiNER normalization is not idempotent for {row['id']}")
        value = deepcopy(row)
        value.pop("content_sha256", None)
        value["messages"][row["target_index"]]["content"] = normalized
        changed += normalized != text
        out.append(value)
    return out, changed


def write(corpus, out_dir, normalizer):
    rows = [json.loads(line) for line in Path(corpus).read_text().splitlines() if line.strip()]
    values, changed = normalize(rows, normalizer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / "quality.jsonl").open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    spec = {"schema_version": 4, "kind": "quality", "source": str((out_dir / "quality.jsonl").resolve()),
            "fields": {k: k for k in ("id", "messages", "target", "target_role", "target_index", "slices")},
            "version": "tails_v2-gliner-normalized",
            "provenance": {"owner": "U2", "dataset": "AutotuneGuardrails/tails", "revision": "v2",
                           "normalizer": "gliner2 processor.collate_fn_inference (idempotent, same input_ids)",
                           "normalized_rows": changed, "scope": "labeled evaluation"},
            "include_texts": False}
    path = out_dir / "import.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    return path


if __name__ == "__main__":
    print(write(sys.argv[1], sys.argv[2], sdk_normalizer()))
