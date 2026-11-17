"""U1 v4 configs for bf16c (bf16 + compile, the reference) and fp8, derived from configs/v4/<model>.json.

python -m u2_quant.configs [--speed-manifest SHA] [--quality-manifest SHA]  # пишет configs/u2/*.json
"""
import argparse
from copy import deepcopy

from u1_benchmark.core import ROOT, read, sha, write
from u1_benchmark.v4_config import variant_identity
from u2_quant.lock import TORCHAO

MODELS, VARIANTS = ("gliner", "hive"), ("bf16c", "fp8")
ADAPTER = "u2_quant/adapter.py"
# gliner — encoder, батч 64; hive — generate с KV-кэшем: худший батч tails при 64 → ~53 ГиБ, при 16 → ~13 ГиБ.
QUALITY_BATCH = {"gliner": [64], "hive": [16]}
LABELS = {"labels": ["safe", "unsafe"], "targets": {"safe": "safe", "unsafe": "unsafe"},
          "verdicts": {"safe": "safe", "unsafe": "unsafe"}, "positive_label": "unsafe"}
SPEED_CORPUS = {"dataset": "vXofi/guardrails-speed-en-ru", "revision": "7324b8823ded971c05076e0c7f4ace5d9b867863"}
PLAN = [(m, "bf16c", "baseline") for m in MODELS] + [(m, "bf16c", "quality") for m in MODELS] + \
       [(m, "fp8", mode) for m in MODELS for mode in ("search", "confirm", "quality")]


def variant(model, name):
    spec = deepcopy(read(ROOT / f"configs/v4/{model}.json")["baseline"])
    spec["adapter"] = {"module": ADAPTER, "source_sha256": sha(ROOT / ADAPTER),
                       "dependencies": {TORCHAO["name"]: TORCHAO["version"]}}
    spec["execution"] = {**spec["execution"], "runtime": "compile", "dtype": "bfloat16"}
    if name == "fp8":
        reference = variant(model, "bf16c")
        spec["quantization"] = {"method": "fp8", "granularity": "per_row", "skip_modules": [],
                                "quantize_sha256": sha(ROOT / "u2_quant/quantize.py"),
                                "baseline_identity": variant_identity(reference)}
    elif name != "bf16c":
        raise ValueError(f"Unknown variant: {name}")
    return spec


def build(model, name, mode, *, environment, data, quality_data, baseline_ref=None):
    v4 = read(ROOT / f"configs/v4/{model}.json")
    cfg = {"schema_version": 4, "baseline": variant(model, "bf16c"), "buckets": v4["buckets"],
           "batch_sizes_by_bucket": v4["batch_sizes_by_bucket"], "mode": mode, "comparison": "quantization",
           "seed": 17, "timeout_seconds": 36000, "environment": environment, "quality_batch_sizes": [1]}
    if name != "bf16c":
        cfg["candidate"] = variant(model, name)
    if mode == "quality":
        # Батч > 1 — штатная опция У1; эталон и FP8 идут на одинаковых батчах.
        cfg.update(quality_data=quality_data, label_mapping=deepcopy(LABELS), quality_batch_sizes=QUALITY_BATCH[model])
    else:
        cfg["data"] = data
    if mode == "search":
        cfg["baseline_ref"] = baseline_ref
    return cfg


def speed_import(corpus, configs):
    return {"schema_version": 4, "kind": "speed", "source": corpus,
            "fields": {"id": "id", "messages": "messages", "provenance": "provenance", "language": "language",
                       "source": "source"},
            "version": "u2-speed-indices-v1",
            "provenance": {"description": "U2 bf16c/fp8 input index over the shared role-message corpus", **SPEED_CORPUS},
            "configs": [{"config": path, "variant": "baseline", "name": f"u2-{model}"} for model, path in configs.items()],
            "seed": 42, "include_texts": False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-manifest", default="REPLACE_SPEED_MANIFEST_SHA256")
    parser.add_argument("--quality-manifest", default="REPLACE_QUALITY_MANIFEST_SHA256")
    parser.add_argument("--quality-manifest-gliner", default="REPLACE_GLINER_QUALITY_MANIFEST_SHA256")
    args = parser.parse_args(argv)
    data = {"path": "/workspace/u2-data/speed", "manifest_sha256": args.speed_manifest}
    # gliner: тексты, нормализованные SDK (u2_quant.gliner_corpus); hive: исходные тексты tails_v2.
    quality = {"gliner": {"path": "/workspace/u2-data/quality-gliner", "manifest_sha256": args.quality_manifest_gliner},
               "hive": {"path": "/workspace/u2-data/quality", "manifest_sha256": args.quality_manifest}}
    out = ROOT / "configs/u2"
    for model, name, mode in PLAN:
        write(out / f"{model}_{name}_{mode}.json",
              build(model, name, mode, environment="/workspace/u2-env", data=data, quality_data=quality[model],
                    baseline_ref=f"/workspace/u2-runs/{model}-baseline"))
    write(out / "speed_import.json", speed_import("/workspace/u2-data/role-corpus/corpus.jsonl",
                                                  {m: f"configs/u2/{m}_bf16c_baseline.json" for m in MODELS}))


if __name__ == "__main__":
    main()
