---
pretty_name: Guardrails Speed EN-RU Role Corpus v4
language:
- en
- ru
license: cc-by-sa-3.0
size_categories:
- 10K<n<100K
tags:
- benchmark
- latency
- guardrails
- synthetic
configs:
- config_name: default
  data_files:
  - split: pool
    path: corpus.jsonl
---

# Guardrails Speed EN-RU: Native Role Corpus v4

**A speed workload, not a labeled safety benchmark.** The common pool contains
12,000 distinct records with `user`, `assistant`, or `user → assistant` messages.
There are 400 unique effective inputs per applicable model bucket: 20 model/bucket
combinations, or 8,000 selections with overlap across models. Do not interpret those
selections as 8,000 globally distinct conversations or as real production traffic.

## Native Preparation

Model | Input policy | Required buckets
--- | --- | ---
GLiNER Guard Omni, Koala Text-Moderation, Llama Prompt Guard, Laya | Assistant text when present; otherwise user text; native tokenizer/schema/SDK | B1
HaloGuard 0.8B | Last user message with author prompt and `Safety: `; assistant-only not applicable | B1–B4
HiveTraceGuard-Pro | Native template; final message is the target, preceding user is context | B1–B4
ShieldGemma 2B | User-only or user+assistant; native prompt/response Dangerous Content wording | B1–B4
OpenGuardrails 0124 | User-only with its own pinned template, not the 2510 template; response formats unconfirmed | B1–B4

All checkpoint/tokenizer revisions, native preparation checksums, dependency versions,
and CPU parity checks are pinned in `manifest.json`. Llama uses the explicitly
selected hipocap mirror; equivalence to the original Meta checkpoint is not claimed.
Technical acceptance by a template does not establish classification quality.

This table describes the **published v4 snapshot**, not a new approval of its
OpenGuardrails preparation. The current U1 script uses the separately agreed
0124 weights + 2510 task-prompt contract. Its Open index must be rebuilt locally;
the published native-v4 index is not silently reused or overwritten. See the
[U1 instructions](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/results/haloguard-gliner-u1/docs/u1_handoff.md#2-получение-данных-и-open-индекс).

## Model-Specific Buckets

- B1: 1–512 full prepared-input tokens.
- B2: 513–1024.
- B3: 1025–2048.
- B4: 2049–4096.

Bucket membership is determined separately by each model tokenizer after native
preparation, including policy/template/schema and special tokens. One source ID
may belong to different buckets across models. The baseline and candidate **of the
same model** must use identical IDs, messages, prepared inputs and request order.
Absolute performance across models is not a matched-input comparison.

Eight equal length intervals per bucket target 50 records each; shortages are
redistributed among available intervals. Unreachable lengths are not filled by
padding or by shortening mandatory policies. Some decoder B1 inputs necessarily
start near 400 tokens. Histograms, min/median/p95/max, language/format composition,
rejected formats and length-limit exclusions are in `REPORT.md` and the manifest.
There is no silent truncation. Seed: 42.

## Files and Loading

- `corpus.jsonl`: the single common source pool; role messages, stable IDs, content
  hashes, language, source and nested attribution/transformation metadata.
- `indices/{model}.jsonl`: full pool indices, actual lengths, target/context,
  token-ID/mask hashes, applicability and explicit rejection reasons.
- `buckets/{model}/{B1..B4}.json`: selected source IDs; no duplicated text files per model.
- `manifest.json`: complete file checksums, native input contracts and distributions.
- `REPORT.md`: CPU assembly report. `LICENSE.md`: attribution and licensing rules.

```python
from huggingface_hub import snapshot_download

root = snapshot_download(
    "vXofi/guardrails-speed-en-ru",
    repo_type="dataset",
    revision="7324b8823ded971c05076e0c7f4ace5d9b867863",
)
```

The exact corpus manifest SHA256 is
`760adad3e5946384de9fec98aebd9327229cc02c706af094322f2f37e95fc02c`.
The U1 repository pins the immutable HF commit in `configs/data/role_speed_v4.json`.

## Using the U1 Measurement Script

The console entry point is `tools/benchmark.py` in the
[U1 Git branch](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/tree/results/haloguard-gliner-u1).
Use the [operational guide](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/results/haloguard-gliner-u1/docs/u1_handoff.md)
for environment setup, HF authorization, verified corpus loading, Open reindexing,
preflight/pilot, baseline or candidate runs, and offline reports. A notebook or
separate source bundle is not required. Record the Git commit used for a run.

- **Speed:** `batch_sizes_by_bucket` selects sizes for each model-specific bucket.
  Baseline and candidate receive the same IDs and batches. Padding extends each
  batch only to its longest native input; it does not change bucket membership.
  The current speed grids are provisional, not established maximum-safe sizes.
- **Quality:** use a separate labeled dataset via `quality_data` and explicit label
  mapping. `quality_batch_sizes` selects sizes independently, defaults to `[1]`, and
  has no artificial upper bound. Each size processes the same labeled set, including
  its partial final batch. No speed buckets or 400-example quotas are imposed.
- `quality` does not need this speed corpus. `report` rebuilds a run's summary;
  `table` aggregates speed baselines, not labeled quality metrics. OOM never silently
  reduces a batch size or changes precision. Native batch capability and GPU support
  still require validation for the selected model and configuration.

Quality-batching has CPU interface tests but no GPU-quality validation at this stage.
There are no safety labels in this corpus; do not use it as `quality_data` or infer
quality preservation from unchanged verdicts. Updating these usage notes does not
change the pinned snapshot, messages, indices, or historical approvals.

## Provenance and Limits

The historical 1,600 texts are retained as user-message seed material. Additional
records use model-independent prefixes from the previously pinned EN/RU source pool,
with deterministic synthetic role assignment and reference exchanges. Assistant
records are **not claimed to be observed LLM responses**. No external LLM APIs were used.

Wikipedia source: `wikimedia/wikipedia`, snapshot `20231101`, commit
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Nested provenance preserves article URL,
title, contributor-history link, original excerpt offsets and hashes, and subsequent
prefix/role transformations. Wikipedia-derived text remains CC BY-SA 3.0; original
template-generated synthetic text is CC0. See `LICENSE.md` for scope and attribution.

CPU preparation was checked for eight models, all 8,000 selected inputs were
revalidated, and repeat builds produced identical manifests and data files.
**GPU compatibility, latency, classification accuracy and safety are not evaluated.**
There are no safety targets; do not compute accuracy/F1/FNR from this pool or interpret
unchanged verdicts as evidence that safety quality is preserved.

The historical `v3` tag and commit `49c2c0b069290c29d45c9c42e41cac4bea344649`
remain available. v3 used intersections and single text inputs; its results and
approvals do not transfer to this release.
