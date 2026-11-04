# vLLM adapters for eight guardrail models

Scenario: batch throughput on **Tesla T4**, without quantization. This follows
the [Torch optimization report](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/research/torch-optimizations/docs/optimizations.md),
but does not assume that vLLM automatically optimizes every architecture.

**Result: eight experimental adapter paths, six completed small-sample
comparisons, two unresolved.** No controlled all-model speed comparison or
production-quality equivalence is claimed.

## Main findings

- Native decisions matched on all paired rows for six models. Coverage matters:
  rejected and unsupported inputs remain in the denominators below.
- The custom DeBERTa integration runs the native HF encoder under vLLM
  scheduling. It is **not** fused vLLM attention and can be slower.
- GLiNER needs its native schema/head, and Laya its decision head/calibration;
  returning encoder embeddings alone is not a valid replacement.
- ShieldGemma FP32 needs a narrow Triton tile workaround on the tested T4
  stack. Both attention and final-logit soft caps remain enabled.
- Earlier controlled Laya measurements found no consistent throughput win
  over length-bucketed Torch. These new parity checks do not establish speedups.

## Models and measured status

“Paired” means both native reference and vLLM returned valid comparable outputs.
Native precision and checkpoint revisions are pinned in
`src/vllm_probe/adapter_contracts.py`.

| Model | Integration / native contract | Paired / selected | Verdict changes | Max probability delta |
|---|---|---:|---:|---:|
| HiveTraceGuard-Pro | native decoder; FP16; global greedy, 1 token | 32/48 | 0 | no native probability output |
| OpenGuardrails-Text-4B-0124 | native decoder; FP16; pinned 2510 task template; categories | 0/12 | not measured | not measured |
| ShieldGemma-2b | patched native decoder; FP32; four independent official policies | 12/12 | 0 | 2.1381e-6 |
| HaloGuard1-Gen-0.8B | native decoder pair scoring; FP16; unsafe >=0.65 | 0/12 | not measured | not measured |
| Laya | ModernBERT/custom decision pooler; FP32; native choice temperature | 39/48 | 0 | 5.6427e-6 |
| KoalaAI/Text-Moderation | custom dense HF DeBERTa; FP32; all nine native classes | 34/48 | 0 | 1.1921e-7 |
| Prompt-Guard-2-22M | custom dense HF DeBERTa; FP32; native label order | 40/48 | 0 | 1.7882e-7 |
| GLiNER guard omni | custom encoder + native schema/head; FP32; argmax, tie=safe | 28/48 | 0 | 1.4604e-6 |

Prompt-Guard uses the explicitly selected **hipocap mirror**, whose config
exposes `LABEL_0`/`LABEL_1`; equivalence to Meta's original is unverified.
We preserve these labels rather than inventing a benchmark mapping.

## A1. Native decoder adapters

- **Condition:** the architecture and its kernels must actually execute on
  T4 under vLLM **0.32.0**; successful loading alone is insufficient.
- **How:** `decoder_adapter.py` consumes exact native token IDs from
  `native_preparation.py`. Initial profile: eager engine, capacity 4, token
  budget 1024, prefix caching disabled. No CUDA-graph/prefix-cache win is claimed.
- **Semantics:** Hive/Open use global greedy decoding, not a vocabulary
  restricted to safe/unsafe. Open retains categories and rejects length-limited
  completions even when a partial category happens to parse.
- **Special scoring:** Halo retains the author system prompt, thinking-off,
  `Safety: ` suffix and 0.65 threshold. Shield requires four finite policy
  scores and declares unsafe if any conditional `P(Yes)>=0.5`.
- **Check:** verdict/category parity; finite native probabilities for pair
  models; missing/unexpected outputs are never mapped to safe.

## A2. Native DeBERTa inside vLLM

- **How:** `hf_encoder_vllm_model.py` reconstructs full right-padded batches,
  runs unchanged native HF DeBERTa, and preserves its classifier/context pooler.
  GLiNER instead receives full token hidden states.
- **Conditions:** FP32; complete, contiguous sequences; chunked prefill off;
  `attn_type='encoder_only'`; no external embeddings or pipeline parallelism.
  Checkpoint loading is strict. Only audited DeBERTa v1/v2 types are accepted.
- **Cost:** CPU position inspection, dense padding and native eager kernels.
  This is a compatibility baseline, not evidence of fused attention or compile
  optimizations. Koala/Prompt comparisons used candidate batches of four
  against native singleton forward.
- **Registration:** `registration.py` is a standard `vllm.general_plugins`
  entry point, so spawned engine processes register the custom architectures.
  Parent-only registration did not survive the actual spawned-engine path.

## A3. Preserve custom heads

- **Laya:** the existing `laya_vllm_model.py` executes the decision head inside
  the engine. `LayaChoicesAdapter` applies checkpoint choice temperature to
  marker logits. This is **decision-only**, not the action head or full SDK.
- **GLiNER:** `GlinerSafetyAdapter` keeps native schema processing and heads,
  verifies public label against native classifier logits, and substitutes only
  the encoder via `gliner_bridge.py`. The bridge copies hidden states through
  host memory; it is not zero-copy or claimed faster.
- **Shield:** the version-guarded `shield_t4_patch.py` selects tile16 only for
  SM75 / FP32-sized elements / head256. Weights, masks and soft caps are not
  modified. The measured run records before/after patch source hashes.

## Dataset protocol and coverage

The parent sample has **48 records**, eight selected without replacement per
safe/harm split of the three pinned benchmarks, seed 17. The heavier final
decoder checks use **12 records**, parent selection positions 0 and 4 in each
split, fixed before inference. Exact revisions/indices are committed in
`results/all-model-vllm/sample-manifest*.json`; private texts are not.

Inputs are never silently truncated or replaced with shorter examples. Output
benchmarks lack original requests: encoder/Laya results are standalone-response
diagnostics; Shield uses explicitly empty user context plus output-policy
wording. Hive/Open/Halo reject output-only roles rather than relabel them as
user requests. Pooled input/output results therefore mix normal input and
missing-context diagnostics.

### Per-split native/vLLM pairs

All paired rows in these tables had **zero verdict changes**.

| Dataset / split | Prompt-Guard | KoalaAI | GLiNER | Laya | Hive |
|---|---:|---:|---:|---:|---:|
| input / safe | 8/8 | 7/8 | 8/8 | 8/8 | 8/8 |
| input / harm | 8/8 | 7/8 | 5/8 | 8/8 | 8/8 |
| input-robust / safe | 8/8 | 8/8 | 6/8 | 8/8 | 8/8 |
| input-robust / harm | 8/8 | 8/8 | 3/8 | 8/8 | 8/8 |
| output / safe | 3/8 | 2/8 | 4/8 | 3/8 | 0/8 |
| output / harm | 5/8 | 2/8 | 2/8 | 4/8 | 0/8 |
| **Pooled** | **40/48** | **34/48** | **28/48** | **39/48** | **32/48** |

Prompt-Guard, Koala and Laya rejected 8, 14 and 9 over-limit rows respectively.
GLiNER rejected 6 over-limit rows and 14 whose processor changed original text.
Hive rejected 16 unsupported output-only roles.

Shield paired **2/2 in every split, 12/12 pooled**, scoring all four policies
independently per record. Open and Halo paired **0/2 in every split, 0/12
pooled**: their drift is unmeasured, not zero. Different sample sizes and
coverage rates are not interchangeable. These are **parity checks, not
ground-truth accuracy/F1** or evidence of full-tail equivalence.

## Unresolved cases and experimental limits

The last shared T4 allocation had an explicitly approved twenty-minute cap
and ended after **1093.94 seconds**, including setup and confirmed cleanup.

- **Open:** native reference completed eight supported input rows, rejecting
  four output-only roles. Native loading took about 203.8 seconds; candidate
  startup began at worker elapsed 211.354 seconds. The 300-second whole-worker
  limit left roughly 89 seconds for startup, which did not finish. No candidate
  verdict/category was compared; this does not prove architecture incompatibility.
- **Halo:** FP16 reference loaded but its first forward raised `RuntimeError`
  in `modeling_qwen3_5.py:270`, native `causal_conv1d_fn` / `F.conv1d`.
  No reference verdict or vLLM candidate ran. Only sanitized type/stack locations
  were retained; the precise backend message/root cause is unknown.
- **Isolation caveat:** the shared run preceded a local fix for parent-only
  timeout termination leaving child processes alive. Halo followed timed-out
  Open, so this is not proof of an isolated native-stack defect. The temporary
  orchestration was removed from the push payload; no GPU recheck followed.
- No repeated, identical-workload throughput or peak-memory comparison was
  completed for the new adapters. One-pass timings must not be used as a ranking.
  Near-limit fixtures, A-B-A checks and production edge-case approval remain open.

Measured stack: Tesla T4, vLLM 0.32.0, PyTorch 2.13.0, Transformers 5.19.0;
GLiNER2 2.1.0. Final per-model reports retain exact revisions, source hashes,
versions and coverage. Hashes describe the **executed pre-cleanup snapshot**;
obsolete script paths in those manifests are historical provenance, not current
dependencies. The report JSONs and their hashes were not rewritten during cleanup.

## Historical evidence

- [Laya's earlier full study](../results/vllm-t4/LAYA-FINAL.md): decisions
  matched on 9,711 records twice; 547 were truncated in that older protocol.
  Best matched throughput on two T4s: native 14.66/15.98 vs vLLM 14.23/16.64
  records/s across 96 trials. No consistent vLLM advantage.
- `results/shieldgemma-vllm-t4/` retains the original first-request shared-memory
  failure (81,920 required vs 65,536 available), six-case patched smoke and
  24-record four-policy pilot. Smoke max Δp 8.3819e-9, pilot max Δp <1e-6;
  neither is a complete quality/throughput benchmark.

## Reproduction: only the retained scripts

Build the exact private samples from the separately downloaded snapshots.
The output path must be **outside any Git working tree**; existing files are not
overwritten. This step uses PyArrow but no network or GPU:

```sh
python -m src.vllm_probe.prepare_sample \
  --dataset-root /path/to/datasets/hivetrace \
  --manifest results/all-model-vllm/sample-manifest.json \
  --output /path/outside/repository/sample-48.json
```

Repeat with `sample-manifest-2-per-split.json` and `sample-12.json`. Supply
gated-model credentials privately. In an **existing** T4 runtime, install the
measured dependencies and this package; no retained script provisions a VM:

```sh
pip install vllm==0.32.0 transformers==5.19.0 gliner2==2.1.0
pip install --no-deps .
export VLLM_PLUGINS=guardrail_adapters

for model in koala llama; do
  python -m src.vllm_probe.encoder_gpu_check "$model" /content/sample-48.json /content/check-results
done
for model in gliner laya; do
  python -m src.vllm_probe.custom_gpu_check "$model" /content/sample-48.json /content/check-results
done
python -m src.vllm_probe.decoder_gpu_check hive /content/sample-48.json /content/check-results
for model in open halo shield; do
  python -m src.vllm_probe.decoder_gpu_check "$model" /content/sample-12.json /content/check-results
done
```

Workers intentionally have no resource-lifecycle manager: set your own deadline,
isolate model processes and stop your runtime. Keep terminal logs and temporary
model artifacts private; publish/download **aggregate report JSONs only**.
CPU protocol tests: `python -m pytest -q` (PyArrow enables sample-builder tests).

Evidence index: `results/all-model-vllm/README.md` and `model-status.json`.
Conclusion: six sample comparisons support experimental integration, not a
blanket recommendation to replace every Torch guardrail with vLLM.
