# Historical ShieldGemma T4 evidence

These aggregate artifacts support the historical section of
[the current report](../../docs/vllm-adapters.md). The final 12-record wrapper
comparison is in `results/all-model-vllm/shield-report.json` instead.

- `smoke-unpatched-status.json`: reproduced FP32 first-request shared-memory
  failure (81,920 bytes required versus the T4 limit of 65,536).
- `patch-manifest.json`, `smoke-patched-status.json`: narrow SM75/FP32/head256
  tile16 patch; caps unchanged; patched scoring completed.
- `smoke-comparison.json`, `smoke-environment.json`: six synthetic comparisons,
  max P(Yes) delta 8.3819e-9, no verdict flips, pinned environment/source hashes.
- `pilot-aggregate-report.json`: 24-record four-policy pilot, zero flips and
  max policy delta <1e-6. Output rows lack original user context and are diagnostic.

No complete dataset-quality or repeated-throughput claim follows from these
artifacts. Cancelled full runs, failed preparation attempts, progress journals
and their temporary scripts are not part of the push payload. Historical
source hashes are preserved as captured, not regenerated after cleanup.
