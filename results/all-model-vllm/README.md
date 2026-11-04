# Evidence for the eight-model vLLM report

Main report: [docs/vllm-adapters.md](../../docs/vllm-adapters.md).

- `model-status.json`: final status, coverage and SHA256 of each model report.
  Missing comparisons use null drift fields rather than zero-error claims.
- `*-report.json`: eight final aggregate reports, including incomplete
  OpenGuardrails and HaloGuard checks. No private texts or record predictions.
- `sample-manifest.json`, `sample-manifest-2-per-split.json`: exact 48/12-record
  selections. Rebuild private JSON with `src.vllm_probe.prepare_sample`.
- `halo-diagnostics.json`: sanitized native-reference failure stack.
- `vllm-small-1793806511-controller.json`: final shared-run deadline, model order
  and cleanup evidence; not an executable controller or reproduction dependency.
- `final-server-status.json`: final readback, zero active server assignments.

The reports are immutable experimental evidence. Their source hashes refer
to the executed workspace before cleanup; removed orchestration/prototype
paths are not current dependencies. Superseded attempts, duplicate summaries,
allocation bookkeeping and old long-run artifacts are omitted from this payload.

Reproduction code consists of the retained adapters, three GPU workers,
private-sample builder and ShieldGemma patch. No VM provisioning or credential
management scripts are shipped.
