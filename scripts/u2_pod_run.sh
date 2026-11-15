#!/bin/bash
# Стадии U2 FP8 на поде; результаты — /workspace/u2-runs/<model>-<stage>.
#   bash scripts/u2_pod_run.sh <stage> <gliner|hive> [--dry-run]
# Порядок: preflight → baseline → quality-bf16c → pack → search-fp8 → quality-fp8 → confirm → pack
set -euo pipefail
STAGE=${1:?stage} MODEL=${2:?model} DRY=${3:-}
case "$MODEL" in gliner|hive) ;; *) echo "unknown model: $MODEL" >&2; exit 2 ;; esac
ENV=/workspace/u2-env RUNS=${U2_RUNS:-/workspace/u2-runs}
export HF_HOME=/workspace/u2-hf COPYFILE_DISABLE=1   # без ._* в tar на macOS
cd "$(dirname "$0")/.."
shopt -s nullglob
PY="$ENV/venv/bin/python"
BENCH="$PY tools/benchmark.py"
C=configs/u2/$MODEL
run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || "$@"; }
[ "$DRY" = "--dry-run" ] || mkdir -p "$RUNS"

case "$STAGE" in
  preflight)
    run $PY -m u2_quant.preflight
    run $BENCH preflight --config "${C}_fp8_confirm.json" --environment "$ENV" --output "$RUNS/$MODEL-preflight" ;;
  baseline)
    run $BENCH run --config "${C}_bf16c_baseline.json" --environment "$ENV" --output "$RUNS/$MODEL-baseline" ;;
  quality-bf16c)
    run $BENCH quality --config "${C}_bf16c_quality.json" --environment "$ENV" --output "$RUNS/$MODEL-quality-bf16c" ;;
  search-fp8)
    run $BENCH run --config "${C}_fp8_search.json" --environment "$ENV" --baseline-ref "$RUNS/$MODEL-baseline" \
      --output "$RUNS/$MODEL-search-fp8" ;;
  quality-fp8)
    run $BENCH quality --config "${C}_fp8_quality.json" --environment "$ENV" --output "$RUNS/$MODEL-quality-fp8" ;;
  confirm)
    run $BENCH run --config "${C}_fp8_confirm.json" --environment "$ENV" --mode confirm --output "$RUNS/$MODEL-confirm-fp8" ;;
  pack)
    if [ "$DRY" = "--dry-run" ]; then ITEMS=("$MODEL-*"); else ITEMS=(); for d in "$RUNS/$MODEL"-*; do
      [ -d "$d" ] && ITEMS+=("$(basename "$d")"); done; fi
    [ ${#ITEMS[@]} -gt 0 ] || { echo "nothing to pack for $MODEL" >&2; exit 1; }
    run tar czf "$RUNS/$MODEL-results-$(date +%Y%m%d-%H%M).tgz" -C "$RUNS" --exclude='*.tgz' "${ITEMS[@]}" ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
