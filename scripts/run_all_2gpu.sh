#!/usr/bin/env bash
# Для двух GPU (Kaggle 2×T4).
# Фаза 1 — качество: модели делятся между GPU 0 и 1 и считаются параллельно (без замера скорости).
# Фаза 2 — скорость: по очереди на GPU 0, пока вторая простаивает (параллельные замеры мешают друг другу через CPU).
# Качество во второй фазе берётся из кэша, поэтому она пересчитывает только скорость и отчёты.
#   ./scripts/run_all_2gpu.sh
#   MODELS="koala gliner-guard" ./scripts/run_all_2gpu.sh --limit-per-split 50
set -uo pipefail
MODELS="${MODELS:-gliner-guard koala hivetrace-guard-pro openguardrails}"
PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."
mkdir -p logs
$PY -m guard_eval check | tee logs/check.log

read -r -a list <<< "$MODELS"
gpu0=() gpu1=()
for i in "${!list[@]}"; do
  if (( i % 2 == 0 )); then gpu0+=("${list[$i]}"); else gpu1+=("${list[$i]}"); fi
done

phase_quality() {   # $1 — номер GPU, остальные — модели
  local gpu=$1; shift
  for model in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu SPEED=0 ./scripts/quantize_model.sh "$model" "${EXTRA[@]}" > "logs/${model}.quality.log" 2>&1
    echo "[GPU $gpu] качество $model: код $?"
  done
}
EXTRA=("$@")
echo "=== Фаза 1: качество. GPU 0: ${gpu0[*]:-—}; GPU 1: ${gpu1[*]:-—}"
phase_quality 0 "${gpu0[@]}" &
phase_quality 1 "${gpu1[@]}" &
wait

echo "=== Фаза 2: скорость на GPU 0"
status=0
for model in "${list[@]}"; do
  CUDA_VISIBLE_DEVICES=0 SPEED=1 ./scripts/quantize_model.sh "$model" "$@" 2>&1 | tee "logs/${model}.speed.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
done
$PY -m guard_eval summary
echo "логи: logs/, сводка: reports/SUMMARY.md, прогоны для коммита: results/quality/runs/"
exit $status
