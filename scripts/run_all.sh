#!/usr/bin/env bash
# Все модели × все методы на одной GPU: качество, скорость, сравнения и общая сводка reports/SUMMARY.md.
#   ./scripts/run_all.sh
#   MODELS="koala gliner-guard" ./scripts/run_all.sh --limit-per-split 50
# На T4 по умолчанию без WildGuard: эталон в fp16 (14.5 ГБ) не помещается в 15 ГБ.
set -uo pipefail
MODELS="${MODELS:-gliner-guard koala hivetrace-guard-pro openguardrails}"
PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."
mkdir -p logs

$PY -m guard_eval check | tee logs/check.log
status=0
for model in $MODELS; do
  ./scripts/quantize_model.sh "$model" "$@" 2>&1 | tee -a "logs/${model}.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
done
$PY -m guard_eval summary
echo "логи: logs/, сводка: reports/SUMMARY.md, прогоны для коммита: results/quality/runs/"
exit $status
