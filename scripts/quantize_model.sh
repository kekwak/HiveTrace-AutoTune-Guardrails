#!/usr/bin/env bash
# Одна модель, все методы динамического квантования: качество (+ скорость) + сравнение с эталоном без квантования.
#   ./scripts/quantize_model.sh koala
#   ./scripts/quantize_model.sh openguardrails --limit-per-split 50     # быстрая проверка на подвыборке
#   METHODS="bnb_int8" ./scripts/quantize_model.sh koala              # только выбранные методы
#   SPEED=0 ./scripts/quantize_model.sh koala                         # только качество, без замера скорости
set -uo pipefail
MODEL="${1:?укажите модель: koala | gliner-guard | hivetrace-guard-pro | openguardrails | wildguard | путь к YAML}"
shift
METHODS="${METHODS:-torchao_int8_dyn torchao_fp8_dyn bnb_int8}"
SPEED_FLAG=$([ "${SPEED:-1}" = "1" ] && echo "--speed" || echo "")
PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."

status=0
echo "=== $MODEL: эталон без квантования"
$PY -m guard_eval run --model "$MODEL" --quant none $SPEED_FLAG "$@" || status=1
for method in $METHODS; do
  echo "=== $MODEL: $method"
  # каждый метод в отдельном процессе: память GPU после предыдущего гарантированно освобождена
  $PY -m guard_eval compare --model "$MODEL" --a none --b "$method" $SPEED_FLAG "$@" \
    || { echo "!!! $MODEL / $method: не выполнен (ошибка или метод не поддерживается) — продолжаю"; status=1; }
done
$PY -m guard_eval summary
exit $status
