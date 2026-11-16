#!/bin/bash
# Подготовка пода RTX 5090 для U2 FP8: код, среда team-rtx-quant, данные, конфиги.
# Только /workspace/u2-*; сетевой том и среды коллег не используются.
# Нужен HF_TOKEN с доступом к приватному AutotuneGuardrails/tails.
#   bash scripts/u2_pod_setup.sh <commit>
set -euo pipefail
COMMIT=${1:?usage: u2_pod_setup.sh <commit>}
REPO=/workspace/u2-repo ENV=/workspace/u2-env DATA=/workspace/u2-data LOGS=/workspace/u2-logs
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/u2-hf
mkdir -p "$LOGS" "$DATA"
step() { echo "== $1 (+${SECONDS}s)" | tee -a "$LOGS/setup-timing.txt"; }
# Шаг пропускается только при готовом результате; частичные каталоги после сбоя удаляются.
ready() { grep -Eqs '"status": *"ready"' "$1/manifest.json"; }

step clone
[ -d "$REPO/.git" ] || git clone --branch research/quantization https://github.com/kekwak/HiveTrace-AutoTune-Guardrails "$REPO"
cd "$REPO"
git fetch -q origin research/quantization
git checkout -q -f "$COMMIT"   # configs/u2 перегенерируются ниже

step uv-python
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
PY13=$(uv python find 3.13)

step env-team-rtx-quant
grep -qs '"status": "passed"' "$ENV/setup/status.json" || {
  rm -rf "$ENV"; "$PY13" tools/benchmark.py setup --profile team-rtx-quant --environment "$ENV"; }
grep -q '"status": "passed"' "$ENV/setup/status.json"
PY="$ENV/venv/bin/python"

step speed-corpus
[ -d "$DATA/role-corpus" ] || "$PY" tools/benchmark.py fetch-data --output "$DATA/role-corpus"

step tails-v2
# pyarrow нет в lock У1: parquet читается во временной uv-среде, u2-env не меняется.
ready "$DATA/quality" || {
  rm -rf "$DATA/quality-src" "$DATA/quality"
  uv run --no-project --python 3.13 --with pandas==2.2.3 --with pyarrow --with huggingface_hub python -c \
    "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('AutotuneGuardrails/tails', 'data/tails.parquet', repo_type='dataset', revision='v2'), '$DATA/tails_v2.parquet')"
  uv run --no-project --python 3.13 --with pandas==2.2.3 --with pyarrow python -m u2_quant.tails "$DATA/tails_v2.parquet" "$DATA/quality-src"
  "$PY" tools/benchmark.py prepare-data --spec "$DATA/quality-src/import.json" --output "$DATA/quality"; }
ready "$DATA/quality"

step gliner-quality
# SDK GLiNER дописывает точку к ~34% текстов; У1 такие изменения запрещает — даём gliner уже нормализованные тексты.
ready "$DATA/quality-gliner" || {
  rm -rf "$DATA/quality-gliner-src" "$DATA/quality-gliner"
  "$PY" -m u2_quant.gliner_corpus "$DATA/quality/corpus.jsonl" "$DATA/quality-gliner-src"
  "$PY" tools/benchmark.py prepare-data --spec "$DATA/quality-gliner-src/import.json" --output "$DATA/quality-gliner"; }
ready "$DATA/quality-gliner"

step speed-indices
# Индексы привязаны к identity подготовки, включающей sha адаптера: при смене adapter.py — пересборка.
ADAPTER_SHA=$(sha256sum u2_quant/adapter.py | cut -d' ' -f1)
[ "$(cat "$DATA/speed.adapter_sha256" 2>/dev/null)" = "$ADAPTER_SHA" ] || rm -rf "$DATA/speed"
ready "$DATA/speed" || {
  rm -rf "$DATA/speed"
  "$PY" -m u2_quant.configs   # speed_import ссылается на конфиги с текущим sha адаптера
  "$PY" tools/benchmark.py prepare-data --spec configs/u2/speed_import.json --output "$DATA/speed"; }
ready "$DATA/speed"
echo "$ADAPTER_SHA" > "$DATA/speed.adapter_sha256"

step configs
"$PY" -m u2_quant.configs --speed-manifest "$(sha256sum "$DATA/speed/manifest.json" | cut -d' ' -f1)" \
  --quality-manifest "$(sha256sum "$DATA/quality/manifest.json" | cut -d' ' -f1)" \
  --quality-manifest-gliner "$(sha256sum "$DATA/quality-gliner/manifest.json" | cut -d' ' -f1)"
step done
