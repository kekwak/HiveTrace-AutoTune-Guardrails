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
[ -f "$ENV/setup/status.json" ] || "$PY13" tools/benchmark.py setup --profile team-rtx-quant --environment "$ENV"
grep -q '"status": "passed"' "$ENV/setup/status.json"
PY="$ENV/venv/bin/python"

step speed-corpus
[ -d "$DATA/role-corpus" ] || "$PY" tools/benchmark.py fetch-data --output "$DATA/role-corpus"

step tails-v2
# pyarrow нет в lock У1: parquet читается во временной uv-среде, u2-env не меняется.
if [ ! -d "$DATA/quality" ]; then
  uv run --no-project --python 3.13 --with pandas==2.2.3 --with pyarrow --with huggingface_hub python -c \
    "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('AutotuneGuardrails/tails', 'data/tails.parquet', repo_type='dataset', revision='v2'), '$DATA/tails_v2.parquet')"
  uv run --no-project --python 3.13 --with pandas==2.2.3 --with pyarrow python -m u2_quant.tails "$DATA/tails_v2.parquet" "$DATA/quality-src"
  "$PY" tools/benchmark.py prepare-data --spec "$DATA/quality-src/import.json" --output "$DATA/quality"
fi

step speed-indices
[ -d "$DATA/speed" ] || "$PY" tools/benchmark.py prepare-data --spec configs/u2/speed_import.json --output "$DATA/speed"

step configs
"$PY" -m u2_quant.configs --speed-manifest "$(sha256sum "$DATA/speed/manifest.json" | cut -d' ' -f1)" \
  --quality-manifest "$(sha256sum "$DATA/quality/manifest.json" | cut -d' ' -f1)"
step done
