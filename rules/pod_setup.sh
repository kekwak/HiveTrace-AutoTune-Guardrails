#!/bin/bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf
cd /workspace

if [ ! -d guardrails/.git ]; then
  git clone --branch results/haloguard-gliner-u1 https://github.com/kekwak/HiveTrace-AutoTune-Guardrails guardrails-src
  cp -rn guardrails-src/. guardrails/ && rm -rf guardrails-src
fi
cd guardrails
git checkout -q 2b09a3e

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
"$(uv python find 3.13)" tools/benchmark.py setup --profile team-rtx --environment ./u1-rtx-env
./u1-rtx-env/venv/bin/python tools/benchmark.py fetch-data

./u1-rtx-env/venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
from u1_benchmark.contracts import MODEL_SPECS
for m in ("koala", "llama", "laya", "gliner"):
    print(m, snapshot_download(MODEL_SPECS[m]["repo"], revision=MODEL_SPECS[m]["revision"]))
EOF
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
