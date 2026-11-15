#!/bin/bash
# Локальная CPU-среда для тестов: версии из env/team_rtx.lock.json У1 + torchao,
# без CUDA-пакетов (nvidia-*, cuda-*, triton); torch* — CPU-сборки тех же версий.
set -euo pipefail
cd "$(dirname "$0")/.."
LOCK=${LOCK:-env/team_rtx.lock.json}
VENV=.venv-u2-tests

uv venv --python 3.13 "$VENV"
python3 - "$LOCK" > "$VENV/requirements.txt" <<'EOF'
import json, re, sys
for p in json.load(open(sys.argv[1]))["packages"]:
    name, version = p["name"], p["version"]
    if re.match(r"(nvidia-|cuda-)", name) or name == "triton":
        continue
    if name == "onnxruntime-gpu":
        name = "onnxruntime"
    print(f"{name}=={version.split('+')[0]}")
print("torchao==0.17.0")
print("nbformat")   # только для тестов ноутбуков У1
print("pip")        # тесты У1 вызывают python -m pip
EOF
uv pip install --python "$VENV/bin/python" -r "$VENV/requirements.txt"
