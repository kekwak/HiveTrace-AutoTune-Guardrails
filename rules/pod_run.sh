#!/bin/bash
cd /workspace/guardrails
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
B=${1:-1}
shift
models=${*:-llama laya koala gliner}
R=runs/tune-v4; [ "$B" = 1 ] || R=$R-b$B
./u1-rtx-env/venv/bin/python rules/drive.py --batch "$B" $models
./u1-rtx-env/venv/bin/python rules/table.py "$R" > /dev/null
