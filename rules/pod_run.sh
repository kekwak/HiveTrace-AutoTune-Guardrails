#!/bin/bash
cd /workspace/guardrails
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
./u1-rtx-env/venv/bin/python rules/drive.py llama laya koala gliner
./u1-rtx-env/venv/bin/python rules/table.py > /dev/null
