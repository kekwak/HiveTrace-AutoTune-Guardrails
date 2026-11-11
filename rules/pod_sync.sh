#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
tar c --exclude=__pycache__ rules | ssh -p "$2" -o StrictHostKeyChecking=accept-new "$1" 'mkdir -p /workspace/guardrails && tar x -C /workspace/guardrails && ls /workspace/guardrails/rules'
