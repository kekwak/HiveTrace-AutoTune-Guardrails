#!/bin/bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf
models=${*:-koala llama laya gliner}
cd /workspace

if [ ! -d guardrails/.git ]; then
  git clone --branch results/haloguard-gliner-u1 https://github.com/kekwak/HiveTrace-AutoTune-Guardrails guardrails-src
  cp -rn guardrails-src/. guardrails/ && rm -rf guardrails-src
fi
cd guardrails
git fetch -q origin results/haloguard-gliner-u1
git checkout -q 100ce48

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
"$(uv python find 3.13)" tools/benchmark.py setup --profile team-rtx --environment ./u1-rtx-env
py=./u1-rtx-env/venv/bin/python
[ -d data/speed/role-corpus ] || $py tools/benchmark.py fetch-data --output data/speed/role-corpus

$py - <<EOF
import json
from huggingface_hub import snapshot_download
for m in "$models".split():
    spec = json.load(open(f"configs/v4/{m}.json"))["baseline"]
    for source in {json.dumps(s, sort_keys=True) for s in (spec["weights"], spec["tokenizer"], *spec.get("sdk_assets", {}).values())}:
        source = json.loads(source)
        print(m, snapshot_download(source["repo"], revision=source["revision"]))
EOF
if [ ! -f data/speed/rules-v4.json ]; then
  $py -c "
import json
spec = json.load(open('configs/v4/reindex.json'))
spec['configs'] = [c for c in spec['configs'] if c['name'] in '$models'.split()]
json.dump(spec, open('data/speed/rules-reindex.json', 'w'), indent=2)"
  $py tools/benchmark.py prepare-data --spec data/speed/rules-reindex.json --output data/speed/rules-v4 > data/speed/rules-v4.tmp
  mv data/speed/rules-v4.tmp data/speed/rules-v4.json && cat data/speed/rules-v4.json
fi

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
