"""Create explicit run configs and CPU plans; never starts a GPU worker."""
from copy import deepcopy
import json
from pathlib import Path
import sys

base = Path('/workspace/u1-new-families-20261125')
repo = base / 'repo'
sys.path.insert(0, str(repo))
from u1_benchmark.core import sha, write
from u1_benchmark.v4_runner import plan

data = repo / 'data/speed/new-families-d3-20261126'
expected = '846e234bd0dff69ca1d22d460e49b9c4dfdc85e08576f10f01f46bc0dd5cbc2c'
assert sha(data / 'manifest.json') == expected
out = base / 'baselines-20261126'
out.mkdir(exist_ok=False)
envs = {'protectai': '/workspace/u1-encoder-smoke-2026-11-09/u1-env',
        'accuknox': str(base / 'd3-env')}
for name in ('protectai', 'accuknox'):
    config = json.loads((repo / 'configs/v4' / (name + '.json')).read_text())
    config.update(data={'path': str(data), 'manifest_sha256': expected},
                  environment=envs[name], mode='baseline')
    write(out / (name + '.json'), config)
    report = plan(config)
    assert report['status'] == 'ready'
    write(out / (name + '.plan.json'), report)
    print(name, json.dumps({k: report[k] for k in ('model_loads', 'timed_batches', 'warmup_batches')}))
    if name == 'accuknox':
        smoke = deepcopy(config)
        smoke.update(mode='smoke', buckets={'B4': config['buckets']['B4']},
                     batch_sizes_by_bucket={'B4': [2]}, timeout_seconds=600)
        write(out / 'accuknox-b4-smoke.json', smoke)
