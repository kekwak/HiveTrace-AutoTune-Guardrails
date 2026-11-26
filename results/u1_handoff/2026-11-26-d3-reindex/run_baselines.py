"""Sequential standard CLI baselines under one explicitly approved time budget."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BASE = Path('/workspace/u1-new-families-20261125')
ROOT = BASE / 'baselines-20261126'
STATE = ROOT / 'baseline-launcher.json'
assert not STATE.exists()
assert json.loads((ROOT / 'accuknox-b4-smoke/summary.json').read_text())['status'] == 'preflight_passed'
os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HOME=str(BASE / 'hf-cache'),
                  HF_HUB_DISABLE_IMPLICIT_TOKEN='1', USE_HUB_KERNELS='NO')
for key in ('CUDA_LAUNCH_BLOCKING', 'HF_DEACTIVATE_ASYNC_LOAD', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ.pop(key, None)
start = time.monotonic()
deadline = start + 12300  # Leave time for download and Pod stop inside the $5 budget.
state = {'started_at': time.time(), 'status': 'running', 'budget_usd': 5,
         'hourly_usd': 1.39, 'work_deadline_seconds': 12300, 'runs': []}

def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATE)

save()
try:
    for name in ('protectai', 'accuknox'):
        remaining = int(deadline - time.monotonic()) - 90
        row = {'model': name, 'status': 'not_run'}
        state['runs'].append(row)
        if remaining <= 120:
            save()
            continue
        config = json.loads((ROOT / (name + '.json')).read_text())
        assert config['mode'] == 'baseline' and 'candidate' not in config
        assert not config.get('pilot', False) and not config.get('profile_trace', False)
        config['timeout_seconds'] = min(config['timeout_seconds'], remaining)
        config_path = ROOT / (name + '-bounded.json')
        config_path.write_text(json.dumps(config, indent=2))
        output = ROOT / (name + '-baseline')
        assert not output.exists() and not Path(str(output) + '.zip').exists()
        row.update(status='running', started_at=time.time())
        save()
        print('START', name, flush=True)
        with (ROOT / (name + '-baseline.log')).open('w') as log:
            process = subprocess.Popen([str(Path(config['environment']) / 'venv/bin/python'),
                'tools/benchmark.py', 'run', '--config', str(config_path), '--environment',
                config['environment'], '--output', str(output)], cwd=BASE / 'repo',
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                row['exit_code'] = process.wait(timeout=remaining + 30)
            except BaseException as error:
                row['interruption'] = type(error).__name__
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        report = output / 'summary.json'
        row['status'] = json.loads(report.read_text())['status'] if report.exists() else 'incomplete'
        row['finished_at'] = time.time()
        save()
        print(json.dumps(row), flush=True)
    state['status'] = 'finished'
finally:
    state['finished_at'] = time.time()
    save()
