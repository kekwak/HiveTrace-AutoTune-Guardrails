"""Package audited sources and a thin private Kaggle launcher; no measurement logic."""
import argparse
import ast
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output, reuse_environment=None):
    files = []
    for package in ("u1_benchmark", "experiments", "followup", "speed_dataset"):
        files += list((ROOT / package).glob("*.py"))
    files += [ROOT / n for n in ("tools/benchmark.py", "speed_dataset/contracts.json", "env/team_environment.lock.json", "env/team_environment.lock.txt")]
    payload = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            if path.suffix == ".py":
                ast.parse(data, filename=name)
            manifest[name] = hashlib.sha256(data).hexdigest()
            z.writestr(name, data)
        z.writestr("bundle_manifest.json", json.dumps(manifest, sort_keys=True))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".ipynb").exists():
        raise ValueError("Use a new bundle name; do not overwrite attempts")
    output.write_bytes(payload.getvalue())
    checksum = hashlib.sha256(payload.getvalue()).hexdigest()
    code = '''import base64, hashlib, io, json, os, subprocess, sys, zipfile, uuid, shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''
    code += f"PAYLOAD = {base64.b64encode(payload.getvalue()).decode()!r}\nSHA256 = {checksum!r}\n"
    code += '''raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == SHA256
ROOT = Path('/kaggle/working') / ('u1-handoff-' + uuid.uuid4().hex[:8])
ROOT.mkdir()
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for info in archive.infolist():
        assert (ROOT / info.filename).resolve().is_relative_to(ROOT)
    archive.extractall(ROOT)
manifest = json.loads((ROOT / 'bundle_manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
ENV = Path('/kaggle/temp') / (ROOT.name + '-env')
print('Verified source bundle', SHA256, 'files', len(manifest))
print('Results', ROOT)
with (ROOT / 'setup-launcher.log').open('w') as log:
    result = subprocess.run([sys.executable, 'tools/benchmark.py', 'setup', '--environment', str(ENV)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
print((ROOT / 'setup-launcher.log').read_text()[-2000:])
if ENV.with_suffix('.zip').exists():
    shutil.copyfile(ENV.with_suffix('.zip'), ROOT / 'setup-diagnostics.zip')
assert result.returncode == 0, 'Setup failed: preserve diagnostics; do not change lock automatically'
'''
    if reuse_environment:
        begin = code.index("ENV = Path('/kaggle/temp')")
        code = code[:begin] + f"ENV = Path({reuse_environment!r})\n" + '''print('Results', ROOT)
assert json.loads((ENV / 'setup/status.json').read_text())['status'] == 'passed'
assert (ENV / 'setup/lock.json').read_bytes() == (ROOT / 'env/team_environment.lock.json').read_bytes()
subprocess.run([str(ENV / 'venv/bin/python'), 'tools/benchmark.py', '_probe',
    '--environment', str(ENV), '--output', str(ROOT / 'reuse-probe')], cwd=ROOT, check=True)
if ENV.with_suffix('.zip').exists():
    shutil.copyfile(ENV.with_suffix('.zip'), ROOT / 'setup-diagnostics.zip')
'''
    run = '''import time, signal
results = []
for model in ('gliner', 'halo', 'koala', 'llama', 'laya', 'hive', 'open', 'shield'):
    cfg = {'schema_version': 2, 'model': model, 'baseline': {'backend': 'eager'},
           'data': {'profile': 'published'}, 'mode': 'smoke', 'timeout_seconds': 3600}
    config = ROOT / (model + '-smoke.json')
    config.write_text(json.dumps(cfg))
    output = ROOT / 'runs' / model
    print('START', model, flush=True)
    log_path = ROOT / (model + '-launcher.log')
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, 'tools/benchmark.py', 'run', '--config', str(config),
            '--environment', str(ENV), '--output', str(output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            process.wait(timeout=3700)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    summary = json.loads((output / 'summary.json').read_text()) if (output / 'summary.json').exists() else {'status': 'incomplete'}
    results.append({'model': model, 'status': summary['status'], 'errors': summary.get('errors', [])})
    print(results[-1], flush=True)
    for error in output.rglob('error.txt'):
        if '.private' not in error.parts:
            print(error.read_text()[-3500:], flush=True)
    (ROOT / 'smoke-matrix.json').write_text(json.dumps(results, indent=2))
with zipfile.ZipFile(ROOT / 'handoff-smoke-results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in [ROOT / 'smoke-matrix.json', ROOT / 'setup-diagnostics.zip', *sorted((ROOT / 'runs').glob('*.zip'))]:
        if p.exists():
            z.write(p, p.name)
print('Download', ROOT / 'handoff-smoke-results.zip')
print('No full screening was run. Stop the GPU session after downloading results.')
'''
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# У1: smoke восьми адаптеров\nПриватный launcher консольного скрипта. T4 ×2, используется только GPU 0; Internet On. Не запускает полный screening. Отдельный venv, lock не меняется. Максимум час на модель; ошибка сохраняется и не превращается в pass."]},
        *[{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [source]} for source in (code, run)]]
    for i, cell in enumerate(cells):
        cell["id"] = "u1-handoff-" + str(i)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "cells": cells}
    for source in (code, run):
        ast.parse(source)
    output.with_suffix(".ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    print(json.dumps({"bundle": str(output), "sha256": checksum, "files": len(manifest)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-environment", help="Existing verified environment in the same Kaggle session")
    args = parser.parse_args()
    build(args.output, args.reuse_environment)
