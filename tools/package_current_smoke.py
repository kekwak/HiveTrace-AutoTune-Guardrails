"""Build a private Kaggle launcher for the current CLI in smoke mode only."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_overlay(source, folder):
    from u1_benchmark.contracts import model_contract
    from u1_benchmark.datasets import load_speed

    source, folder = Path(source), Path(folder)
    expected = json.loads((source / "configs/benchmark-v3/open_baseline.json").read_text())["data"]["manifest_sha256"]
    _, manifest = load_speed(folder, "open", "screening", expected)
    release = json.loads((source / "configs/data/role_speed_v4.json").read_text())
    if (manifest["status"] != "ready"
            or manifest["provenance"]["reindexed_models"] != ["open"]
            or manifest["provenance"]["parent_manifest_sha256"] != release["manifest_sha256"]
            or manifest["indices"]["open"]["contract"] != model_contract("open", 3)):
        raise ValueError("Expected the pinned, ready Open task overlay of the common corpus")
    names = ["manifest.json", "REPORT.md", "indices/open.jsonl",
             *[f"buckets/open/B{i}.json" for i in range(1, 5)]]
    for name in names:
        if not (folder / name).is_file():
            raise ValueError("Open overlay file missing: " + name)
    return manifest, names


def build(source, output, reuse_environment=None, previous_root=None, open_diagnostic=False, models=None, dataset_overlay=None):
    import nbformat

    source, output = Path(source).resolve(), Path(output).resolve()
    selected = tuple(models or ("gliner", "halo", "koala", "llama", "laya", "hive", "open", "shield"))
    if not selected or len(set(selected)) != len(selected) or set(selected) - {
            "gliner", "halo", "koala", "llama", "laya", "hive", "open", "shield"}:
        raise ValueError("Unknown or repeated smoke model")
    if open_diagnostic and dataset_overlay is not None:
        raise ValueError("Native Open diagnostic uses the pinned source corpus, not a smoke overlay")
    if not open_diagnostic and "open" in selected and dataset_overlay is None:
        raise ValueError("Open smoke requires --dataset-overlay with the current task-contract index")
    overlay_data = validate_overlay(source, dataset_overlay) if dataset_overlay is not None else None
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for package in ("u1_benchmark", "experiments", "followup", "speed_dataset"):
        paths.extend((source / package).glob("*.py"))
    paths.extend((source / "configs/benchmark-v3").glob("*.json"))
    paths.extend(source / name for name in (
        "tools/benchmark.py", "speed_dataset/contracts.json", "configs/data/role_speed_v4.json",
        "env/team_environment.lock.json", "env/team_environment.lock.txt",
        "env/team_rtx.lock.json", "env/team_rtx.lock.txt"))
    if open_diagnostic:
        paths.append(source / "tools/diagnose_open_precision.py")
    checksums = {}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            raw = path.read_bytes()
            if path.suffix == ".py":
                ast.parse(raw, filename=str(path.relative_to(source)))
            name = str(path.relative_to(source))
            checksums[name] = hashlib.sha256(raw).hexdigest()
            archive.writestr(name, raw)
        if dataset_overlay:
            folder = Path(dataset_overlay)
            manifest, names = overlay_data
            overlay = {"base_manifest_sha256": manifest["provenance"]["parent_manifest_sha256"],
                       "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
                       "corpus_sha256": manifest["files"]["corpus.jsonl"], "files": names}
            for name in names:
                raw = (folder / name).read_bytes()
                checksums["data_overlay/" + name] = hashlib.sha256(raw).hexdigest()
                archive.writestr("data_overlay/" + name, raw)
            raw = json.dumps(overlay, sort_keys=True).encode()
            checksums["data_overlay/overlay.json"] = hashlib.sha256(raw).hexdigest()
            archive.writestr("data_overlay/overlay.json", raw)
        archive.writestr("bundle_manifest.json", json.dumps(checksums, sort_keys=True, indent=2))
    raw = stream.getvalue()
    checksum = hashlib.sha256(raw).hexdigest()
    (output / "sources.zip").write_bytes(raw)
    code = """import base64, hashlib, io, json, os, subprocess, sys, zipfile, uuid, shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
PAYLOAD = __PAYLOAD__
SHA256 = __SHA__
raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == SHA256
ROOT = Path('/kaggle/working') / ('u1-current-smoke-' + uuid.uuid4().hex[:8])
ROOT.mkdir()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for info in z.infolist():
        assert (ROOT / info.filename).resolve().is_relative_to(ROOT)
    z.extractall(ROOT)
for name, expected in json.loads((ROOT / 'bundle_manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
ENV = __ENVIRONMENT__
PREVIOUS = __PREVIOUS__
DATA_OVERRIDE = None
if PREVIOUS is not None:
    previous_archive = PREVIOUS / 'interrupted-smoke.zip'
    assert not previous_archive.exists()
    with zipfile.ZipFile(previous_archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in [PREVIOUS / 'smoke-matrix.json', PREVIOUS / 'bundle_manifest.json',
                     PREVIOUS / 'setup-diagnostics.zip', PREVIOUS / 'fetch-data.log',
                     PREVIOUS / 'setup-launcher.log', *sorted((PREVIOUS / 'runs').glob('*.zip'))]:
            if path.exists():
                z.write(path, path.name)
    from IPython.display import FileLink, display
    display(FileLink(str(previous_archive.relative_to('/kaggle/working'))))
print('Current source bundle:', SHA256)
print('Results:', ROOT)
""".replace("__PAYLOAD__", repr(base64.b64encode(raw).decode())).replace("__SHA__", repr(checksum))
    code = code.replace("__ENVIRONMENT__", "Path(" + repr(reuse_environment) + ")" if reuse_environment
                        else "Path('/kaggle/temp') / (ROOT.name + '-env')")
    code = code.replace("__PREVIOUS__", "Path(" + repr(previous_root) + ")" if previous_root else "None")
    setup = """def command(args, name, timeout):
    path = ROOT / name
    with path.open('w') as log:
        result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        print(path.read_text()[-5000:])
        raise RuntimeError('Command failed; log: ' + name)
    return path

if (ENV / 'setup/status.json').exists():
    assert json.loads((ENV / 'setup/status.json').read_text())['status'] == 'passed'
    assert json.loads((ENV / 'setup/profile.json').read_text())['profile'] == 'team-t4-laya'
    assert (ENV / 'setup/lock.json').read_bytes() == (ROOT / 'env/team_rtx.lock.json').read_bytes()
    command([str(ENV / 'venv/bin/python'), 'tools/benchmark.py', '_probe', '--environment', str(ENV),
             '--output', str(ROOT / 'reuse-probe')], 'reuse-probe.log', 180)
else:
    command([sys.executable, 'tools/benchmark.py', 'setup', '--profile', 'team-t4-laya',
             '--environment', str(ENV)], 'setup-launcher.log', 4200)
assert json.loads((ENV / 'setup/status.json').read_text())['status'] == 'passed'
PYTHON = str(ENV / 'venv/bin/python')
SETUP_ARCHIVE = ENV.parent / (ENV.name + '.zip')
if SETUP_ARCHIVE.exists():
    shutil.copyfile(SETUP_ARCHIVE, ROOT / 'setup-diagnostics.zip')
print('Isolated T4 environment ready. Host packages unchanged.')
"""
    data = """# Use only an already available secret. Never paste or print credentials here.
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret('HF_TOKEN')
except Exception:
    print('No HF_TOKEN secret available; Shield access will be checked explicitly.')
else:
    os.environ['HF_TOKEN'] = hf_token
    del hf_token
    print('Existing HF_TOKEN secret available; value is not recorded.')
command([PYTHON, 'tools/benchmark.py', 'fetch-data'], 'fetch-data.log', 900)
access = subprocess.run([PYTHON, 'tools/benchmark.py', 'auth-check', '--model', 'shield'],
                        cwd=ROOT, capture_output=True, text=True, timeout=180)
SHIELD_ACCESS = access.returncode == 0
(ROOT / 'shield-access.json').write_text(json.dumps({'model': 'shield', 'status': 'passed' if SHIELD_ACCESS else 'blocked_auth'}))
print('Pinned role corpus verified. Shield access:', SHIELD_ACCESS)
"""
    if dataset_overlay:
        data += """# Preserve the published snapshot; apply only the explicitly versioned Open index.
BASE_DATA = ROOT / 'data/speed/role-corpus'
assert hashlib.sha256((BASE_DATA / 'manifest.json').read_bytes()).hexdigest() == json.loads((ROOT / 'data_overlay/overlay.json').read_text())['base_manifest_sha256']
overlay = json.loads((ROOT / 'data_overlay/overlay.json').read_text())
assert hashlib.sha256((BASE_DATA / 'corpus.jsonl').read_bytes()).hexdigest() == overlay['corpus_sha256']
CURRENT_DATA = ROOT / 'data/speed/role-corpus-open-task'
shutil.copytree(BASE_DATA, CURRENT_DATA)
for name in overlay['files']:
    path = CURRENT_DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'data_overlay' / name, path)
DATA_OVERRIDE = {'path': str(CURRENT_DATA), 'manifest_sha256': overlay['manifest_sha256']}
print('New Open task contract; shared corpus bytes unchanged:', overlay['manifest_sha256'])
"""
    smoke = """import signal
results = []
for model in ('gliner', 'halo', 'koala', 'llama', 'laya', 'hive', 'open', 'shield'):
    cfg = json.loads((ROOT / 'configs/benchmark-v3' / (model + '_baseline.json')).read_text())
    cfg.update(mode='smoke', timeout_seconds=3600)
    if DATA_OVERRIDE is not None:
        cfg['data'] = DATA_OVERRIDE
    assert cfg['schema_version'] == 3 and not cfg.get('candidate')
    config = ROOT / (model + '-smoke.json')
    config.write_text(json.dumps(cfg))
    output = ROOT / 'runs' / model
    print('START', model, flush=True)
    if model == 'shield' and not SHIELD_ACCESS:
        results.append({'model': model, 'status': 'blocked_auth', 'reason': 'Shield config/weight metadata inaccessible'})
    else:
        with (ROOT / (model + '-launcher.log')).open('w') as log:
            process = subprocess.Popen([sys.executable, 'tools/benchmark.py', 'run', '--config', str(config),
                '--environment', str(ENV), '--output', str(output)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=3700)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        report = output / 'summary.json'
        summary = json.loads(report.read_text()) if report.exists() else {'status': 'incomplete'}
        results.append({'model': model, 'status': summary['status'], 'errors': summary.get('errors', [])})
        for error in output.rglob('error.txt'):
            if '.private' not in error.parts:
                print(error.read_text()[-3500:], flush=True)
    print(results[-1], flush=True)
    (ROOT / 'smoke-matrix.json').write_text(json.dumps(results, indent=2))

archive = ROOT / 'current-smoke-results.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in [ROOT / 'smoke-matrix.json', ROOT / 'bundle_manifest.json', ROOT / 'setup-diagnostics.zip', ROOT / 'shield-access.json',
                 *sorted((ROOT / 'runs').glob('*.zip'))]:
        if path.exists():
            z.write(path, path.name)
print('Smoke complete. No screening or benchmark series were run.')
print('Download:', archive)
from IPython.display import FileLink, display
display(FileLink(str(archive.relative_to('/kaggle/working'))))
if PREVIOUS is not None:
    display(FileLink(str((PREVIOUS / 'interrupted-smoke.zip').relative_to('/kaggle/working'))))
"""
    smoke = smoke.replace("('gliner', 'halo', 'koala', 'llama', 'laya', 'hive', 'open', 'shield')", repr(selected))
    if open_diagnostic:
        data = """command([PYTHON, 'tools/benchmark.py', 'fetch-data'], 'fetch-data.log', 900)
print('Pinned role corpus verified. No credentials needed for OpenGuardrails.')
"""
        smoke = """# Diagnostic only: fresh process and original weights for each precision.
try:
    for dtype in ('float16', 'bfloat16'):
        command([PYTHON, 'tools/diagnose_open_precision.py', '--environment', str(ENV),
                 '--dataset', str(ROOT / 'data/speed/role-corpus'), '--dtype', dtype,
                 '--input-profile', 'native',
                 '--output', str(ROOT / 'diagnostic' / dtype)], dtype + '.log', 900)
        state = json.loads((ROOT / 'diagnostic' / dtype / 'precision.json').read_text())
        print('PRECISION', dtype, state['status'], flush=True)
        for case in state.get('cases', []):
            print(case['id'], case['index']['total_tokens'], repr(case['raw_output']),
                  case.get('parse_error'), flush=True)
        if state.get('error'):
            print(state['error'], flush=True)
finally:
    archive = ROOT / 'open-precision-diagnostic.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if path.is_file() and (relative.parts[0] == 'diagnostic' or
                    relative.as_posix() in ('bundle_manifest.json', 'setup-launcher.log',
                        'float16.log', 'bfloat16.log', 'setup-diagnostics.zip')):
                z.write(path, str(relative))
    from IPython.display import FileLink, display
    display(FileLink(str(archive.relative_to('/kaggle/working'))))
print('No screening or benchmark series were run.')
"""
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# У1: актуальный GPU-smoke\n\nТолько `mode=smoke`, восемь baseline-адаптеров, ролевой корпус v4. "
            "Приватный launcher текущего консольного скрипта, T4 ×2, используется GPU 0. "
            "Без screening и серий измерений. Internet On. Новый venv, прежние 97 закреплённых wheels. "
            "HF_TOKEN берётся только из уже доступного Kaggle Secret; отсутствие доступа сохраняется как blocker."),
        *[nbformat.v4.new_code_cell(text) for text in (code, setup, data, smoke)]],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}})
    if open_diagnostic:
        notebook.cells[0].source = ("# OpenGuardrails 0124: диагностика FP16/BF16\n\n"
            "Четыре неизменных входа, нативный шаблон 0124, до 10 токенов. "
            "Отдельный процесс на точность; никаких benchmark-серий и изменения parser. "
            "T4, только GPU 0. Диагностический результат не является smoke approval.")
    elif models:
        notebook.cells[0].source = ("# У1: адресный GPU-smoke\n\n"
            "Только `mode=smoke`: " + ", ".join(selected) + ". Ролевой корпус v4, T4 GPU 0. "
            "Без benchmark-серий; остальные модели не запускаются. "
            "Авторизация только через разрешённый Kaggle Secret, значение не сохраняется.")
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nbformat.write(notebook, output / "current-smoke.ipynb")
    (output / "bundle_manifest.json").write_text(json.dumps(checksums, sort_keys=True, indent=2))
    print(json.dumps({"notebook": str(output / "current-smoke.ipynb"), "sha256": checksum,
                      "files": len(checksums), "bytes": len(raw)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-environment")
    parser.add_argument("--previous-root")
    parser.add_argument("--open-diagnostic", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--dataset-overlay", type=Path)
    args = parser.parse_args()
    build(args.source, args.output, args.reuse_environment, args.previous_root, args.open_diagnostic, args.models, args.dataset_overlay)
