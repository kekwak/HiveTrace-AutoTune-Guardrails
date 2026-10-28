"""Isolated team-environment smoke: no benchmarks, cloud provisioning or publication."""

import gzip
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile


TEXTS = ["What time does the public library open?", "Объясни, как работает библиотечный каталог.",
         " ".join(["The library keeps a catalogue of books."] * 18)]
STAGES = ("cuda", "ort_cuda", "gliner", "halo")


@contextmanager
def isolated_python_env():
    # Kaggle's inherited PYTHONPATH injects host sitecustomize into a new venv.
    keys = ('PYTHONPATH', 'PYTHONHOME')
    saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
    try:
        yield
    finally:
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(saved)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pins(spec):
    return {**spec['package_pins'], **spec['extension_pins_selected_not_yet_validated'],
            **spec['ort_cuda12_runtime_pins_from_our_successful_environment'],
            'nvidia-cuda-nvrtc': '13.0.88', 'nvidia-cudnn-cu13': '9.19.0.56', 'triton': '3.6.0'}


def package_snapshot():
    return dict(sorted((p.metadata['Name'].lower().replace('_', '-'), p.version)
                       for p in importlib.metadata.distributions() if p.metadata.get('Name')))


def target_pip(python):
    # pip >=22.3 can manage a pip-less venv without relying on host ensurepip.
    return [sys.executable, '-m', 'pip', '--python', str(python)]


def check_versions(spec, actual):
    differences = {k: {'expected': v, 'actual': actual.get(k.replace('_', '-'))}
                   for k, v in pins(spec).items() if actual.get(k.replace('_', '-')) != v}
    if differences:
        raise RuntimeError('Installed pins differ: ' + json.dumps(differences))


def archive(work):
    work = Path(work)
    target = work.with_suffix('.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(work.rglob('*')):
            if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:
                continue
            if p.suffix in ('.onnx', '.data', '.pt', '.bin', '.safetensors') or '.onnx.' in p.name:
                continue
            relative = str(p.relative_to(work))
            if p.name.startswith('ort-') and p.suffix == '.json':
                z.writestr(relative + '.gz', gzip.compress(p.read_bytes(), mtime=0))
            else:
                z.write(p, relative)
    return target


def setup_torch(directory):
    import ctypes
    import site
    import torch
    import numpy as np
    import random
    files = sorted({str(p) for base in site.getsitepackages()
                    for p in Path(base).glob('nvidia/cu13/lib/libnvrtc-builtins.so.13.0')})
    if not files:
        raise RuntimeError('Missing CUDA 13 NVRTC builtins required by the team recipe')
    ctypes.CDLL(files[0], mode=ctypes.RTLD_GLOBAL)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('One CUDA T4 required; no CPU fallback')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    write(Path(directory) / 'runtime.json', {'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0), 'threads': torch.get_num_threads(),
        'nvrtc_preload': files[0], 'seed': 42, 'deterministic': 'warn_only'})
    if torch.version.cuda != '13.0' or torch.backends.cudnn.version() != 91900:
        raise RuntimeError('CUDA/cuDNN differs from selected team base; see runtime.json')
    return torch


def preload_ort(directory):
    import onnxruntime as ort
    # CUDA 12 libraries explicitly coexist with CUDA 13 PyTorch in this candidate environment.
    ort.preload_dlls(cuda=True, cudnn=True, directory='')
    ort.print_debug_info()
    write(Path(directory) / 'ort_available.json', {'version': ort.__version__,
                                                'providers': ort.get_available_providers()})
    return ort


def check_ort_events(events):
    nodes = [e for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider')]
    cuda = [e for e in nodes if e['args']['provider'] == 'CUDAExecutionProvider']
    if not cuda or any(e['args']['provider'] != 'CUDAExecutionProvider' for e in nodes):
        raise RuntimeError('Toy graph did not execute exclusively on CUDA')
    return {'executed_cuda_nodes': len(cuda), 'cpu_nodes': 0}


def ort_probe(directory):
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    ort = preload_ort(directory)
    graph = helper.make_graph([helper.make_node('MatMul', ['a', 'b'], ['c'])], 'cuda-smoke',
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [32, 32]) for n in ('a', 'b')],
        [helper.make_tensor_value_info('c', TensorProto.FLOAT, [32, 32])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10)
    path = Path(directory) / 'probe.onnx'
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(directory) / 'ort-toy')
    session = ort.InferenceSession(str(path), sess_options=options,
        providers=[('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0})])
    rng = np.random.default_rng(42)
    a, b = (rng.normal(size=(32, 32)).astype(np.float32) for _ in range(2))
    output = session.run(None, {'a': a, 'b': b})[0]
    np.testing.assert_allclose(output, a @ b, rtol=1e-4, atol=1e-4)
    profile = Path(session.end_profiling())
    result = check_ort_events(read(profile))
    write(Path(directory) / 'ort_executed.json', {**result, 'profile': profile.name})


def check_scores(reference, actual, dtype):
    from experiments.common import check_score
    if len(reference) != len(actual) or not reference:
        raise ValueError('Missing smoke scores')
    for score in reference + actual:
        check_score(score)
    tolerance = .05 if dtype == 'float16' else .001
    differences = [abs(a['p_unsafe'] - b['p_unsafe']) for a, b in zip(reference, actual)]
    flips = [i for i, (a, b) in enumerate(zip(reference, actual)) if a['label'] != b['label']]
    result = {'n': len(reference), 'max_delta': max(differences), 'label_flips': flips,
              'max_delta_tolerance': tolerance, 'passed': max(differences) <= tolerance and not flips,
              'note': 'Tiny environment smoke only; not numerical-regression approval or safety quality'}
    return result


def evaluate(adapter, directory=None):
    import torch
    from experiments.common import check_score
    rows = []
    with torch.inference_mode():
        for index, text in enumerate(TEXTS):
            if directory is not None:
                write(Path(directory) / 'progress.json', {'input_index': index, 'text': text,
                                                         'completed_inputs': len(rows)})
            prepared = adapter.prepare([text])
            score = adapter.scores([text])[0]
            check_score(score)
            rows.append({'text': text, 'input_hash': adapter.input_hash([text]),
                         'total_tokens': int(prepared['attention_mask'].sum()), **score})
            if directory is not None:
                write(Path(directory) / 'scores_partial.json', rows)
        repeat = adapter.scores([TEXTS[0]])[0]
    check_score(repeat)
    if abs(rows[0]['p_unsafe'] - repeat['p_unsafe']) > 1e-5 or rows[0]['label'] != repeat['label']:
        raise RuntimeError('A-B-C-A repeat differs')
    return rows


def model_probe(model, directory, work):
    from experiments.adapters import load_adapter
    from experiments.variants import spec_for, apply_variant
    from experiments import onnx_service
    import torch
    directory = Path(directory)
    variants = ('B0', 'P1', 'H2_linear') if model == 'halo' else ('B0', 'G1_service')
    reference, identity = None, None
    for name in variants:
        target = directory / name
        target.mkdir()
        if name == 'G1_service' and read(Path(work) / 'stages/ort_cuda/status.json')['status'] != 'passed':
            write(target / 'status.json', {'status': 'not_run', 'reason': 'ORT CUDA primitive failed'})
            raise RuntimeError('GLiNER eager passed, but ORT CUDA primitive failed; export skipped')
        adapter = None
        try:
            spec = spec_for(name, model)
            if name == 'G1_service':
                preload_ort(target)
                onnx_service.VALIDATION_TEXTS[:] = TEXTS
            if model == 'halo':
                from .team_halo import TeamHaloAdapter
                adapter = TeamHaloAdapter(spec, target)
            else:
                adapter = load_adapter(model, spec, target)
            apply_variant(adapter, spec, target)
            write(target / 'contract.json', adapter.contract)
            write(target / 'metadata.json', adapter.metadata)
            try:
                actual = evaluate(adapter, target)
            finally:
                if model == 'halo':
                    write(target / 'dispatch_calls.json', adapter.calls)
            write(target / 'scores.json', actual)
            if reference is None:
                reference, identity = actual, [r['input_hash'] for r in actual]
            if identity != [r['input_hash'] for r in actual]:
                raise RuntimeError('Input/scoring contract changed between variants')
            result = check_scores(reference, actual, spec['dtype'])
            write(target / 'numerical_smoke.json', result)
            if not result['passed']:
                raise RuntimeError('Numerical smoke failed')
            if name == 'G1_service':
                with torch.inference_mode():
                    onnx_service.finish(adapter, target, TEXTS[0])
            if model == 'halo':
                from .team_halo import check_dispatch
                write(target / 'backend_evidence.json', check_dispatch(adapter.calls, spec.get('fast', 'none')))
            write(target / 'status.json', {'status': 'passed', 'quality_evaluated': False})
        finally:
            if adapter is not None:
                adapter.close()


def worker(work, stage):
    work = Path(work)
    target = work / 'stages' / stage
    target.mkdir(parents=True, exist_ok=True)
    state = {'status': 'in_progress', 'stage': stage, 'quality_evaluated': False}
    write(target / 'status.json', state)
    try:
        os.environ['USE_HUB_KERNELS'] = 'NO'
        spec = read(work / 'team_base.json')
        actual = package_snapshot()
        write(target / 'packages.json', actual)
        check_versions(spec, actual)
        torch = setup_torch(target)
        if stage == 'cuda':
            x = torch.ones((32, 32), device='cuda')
            assert torch.equal(x @ x, torch.full_like(x, 32))
            torch.cuda.synchronize()
        elif stage == 'ort_cuda':
            ort_probe(target)
        elif stage in ('gliner', 'halo'):
            model_probe(stage, target, work)
        else:
            raise ValueError('Unknown stage')
        state['status'] = 'passed'
    except Exception as error:
        state.update(status='failed', error=str(error))
        (target / 'error.txt').write_text(traceback.format_exc())
    finally:
        write(target / 'status.json', state)
    return state


def supervise(work, total_seconds=5400):
    from experiments.round2 import run_process
    work = Path(work).resolve()
    spec = read(work / 'team_base.json')
    start = time.monotonic()
    deadline = start + total_seconds
    state = {'status': 'in_progress', 'stages': {}, 'benchmark_performed': False,
             'quality_evaluated': False, 'environment_validated': False}
    venv = work.parent / ('_' + work.name + '_venv')
    install = work / 'installation'
    install.mkdir()
    def command(args, name, limit=1800):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Overall smoke budget exhausted')
        with isolated_python_env():
            result = run_process(args, install, name, min(limit, remaining), cwd=work)
        if result['status'] != 'completed':
            raise RuntimeError(name + ': ' + result['status'] + '; see installation logs')
    try:
        if platform.system() != 'Linux' or sys.version_info[:2] != (3, 13):
            raise RuntimeError('Use Linux Kaggle/Colab with Python 3.13; host runtime is not modified')
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=15, check=True)
        (install / 'gpu.txt').write_text(gpu.stdout)
        if 'T4' not in gpu.stdout.splitlines()[0]:
            raise RuntimeError('Select T4 before installing; no CPU fallback')
        write(install / 'host.json', {'python': platform.python_version(), 'packages': package_snapshot()})
        if venv.exists():
            raise RuntimeError('Refuse to reuse an existing virtual environment')
        command([sys.executable, '-m', 'venv', '--without-pip', str(venv)], 'venv', 120)
        python = str(venv / 'bin/python')
        command([python, '-c',
                 'import json,sys,site; from pathlib import Path; '
                 'assert sys.prefix != sys.base_prefix, "Not an isolated venv"; '
                 f'assert Path(sys.prefix).resolve() == Path({str(venv)!r}).resolve(); '
                 'assert site.ENABLE_USER_SITE is False; '
                 'print(json.dumps(dict(prefix=sys.prefix, base_prefix=sys.base_prefix, '
                 'user_site=site.ENABLE_USER_SITE)))'], 'venv_isolation', 60)
        pip = target_pip(python)
        command([*pip, '--version'], 'target_pip', 60)
        versions = pins(spec)
        constraints = install / 'constraints.txt'
        constraints.write_text(''.join(f'{k}=={v}\n' for k, v in sorted(versions.items())))
        base = [*pip, 'install', '--disable-pip-version-check',
                '--index-url', 'https://pypi.org/simple', '--extra-index-url', spec['installation_notes']['torch_index'],
                '--constraint', str(constraints), *[f'{k}=={v}' for k, v in versions.items()], 'gliner2[local]==2.1.0']
        command([*base, '--dry-run', '--report', str(install / 'resolve.json')], 'resolve')
        command([*base, '--report', str(install / 'install.json')], 'install', 2700)
        command([*pip, 'check'], 'pip_check', 120)
        command([*pip, 'freeze', '--all'], 'resolved_versions', 120)
        for stage in STAGES:
            target = work / 'stages' / stage
            target.mkdir(parents=True, exist_ok=True)
            if stage != 'cuda' and state['stages']['cuda']['status'] != 'passed':
                outcome = {'status': 'not_run', 'reason': 'CUDA base failed'}
            elif deadline <= time.monotonic():
                outcome = {'status': 'not_run', 'reason': 'budget_exhausted'}
            else:
                print('Smoke:', stage, flush=True)
                with isolated_python_env():
                    process = run_process([python, '-m', 'followup.team_smoke', 'worker', str(work), stage],
                                          target, 'worker', min(1200, deadline-time.monotonic()), cwd=work)
                outcome = (read(target / 'status.json') if process['status'] == 'completed'
                           else {'status': process['status'], 'reason': 'worker did not finish'})
            write(target / 'status.json', outcome)
            state['stages'][stage] = outcome
            write(work / 'status.json', state)
            archive(work)
        state['environment_validated'] = all(x['status'] == 'passed' for x in state['stages'].values())
        state['status'] = 'passed' if state['environment_validated'] else 'partial'
    except Exception as error:
        state.update(status='failed', error=str(error))
        (work / 'error.txt').write_text(traceback.format_exc())
    finally:
        state['seconds'] = time.monotonic() - start
        write(work / 'status.json', state)
        archive(work)
    return state


def build_notebook(root):
    root = Path(root)
    from .prepare import cell
    from .runtime import digest
    measured = read(root / 'followup/frozen/gliner_onnx.json')
    sources = measured['sources']
    assert digest(sources) == measured['manifest']['source_hash']
    spec = read(root / 'env/team_base.json')
    module = (root / 'followup/team_smoke.py').read_text()
    payload = {'spec': spec, 'sources': sources, 'module': module,
               'halo_adapter': (root / 'followup/team_halo.py').read_text()}
    code = """import hashlib, json, os, sys, uuid
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL='https://pypi.org/simple')
os.environ.pop('PIP_EXTRA_INDEX_URL', None)
ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
WORK = ROOT / ('team_env_smoke_' + uuid.uuid4().hex[:8])
WORK.mkdir()
""" + f"PAYLOAD = {payload!r}\nCHECKSUM = {digest(payload)!r}\n" + """assert hashlib.sha256(json.dumps(PAYLOAD, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest() == CHECKSUM
for package in ('experiments', 'followup'):
    (WORK / package).mkdir()
for name, text in PAYLOAD['sources'].items():
    (WORK / 'experiments' / name).write_text(text)
(WORK / 'followup/__init__.py').write_text('')
(WORK / 'followup/team_smoke.py').write_text(PAYLOAD['module'])
(WORK / 'followup/team_halo.py').write_text(PAYLOAD['halo_adapter'])
(WORK / 'team_base.json').write_text(json.dumps(PAYLOAD['spec'], indent=2))
for name in list(sys.modules):
    if name.split('.')[0] in ('experiments', 'followup'):
        del sys.modules[name]
sys.path.insert(0, str(WORK))
from followup.team_smoke import supervise
print(WORK)
"""
    intro = """# Проверка общей среды: T4 + сохранённые ORT / GLiNER2 / FLA

Чистая Kaggle/Colab Linux-сессия, Python 3.13, T4 + Internet → Run All.
Установка в отдельный venv без ensurepip; host pip управляет только целевым venv через --python.
Python/PyTorch основного notebook не заменяются, ручной restart не нужен.
База коллег: torch 2.11.0+cu130, Transformers 5.20.1. Дополнения: ORT 1.23.2,
GLiNER2 2.1.0, FLA/fla-core 0.4.2; ONNX 1.20.1. ORT получает явно закреплённые CUDA 12 библиотеки.
Это проверка совместимости двух CUDA-стеков, не предположение, что они уже совместимы.
Установка скачивает PyTorch/зависимости и веса: потребуется несколько ГБ диска и интернет.
Лимит 90 минут (большая часть возможного времени — установка/JIT/скачивание), не ожидаемая длительность.

Проверки: CUDA MatMul → ORT MatMul с доказанным CUDA-исполнением → GLiNER eager/ORT через настоящую голову
→ HaloGuard FP32/FP16/FLA с фиксацией вызовов FLA. По три синтетических текста и повтор первого.
Без 300-примерной регрессии, замеров latency, compile, датасетов качества и holdout.
Ошибка ORT не отменяет проверку HaloGuard. Никаких автоматических обновлений версий или CPU fallback.
Замороженные исходники сохранены. Для Halo добавлен отдельный адаптер Transformers 5.20.1:
явный PyTorch reference или только FLA linear attention; conv/norm остаются reference.
Автоматический выбор hub kernels отключён, реальные вызовы функций сохраняются для каждого варианта.
Запуск team_env_smoke_45555111 прошёл все стадии на T4: установка, CUDA/ORT и обе модели.
Это воспроизводящий ноутбук; повтор сейчас не требуется. Скорость и качество этим smoke не подтверждаются.
"""
    return {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                     'language_info': {'name': 'python'}},
        'cells': [cell('markdown', intro, 'intro'), cell('code', code, 'unpack'),
                  cell('code', 'STATUS = supervise(WORK)\nprint(json.dumps(STATUS, indent=2, ensure_ascii=False))\n', 'run'),
                  cell('code', """ARCHIVE = WORK.with_suffix('.zip')
print('Скачать:', ARCHIVE)
try:
    from google.colab import files
except ImportError:
    print('Kaggle Output: team_env_smoke_*.zip. Папку _team_env_smoke_*_venv скачивать не нужно.')
else:
    files.download(str(ARCHIVE))
""", 'download')]}


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'worker':
        worker(sys.argv[2], sys.argv[3])
    elif sys.argv[1:] == ['build']:
        root = Path(__file__).resolve().parents[1]
        write(root / 'notebooks/team_environment_smoke.ipynb', build_notebook(root))
    else:
        raise SystemExit('Use build or worker WORK STAGE')
