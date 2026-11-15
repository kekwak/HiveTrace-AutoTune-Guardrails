"""Standalone locked environment and user-script launcher, not a benchmark runner."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import signal
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

from .team_lock import read, requirements, verify_inventory


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def child_env():
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_EXTRA_INDEX_URL', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               USE_HUB_KERNELS='NO', PIP_CONFIG_FILE=os.devnull, PYTHONUNBUFFERED='1')
    return env


def command(args, cwd, log, timeout):
    """Keep stdout/stderr; stop the process group on failure, timeout or interruption."""
    if timeout <= 0:
        raise ValueError('Timeout must be positive')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = None
    status = 'failed'
    print('Запуск:', log.name, '| лог:', log, flush=True)
    try:
        with log.open('w') as stream:
            proc = subprocess.Popen(args, cwd=cwd, env=child_env(), stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
                status = 'passed' if proc.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                status = 'timeout'
            except KeyboardInterrupt:
                status = 'interrupted'
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        result = {'status': status, 'returncode': proc.returncode if proc else None,
                  'seconds': time.monotonic() - started}
        write(log.with_suffix('.process.json'), result)
    if status != 'passed':
        print(log.read_text(errors='replace')[-6000:])
        raise RuntimeError(f'{log.name}: {status}; полная диагностика в логе')
    return result


def archive(work, max_file=50 * 1024**2, max_total=200 * 1024**2):
    work = Path(work)
    target = work.with_suffix('.zip')
    total, skipped = 0, []
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as output:
        # Arbitrary input directories, credentials and venv are deliberately excluded.
        paths = [work / 'status.json', *sorted(work.glob('*.py'))]
        for name in ('setup', 'followup', 'runs'):
            paths += sorted((work / name).rglob('*'))
        for path in paths:
            if not path.is_file():
                continue
            rel = path.relative_to(work)
            reason = None
            if any((work / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
                reason = 'symlink'
            elif any(p.startswith('.') or p == '__pycache__' for p in rel.parts):
                reason = 'hidden_or_cache'
            elif path.suffix in ('.onnx', '.data', '.bin', '.pt', '.pth', '.safetensors', '.zip'):
                reason = 'weights_or_nested_archive'
            elif path.stat().st_size > max_file or total + path.stat().st_size > max_total:
                reason = 'size_limit'
            if reason:
                skipped.append({'path': str(rel), 'reason': reason})
                continue
            output.write(path, str(rel))
            total += path.stat().st_size
        output.writestr('archive_contents.json', json.dumps({'uncompressed_bytes': total,
            'max_file_bytes': max_file, 'max_total_bytes': max_total, 'excluded': skipped}, indent=2))
    return target


def precheck():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 13):
        raise RuntimeError('Нужны Linux x86_64 и Python 3.13. Этот lock не поддерживает текущий runtime; версии автоматически не меняем.')
    pip = importlib.metadata.version('pip')
    if tuple(int(x) for x in pip.split('.')[:2]) < (22, 3):
        raise RuntimeError('Host pip >=22.3 required for --python; host pip is not auto-updated')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=20)
    if not gpu.stdout.splitlines() or 'T4' not in gpu.stdout.splitlines()[0]:
        raise RuntimeError('Выберите T4. GPU 0 должна быть T4; CPU fallback запрещён.')
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'pip': pip, 'visible_device': 0, 'nvidia_smi': gpu.stdout.strip()}


def probe(work, directory):
    """Runs only under the locked venv; no model downloads or benchmarks."""
    import ctypes
    import random
    import site
    import numpy as np
    import torch
    work, directory = Path(work), Path(directory)
    if sys.prefix == sys.base_prefix or sys.version_info[:2] != (3, 13):
        raise RuntimeError('Expected isolated Python 3.13 venv')
    actual = {p.metadata['Name']: p.version for p in importlib.metadata.distributions() if p.metadata.get('Name')}
    write(directory / 'packages.json', actual)
    verify_inventory(read(work / 'setup/lock.json'), actual)
    libs = [p for base in site.getsitepackages() for p in Path(base).glob('nvidia/cu13/lib/libnvrtc-builtins.so.13.0')]
    if not libs:
        raise RuntimeError('CUDA 13 NVRTC builtins not found')
    ctypes.CDLL(str(libs[0]), mode=ctypes.RTLD_GLOBAL)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('T4 CUDA unavailable; no CPU fallback')
    if torch.version.cuda != '13.0' or torch.backends.cudnn.version() != 91900:
        raise RuntimeError('CUDA/cuDNN differs from the selected environment')
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
    x = torch.ones(16, 16, device='cuda')
    if not torch.equal(x @ x, torch.full_like(x, 16)):
        raise RuntimeError('CUDA MatMul failed')
    torch.cuda.synchronize()
    write(directory / 'runtime.json', {'python': platform.python_version(), 'executable': sys.executable,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0), 'seed': 42, 'cpu_threads': 2, 'tf32': False,
        'deterministic': 'warn_only', 'hub_kernels': False, 'nvrtc_preload': str(libs[0]),
        'cuda_primitive_passed': True, 'ort_cuda_tested': False, 'model_tested': False})


class Session:
    def __init__(self, work):
        self.work = Path(work).resolve()
        self.venv = self.work.parent / ('_' + self.work.name + '_venv')
        self.python = self.venv / 'bin/python'

    def setup(self):
        folder = self.work / 'setup'
        folder.mkdir(exist_ok=True)
        state = {'stage': 'setup', 'status': 'running', 'model_tested': False}
        write(self.work / 'status.json', state)
        try:
            write(folder / 'host.json', precheck())
            manifest = read(folder / 'lock.json')
            text = (folder / 'requirements.txt').read_text()
            if text != requirements(manifest):
                raise RuntimeError('Embedded lock text and manifest disagree')
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            marker = self.venv / 'u1-lock.sha256'
            if self.venv.exists():
                if not marker.exists() or marker.read_text() != fingerprint:
                    raise RuntimeError('Неполная или другая среда: повторите ячейку подготовки нового рабочего каталога.')
            else:
                command([sys.executable, '-m', 'venv', '--without-pip', str(self.venv)], self.work, folder / 'venv.log', 120)
                command([sys.executable, '-m', 'pip', '--python', str(self.python), 'install',
                         '--disable-pip-version-check', '--no-deps', '--require-hashes', '--only-binary=:all:',
                         '-r', str(folder / 'requirements.txt'), '--report', str(folder / 'install.json')],
                        self.work, folder / 'install.log', 3600)
            command([sys.executable, '-m', 'pip', '--python', str(self.python), 'check'],
                    self.work, folder / 'pip-check.log', 120)
            command([str(self.python), '-u', '-m', 'followup.team_skeleton', 'probe', str(self.work), str(folder)],
                    self.work, folder / 'probe.log', 180)
            marker.write_text(fingerprint)
            state.update(status='passed', lock_inventory_verified=True, cuda_primitive_passed=True)
        except (Exception, KeyboardInterrupt) as error:
            state.update(status='failed', error=str(error))
            (folder / 'error.txt').write_text(traceback.format_exc())
        finally:
            write(folder / 'status.json', state)
            write(self.work / 'status.json', state)
            archive(self.work)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return state

    def run(self, timeout=3600):
        setup_status = self.work / 'setup/status.json'
        if (not setup_status.exists() or read(setup_status).get('status') != 'passed'
                or not (self.venv / 'u1-lock.sha256').exists()
                or not (self.work / 'setup/runtime.json').exists()):
            print('Эксперимент не запущен: сначала исправьте setup. Архив:', archive(self.work))
            return {'status': 'not_run', 'reason': 'setup_not_ready'}
        target = self.work / 'runs' / ('run_' + uuid.uuid4().hex[:8])
        target.mkdir(parents=True)
        (target / 'results').mkdir()
        state = {'stage': 'experiment', 'status': 'running', 'run': target.name}
        try:
            source = self.work / 'experiment.py'
            (target / 'experiment.py').write_bytes(source.read_bytes())
            for helper in self.work.glob('*.py'):
                if helper.name != 'experiment.py' and not helper.is_symlink():
                    (target / 'source_helpers').mkdir(exist_ok=True)
                    (target / 'source_helpers' / helper.name).write_bytes(helper.read_bytes())
            state['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            command([str(self.python), '-u', '-m', 'followup.team_skeleton', 'experiment',
                     str(self.work), str(target)], self.work, target / 'stdout.log', timeout)
            state['status'] = 'passed'
        except (Exception, KeyboardInterrupt) as error:
            state.update(status='failed', error=str(error))
            (target / 'error.txt').write_text(traceback.format_exc())
        finally:
            write(target / 'status.json', state)
            write(self.work / 'status.json', state)
            archive(self.work)
        if (target / 'stdout.log').exists():
            print((target / 'stdout.log').read_text(errors='replace')[-6000:])
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print('Результаты:', target / 'results')
        return state


def build_notebook(root):
    root = Path(root)
    from .prepare import cell
    from .runtime import digest
    payload = {'lock': read(root / 'env/team_environment.lock.json'),
               'requirements': (root / 'env/team_environment.lock.txt').read_text(),
               'sources': {n: (root / 'followup' / n).read_text()
                           for n in ('team_lock.py', 'team_skeleton.py')}}
    unpack = '''import hashlib, json, os, sys, uuid
from pathlib import Path
''' + f'PAYLOAD = {payload!r}\nCHECKSUM = {digest(payload)!r}\n' + '''assert hashlib.sha256(json.dumps(PAYLOAD, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest() == CHECKSUM
ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content') if Path('/content').exists() else Path.cwd()
WORK = ROOT / ('u1_workspace_' + uuid.uuid4().hex[:8])
(WORK / 'setup').mkdir(parents=True)
(WORK / 'followup').mkdir()
(WORK / 'followup/__init__.py').write_text('')
for name, source in PAYLOAD['sources'].items():
    (WORK / 'followup' / name).write_text(source)
(WORK / 'setup/lock.json').write_text(json.dumps(PAYLOAD['lock'], indent=2))
(WORK / 'setup/requirements.txt').write_text(PAYLOAD['requirements'])
for name in list(sys.modules):
    if name == 'followup' or name.startswith('followup.'):
        del sys.modules[name]
sys.path.insert(0, str(WORK))
os.chdir(WORK)
from followup.team_skeleton import Session, archive
SESSION = Session(WORK)
ENV_PYTHON = str(SESSION.python)
print('Рабочая папка:', WORK)
print('Python экспериментов:', ENV_PYTHON)
'''
    intro = '''# У1: окружение из lock и свой эксперимент

**Один файл, без клона репозитория.** Встроены 96 точных wheels с SHA256 и вспомогательный код.
Kaggle: новый notebook → Import/Upload этот файл → Accelerator T4 → Internet On → Run All.
Colab: загрузить этот файл → выбрать GPU T4 → Run All. Требуются Linux x86_64 / Python 3.13;
если Colab выдаёт другую версию, установка остановится, автоматической замены Python нет.
Исходная среда проверена на Kaggle, этот новый skeleton и установка именно из lock пока проверены только CPU-тестами.

Установка скачивает несколько ГБ в **отдельный venv**, веса моделей не скачиваются автоматически.
Host PyTorch/kernel не меняется, restart не нужен. Проверка включает точные версии и CUDA MatMul,
но не запуск Halo/GLiNER/ORT и не подтверждение скорости или качества.
Лимит установки 60 минут; каждый пользовательский запуск по умолчанию ограничен 60 минутами.
GPU-сессию выбирает пользователь: notebook не арендует ресурсы, но работа сессии расходует её квоту.
'''
    guide = '''## Ваш код

Меняйте только следующую ячейку `%%writefile experiment.py`, затем выполняйте её и ячейку запуска.
Первая строка **записывает файл, а не выполняет эксперимент в kernel**. Код выполняется отдельным Python из venv.
Не запускайте свои импорты обычной notebook-ячейкой или через `!python`: это будет host-среда.
Повторять установку при редактировании эксперимента не нужно; повтор полного Run All создаст новый venv.

Между запусками Python-переменные не сохраняются. Рабочая папка — WORK; датасеты Kaggle доступны по `/kaggle/input/...`.
Результаты пишите в `RESULT_DIR`; дополнительные модули можно записывать рядом с experiment.py.
Новые зависимости автоматически не устанавливаются: изменение lock нужно согласовать отдельно.
Каждый запуск получает seed 42, два CPU-потока, TF32 off и CUDA 13 NVRTC preload;
Hub kernels отключены, выбор backend конкретной модели остаётся ответственностью автора.
Для ORT перед созданием session используйте `onnxruntime.preload_dlls(cuda=True, cudnn=True, directory='')`;
эта операция сама по себе не доказывает CUDA-исполнение графа.

Не вставляйте токены в код/print: исходник и stdout попадут в архив. При необходимости задайте HF_TOKEN через
secrets платформы в окружении host перед запуском; он наследуется процессом, но полный env не собирается.
'''
    sample = '''%%writefile experiment.py
import json
import os
from pathlib import Path
import torch

OUTPUT = Path(os.environ['RESULT_DIR'])
print('Torch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))

# Замените этот пример своим кодом. Проверка окружения уже выполнена вне эксперимента.
with torch.inference_mode():
    x = torch.ones((32, 32), device='cuda')
    y = x @ x
    result = {'example_sum': float(y.sum().cpu()), 'note': 'Not a latency benchmark'}
(OUTPUT / 'result.json').write_text(json.dumps(result, indent=2))
print(result)
'''
    download = '''ARCHIVE = archive(WORK)
print('Архив:', ARCHIVE)
print('Статус:', (WORK / 'status.json').read_text() if (WORK / 'status.json').exists() else 'setup не запускался')
try:
    from google.colab import files
except ImportError:
    from IPython.display import FileLink, display
    display(FileLink(str(ARCHIVE)))
    print('Kaggle: файл также доступен в Output / /kaggle/working.')
else:
    files.download(str(ARCHIVE))
'''
    return {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                     'language_info': {'name': 'python'}},
        'cells': [cell('markdown', intro, 'intro'), cell('code', unpack, 'unpack'),
                  cell('code', 'SETUP = SESSION.setup()\n', 'setup'), cell('markdown', guide, 'guide'),
                  cell('code', sample, 'experiment'),
                  cell('code', 'RUN = SESSION.run(timeout=60 * 60)\n', 'run'),
                  cell('markdown', '''## Результаты и диагностика

Архив обновляется после setup и каждого эксперимента, в том числе при ошибке/таймауте.
Внутри: lock, установочные логи, runtime, копия кода каждого запуска, stdout и `results/`.
Ошибки не маскируются успехом: смотрите status.json и *.process.json. Потеря самой VM не перехватывается.
Venv, входные датасеты, скрытые файлы и веса не архивируются; лимит одного файла 50 MiB, общего содержимого 200 MiB.
Список исключений — archive_contents.json. Большие traces/веса скачивайте отдельно из папки запуска.
При ошибке setup скачайте архив; если нужна повторная установка, начните с ячейки рабочего каталога.
''', 'results'), cell('code', download, 'download')]}


if __name__ == '__main__':
    if sys.argv[1:] == ['build']:
        root = Path(__file__).resolve().parents[1]
        write(root / 'notebooks/team_environment_skeleton.ipynb', build_notebook(root))
    elif len(sys.argv) == 4 and sys.argv[1] in ('probe', 'experiment'):
        work, target = Path(sys.argv[2]), Path(sys.argv[3])
        probe(work, target)
        if sys.argv[1] == 'experiment':
            os.environ['RESULT_DIR'] = str(target / 'results')
            sys.path.insert(0, str(target / 'source_helpers'))
            runpy.run_path(str(target / 'experiment.py'), run_name='__main__')
    else:
        raise SystemExit('Use build, probe WORK OUTPUT or experiment WORK RUN_DIR')
