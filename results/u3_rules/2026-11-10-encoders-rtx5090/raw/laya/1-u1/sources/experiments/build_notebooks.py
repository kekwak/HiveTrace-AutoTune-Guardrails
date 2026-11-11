"""Generate self-contained notebooks from auditable local Python sources.

Run from the repository root: python -m experiments.build_notebooks
Historical notebooks are never read or overwritten.
"""

import hashlib
import json
from pathlib import Path


def cell(kind, name, source):
    result = {"cell_type": kind, "id": name, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(model, probe=False):
    root = Path(__file__).parent
    sources = {p.name: p.read_text() for p in sorted(root.glob("*.py")) if p.name != "build_notebooks.py"}
    checksum = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    config = f'''MODEL = {model!r}
CONFIG = {{
    "model": MODEL, "candidate": "P1", "control": "B0",
    "phase": "screening",  # затем confirmation; throughput отдельно
    "session_seed": 17,  # три НОВЫЕ сессии confirmation: 17, 42, 123
    "profile": True,
    "approval_file": "",  # путь к approval.json успешного screening
}}
# H2/G1: зависимости устанавливаются только здесь, с точными версиями.
# Сначала проверить совместимость с T4 в отдельной среде. Пустой список не включает optional backend.
EXTRA_REQUIREMENTS = []
INSTALL_BASE_DEPENDENCIES = {not probe!r}
'''
    setup = '''import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_LOGS"] = "graph_breaks,recompiles,perf_hints"
import torch
ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
WORK = ROOT / ("experiment_session_" + uuid.uuid4().hex[:8])
WORK.mkdir()
constraint = WORK / "torch_constraint.txt"
constraint.write_text("torch==" + metadata.version("torch") + "\\n")
before = (metadata.version("torch"), torch.version.cuda)
try:
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Нужна T4; CPU fallback запрещён")
    if INSTALL_BASE_DEPENDENCIES:
        if any(name in sys.modules for name in ("transformers", "gliner2")):
            raise RuntimeError("Библиотеки уже импортированы. Начните новую runtime-сессию")
        requirements = (["transformers @ git+https://github.com/huggingface/transformers.git@394b1a0eaa8e6199e372334da0aff3753a117fdb", "accelerate", "sentencepiece"]
                        if MODEL == "halo" else ["gliner2[local]==2.1.0", "transformers==5.5.3", "sentencepiece==0.2.2"])
        command = [sys.executable, "-m", "pip", "install", "--constraint", str(constraint), *requirements, *EXTRA_REQUIREMENTS]
        with (WORK / "installation.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    elif EXTRA_REQUIREMENTS:
        raise ValueError("Probe использует вручную подготовленную среду; автоматическая установка выключена")
    if (metadata.version("torch"), torch.version.cuda) != before:
        raise RuntimeError("Установщик изменил PyTorch/CUDA")
except Exception:
    (WORK / "setup_status.json").write_text(json.dumps({"status": "failed", "stage": "dependencies"}))
    with zipfile.ZipFile(str(WORK) + ".zip", "w") as out:
        for path in WORK.iterdir():
            if path.is_file(): out.write(path, path.name)
    print("Диагностика установки:", str(WORK) + ".zip")
    raise
print("Среда готова. Каталог:", WORK)
'''
    bundle = "# Исходники включены в ноутбук; GitHub и локальный клон не нужны.\n"
    bundle += "SOURCES = " + repr(sources) + "\n"
    bundle += f"SOURCE_SHA256 = {checksum!r}\n"
    bundle += '''if hashlib.sha256(json.dumps(SOURCES, sort_keys=True).encode()).hexdigest() != SOURCE_SHA256:
    raise ValueError("Повреждён встроенный комплект исходников")
package = WORK / "experiments"
package.mkdir()
for name, source in SOURCES.items():
    compile(source, name, "exec")
    (package / name).write_text(source)
for name in list(sys.modules):
    if name == "experiments" or name.startswith("experiments."):
        del sys.modules[name]
sys.path.insert(0, str(WORK))
'''
    execute = ('''from experiments.vllm_probe import run_probe
RUN_DIR = run_probe(WORK / "runs")
''' if probe else '''from experiments.runner import run
RUN_DIR = run(CONFIG, WORK / "runs")
''')
    execute += '''print((RUN_DIR / "manifest.json").read_text())
'''
    finish = '''# Добавляем установочные логи и исходники к результатам; никаких весов/токенов.
from experiments.common import archive
import shutil
shutil.copytree(package, RUN_DIR / "source", ignore=shutil.ignore_patterns("__pycache__"))
for name in ("installation.log", "torch_constraint.txt"):
    if (WORK / name).exists(): shutil.copy2(WORK / name, RUN_DIR / name)
ARCHIVE = archive(RUN_DIR)
print("Сохранить:", ARCHIVE)
try:
    from google.colab import files
except ImportError:
    print("Kaggle: Output → experiment_session_* → runs → ZIP")
else:
    files.download(str(ARCHIVE))
'''
    intro = ("# HaloGuard: H3 / vLLM feasibility\n\nТолько проверка совместимости, не benchmark. "
             "Новая T4-сессия с заранее установленным vLLM. Этот ноутбук НЕ обновляет CUDA PyTorch автоматически. "
             "Версии сохраняются; после успешного smoke их нужно закрепить для отдельного paired benchmark.\n"
             if probe else f"# {'HaloGuard' if model == 'halo' else 'GLiNER Guard'}: контролируемые эксперименты T4\n\n"
             "**Новый экспериментальный код: GPU-проверка ещё не выполнена.** Начать с P1 против B0, "
             "screening, новой Kaggle/Colab T4-сессии с Internet. Не смешивать две модели в одной runtime.\n"
             "600 синтетических примеров: 300 development используются здесь; 300 holdout только в confirmation. "
             "Это не разметка качества. Перезапускать весь ноутбук в новой runtime, не отдельные ячейки замеров.\n")
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}, "source_sha256": checksum},
            "cells": [cell("markdown", "intro", intro), cell("code", "configuration", config),
                      cell("code", "environment", setup), cell("code", "sources", bundle),
                      cell("markdown", "execution-note", "## Запуск\nОшибки/несовместимость сохраняются в ZIP; `completed` не означает доказанное ускорение.\n"),
                      cell("code", "experiment", execute), cell("code", "download", finish)]}


def main():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for model, name, probe in [("halo", "haloguard_experiments", False),
                               ("gliner", "gliner_guard_experiments", False),
                               ("halo", "haloguard_vllm_probe", True)]:
        (root / (name + ".ipynb")).write_text(json.dumps(notebook(model, probe), ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
