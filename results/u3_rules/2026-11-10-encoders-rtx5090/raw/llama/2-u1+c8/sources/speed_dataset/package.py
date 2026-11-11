"""Make the colleague's CPU-only Llama indexing handoff."""

import json
import zipfile
from pathlib import Path

from .core import CONTRACTS, ROOT, file_hash, validate_pool, write_json


def cell(kind, text, number):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True), "id": f"llama-{number}"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def package(directory):
    directory = Path(directory).resolve()
    _, manifest = validate_pool(directory)
    dest = directory / "handoff"
    dest.mkdir(exist_ok=True)
    paths = {"pool.jsonl": directory / "pool.jsonl", "pool_manifest.json": directory / "pool_manifest.json"}
    paths.update({"speed_dataset/" + p.name: p for p in ROOT.iterdir() if p.suffix in (".py", ".json", ".txt")})
    checksums = {name: file_hash(path) for name, path in paths.items()}
    archive = dest / "llama_index_request.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(paths.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    expected_zip = file_hash(archive)
    spec = CONTRACTS["models"]["llama"]
    intro = f"""# Индекс Llama Prompt Guard: только CPU

Выбран публичный источник `{spec['repo']}`, revision `{spec['revision']}`.
Это явно выбранное зеркало команды, не автоматический fallback. Идентичность оригиналу Meta не проверена.
GPU не включать. Загружаются tokenizer/config, не веса. Пул не размечается и не изменяется.
Загрузите рядом с ноутбуком `llama_index_request.zip` (в Kaggle можно как приватный Dataset),
включите Internet. Укажите путь ниже. HF token не требуется и не передаётся в процесс индексации.
При ошибке доступа остановитесь; не меняйте источник вручную. Верните `llama_index_response.zip`.
"""
    setup = f'''from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile, tempfile

REQUEST_ARCHIVE = Path("llama_index_request.zip")  # Kaggle: /kaggle/input/.../llama_index_request.zip
EXPECTED_SHA256 = {expected_zip!r}
EXPECTED_FILES = {checksums!r}
assert hashlib.sha256(REQUEST_ARCHIVE.read_bytes()).hexdigest() == EXPECTED_SHA256, "Другой архив"
WORK = Path(tempfile.mkdtemp(prefix="llama-index-"))
with zipfile.ZipFile(REQUEST_ARCHIVE) as z:
    assert set(z.namelist()) == set(EXPECTED_FILES), "Неожиданные файлы"
    for name, expected in EXPECTED_FILES.items():
        content = z.read(name)
        assert hashlib.sha256(content).hexdigest() == expected, name
        target = WORK / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
subprocess.run([sys.executable, "-m", "venv", str(WORK / "venv")], check=True)
PYTHON = WORK / "venv" / "bin" / "python"
subprocess.run([str(PYTHON), "-m", "pip", "install", "transformers==5.20.1", "tokenizers==0.23.3", "huggingface-hub==1.33.0"], check=True)
print("Изолированная CPU-среда готова. Основной runtime не менялся.")
'''
    run = '''env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
           HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    env.pop(key, None)
result = subprocess.run([str(PYTHON), "-m", "speed_dataset", "index", str(WORK), "--model", "llama"],
                        cwd=WORK, env=env)
response = Path("llama_index_response.zip")
with zipfile.ZipFile(response, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("llama.meta.json", "llama.jsonl"):
        p = WORK / "indices" / name
        if p.exists():
            z.write(p, name)
print("Код завершения:", result.returncode, "Архив:", response.resolve())
if result.returncode:
    print("Индекс НЕ готов. Верните диагностику; не редактируйте метаданные вручную.")
'''
    download = '''try:
    from google.colab import files
except ImportError:
    print("Kaggle/Jupyter: скачайте llama_index_response.zip из output/файлов.")
else:
    files.download(str(response))
'''
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "cells": [cell(k, s, i) for i, (k, s) in enumerate([
            ("markdown", intro), ("code", setup), ("code", run), ("code", download)])]}
    write_json(dest / "llama_index_cpu.ipynb", nb)
    write_json(dest / "handoff_manifest.json", {"pool_sha256": manifest["pool_sha256"],
               "archive_sha256": expected_zip, "files": checksums, "notebook_sha256": file_hash(dest / "llama_index_cpu.ipynb")})
    print(dest)
