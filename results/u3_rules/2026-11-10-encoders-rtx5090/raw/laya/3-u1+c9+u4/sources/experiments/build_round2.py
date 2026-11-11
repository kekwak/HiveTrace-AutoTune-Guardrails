"""Generate two model suites; earlier single-experiment notebooks remain frozen."""

import hashlib
import json
from pathlib import Path

from .build_notebooks import cell
from .suites import SUITES


def notebook(suite):
    if suite not in SUITES:
        raise ValueError("Unknown suite")
    sources = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))
               if not p.name.startswith("build_")}
    checksum = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    title = {"haloguard": "HaloGuard: SDPA + FLA за один запуск",
             "gliner": "GLiNER: compile + ONNX за один запуск"}[suite]
    intro = f"""# {title}

**Этот комплект ещё не проверен на T4.** Старые B0/P1 успешно запускались, но не подтверждают его совместимость.

Новая Kaggle/Colab-сессия: T4, Internet включён. Run All. Не запускайте другие модели на той же GPU.
GPU не арендуется автоматически. Два эксперимента выполняются последовательно в отдельных процессах.
Общий лимит на установку и оба эксперимента — 10 часов; установки — до часа на шаг, preflight — до двух часов.
Первый эксперимент получает половину оставшегося времени, второй — остаток. Ошибка/таймаут первого не блокирует второй.
При несовместимости сохраняются диагностика и статус, скрытого fallback нет.

Порядок: одна установка общего стека → для каждого эксперимента preflight → регрессия на 300 development-примерах
→ A/A (два блока по 30 запросов) → A/B (три блока по 100) → отдельный и общий ZIP.
В каждой паре сначала измеряются оба варианта на одном сценарии, затем следующий сценарий.
Контрольные замеры не переиспользуются между кандидатами. Шумная A/A не останавливает запуск:
вывод о скорости отдельно записывается в noise_assessment.json и REPORT.md.
При ошибке общей установки оба эксперимента будут пропущены. Holdout, safety-разметка и квантование не используются.
Для SDPA штатный math dispatch сохраняется как отдельный отрицательный результат, затем проверяется explicit KV repeat.
Не редактируйте исходники внутри notebook и не перезапускайте отдельные измерительные ячейки.
"""
    config = f'''SUITE = {suite!r}
TOTAL_SECONDS = 10 * 60 * 60
PREFLIGHT_SECONDS = 2 * 60 * 60
'''
    bundle = '''import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_LOGS"] = "graph_breaks,recompiles,perf_hints"
ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
WORK = ROOT / ("round2_session_" + uuid.uuid4().hex[:8])
WORK.mkdir()
'''
    bundle += "SOURCES = " + repr(sources) + "\n"
    bundle += f"SOURCE_SHA256 = {checksum!r}\n"
    bundle += '''if hashlib.sha256(json.dumps(SOURCES, sort_keys=True).encode()).hexdigest() != SOURCE_SHA256:
    raise ValueError("Повреждён комплект исходников")
package = WORK / "experiments"
package.mkdir()
for name, source in SOURCES.items():
    compile(source, name, "exec")
    (package / name).write_text(source)
for name in list(sys.modules):
    if name == "experiments" or name.startswith("experiments."):
        del sys.modules[name]
sys.path.insert(0, str(WORK))
print("Рабочая папка:", WORK)
'''
    execute = '''from experiments.suites import run_suite
RUN_DIR = run_suite(SUITE, WORK / SUITE, total_seconds=TOTAL_SECONDS,
                    stage_seconds=PREFLIGHT_SECONDS)
print((RUN_DIR / "suite.json").read_text())
'''
    finish = '''ARCHIVE = RUN_DIR.with_suffix(".zip")
print("Скачать результат:", ARCHIVE)
try:
    from google.colab import files
except ImportError:
    print("Kaggle: Output → round2_session_* → ZIP. Сохраните архив и при ошибке.")
else:
    files.download(str(ARCHIVE))
'''
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "source_sha256": checksum},
            "cells": [cell("markdown", "intro", intro), cell("code", "configuration", config),
                      cell("code", "sources", bundle), cell("code", "run-all", execute),
                      cell("markdown", "interpretation", "## Результат\n`completed` означает, что оба screening выполнены, не гарантирует ускорение или качество.\n`completed_with_blockers`: очередь обработана, но один или оба варианта не прошли проверки; скачайте ZIP с диагностикой.\nВ suite.json видны оба результата; детали — в <эксперимент>/workflow.json и <эксперимент>/screening/runs/*/REPORT.md.\n`noise_assessment.json`: inconclusive означает, что эффект по скорости не установлен; promising_requires_confirmation — только кандидат на независимое подтверждение. A/A — короткая диагностика, не гарантия отсутствия дрейфа.\nУстановка общая: environment/. При аварийной остановке runtime сохраните уже записанные файлы; полный ZIP может не успеть создаться.\n"),
                      cell("code", "download", finish)]}


def main():
    target = Path(__file__).resolve().parents[1] / "notebooks"
    for suite in SUITES:
        (target / (suite + "_round2.ipynb")).write_text(
            json.dumps(notebook(suite), ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
