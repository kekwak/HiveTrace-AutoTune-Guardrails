"""Two model-specific sessions with one installation and a shared wall-clock budget."""

import json
from pathlib import Path
import sys
import time
import traceback

from .common import archive, digest, write_json
from .round2 import RECIPES, repair_incomplete, run_process, run_workflow, setup_environment

SUITES = {
    "haloguard": ("haloguard_sdpa", "haloguard_fla"),
    "gliner": ("gliner_compile", "gliner_onnx"),
}


def suite_recipe(name):
    recipes = [RECIPES[key] for key in SUITES[name]]
    if len({r["model"] for r in recipes}) != 1:
        raise ValueError("A suite must use one model/environment")
    return {"model": recipes[0]["model"], "experiments": list(SUITES[name]),
            "extra": list(dict.fromkeys(p for r in recipes for p in r["extra"]))}


def run_suite(name, directory, total_seconds=36000, stage_seconds=7200,
              execute=run_process, install=setup_environment, workflow=run_workflow):
    if name not in SUITES or not 0 < total_seconds <= 36000 or not 0 < stage_seconds <= 7200:
        raise ValueError("Unknown suite or invalid time budget")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + total_seconds
    status = {"status": "in_progress", "suite": name, "total_seconds": total_seconds,
              "quality_evaluated": False, "environment": "environment/",
              "results": [{"experiment": key, "status": "pending"} for key in SUITES[name]]}
    source = directory / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        if not path.name.startswith("build_"):
            (source / path.name).write_bytes(path.read_bytes())
    status["source_hash"] = digest({p.name: p.read_text() for p in source.glob("*.py")})

    def save():
        status["elapsed_seconds"] = time.monotonic() - start
        write_json(directory / "suite.json", status)

    save()
    try:
        environment = directory / "environment"
        environment.mkdir()
        result = execute([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
                          "assert 'T4' in torch.cuda.get_device_name(0); print(torch.__version__, torch.version.cuda)"],
                         environment, "hardware", min(60, total_seconds))
        if result["status"] != "completed":
            raise RuntimeError("T4 hardware check failed; no CPU fallback")
        print("Одна установка зависимостей для", name, flush=True)
        install(suite_recipe(name), environment, deadline, execute)
        save()
        archive(directory)
        for index, entry in enumerate(status["results"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Shared session budget exhausted")
            # Reserve time for the second direction instead of letting a stuck first one consume all 10h.
            allocation = remaining / (len(status["results"]) - index)
            key = entry["experiment"]
            entry.update(status="in_progress", allocated_seconds=allocation,
                         directory=key + "/", environment="../environment/")
            save()
            print("Эксперимент:", key, "бюджет, мин:", round(allocation / 60, 1), flush=True)
            target = directory / key
            try:
                result_dir = workflow(key, target, total_seconds=allocation, stage_seconds=stage_seconds,
                                      execute=execute, install=None)
                result = json.loads((result_dir / "workflow.json").read_text())
                if result.get("status") not in {"completed", "unsupported", "regression_failed", "failed", "timeout"}:
                    raise RuntimeError("Experiment returned without a terminal status")
                entry.update(status=result["status"], error=result.get("error"))
            except Exception as error:
                entry.update(status="failed", error=str(error))
                target.mkdir(exist_ok=True)
                (target / "suite_worker_error.txt").write_text(traceback.format_exc())
                repair_incomplete(target, "failed")
                archive(target)
            save()
            archive(directory)
        status["status"] = ("completed" if all(r["status"] == "completed" for r in status["results"])
                            else "completed_with_blockers")
    except TimeoutError as error:
        status.update(status="timeout", error=str(error))
    except Exception as error:
        status.update(status="failed", error=str(error))
        (directory / "suite_error.txt").write_text(traceback.format_exc())
    finally:
        for entry in status["results"]:
            if entry["status"] == "pending":
                entry.update(status="skipped", reason=status["status"])
        save()
        lines = ["# Результат " + name, "", "Статус: **" + status["status"] + "**.", "",
                 "| Эксперимент | Статус |", "| --- | --- |"]
        lines += ["| " + r["experiment"] + " | " + r["status"] + " |" for r in status["results"]]
        lines += ["", "Общая установка: environment/. Каждый эксперимент: <имя>/workflow.json и screening/runs/*/REPORT.md.",
                  "completed_with_blockers означает, что очередь обработана, но не все варианты прошли проверки.",
                  "Парные контроли измеряются заново для каждого кандидата. Screening не подтверждает устойчивое ускорение.",
                  "Качество по разметке не оценивалось; завершённые и неуспешные эксперименты сохранены отдельно."]
        (directory / "README.md").write_text("\n".join(lines) + "\n")
        archive(directory)
    return directory
