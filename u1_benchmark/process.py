import os
import signal
import subprocess
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

from .core import ROOT, write


def child_environment():
    from followup.team_skeleton import child_env
    env = child_env()
    env.update(PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false",
               TORCH_LOGS="recompiles,graph_breaks")
    return env


def invoke(args, log, timeout):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    if timeout <= 0:
        raise TimeoutError("Comparison deadline exhausted")
    start = time.monotonic()
    status, proc = "failed", None
    try:
        with log.open("w") as stream:
            proc = subprocess.Popen([str(a) for a in args], cwd=ROOT, env=child_environment(),
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
                status = "passed" if proc.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
            except KeyboardInterrupt:
                status = "interrupted"
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        write(log.with_suffix(".process.json"), {"status": status, "pid": proc.pid if proc else None,
              "returncode": proc.returncode if proc else None, "seconds": time.monotonic() - start})
    if status == "timeout":
        raise TimeoutError(str(log))
    if status == "interrupted":
        raise KeyboardInterrupt()
    if status != "passed":
        raise RuntimeError("Worker failed; see " + str(log))


OPTIONAL_FILE_LIMIT = 64 * 1024**2
OPTIONAL_ARCHIVE_LIMIT = 200 * 1024**2


def archive_path(directory):
    directory = Path(directory).resolve()
    return directory.parent / (directory.name + ".zip")


def check_archive_target(directory, *, target=None):
    target = Path(target) if target is not None else archive_path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Archive target already exists: " + str(target))
    return target


def finish_archive(directory, state, status_path):
    """Keep the experiment's original failure even if final packaging also fails."""
    directory = Path(directory)
    result = {"archive_status": "created", "archive_path": str(archive_path(directory))}
    try:
        write(directory / "archive_status.json", result)
        archive(directory)
    except Exception as error:
        result.update(archive_status="failed", archive_error_type=type(error).__name__)
        state.update(result)
        if state["status"] not in ("failed", "timeout", "interrupted"):
            state.update(status="failed", phase="archive")
        try:
            write(directory / "archive_status.json", result)
            write(status_path, state)
        except OSError:
            pass  # A full/unavailable filesystem must not replace the original failure.
        return False
    return True


@contextmanager
def exclusive_zip(target):
    z = zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED)
    try:
        with z:
            yield z
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def archive(directory, *, target=None):
    directory = Path(directory).resolve()
    skipped, size, required_size, optional_size = [], 0, 0, 0
    target = check_archive_target(directory, target=target)
    with exclusive_zip(target) as z:
        for p in sorted(directory.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(directory)
            required = (rel.parts[0] in ("dataset", "sources", "setup")
                        or len(rel.parts) == 1
                        or p.name in ("result.json", "runtime.json", "job.json", "scores.jsonl",
                                      "latencies.jsonl", "memory.json", "metadata_dataflow.json",
                                      "onnx_graph.json", "telemetry_before.json", "telemetry_after.json",
                                      "error.txt")
                        or p.name.endswith((".process.json", ".trace.json.gz", ".operators.txt", ".input.json")))
            if (any(part.startswith(".") or part in ("venv", "__pycache__") for part in rel.parts)
                or any((directory / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1))
                or p.suffix not in (".json", ".jsonl", ".txt", ".md", ".csv", ".log", ".py", ".gz")
                or ".onnx." in p.name or p.name == "archive_contents.json"
                or (not required and (p.stat().st_size > OPTIONAL_FILE_LIMIT
                    or optional_size + p.stat().st_size > OPTIONAL_ARCHIVE_LIMIT))):
                skipped.append(str(rel))
                continue
            size += p.stat().st_size
            if required:
                required_size += p.stat().st_size
            else:
                optional_size += p.stat().st_size
            z.write(p, str(rel))
        import json
        z.writestr("archive_contents.json", json.dumps({"uncompressed_bytes": size,
            "required_bytes": required_size, "optional_bytes": optional_size,
            "optional_budget_bytes": OPTIONAL_ARCHIVE_LIMIT, "excluded": skipped}))
    return target
