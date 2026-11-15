"""Confirmation supervisor, kept outside the approved experiments package."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_bundle(bundle):
    sources, approval = bundle["sources"], bundle["approval"]
    if not sources or any(Path(name).name != name or not name.endswith(".py") for name in sources):
        raise ValueError("Invalid frozen source paths")
    if digest(sources) != approval["source_hash"]:
        raise ValueError("Frozen sources differ from screening approval")
    m = bundle["manifest"]
    if (m["status"] != "completed" or not approval.get("regression_passed")
            or approval["source_hash"] != m["source_hash"]
            or approval["model"] != m["config"]["model"]
            or approval["spec_hash"] != digest(m["candidate_spec"])
            or approval["measurement_protocol"] != m["measurement_protocol"]):
        raise ValueError("Invalid screening provenance")
    if m["config"]["candidate"] not in ("H2_linear", "C1_dynamic"):
        raise ValueError("Only screened FLA/compile finalists are allowed")
    for name, source in sources.items():
        compile(source, name, "exec")


def restore_bundle(bundle, work):
    verify_bundle(bundle)
    package = Path(work) / "experiments"
    package.mkdir(exist_ok=True)
    if list(package.glob("*.py")):
        raise ValueError("Use a fresh work directory, not an existing source package")
    for name, source in bundle["sources"].items():
        (package / name).write_text(source)
    write(Path(work) / "bundle.json", bundle)
    write(Path(work) / "screening_approval.json", bundle["approval"])


def environment_diff(expected, actual):
    keys = ("python", "torch", "cuda", "cudnn", "gpu", "packages")
    differences = {}
    for key in keys:
        if expected[key] != actual[key]:
            if key == "packages":
                differences[key] = {name: {"expected": expected[key].get(name), "actual": actual[key].get(name)}
                                    for name in sorted(expected[key].keys() | actual[key].keys())
                                    if expected[key].get(name) != actual[key].get(name)}
            else:
                differences[key] = {"expected": expected[key], "actual": actual[key]}
    return differences


def check_environment(bundle, directory):
    # This runs in a fresh child after installation, before importing any holdout data.
    from experiments.runner import hardware
    actual = hardware()
    write(Path(directory) / "environment.json", actual)
    differences = environment_diff(bundle["environment"], actual)
    write(Path(directory) / "environment_diff.json", differences)
    if differences:
        raise RuntimeError("Environment differs from screening; holdout was not opened")
    return actual


def provenance(fresh_runtime, provider_run):
    if fresh_runtime is not True or not isinstance(provider_run, str) or not provider_run.strip():
        raise ValueError("Confirm a newly allocated runtime and supply its provider run reference")
    facts = {"fresh_runtime_declared": True, "provider_run": provider_run.strip(),
             "recorded_unix": time.time(), "independence_note":
             "Fingerprints reject duplicates; distinct values do not prove independent hardware"}
    for name, file in (("boot", "/proc/sys/kernel/random/boot_id"), ("host", "/etc/hostname")):
        try:
            value = Path(file).read_text().strip()
            facts[name + "_sha256"] = digest(value) if value else None
        except OSError:
            facts[name + "_sha256"] = None
    try:
        value = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        facts["gpu_driver_sha256"] = digest(value)
    except (OSError, subprocess.SubprocessError):
        facts["gpu_driver_sha256"] = None
    return facts


def install_locked(bundle, directory, deadline):
    from experiments.round2 import protected_packages, run_process
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    required = {k: bundle["environment"][k] for k in ("python", "torch", "cuda", "cudnn", "gpu")}
    probe = ("import json, platform, torch; "
             "assert torch.cuda.is_available(), 'CUDA T4 required'; "
             "actual = dict(python=platform.python_version(), torch=torch.__version__, "
             "cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0)); "
             "print(json.dumps(actual)); " + f"assert actual == {required!r}, 'Core runtime differs from screening'")
    checked = run_process([sys.executable, "-c", probe], directory, "hardware",
                          min(120, max(.01, deadline - time.monotonic())))
    if checked["status"] != "completed":
        raise RuntimeError("Preinstalled GPU/Python/PyTorch mismatch; see installation/hardware.log")
    before = protected_packages()
    if "torch" not in before:
        raise RuntimeError("A preinstalled CUDA PyTorch is required")
    expected = {k.lower().replace("_", "-"): v for k, v in bundle["environment"]["packages"].items()}
    if any(expected.get(k) != v for k, v in before.items()):
        raise RuntimeError("Protected CUDA packages differ; automatic replacement is forbidden")
    write(directory / "protected_packages.json", before)
    constraints = directory / "constraints.txt"
    constraints.write_text("\n".join(f"{k}=={v}" for k, v in sorted(expected.items())) + "\n")
    requirements = []
    for item in bundle["install_report"]["install"]:
        name, version = item["metadata"]["name"], item["metadata"]["version"]
        info = item["download_info"]
        if "vcs_info" in info:
            vcs = info["vcs_info"]
            if vcs["vcs"] != "git":
                raise ValueError("Unsupported installation provenance")
            requirements.append(f"{name} @ git+{info['url']}@{vcs['commit_id']}")
        else:
            requirements.append(f"{name}=={version}")
    write(directory / "requirements.json", requirements)
    for dry in (True, False):
        if time.monotonic() >= deadline:
            raise TimeoutError("Installation budget exhausted")
        name = "resolve" if dry else "install"
        command = [sys.executable, "-m", "pip", "install", "--constraint", str(constraints),
                   "--report", str(directory / (name + "_report.json"))]
        if dry:
            command.append("--dry-run")
        result = run_process(command + requirements, directory, name,
                             min(3600, deadline - time.monotonic()))
        if result["status"] != "completed":
            raise TimeoutError(name) if result["status"] == "timeout" else RuntimeError(name + " failed")
        if dry:
            for item in read(directory / "resolve_report.json")["install"]:
                key = item["metadata"]["name"].lower().replace("_", "-")
                if key in before:
                    raise RuntimeError("Resolver proposes touching protected package " + key)
    if protected_packages() != before:
        raise RuntimeError("Protected packages changed")
    vcs_check = {}
    for item in bundle["install_report"]["install"]:
        expected_vcs = item["download_info"].get("vcs_info")
        if expected_vcs:
            name = item["metadata"]["name"]
            raw = importlib.metadata.distribution(name).read_text("direct_url.json")
            installed = json.loads(raw) if raw else {}
            actual_commit = installed.get("vcs_info", {}).get("commit_id")
            vcs_check[name] = {"expected_commit": expected_vcs["commit_id"], "actual_commit": actual_commit}
            write(directory / "vcs_verification.json", vcs_check)
            if actual_commit != expected_vcs["commit_id"]:
                raise RuntimeError("Installed VCS revision differs: " + name)


def worker(work):
    work = Path(work)
    bundle = read(work / "bundle.json")
    verify_bundle(bundle)
    check_environment(bundle, work)
    from experiments.runner import run
    result = run(read(work / "config.json"), work / "runs")
    write(work / "result.json", {"run": str(result.relative_to(work)),
                                  "status": read(result / "manifest.json")["status"]})


def supervise(work, seed, fresh_runtime, provider_run, total_seconds=36000,
              diagnostic=False):
    from experiments.common import archive
    from experiments.round2 import repair_incomplete, run_process
    work = Path(work).resolve()
    state = {"status": "in_progress", "diagnostic_only": diagnostic,
             "quality_evaluated": False, "confirmation_complete": False}
    start = time.monotonic()
    deadline = start + total_seconds
    try:
        if seed not in (17, 42, 123) or not 0 < total_seconds <= (7200 if diagnostic else 36000):
            raise ValueError("Invalid seed/time budget")
        bundle = read(work / "bundle.json")
        verify_bundle(bundle)
        write(work / "runtime_provenance.json", provenance(fresh_runtime, provider_run))
        state["stage"] = "environment"
        write(work / "status.json", state)
        install_locked(bundle, work / "installation", deadline)
        config = dict(bundle["manifest"]["config"], phase="confirmation", session_seed=seed,
                      approval_file=str(work / "screening_approval.json"))
        write(work / "config.json", config)
        state["stage"] = "diagnostics" if diagnostic else "confirmation"
        write(work / "status.json", state)
        if time.monotonic() >= deadline:
            raise TimeoutError("Budget exhausted before worker")
        module = "followup.onnx_diagnostic" if diagnostic else "followup.runtime"
        outcome = run_process([sys.executable, "-m", module, "worker", str(work)], work,
                              "worker", deadline - time.monotonic(), cwd=work)
        state.update(status=outcome["status"])
        if outcome["status"] == "completed":
            state.update(read(work / "result.json"))
        else:
            repair_incomplete(work, outcome["status"])
    except Exception as error:
        state.update(status="timeout" if isinstance(error, TimeoutError) else "failed", error=str(error))
        (work / "error.txt").write_text(traceback.format_exc())
    finally:
        state["seconds"] = time.monotonic() - start
        write(work / "status.json", state)
        archive(work)
    return state


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "worker":
        raise SystemExit("Internal worker: python -m followup.runtime worker WORK")
    worker(sys.argv[2])
