"""Explicit environment profiles; never repair dependencies or select a CPU fallback."""
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import traceback

from .core import ROOT, read, sha, write


def rtx_name(name):
    import re
    return bool(re.fullmatch(r"(?:NVIDIA\s+)?(?:GeForce\s+)?RTX\s+(?:4090|5090)", name.strip()))


def profile(environment):
    path = Path(environment) / "setup/profile.json"
    value = read(path)["profile"] if path.exists() else "team-t4"
    if value not in ("team-t4", "team-rtx", "team-t4-laya"):
        raise ValueError("Unknown saved environment profile")
    return value


def expected_lock(environment):
    name = "team_environment.lock.json" if profile(environment) == "team-t4" else "team_rtx.lock.json"
    return read(ROOT / "env" / name)


def rtx_requirements(lock):
    from followup.team_lock import requirements
    return requirements(lock).replace("# Original smoke passed; redeployment from this lock has NOT been verified.",
        "# Derived RTX lock: installation, Laya runtime and GPU compatibility NOT verified.")


def extended_gpu(name, selected):
    if selected == "team-t4-laya":
        return name.strip() in ("Tesla T4", "NVIDIA T4")
    if selected == "team-rtx":
        return rtx_name(name)
    raise ValueError("Unknown extended environment profile")


def precheck_rtx(selected="team-rtx"):
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 13):
        raise RuntimeError(selected + " requires Linux x86_64 / Python 3.13; no automatic version changes")
    pip = importlib.metadata.version("pip")
    if tuple(map(int, pip.split(".")[:2])) < (22, 3):
        raise RuntimeError("Host pip >=22.3 required")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=20)
    first = result.stdout.splitlines()[0].split(",")[0] if result.stdout.splitlines() else ""
    if not extended_gpu(first, selected):
        raise RuntimeError("GPU 0 does not match " + selected + "; no fallback")
    return {"python": platform.python_version(), "platform": platform.platform(), "pip": pip,
            "visible_device": 0, "nvidia_smi": result.stdout.strip()}


def probe(environment, directory):
    selected = profile(environment)
    if selected == "team-t4":
        from followup.team_skeleton import probe as old_probe
        return old_probe(environment, directory)
    import ctypes
    import random
    import site
    import numpy as np
    import torch
    from followup.team_lock import verify_inventory
    if sys.prefix == sys.base_prefix or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Expected isolated Python 3.13 venv")
    packages = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions() if p.metadata.get("Name")}
    write(Path(directory) / "packages.json", packages)
    lock = expected_lock(environment)
    if read(Path(environment) / "setup/lock.json") != lock:
        raise ValueError("Environment lock differs from the requested profile")
    verify_inventory(lock, packages)
    if not torch.cuda.is_available() or not extended_gpu(torch.cuda.get_device_name(0), selected):
        raise RuntimeError(selected + " CUDA unavailable; no fallback")
    if torch.version.cuda != "13.0" or torch.backends.cudnn.version() != 91900:
        raise RuntimeError("Pinned CUDA/cuDNN mismatch")
    libs = [p for base in site.getsitepackages() for p in Path(base).glob("nvidia/cu13/lib/libnvrtc-builtins.so.13.0")]
    if not libs:
        raise RuntimeError("CUDA 13 NVRTC builtins missing")
    ctypes.CDLL(str(libs[0]), mode=ctypes.RTLD_GLOBAL)
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
    x = torch.ones(16, 16, device="cuda:0")
    if not torch.equal(x @ x, torch.full_like(x, 16)):
        raise RuntimeError("CUDA primitive failed")
    torch.cuda.synchronize()
    gpu = torch.cuda.get_device_properties(0)
    write(Path(directory) / "runtime.json", {"profile": selected, "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu.name, "compute_capability": [gpu.major, gpu.minor], "total_memory_bytes": gpu.total_memory,
        "seed": 42, "cpu_threads": 2, "tf32": False, "deterministic": "warn_only", "hub_kernels": False,
        "visible_device": 0, "cuda_primitive_passed": True, "model_tested": False, "ort_cuda_tested": False})


def setup_rtx(directory, selected="team-rtx"):
    from followup.team_lock import requirements
    from .process import archive, invoke
    if selected not in ("team-rtx", "team-t4-laya"):
        raise ValueError("Unknown extended environment profile")
    directory = Path(directory).resolve()
    folder = directory / "setup"
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "profile.json", {"profile": selected})
    state = {"status": "running", "profile": selected, "gpu_verified_for_models": False}
    try:
        write(folder / "host.json", precheck_rtx(selected))
        lock = expected_lock(directory)
        write(folder / "lock.json", lock)
        text = rtx_requirements(lock)
        if text != (ROOT / "env/team_rtx.lock.txt").read_text():
            raise ValueError("RTX lock text differs from manifest")
        (folder / "requirements.txt").write_text(text)
        python = directory / "venv/bin/python"
        invoke([sys.executable, "-m", "venv", "--without-pip", directory / "venv"], folder / "venv.log", 120)
        invoke([sys.executable, "-m", "pip", "--python", python, "install", "--no-deps", "--require-hashes",
                "--only-binary=:all:", "-r", folder / "requirements.txt", "--report", folder / "install.json"],
               folder / "install.log", 3600)
        invoke([sys.executable, "-m", "pip", "--python", python, "check"], folder / "pip-check.log", 120)
        invoke([python, ROOT / "tools/benchmark.py", "_probe", "--environment", directory, "--output", folder],
               folder / "probe.log", 180)
        state.update(status="passed", lock_sha256=sha(folder / "lock.json"), cuda_primitive_passed=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        (folder / "error.txt").write_text(traceback.format_exc())
    finally:
        write(folder / "status.json", state)
        archive(directory)
    return state


def auth_check(model="shield"):
    from .contracts import MODEL_SPECS
    if model not in MODEL_SPECS:
        raise ValueError("Unknown model")
    spec = MODEL_SPECS[model]
    result = {"status": "failed", "model": model, "revision": spec["revision"], "weights_downloaded": False}
    try:
        from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url, HfApi
        token = get_token()
        result["credential_available"] = bool(token)
        files = HfApi().list_repo_files(spec["repo"], revision=spec["revision"], token=token)
        weight = next((f for f in files if f.endswith(".safetensors")), None)
        if weight is None:
            raise ValueError("No checkpoint weight metadata found")
        for name in ("config.json", weight):
            get_hf_file_metadata(hf_hub_url(spec["repo"], name, revision=spec["revision"]), token=token)
        result.update(status="passed", access="config_and_weight_metadata", gpu_checked=False)
    except Exception as error:
        result["error_type"] = type(error).__name__
    return result
