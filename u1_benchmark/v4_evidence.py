"""Immutable snapshots and non-executable reference evidence."""
from contextlib import contextmanager
import importlib.metadata
import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys
import tempfile
import time
import zipfile

from .core import ROOT, digest, read, sha, write
from .v4_config import MONITOR_VERSION, PROTOCOL_VERSION, ConfigurationError


def auth_check(config):
    from .v4_config import validate_config
    from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url
    cfg = validate_config(config)
    token = get_token()
    result = {"status": "failed", "credential_available": bool(token), "weights_downloaded": False,
              "gpu_checked": False}
    try:
        for name in ("baseline", "candidate"):
            if name not in cfg:
                continue
            source = cfg[name]["weights"]
            if "repo" not in source:
                continue
            info = HfApi().model_info(source["repo"], revision=source["revision"], token=token)
            if info.sha != source["revision"]:
                raise ConfigurationError("HF source resolved to a different revision")
            filename = next(f.rfilename for f in info.siblings if f.rfilename.endswith((".safetensors", ".bin", ".pt")))
            get_hf_file_metadata(hf_hub_url(source["repo"], filename, revision=source["revision"]), token=token)
        result.update(status="passed", access="pinned_config_and_weight_metadata")
    except Exception as error:
        result["error_type"] = type(error).__name__
    return result


def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def source_inventory():
    paths = [ROOT / "tools/benchmark.py", ROOT / "tools/check_v4_monitoring.py"]
    for directory in ("u1_benchmark", "experiments", "speed_dataset"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend(ROOT / "followup" / name for name in (
        "__init__.py", "onnx_service.py", "prepare.py", "runtime.py", "team_halo.py",
        "team_lock.py", "team_skeleton.py", "team_smoke.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file() and "__pycache__" not in p.parts}


def check_sources(expected):
    if source_inventory() != expected:
        raise ConfigurationError("Measurement source inventory changed")


def runtime():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is forbidden")
    gpu = torch.cuda.get_device_properties(0)
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, timeout=20, check=True)
    first = next(line for line in query.stdout.splitlines() if line.split(",")[0].strip() == "0")
    _, uuid, driver = [s.strip() for s in first.split(",")]
    packages = {p.metadata["Name"].lower().replace("_", "-"): p.version
                for p in importlib.metadata.distributions() if p.metadata.get("Name")}
    return {"gpu": gpu.name, "vram_bytes": gpu.total_memory, "gpu_uuid": uuid, "driver": driver,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(), "packages": packages,
            "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "host_sha256": digest(platform.node()), "pid": os.getpid(), "timestamp": time.time(),
            "monitor_method": MONITOR_VERSION, "visible_device": 0}


def environment_identity(value):
    return {k: v for k, v in value.items() if k not in {"gpu_uuid", "host_sha256", "pid", "timestamp", "session_id"}}


def assert_environment(actual, expected):
    if environment_identity(actual) != environment_identity(expected):
        raise ConfigurationError("Worker environment differs from resolved environment")


def configure_gpu(environment):
    from .environment import expected_lock, extended_gpu, profile
    from followup.team_lock import verify_inventory
    import random
    import numpy as np
    import torch
    selected = profile(environment)
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != (Path(environment) / "venv").resolve():
        raise RuntimeError("Expected the selected isolated environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    name = torch.cuda.get_device_name(0)
    if not (name in ("Tesla T4", "NVIDIA T4") if selected == "team-t4" else extended_gpu(name, selected)):
        raise RuntimeError("GPU does not match installed environment profile")
    packages = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions() if p.metadata.get("Name")}
    verify_inventory(expected_lock(environment), packages)
    if read(Path(environment) / "setup/lock.json") != expected_lock(environment):
        raise ConfigurationError("Installed lock does not match the selected profile")
    if platform.system() != "Linux" or platform.python_version_tuple()[:2] != ("3", "13"):
        raise RuntimeError("Pinned environment requires Linux/Python 3.13")
    if torch.version.cuda != "13.0" or torch.backends.cudnn.version() != 91900:
        raise RuntimeError("Pinned CUDA/cuDNN mismatch")
    builtins = [p for base in site.getsitepackages()
                for p in Path(base).glob("nvidia/cu13/lib/libnvrtc-builtins.so.13.0")]
    if not builtins:
        raise RuntimeError("CUDA 13 NVRTC builtins missing")
    ctypes.CDLL(str(builtins[0]), mode=ctypes.RTLD_GLOBAL)
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
    return runtime()


@contextmanager
def reference_folder(path):
    path = Path(path).resolve()
    if path.is_dir():
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="u1-reference-") as folder:
        root = Path(folder).resolve()
        with zipfile.ZipFile(path) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 2 * 1024**3:
                raise ConfigurationError("Reference evidence archive too large")
            for entry in archive.infolist():
                target = (root / entry.filename).resolve()
                if not target.is_relative_to(root) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ConfigurationError("Unsafe archive member")
                if entry.is_dir() or target.suffix not in {".json", ".jsonl"}:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        yield root


def read_reference(path):
    with reference_folder(path) as root:
        resolved = read(root / "resolved_run.json")
        state = read(root / "status.json")
        if resolved.get("schema_version") != 4 or resolved.get("protocol_version") != PROTOCOL_VERSION:
            raise ConfigurationError("Unsupported reference format")
        if state.get("resolved_sha256") != sha(root / "resolved_run.json"):
            raise ConfigurationError("Reference snapshot checksum mismatch")
        cfg = resolved["config"]
        if cfg["mode"] != "baseline" or cfg.get("pilot") or state.get("status") != "baseline_complete":
            raise ConfigurationError("Reference must be a complete full baseline")
        from .v4_report import build
        verified = build(root, save=False)
        if verified.get("status") != "baseline_complete":
            raise ConfigurationError("Reference evidence is incomplete")
        evidence = {}
        for p in sorted(root.rglob("*")):
            rel = str(p.relative_to(root))
            if p.is_file() and not p.is_symlink() and (rel in {"resolved_run.json", "status.json", "summary.json"}
                    or rel.startswith("jobs/") and p.name in {"job.json", "result.json"}):
                evidence[rel] = p.read_text()
        return {"resolved": resolved, "files": evidence,
                "sha256": sha(path) if Path(path).is_file() else digest(evidence)}


def check_reference(cfg, reference, sources, inputs, environment=None):
    from .v4_privacy import public_config
    cfg = public_config(cfg)
    old = reference["resolved"]
    previous = old["config"]
    keys = ("baseline", "buckets", "batch_sizes_by_bucket", "seed")
    if any(previous[k] != cfg[k] for k in keys):
        raise ConfigurationError("Reference model, grid or seed differs")
    if old["source_inventory"] != sources or old["input_indices"] != inputs:
        raise ConfigurationError("Reference engine or prepared inputs differ")
    if environment is not None:
        assert_environment(environment, old["environment"])


def copy_reference(reference, directory):
    for name, value in reference["files"].items():
        target = Path(directory) / "reference" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(value)
    write(Path(directory) / "reference.json", {"sha256": reference["sha256"],
          "files": {name: sha(Path(directory) / "reference" / name) for name in reference["files"]}})
