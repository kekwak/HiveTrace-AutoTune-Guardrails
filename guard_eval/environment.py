"""Паспорт среды: GPU, драйвер, версии пакетов."""

import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("torch", "transformers", "torchao", "bitsandbytes", "gliner2", "numpy", "pandas", "huggingface_hub")


def _version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def describe(device):
    import torch
    env = {
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in PACKAGES},
        "cpu_count": os.cpu_count(),
        "device": str(device),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(index)
        env.update(
            gpu=props.name,
            capability=f"sm_{props.major}{props.minor}",
            gpu_memory_gb=round(props.total_memory / 2**30, 1),
            cuda_runtime=torch.version.cuda,
            driver=_driver(),
        )
        env["device_class"] = f"{props.name} ({env['capability']})"
    else:
        env["device_class"] = "cpu"
    return env


def _driver():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return out[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None
