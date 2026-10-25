import platform, subprocess, sys, time
from pathlib import Path

import torch, transformers

OUT = Path("env")
OUT.mkdir(parents=True, exist_ok=True)
TS = time.strftime("%Y%m%d-%H%M%S")

def sh(cmd, name):
    (OUT / name).write_text(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)

sh("pip freeze", "pip-freeze.txt")
sh("lscpu", "cpu.txt")
sh("free -h", "mem.txt")
sh("nvidia-smi", f"nvidia-smi-{TS}.txt")

(OUT / "versions.txt").write_text(
    f"python={sys.version.split()[0]}\n"
    f"platform={platform.platform()}\n"
    f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}\n"
    f"transformers={transformers.__version__}\n"
    f"gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}\n"
    "model=hipocap/Llama-Prompt-Guard-2-22M (mirror of meta-llama/Llama-Prompt-Guard-2-22M, model.safetensors sha256 5120e30b...) seed=42"
)
print((OUT / "versions.txt").read_text())
