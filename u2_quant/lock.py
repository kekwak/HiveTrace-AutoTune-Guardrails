"""team-rtx-quant lock: the unchanged U1 RTX lock plus exactly one audited torchao wheel.

python -m u2_quant.lock  # пишет env/team_rtx_quant.lock.json и .txt
"""
import copy

from u1_benchmark.core import ROOT, read, sha, write
from u1_benchmark.environment import rtx_requirements

BASE = ROOT / "env/team_rtx.lock.json"
# Собран против torch 2.11 (pytorch/ao#2919); чистый Python-wheel, без C++/CUDA-расширений.
TORCHAO = {"name": "torchao", "version": "0.17.0",
           "url": "https://files.pythonhosted.org/packages/c9/37/ef37ca885265e5f79a168616767dd416a3cea1cc3b28bb6b503ce4a5b652/torchao-0.17.0-py3-none-any.whl",
           "sha256": "02eba449036715b9ae784fbaa1a6f97994bb7b0421ce92d1d5d1c08e5bd6d349"}


def derive(base, extra):
    if any(p["name"] == extra["name"] for p in base["packages"]):
        raise ValueError("Base lock already includes " + extra["name"])
    lock = copy.deepcopy(base)
    lock["packages"] = sorted(lock["packages"] + [dict(extra)], key=lambda p: p["name"])
    lock.update(run="team-rtx-quant-derived", package_count=len(lock["packages"]), base_lock_sha256=sha(BASE),
                profile="team-rtx-quant",
                modification=f"Only {extra['name']}=={extra['version']} added to team_rtx.lock.json; base artifacts unchanged",
                lock_redeployment_verified=False, gpu_performance_verified=False)
    return lock


def main():
    lock = derive(read(BASE), TORCHAO)
    write(ROOT / "env/team_rtx_quant.lock.json", lock)
    (ROOT / "env/team_rtx_quant.lock.txt").write_text(rtx_requirements(lock))
    print(lock["package_count"], "packages")


if __name__ == "__main__":
    main()
