"""Generate a new RTX lock from the unchanged team lock plus one audited Laya wheel."""
import copy
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.core import ROOT, read, sha, write
from u1_benchmark.environment import rtx_requirements


def build(metadata):
    base = ROOT / "env/team_environment.lock.json"
    lock = copy.deepcopy(read(base))
    wheel = next(x for x in metadata["urls"] if x["filename"] == "laya-0.3.20-py3-none-any.whl")
    checksum = "6039e802fa5effb8dd492061cd7ad39a43087beadc4a4fa4a649614e77eb83d4"
    if wheel["digests"]["sha256"] != checksum or metadata["info"]["version"] != "0.3.20":
        raise ValueError("Unexpected Laya artifact")
    if any(x["name"] == "laya" for x in lock["packages"]):
        raise ValueError("Base lock already includes Laya")
    lock["packages"].append({"name": "laya", "version": "0.3.20", "url": wheel["url"], "sha256": checksum})
    lock["packages"].sort(key=lambda x: x["name"])
    lock.update(run="team-rtx-derived", package_count=len(lock["packages"]), base_lock_sha256=sha(base),
        source_smoke_passed=False, lock_redeployment_verified=False, gpu_performance_verified=False,
        profile="team-rtx", modification="Only laya==0.3.20 added; base artifacts unchanged",
        laya_requires_python=metadata["info"]["requires_python"], laya_requires_dist=metadata["info"]["requires_dist"],
        dependency_validation="Declared requirements checked separately; no RTX installation or GPU validation")
    write(ROOT / "env/team_rtx.lock.json", lock)
    text = rtx_requirements(lock)
    # setup checks the same generator output; the run name identifies this derived lock.
    (ROOT / "env/team_rtx.lock.txt").write_text(text)
    print(json.dumps({"status": "prepared", "package_count": lock["package_count"], "gpu_checked": False}))


if __name__ == "__main__":
    with urllib.request.urlopen("https://pypi.org/pypi/laya/0.3.20/json", timeout=30) as response:
        build(json.load(response))
