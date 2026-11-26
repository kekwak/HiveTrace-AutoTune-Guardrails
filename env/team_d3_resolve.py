"""Resolve Linux CPython 3.13 wheels without installing packages or model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
ROOTS = ["torch==2.11.0+cu130", "torchvision==0.26.0+cu130",
         "transformers==5.17.0", "flash-linear-attention==0.5.2",
         "accelerate>=1.10", "safetensors>=0.6", "huggingface_hub>=1.0",
         "einops", "pillow>=10", "opencv-python-headless"]
PLATFORMS = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)] + [
    "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]
UNVERIFIED = "# Prepared d3 lock; installation and GPU compatibility have NOT been verified."


def target_environment():
    env = default_environment()
    env.update(implementation_name="cpython", implementation_version="3.13.0",
               os_name="posix", platform_machine="x86_64", platform_system="Linux",
               platform_release="", platform_version="", python_full_version="3.13.0",
               platform_python_implementation="CPython", python_version="3.13", sys_platform="linux")
    return env


def target_requirements(report, roots):
    """Expand markers and extras for the target, not pip's host environment."""
    rows = {canonicalize_name(r["metadata"]["name"]): r["metadata"] for r in report["install"]}
    wanted = set(roots)
    pending = list(roots)
    visited = set()
    env = target_environment()
    while pending:
        text = pending.pop()
        req = Requirement(text)
        name = canonicalize_name(req.name)
        if name not in rows:
            continue
        for extra in {"", *req.extras}:
            if (name, extra) in visited:
                continue
            visited.add((name, extra))
            for value in rows[name].get("requires_dist", []):
                child = Requirement(value)
                if child.marker and not child.marker.evaluate({**env, "extra": extra}):
                    continue
                child.marker = None
                value = str(child)
                if value not in wanted:
                    wanted.add(value)
                    pending.append(value)
    return wanted


def validate_artifact_url(download):
    url = urlsplit(download["url"])
    hosts = {"files.pythonhosted.org", "pypi.nvidia.com", "download.pytorch.org", "download-r2.pytorch.org"}
    if (url.scheme != "https" or url.hostname not in hosts or url.username or url.password
            or url.query or url.fragment or not url.path.endswith(".whl")):
        raise ValueError("Unverified wheel artifact URL")


def validate_report(report, required, computed=None, *, dependencies_only=False):
    rows = {canonicalize_name(r["metadata"]["name"]): r for r in report["install"]}
    for text in required:
        req = Requirement(text)
        row = rows.get(canonicalize_name(req.name))
        if row is None or not req.specifier.contains(row["metadata"]["version"], prereleases=True):
            raise ValueError("Target dependency unsatisfied: " + text)
    if dependencies_only:
        return
    for row in rows.values():
        if not SpecifierSet(row["metadata"].get("requires_python", "")).contains("3.13.0", prereleases=True):
            raise ValueError("Python 3.13 incompatible: " + row["metadata"]["name"])
        download = row["download_info"]
        validate_artifact_url(download)
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256", "")
        digest = digest or (computed or {}).get(download["url"], "")
        if not re.fullmatch("[a-f0-9]{64}", digest) or row.get("is_yanked"):
            raise ValueError("Unverified wheel artifact: " + row["metadata"]["name"])


def lock_text(lock):
    lines = ["# Generated from team-d3-resolved; Linux x86_64 / CPython 3.13 only.",
             "# Exact wheels; install into a fresh pip-less venv with --no-deps --require-hashes.", UNVERIFIED]
    for row in lock["packages"]:
        lines.extend(["# " + row["name"] + "==" + row["version"],
                      f"{row['name']} @ {row['url']} --hash=sha256:{row['sha256']}"])
    return "\n".join(lines) + "\n"


def build_lock(report_path, output):
    report = json.loads(report_path.read_text())
    validate_report(report, target_requirements(report, ROOTS), dependencies_only=True)
    computed = {}
    for row in report["install"]:
        download = row["download_info"]
        if download.get("archive_info", {}).get("hashes", {}).get("sha256"):
            continue
        # Validate origin before downloading an artifact whose index omitted its digest.
        validate_artifact_url(download)
        checksum = hashlib.sha256()
        with urlopen(download["url"], timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                checksum.update(chunk)
        computed[download["url"]] = checksum.hexdigest()
    validate_report(report, target_requirements(report, ROOTS), computed)
    packages = sorted([{"name": canonicalize_name(r["metadata"]["name"]),
                        "version": r["metadata"]["version"], "url": r["download_info"]["url"],
                        "sha256": r["download_info"].get("archive_info", {}).get("hashes", {}).get(
                            "sha256") or computed[r["download_info"]["url"]]}
                       for r in report["install"]], key=lambda row: row["name"])
    base_path = ROOT / "env/team_rtx.lock.json"
    original = {r["name"]: r for r in json.loads(base_path.read_text())["packages"]}
    for row in packages:
        if row["name"] in original and row["name"] not in {"transformers", "flash-linear-attention", "fla-core"}:
            old = original[row["name"]]
            if (row["version"], row["sha256"]) != (old["version"], old["sha256"]):
                raise ValueError("Preserved artifact changed: " + row["name"])
    lock = {"format_version": 1, "profile": "team-d3", "run": "team-d3-resolved",
            "package_count": len(packages), "packages": packages,
            "source_install_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "source_environment": report["environment"], "target_environment": target_environment(),
            "source_pip_version": report["pip_version"], "root_requirements": ROOTS,
            "computed_download_sha256": computed, "target_marker_closure_verified": True,
            "base_lock_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "source_smoke_passed": False, "lock_redeployment_verified": False,
            "gpu_performance_verified": False, "quality_evaluated": False}
    (output / "team_d3.resolve.json").write_bytes(report_path.read_bytes())
    (output / "team_d3.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    (output / "team_d3.lock.txt").write_text(lock_text(lock))
    (output / "status.json").write_text(json.dumps({"status": "resolved", "packages": len(packages)}))


def resolve(output, completed_report=None):
    output.mkdir(parents=True, exist_ok=False)
    base_path = ROOT / "env/team_rtx.lock.json"
    base = json.loads(base_path.read_text())
    changed = {"transformers", "flash-linear-attention", "fla-core"}
    constraints = output / "constraints.txt"
    constraints.write_text("\n".join(p["name"] + "==" + p["version"]
                                      for p in base["packages"] if p["name"] not in changed) + "\n")
    required = set(ROOTS)
    try:
        if completed_report is not None:
            build_lock(completed_report, output)
            return
        for attempt in range(10):
            report_path = output / f"resolve-{attempt}.json"
            command = [sys.executable, "-m", "pip", "--isolated", "install", "--dry-run",
                       "--ignore-installed", "--only-binary=:all:", "--python-version", "3.13",
                       "--implementation", "cp", "--abi", "cp313", "--index-url", "https://pypi.org/simple",
                       "--extra-index-url", "https://download.pytorch.org/whl/cu130",
                       "--extra-index-url", "https://pypi.nvidia.com", "--report", str(report_path),
                       "-c", str(constraints)]
            for value in PLATFORMS:
                command += ["--platform", value]
            command += sorted(required)
            (output / f"command-{attempt}.json").write_text(json.dumps(command, indent=2) + "\n")
            with (output / f"resolve-{attempt}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
            report = json.loads(report_path.read_text())
            expanded = target_requirements(report, required)
            try:
                validate_report(report, expanded, dependencies_only=True)
            except ValueError:
                if expanded == required:
                    raise
            else:
                required = expanded
                break
            required = expanded
        else:
            raise ValueError("Target dependency closure did not converge")
        build_lock(report_path, output)
    except Exception as error:
        (output / "status.json").write_text(json.dumps({"status": "blocked", "error_type": type(error).__name__,
                                                       "error": str(error)}))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Finalize an existing complete pip report; recheck target closure")
    args = parser.parse_args()
    resolve(args.output, args.report)
