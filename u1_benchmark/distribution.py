"""Install the pinned speed corpus from a local directory or an explicit HF commit."""
from pathlib import Path
import re
import shutil
import tempfile

from .core import ROOT, read, sha
from .contracts import MODEL_SPECS, model_contract
from .datasets import verified


def fetch(output=None, source=None, revision=None):
    release = read(ROOT / "configs/data/role_speed_v4.json")
    expected = release["manifest_sha256"]
    target = Path(output or ROOT / release["destination"]).resolve()
    if target.exists():
        raise FileExistsError("Destination exists; use a new directory, never overwrite a corpus")
    if source is not None and revision is not None:
        raise ValueError("Select a local source OR an HF revision")
    remote = source is None
    if source is None:
        revision = revision or release["revision"]
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Role corpus is not published yet. Supply --source for the validated local corpus; after publication pin its 40-character HF commit, never main or the historical v3 commit.")
        from huggingface_hub import hf_hub_download, snapshot_download
        manifest_path = Path(hf_hub_download(release["repo"], "manifest.json", repo_type="dataset", revision=revision))
        if sha(manifest_path) != expected:
            raise ValueError("HF commit does not contain the pinned role corpus")
        manifest = read(manifest_path)
        source = snapshot_download(release["repo"], repo_type="dataset", revision=revision,
                                   allow_patterns=["manifest.json", *manifest["files"]])
    source = Path(source).resolve()
    if not remote:
        _, manifest = verified(source, expected)
    if manifest.get("schema_version") != 3 or manifest.get("status") != "ready":
        raise ValueError("A complete role corpus is required")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".u1-corpus-", dir=target.parent) as tmp:
        staging = Path(tmp) / "corpus"
        staging.mkdir()
        for name in ("manifest.json", *manifest["files"]):
            relative = Path(name)
            origin = source / name
            if relative.is_absolute() or ".." in relative.parts or any(p.startswith(".") for p in relative.parts):
                raise ValueError("Unexpected source path")
            if not remote and not origin.resolve().is_relative_to(source):
                raise ValueError("Unexpected local source symlink")
            # HF snapshots legitimately symlink into the shared blob cache. Verify the
            # pinned bytes, then materialize regular files before strict dataset checks.
            checksum = expected if name == "manifest.json" else manifest["files"][name]
            if sha(origin) != checksum:
                raise ValueError("Downloaded dataset file mismatch")
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        rows, staged = verified(staging, expected)
        from .role_data import load_speed as load_declared_speed
        compatibility = {}
        for model in MODEL_SPECS:
            declared = staged["indices"][model]["contract"]
            # Fetch verifies the pinned historical source, not a new runtime approval.
            # Actual runs still require the current contract and reject stale indices.
            load_declared_speed(staging, model, "screening", staged, rows, expected_contract=declared)
            compatibility[model] = "current" if declared == model_contract(model, 3) else "requires_reindex"
        staging.rename(target)
    return {"status": "ready", "path": str(target), "manifest_sha256": expected,
            "scope": "pinned_source_integrity", "model_contract_status": compatibility,
            "origin": "local" if revision is None else "huggingface", "revision": revision,
            "gpu_checked": False, "published": False}
