import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from u1_benchmark import distribution
from u1_benchmark.core import ROOT, read, sha
from u1_benchmark.datasets import verified
from test_u1_roles import fixture_data


class RepositoryHandoff(unittest.TestCase):
    def test_eleven_real_configs_not_generated_by_zip(self):
        files = list((ROOT / "configs/benchmark-v3").glob("*.json"))
        self.assertEqual(len(files), 11)
        release = read(ROOT / "configs/data/role_speed_v4.json")
        if release["revision"] is None:
            self.assertEqual(release["status"], "validated_locally_not_published")
        else:
            self.assertRegex(release["revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(release["status"], "published")
        for path in files:
            cfg = read(path)
            self.assertEqual(cfg["schema_version"], 3)
            self.assertEqual(cfg["data"], {"path": release["destination"], "manifest_sha256": release["manifest_sha256"]})
            self.assertNotIn("/Users/", path.read_text())
        self.assertNotIn("add(f\"configs/benchmark-v3/", (ROOT / "tools/package_role_handoff.py").read_text())

    def test_unpublished_or_moving_revision_never_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "corpus"
            release = {**read(ROOT / "configs/data/role_speed_v4.json"), "revision": None}
            with patch.object(distribution, "read", return_value=release):
                for revision in (None, "main", "v4", "49c2c0b"):
                    with self.assertRaises(ValueError):
                        distribution.fetch(target, revision=revision)
                    self.assertFalse(target.exists())

    def test_local_install_verified_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture_data(root)
            release = {"manifest_sha256": data["manifest_sha256"], "revision": None}
            with patch.object(distribution, "read", return_value=release), patch.object(distribution, "MODEL_SPECS", {"gliner": {}}):
                result = distribution.fetch(root / "installed", source=data["path"])
                self.assertEqual(result["status"], "ready")
                self.assertEqual(verified(result["path"], data["manifest_sha256"])[1]["status"], "ready")
                with self.assertRaises(FileExistsError):
                    distribution.fetch(root / "installed", source=data["path"])

    def test_wrong_corpus_cannot_be_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture_data(root)
            with self.assertRaises(ValueError):
                distribution.fetch(root / "installed", source=data["path"])
            self.assertFalse((root / "installed").exists())

    def test_hf_snapshot_symlinks_materialize_before_validation(self):
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest("huggingface_hub unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture_data(root)
            original = Path(data["path"])
            snapshot = root / "snapshot"
            blobs = root / "blobs"
            snapshot.mkdir(); blobs.mkdir()
            for path in original.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(original)
                    blob = blobs / sha(path)
                    blob.write_bytes(path.read_bytes())
                    link = snapshot / relative
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(blob)
            release = {"repo": "owner/fixture", "manifest_sha256": data["manifest_sha256"], "revision": "f" * 40}
            with patch.object(distribution, "read", side_effect=lambda p: release if str(p).endswith("role_speed_v4.json") else read(p)), patch.object(
                    distribution, "MODEL_SPECS", {"gliner": {}}), patch.object(huggingface_hub, "hf_hub_download", return_value=str(snapshot / "manifest.json")), patch.object(
                    huggingface_hub, "snapshot_download", return_value=str(snapshot)):
                result = distribution.fetch(root / "installed")
            self.assertEqual(result["status"], "ready")
            self.assertFalse(any(p.is_symlink() for p in (root / "installed").rglob("*")))

    def test_only_explicit_runtime_locks_escape_env_ignore(self):
        ignore = (ROOT / ".gitignore").read_text()
        for name in ("team_environment.lock.json", "team_environment.lock.txt", "team_rtx.lock.json", "team_rtx.lock.txt"):
            self.assertIn("!env/" + name, ignore)
        self.assertIn("env/*", ignore)


if __name__ == "__main__":
    unittest.main()
