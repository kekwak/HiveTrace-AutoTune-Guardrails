"""Delivery checks for the pinned new-family input snapshots; no HF or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "results/u1_handoff/2026-11-26-d3-reindex/restore_data.py"
spec = importlib.util.spec_from_file_location("restore_indices_fixture", SCRIPT)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class RestoreIndicesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.corpus = b'{"fixture": true}\n'
        (self.source / "corpus.jsonl").write_bytes(self.corpus)
        self.output = self.root / "output"
        self.archive = self.root / "indices.zip"
        self.files = {}
        self.snapshots = {}
        for name in ("old", "current"):
            index = b'{"index": 1}\n'
            manifest = json.dumps({"files": {
                "corpus.jsonl": digest(self.corpus), "indices/test.json": digest(index)
            }}).encode()
            self.files[name + "/manifest.json"] = manifest
            self.files[name + "/indices/test.json"] = index
            self.snapshots[name] = digest(manifest)
        self.write_archive()
        self.addCleanup(patch.stopall)
        patch.object(restore, "ARCHIVE", self.archive).start()
        patch.object(restore, "CORPUS_SHA256", digest(self.corpus)).start()
        patch.object(restore, "SNAPSHOTS", self.snapshots).start()
        self.verifier = patch.object(restore, "verified").start()

    def write_archive(self, extra=None):
        with zipfile.ZipFile(self.archive, "w") as z:
            for name, value in self.files.items():
                z.writestr(name, value)
            if extra is not None:
                z.writestr(extra, b"unexpected")

    def run_restore(self):
        with patch.object(restore, "ARCHIVE_SHA256", digest(self.archive.read_bytes())):
            return restore.restore(self.source, self.output)

    def test_restores_both_exact_snapshots(self):
        result = self.run_restore()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.verifier.call_count, 2)
        for name in self.snapshots:
            self.assertEqual((self.output / name / "corpus.jsonl").read_bytes(), self.corpus)
            self.assertEqual(digest((self.output / name / "manifest.json").read_bytes()),
                             self.snapshots[name])
        self.assertFalse(list(self.output.glob(".restore-*")))

    def test_rejects_existing_output_without_overwrite(self):
        self.run_restore()
        with self.assertRaises(FileExistsError):
            self.run_restore()
        self.assertEqual((self.output / "old/corpus.jsonl").read_bytes(), self.corpus)

    def test_rejects_wrong_corpus(self):
        (self.source / "corpus.jsonl").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "corpus checksum"):
            self.run_restore()
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_archive_checksum(self):
        with patch.object(restore, "ARCHIVE_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "archive checksum"):
                restore.restore(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_path_traversal_and_symlinks(self):
        for name in ("old/../../escape", "/absolute", "old\\escape", "old/.hidden"):
            with self.subTest(name=name):
                self.write_archive(name)
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    self.run_restore()
        link = zipfile.ZipInfo("old/link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.write_archive(link)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.run_restore()
        self.assertFalse(self.output.exists())

    def test_rejects_changed_manifest_and_extra_files(self):
        self.write_archive("old/extra.json")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.run_restore()
        self.files["current/manifest.json"] = b"{}"
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            self.run_restore()

    def test_validation_failure_publishes_neither_snapshot(self):
        self.verifier.side_effect = [None, ValueError("bad index")]
        with self.assertRaisesRegex(ValueError, "bad index"):
            self.run_restore()
        self.assertEqual(list(self.output.iterdir()), [])


class DeliveredArchiveTests(unittest.TestCase):
    def test_checked_in_archive_and_manifest_checksums(self):
        self.assertEqual(digest(restore.ARCHIVE.read_bytes()), restore.ARCHIVE_SHA256)
        with zipfile.ZipFile(restore.ARCHIVE) as z:
            self.assertIsNone(z.testzip())
            members = restore._members(z)
            expected = set()
            for name, checksum in restore.SNAPSHOTS.items():
                raw = z.read(name + "/manifest.json")
                self.assertEqual(digest(raw), checksum)
                expected.add(name + "/manifest.json")
                manifest = json.loads(raw)
                self.assertEqual(manifest["files"]["corpus.jsonl"], restore.CORPUS_SHA256)
                for path, value in manifest["files"].items():
                    if path != "corpus.jsonl":
                        expected.add(name + "/" + path)
                        self.assertEqual(digest(z.read(name + "/" + path)), value)
            self.assertEqual(set(members), expected)


if __name__ == "__main__":
    unittest.main()
