import json
import tempfile
import unittest
from pathlib import Path

from speed_dataset.core import CONTRACTS, file_hash, read_jsonl, write_json
from speed_dataset.pipeline import finalize
from test_speed_dataset import fixture
from tools.export_speed_dataset import export


def complete_fixture(root):
    fixture(root, 150)
    for model in CONTRACTS["models"]:
        path = root / "indices" / f"{model}.meta.json"
        meta = json.loads(path.read_text())
        meta.update(contract=CONTRACTS["models"][model], input_contract={"truncation": False})
        write_json(path, meta)
    return finalize(root)


class SpeedExportTests(unittest.TestCase):
    def test_allowlisted_compact_reproducible_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete_fixture(root / "source")
            (root / "source" / "private.env").write_text("never publish")
            a = export(root / "source", root / "a", "owner/test")
            b = export(root / "source", root / "b", "owner/test")
            self.assertEqual(a, b)
            self.assertFalse((root / "a/private.env").exists())
            self.assertFalse((root / "a/pool.jsonl").exists())
            self.assertEqual(len(read_jsonl(root / "a/corpus.jsonl")), 1600)
            for model in CONTRACTS["models"]:
                expected = 1600 if model in CONTRACTS["buckets"]["B4"]["models"] else 400
                meta = a["indices"][model]
                self.assertEqual(meta["count"], expected)
                self.assertEqual(len(read_jsonl(root / "a" / meta["index_file"])), expected)
                self.assertNotIn("input_contract", meta)
            for name, checksum in a["files"].items():
                self.assertEqual(file_hash(root / "a" / name), checksum)
            with self.assertRaisesRegex(ValueError, "already exists"):
                export(root / "source", root / "a", "owner/test")

    def test_tampered_release_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete_fixture(root / "source")
            latest = json.loads((root / "source/latest.json").read_text())
            (root / "source" / latest["release"] / "corpus.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                export(root / "source", root / "out", "owner/test")
            self.assertFalse((root / "out").exists())

    def test_incomplete_release_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete_fixture(root / "source")
            (root / "source/indices/llama.meta.json").unlink()
            finalize(root / "source")
            with self.assertRaisesRegex(ValueError, "complete release"):
                export(root / "source", root / "out", "owner/test")
            self.assertFalse((root / "out").exists())
