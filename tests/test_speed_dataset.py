import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

from speed_dataset.core import (CONTRACTS, bucket, digest, file_hash, preparation_hash,
                                validate_index, validate_pool, write_json, write_jsonl)
from speed_dataset.pipeline import describe, finalize, select
from speed_dataset.preparation import author_prompt, normalized
from speed_dataset.sources import excerpt, synthetic, text_record


def fixture(root, count=2):
    rows, indices = [], {m: {} for m in CONTRACTS["models"]}
    for b, n in (("B1", 300), ("B2", 700), ("B3", 1400), ("B4", 3000)):
        for lang in ("en", "ru"):
            for source in ("wikipedia", "synthetic"):
                for i in range(count):
                    row = text_record(f"{b} {lang} {source} unique text {i}", lang, source, {}, "test")
                    rows.append(row)
                    for model, spec in CONTRACTS["models"].items():
                        indices[model][row["id"]] = {"id": row["id"], "total_tokens": n, "user_tokens": n - 30,
                            "bucket": b, "supported": n <= spec["limit"], "input_changed": False,
                            "reason": None if n <= spec["limit"] else "outside_input_limit",
                            "ids_sha256": "a" * 64, "mask_sha256": "b" * 64, "input_sha256": "c" * 64}
    write_jsonl(root / "pool.jsonl", rows)
    manifest = {"count": len(rows), "pool_sha256": file_hash(root / "pool.jsonl")}
    write_json(root / "pool_manifest.json", manifest)
    for model, mapping in indices.items():
        path = root / "indices" / f"{model}.jsonl"
        write_jsonl(path, mapping.values())
        meta = {"model": model, "status": "complete", "pool_sha256": manifest["pool_sha256"],
                "contract_sha256": digest(CONTRACTS["models"][model]), "preparation_sha256": preparation_hash(),
                "index_sha256": file_hash(path), "assets": {"tokenizer.json": "d" * 64},
                "parity": {"status": "passed"}, "versions": {**CONTRACTS["dependencies"],
                    **CONTRACTS["models"][model].get("dependencies", {})}}
        write_json(root / "indices" / f"{model}.meta.json", meta)
    return rows, manifest, indices


class SpeedCorpusTests(unittest.TestCase):
    def test_boundaries(self):
        for n, expected in [(0, None), (1, "B1"), (512, "B1"), (513, "B2"),
                            (1024, "B2"), (1025, "B3"), (2048, "B3"),
                            (2049, "B4"), (4096, "B4"), (4097, None)]:
            self.assertEqual(bucket(n), expected)

    def test_unicode_and_no_padding(self):
        t = "Ёж, e\u0301, 🙂\n日本語"
        row = text_record(t, "ru", "synthetic", {}, "test")
        self.assertEqual(row["text"], t)
        self.assertEqual(normalized({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}),
                         {"input_ids": [1, 2], "attention_mask": [1, 1]})
        with self.assertRaises(ValueError):
            normalized({"input_ids": [1, 2], "attention_mask": [1, 0]})

    def test_four_bucket_model_membership(self):
        self.assertEqual(len(CONTRACTS["buckets"]), 4)
        self.assertEqual(set(CONTRACTS["buckets"]["B4"]["models"]), {"halo", "hive", "open", "shield"})
        for m in ("gliner", "koala", "laya", "llama"):
            self.assertEqual(CONTRACTS["models"][m]["limit"], 512)

    def test_mirror_is_explicit_and_original_index_is_stale(self):
        spec = CONTRACTS["models"]["llama"]
        self.assertEqual(spec["repo"], "hipocap/Llama-Prompt-Guard-2-22M")
        self.assertEqual(len(spec["revision"]), 40)
        self.assertFalse(spec["provenance"]["original_equivalence_verified"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, manifest, _ = fixture(root)
            path = root / "indices/llama.meta.json"
            meta = json.loads(path.read_text())
            meta["contract_sha256"] = digest({
                "repo": "meta-llama/Llama-Prompt-Guard-2-22M",
                "revision": "11614a155199674a0a95e6602d6ab0417b790ed0", "limit": 512})
            write_json(path, meta)
            with self.assertRaisesRegex(ValueError, "mismatched contract"):
                validate_index(root / "indices", "llama", rows, manifest)

    def test_long_synthetic_is_deterministic_not_repeated_filler(self):
        for lang, minimum in (("en", 9500), ("ru", 6500)):
            for i in range(4):
                row = synthetic(lang, i, long_only=True)
                self.assertEqual(row, synthetic(lang, i, long_only=True))
                self.assertGreaterEqual(len(row["text"]), minimum)
                lines = row["text"].splitlines()
                self.assertEqual(len(lines), len(set(lines)))
                self.assertTrue(row["provenance"]["long_only"])

    def test_literal_prompt_not_executed(self):
        self.assertEqual(author_prompt('```python\nGUARD_SYSTEM_PROMPT = "hello"\n```'), "hello")
        with self.assertRaises((ValueError, SyntaxError)):
            author_prompt('```python\nGUARD_SYSTEM_PROMPT = __import__("os").system("false")\n```')

    def test_source_offsets_and_determinism(self):
        article = "Первое предложение. Second sentence. " * 15
        a = excerpt(article, 120, 42)
        self.assertEqual(a, excerpt(article, 120, 42))
        text, start, end = a
        self.assertEqual(text, article[start:end])
        for lang in ("en", "ru"):
            for i in range(20):
                self.assertEqual(synthetic(lang, i), synthetic(lang, i))
                lines = synthetic(lang, i)["text"].splitlines()
                self.assertEqual(len(lines), len(set(lines)))

    def test_selection_intersection_and_quotas(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, manifest, indices = fixture(Path(tmp))
            quotas = {(lang, source): 1 for lang in ("en", "ru") for source in ("wikipedia", "synthetic")}
            a = select(rows, indices, quotas=quotas)
            self.assertEqual(a, select(list(reversed(rows)), indices, quotas=quotas))
            ids = [i for result in a.values() for i in result["ids"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(r["status"] == "ready" for r in a.values()))
            del indices["llama"]
            b = select(rows, indices, quotas=quotas)
            self.assertEqual(b["B1"]["status"], "not_ready")
            self.assertEqual(b["B2"]["status"], "ready")
            self.assertNotIn("ids", b["B1"])

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _, indices = fixture(Path(tmp), 1)
            for row in indices["gliner"].values():
                row.update(supported=False, input_changed=True, reason="processor_changed_text")
            quotas = {(lang, source): 1 for lang in ("en", "ru") for source in ("wikipedia", "synthetic")}
            self.assertEqual(select(rows, indices, quotas=quotas)["B1"]["status"], "not_ready")

    def test_pool_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            validate_pool(root)
            with (root / "pool.jsonl").open("a") as f:
                f.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_pool(root)

    def test_stale_missing_and_duplicate_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, manifest, _ = fixture(root)
            validate_index(root / "indices", "llama", pool, manifest)
            meta_path = root / "indices/llama.meta.json"
            good = json.loads(meta_path.read_text())
            for field in ("pool_sha256", "contract_sha256", "preparation_sha256", "status"):
                bad = copy.deepcopy(good)
                bad[field] = "stale"
                write_json(meta_path, bad)
                with self.assertRaises(ValueError):
                    validate_index(root / "indices", "llama", pool, manifest)
            write_json(meta_path, good)
            path = root / "indices/llama.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[-1] = records[0]
            write_jsonl(path, records)
            good["index_sha256"] = file_hash(path)
            write_json(meta_path, good)
            with self.assertRaisesRegex(ValueError, "missing, duplicate or extra"):
                validate_index(root / "indices", "llama", pool, manifest)

    def test_incomplete_finalize_and_histogram(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            report = finalize(root)
            self.assertEqual(report["status"], "not_ready")
            self.assertEqual(report["selected_count"], 0)
            self.assertFalse(report["quality_evaluated"])
            self.assertFalse(report["gpu_verified"])
        stat = describe([1, 2, 511, 512], 1, 512)
        self.assertEqual(sum(bin["count"] for bin in stat["histogram"]), 4)

    def test_handoff_notebook(self):
        from speed_dataset.package import package
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            package(root)
            nb = json.loads((root / "handoff/llama_index_cpu.ipynb").read_text())
            notebook_text = "\n".join("".join(c["source"]) for c in nb["cells"])
            self.assertIn(CONTRACTS["models"]["llama"]["repo"], notebook_text)
            self.assertIn(CONTRACTS["models"]["llama"]["revision"], notebook_text)
            self.assertNotIn("getpass", notebook_text)
            self.assertIn('HF_HUB_DISABLE_IMPLICIT_TOKEN="1"', notebook_text)
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    ast.parse("".join(c["source"]))
            with zipfile.ZipFile(root / "handoff/llama_index_request.zip") as z:
                self.assertIn("pool.jsonl", z.namelist())
                self.assertNotIn("indices/llama.jsonl", z.namelist())
                self.assertFalse(any("token" in p and p.endswith(".json") for p in z.namelist()))

    def test_import_validates_before_replacing(self):
        from speed_dataset.pipeline import import_index
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, manifest, _ = fixture(root)
            old_hash = file_hash(root / "indices/llama.meta.json")
            archive = root / "response.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for name in ("llama.meta.json", "llama.jsonl"):
                    z.write(root / "indices" / name, name)
            import_index(root, archive)
            self.assertEqual(file_hash(root / "indices/llama.meta.json"), old_hash)
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../llama.meta.json", "{}")
                z.writestr("llama.jsonl", "")
            with self.assertRaises(ValueError):
                import_index(root, archive)
            self.assertEqual(file_hash(root / "indices/llama.meta.json"), old_hash)

    def test_full_release_then_missing_model_does_not_leave_stale_b1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, 150)
            good = finalize(root)
            self.assertEqual(good["status"], "ready")
            self.assertEqual(good["selected_count"], 1600)
            (root / "indices/llama.meta.json").unlink()
            partial = finalize(root)
            self.assertEqual(partial["status"], "not_ready")
            self.assertEqual(partial["selected_count"], 1200)
            latest = json.loads((root / "latest.json").read_text())
            self.assertFalse((root / latest["release"] / "B1.ids.json").exists())

    def test_reuse_requires_same_assets(self):
        from unittest.mock import patch
        from speed_dataset.pipeline import index_pool
        class FakePreparer:
            versions = {**CONTRACTS["dependencies"]}
            assets = {"tokenizer.json": "changed"}
            spec = CONTRACTS["models"]["llama"]
            def __init__(self, name):
                pass
            def parity(self):
                return {"status": "passed"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "old")
            fixture(root / "new")
            with patch("speed_dataset.preparation.Preparer", FakePreparer):
                with self.assertRaisesRegex(ValueError, "assets/environment changed"):
                    index_pool(root / "new", "llama", root / "old")
            meta = json.loads((root / "new/indices/llama.meta.json").read_text())
            self.assertEqual(meta["status"], "not_ready")


if __name__ == "__main__":
    unittest.main()
