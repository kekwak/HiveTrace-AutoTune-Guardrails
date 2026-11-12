"""CPU fixtures only: no tokenizer downloads, model weights, GPU or publishing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from u1_benchmark import v4_data as data
from u1_benchmark.core import digest, sha, write
from u1_benchmark.messages import content_hash
from u1_benchmark.v4_config import preparation_identity


BOUNDS = {"B1": {"min": 1, "max": 512}}


def variant(adapter="halo"):
    return {"adapter": adapter, "weights": {"repo": "fixture/weights", "revision": "a" * 40},
            "tokenizer": {"repo": "fixture/tokenizer", "revision": "b" * 40}, "template": None,
            "preparation": {"limit": 512, **({"system_prompt": "fixture prompt", "cue": "Safety: ",
                                               "enable_thinking": False} if adapter == "halo" else {})},
            "scoring": {"threshold": 0.5} if adapter == "halo" else {"max_new_tokens": 10},
            "execution": {"runtime": "eager", "dtype": "float32", "attention": "eager",
                          "sdpa_backend": None, "linear_attention": "reference"}}


class NativeFixture:
    def __init__(self, spec):
        self.spec = spec

    def parity(self):
        return {"status": "passed"}

    def index(self, row):
        n = int(row["id"].split("-")[-1]) % 512 + 1
        ids = [int(row["id"].split("-")[-1]) + 10] * n
        mask = [1] * n
        return {"id": row["id"], "source_sha256": content_hash(row["messages"]),
                "supported": True, "reason": None, "total_tokens": n,
                "input_sha256": digest({"input_ids": ids, "attention_mask": mask}),
                "ids_sha256": digest(ids), "mask_sha256": digest(mask), "source_mode": "user-only",
                "target_index": 0, "target_role": "user", "context_indices": [],
                "user_tokens": n, "content_tokens": n, "input_changed": False}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = variant()
        self.key = preparation_identity(self.spec)

    def source(self, count=512):
        source = self.root / "source.jsonl"
        data._jsonl(source, [{"key": f"row-{i}", "body": f"PRIVATE TEXT {i}",
                              "lang": "en" if i % 2 else "ru", "origin": "one" if i % 3 else "two"}
                             for i in range(count)])
        return source

    def reference(self, spec=None, **kwargs):
        return {"spec": spec or self.spec, "buckets": copy.deepcopy(BOUNDS), "name": "fixture", **kwargs}

    def prepare(self, count=512, factory=NativeFixture, **changes):
        spec = {"schema_version": 4, "kind": "speed", "source": str(self.source(count)),
                "fields": {"id": "key", "text": "body", "language": "lang", "source": "origin"},
                "version": "test", "provenance": "fixture", "preparations": [self.reference()], **changes}
        return data.prepare(spec, self.root / "output", factory)

    def manifest(self, result):
        return data._read(Path(result["path"]) / "manifest.json")

    def mutate(self, result, callback):
        root = Path(result["path"])
        manifest = self.manifest(result)
        callback(root, manifest)
        write(root / "manifest.json", manifest)

    def test_full_round_trip_and_identity_not_weights(self):
        result = self.prepare()
        self.assertEqual(result["status"], "ready")
        changed = copy.deepcopy(self.spec)
        changed["weights"]["revision"] = "c" * 40
        changed["execution"]["runtime"] = "compile"
        selected, manifest = data.load_speed(result["path"], changed, BOUNDS, "baseline",
                                             manifest_sha256=result["manifest_sha256"])
        self.assertEqual(len(selected["B1"]), 400)
        self.assertEqual(manifest["format_version"], 4)
        self.assertEqual(set(manifest["indices"]), {self.key})
        self.assertEqual(manifest["model_buckets"][self.key]["B1"]["histogram_8"], [50] * 8)
        public = json.dumps(data.public_rows(selected["B1"]))
        self.assertNotIn("PRIVATE", public)
        self.assertNotIn("row-", public)

    def test_changed_preparation_is_not_reused(self):
        result = self.prepare(5)
        changed = copy.deepcopy(self.spec)
        changed["template"] = {"text": "different"}
        with self.assertRaisesRegex(data.DataError, "identity"):
            data.load_speed(result["path"], changed, BOUNDS, "smoke")

    def test_full_modes_require_400_smoke_and_pilot_allow_small(self):
        result = self.prepare(5)
        self.assertEqual(result["status"], "not_ready")
        for mode in ("baseline", "search", "confirm"):
            with self.subTest(mode=mode), self.assertRaisesRegex(data.DataError, "N=400"):
                data.load_speed(result["path"], self.spec, BOUNDS, mode)
        self.assertEqual(len(data.load_speed(result["path"], self.spec, BOUNDS, "smoke")[0]["B1"]), 5)
        self.assertEqual(len(data.load_speed(result["path"], self.spec, BOUNDS, "search", pilot=True)[0]["B1"]), 5)

    def test_config_reference_uses_config_bounds(self):
        config = {"schema_version": 4, "baseline": self.spec, "mode": "baseline",
                  "data": {"path": "unused", "manifest_sha256": "0" * 64},
                  "buckets": {"SHORT": {"min": 1, "max": 4}}}
        path = self.root / "config.json"
        write(path, config)
        result = self.prepare(5, preparations=[{"config": str(path), "variant": "baseline", "name": "p"}])
        selected, _ = data.load_speed(result["path"], self.spec, config["buckets"], "smoke")
        self.assertEqual(len(selected["SHORT"]), 4)

    def test_configured_bucket_subset_allowed(self):
        buckets = {"B1": {"min": 1, "max": 400}, "B4": {"min": 401, "max": 512}}
        result = self.prepare(preparations=[self.reference(buckets=buckets)])
        selected, manifest = data.load_speed(result["path"], self.spec, {"B4": buckets["B4"]}, "smoke",
                                             manifest_sha256=result["manifest_sha256"])
        self.assertEqual(set(selected), {"B4"})
        self.assertEqual(len(selected["B4"]), 112)
        self.assertEqual(set(manifest["model_buckets"][self.key]), {"B1", "B4"})
        full, _ = data.load_speed(result["path"], self.spec, {"B1": buckets["B1"]}, "baseline")
        self.assertEqual(set(full), {"B1"})
        self.assertEqual(len(full["B1"]), 400)
        with self.assertRaisesRegex(data.DataError, "N=400"):
            data.load_speed(result["path"], self.spec, {"B4": buckets["B4"]}, "baseline")

    def test_configured_bucket_widened_bounds_rejected(self):
        result = self.prepare(5)
        with self.assertRaisesRegex(data.DataError, "Changed bucket bounds"):
            data.load_speed(result["path"], self.spec, {"B1": {"min": 1, "max": 513}}, "smoke")

    def test_unknown_configured_bucket_rejected(self):
        result = self.prepare(5)
        with self.assertRaisesRegex(data.DataError, "Unknown configured bucket"):
            data.load_speed(result["path"], self.spec, {"B9": BOUNDS["B1"]}, "smoke")

    def test_same_preparation_names_coalesce_but_conflicts_fail(self):
        refs = [self.reference(), self.reference(name="second")]
        result = self.prepare(5, preparations=refs)
        self.assertEqual(self.manifest(result)["indices"][self.key]["names"], ["fixture", "second"])
        refs[1]["buckets"]["B1"]["max"] = 256
        with self.assertRaisesRegex(data.DataError, "Conflicting"):
            data._references(refs)

    def test_effective_input_deduplication(self):
        class SameInput(NativeFixture):
            def index(self, row):
                value = super().index(row)
                value["input_sha256"] = "a" * 64
                return value
        result = self.prepare(512, SameInput)
        self.assertEqual(self.manifest(result)["model_buckets"][self.key]["B1"]["selected"], 1)

    def test_manifest_and_file_tampering(self):
        result = self.prepare(5)
        with self.assertRaisesRegex(data.DataError, "manifest hash"):
            data.verified(result["path"], "0" * 64)
        root = Path(result["path"])
        with (root / "corpus.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(data.DataError, "checksum"):
            data.verified(root)

    def test_content_hash_is_checked_after_rechecksum(self):
        result = self.prepare(5)
        def corrupt(root, manifest):
            rows = data._lines(root / "corpus.jsonl")
            rows[0]["messages"][0]["content"] = "changed"
            data._jsonl(root / "corpus.jsonl", rows)
            manifest["files"]["corpus.jsonl"] = sha(root / "corpus.jsonl")
        self.mutate(result, corrupt)
        with self.assertRaisesRegex(data.DataError, "content identity"):
            data.verified(result["path"])

    def test_unlisted_index_cannot_be_used(self):
        result = self.prepare(5)
        self.mutate(result, lambda root, m: m["files"].pop(m["indices"][self.key]["index_file"]))
        with self.assertRaisesRegex(data.DataError, "not checksummed"):
            data.verified(result["path"])

    def test_unsafe_and_hidden_paths(self):
        for name in ("../x", "/tmp/x", "indices/../x", ".private/x", "a//b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(data.DataError):
                data._safe(self.root, name)

    def test_symlink_rejected(self):
        result = self.prepare(5)
        root = Path(result["path"])
        (root / "alias.jsonl").symlink_to(root / "corpus.jsonl")
        self.mutate(result, lambda root, m: m["files"].update({"alias.jsonl": sha(root / "corpus.jsonl")}))
        with self.assertRaisesRegex(data.DataError, "symlinks"):
            data.verified(root)

    def test_duplicate_json_keys_and_nonfinite_values(self):
        for raw in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(data.DataError):
                data._json(raw)

    def test_all_indices_checked_even_if_not_selected(self):
        result = self.prepare()
        def corrupt(root, manifest):
            file = manifest["indices"][self.key]["index_file"]
            rows = data._lines(root / file)
            selected = set(data._read(root / manifest["model_buckets"][self.key]["B1"]["ids_file"]))
            next(r for r in rows if r["id"] not in selected)["source_sha256"] = "0" * 64
            data._jsonl(root / file, rows)
            manifest["files"][file] = sha(root / file)
        self.mutate(result, corrupt)
        with self.assertRaisesRegex(data.DataError, "source identity"):
            data.verified(result["path"])

    def test_duplicate_selected_inputs_rejected(self):
        result = self.prepare(5)
        def corrupt(root, manifest):
            file = manifest["indices"][self.key]["index_file"]
            rows = data._lines(root / file)
            rows[1]["input_sha256"] = rows[0]["input_sha256"]
            data._jsonl(root / file, rows)
            manifest["files"][file] = sha(root / file)
        self.mutate(result, corrupt)
        with self.assertRaisesRegex(data.DataError, "duplicate effective"):
            data.verified(result["path"])

    def test_explicit_blockers_do_not_leak_exceptions(self):
        def fail(spec):
            raise RuntimeError("PRIVATE INPUT SECRET")
        result = self.prepare(5, fail)
        manifest = self.manifest(result)
        self.assertEqual(manifest["blockers"][self.key]["error_type"], "RuntimeError")
        self.assertNotIn("SECRET", json.dumps(manifest))
        self.assertFalse(manifest["indices"])

    def test_failed_native_parity_blocks(self):
        class Fail(NativeFixture):
            def parity(self):
                return {"status": "failed"}
        result = self.prepare(5, Fail)
        self.assertEqual(result["status"], "not_ready")
        self.assertIn(self.key, result["blockers"])

    def test_missing_parity_interface_is_explicit(self):
        class Missing(NativeFixture):
            parity = None
        result = self.prepare(5, Missing)
        self.assertEqual(result["blockers"][self.key]["reason"], "CPU preparer must expose parity()")

    def test_lazy_default_factory(self):
        with patch.object(data, "_factory", NativeFixture):
            result = self.prepare(5, None)
        self.assertFalse(result["blockers"])

    def test_quality_no_quota_preserves_role_slice_coverage(self):
        source = self.root / "quality.jsonl"
        values = [{"id": "q", "messages": [{"role": "assistant", "content": "private answer"}],
                   "target": "safe", "target_role": "assistant", "slices": {"language": "ru"}}]
        data._jsonl(source, values)
        result = data.prepare({"schema_version": 4, "source": str(source), "kind": "quality",
                               "version": "fixture", "provenance": "test",
                               "fields": {k: k for k in values[0]}}, self.root / "quality")
        rows, manifest = data.verified(result["path"], result["manifest_sha256"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(rows[0]["target_index"], 0)
        self.assertEqual(manifest["target_role_coverage"], {"assistant": {"safe": 1}})
        self.assertEqual(manifest["slice_coverage"], {"language": {"ru": 1}})
        self.assertEqual(manifest["indices"], {})
        self.mutate(result, lambda root, m: m.update(target_roles={"user": 1}))
        with self.assertRaisesRegex(data.DataError, "coverage"):
            data.verified(result["path"])

    def legacy(self, adapter="halo", task_overlay=False):
        result = self.prepare(5)
        root = Path(result["path"])
        manifest = self.manifest(result)
        manifest.pop("format_version")
        manifest["schema_version"] = 3
        meta = manifest["indices"].pop(self.key)
        values = data._lines(root / meta["index_file"])
        oldfile = f"indices/{adapter}.jsonl"
        data._jsonl(root / oldfile, values)
        manifest["files"][oldfile] = sha(root / oldfile)
        manifest["indices"] = {adapter: {"assets": {"task_role_preparation": "a" * 64} if task_overlay else {},
                                         "parity": {"status": "passed"}}}
        manifest["model_buckets"] = {adapter: manifest["model_buckets"][self.key]}
        write(root / "manifest.json", manifest)
        return root

    def test_reindex_preserves_corpus_bytes_and_bucket_ids(self):
        source = self.legacy()
        before = {p: sha(p) for p in source.rglob("*") if p.is_file()}
        result = data.reindex(source, self.root / "reindexed", [self.reference()], NativeFixture)
        self.assertFalse(result["blockers"])
        manifest = self.manifest(result)
        self.assertEqual(sha(source / "corpus.jsonl"), sha(Path(result["path"]) / "corpus.jsonl"))
        old = data._read(source / "manifest.json")["model_buckets"]["halo"]["B1"]
        new = manifest["model_buckets"][self.key]["B1"]
        self.assertEqual(data._read(source / old["ids_file"]), data._read(Path(result["path"]) / new["ids_file"]))
        self.assertEqual(before, {p: sha(p) for p in source.rglob("*") if p.is_file()})
        data.verified(result["path"], result["manifest_sha256"])

    def test_prepare_data_dispatches_reindex_with_full_config_reference(self):
        source = self.legacy()
        config = {"schema_version": 4, "baseline": self.spec, "mode": "baseline",
                  "buckets": BOUNDS, "data": {"path": str(source),
                                               "manifest_sha256": sha(source / "manifest.json")}}
        config_path = self.root / "baseline.json"
        write(config_path, config)
        envelope = {"operation": "reindex", "source": config["data"], "configs": [
            {"config": str(config_path), "variant": "baseline", "name": "halo",
             "old_source": {**config["data"], "model": "halo"}}]}
        result = data.prepare(envelope, self.root / "via-cli", NativeFixture)
        self.assertFalse(result["blockers"])
        self.assertEqual(self.manifest(result)["reindex"]["comparisons"][self.key]["compared_rows"], 5)
        self.assertEqual(sha(source / "corpus.jsonl"), sha(Path(result["path"]) / "corpus.jsonl"))

    def test_invalid_dispatch_never_creates_output(self):
        for envelope in ({"operation": "unknown"}, {"operation": "reindex", "source": "missing"},
                         {"operation": "reindex", "source": "missing", "configs": [], "models": ["halo"]}):
            with self.subTest(envelope=envelope), self.assertRaises(data.DataError):
                data.prepare(envelope, self.root / "never", NativeFixture)
            self.assertFalse((self.root / "never").exists())

    def test_reindex_compares_every_row_and_blocks_mismatch(self):
        source = self.legacy()
        class Changed(NativeFixture):
            def index(self, row):
                value = super().index(row)
                if row["id"] == "row-4":
                    value["ids_sha256"] = "0" * 64
                return value
        result = data.reindex(source, self.root / "reindexed", [self.reference()], Changed)
        self.assertEqual(result["blockers"][self.key]["mismatch_count"], 1)
        self.assertEqual(result["blockers"][self.key]["stage"], "reindex_parity")
        self.assertFalse(self.manifest(result)["indices"])

    def test_open_requires_task_overlay_not_native(self):
        source = self.legacy(adapter="open")
        spec = variant("open")
        key = preparation_identity(spec)
        for name, reference in (("missing", self.reference(spec)), ("native", self.reference(spec,
                 old_source={"path": str(source), "model": "open"}))):
            result = data.reindex(source, self.root / name, [reference], NativeFixture)
            self.assertEqual(result["blockers"][key]["stage"], "old_source")
        manifest = data._read(source / "manifest.json")
        manifest["indices"]["open"]["assets"]["task_role_preparation"] = "a" * 64
        write(source / "manifest.json", manifest)
        result = data.reindex(source, self.root / "overlay-pass", [self.reference(spec,
            old_source={"path": str(source), "model": "open", "manifest_sha256": sha(source / "manifest.json")})], NativeFixture)
        self.assertFalse(result["blockers"])

    def test_existing_output_and_old_subdirectory_are_never_written(self):
        source = self.legacy()
        with self.assertRaises(data.DataError):
            data.reindex(source, source, [self.reference()], NativeFixture)
        with self.assertRaises(data.DataError):
            data.reindex(source, source / "new", [self.reference()], NativeFixture)
        self.assertFalse((source / "new").exists())
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            data.reindex(source, existing, [self.reference()], NativeFixture)

    def test_overlay_output_guard_runs_before_any_write(self):
        source = self.legacy()
        overlay = self.root / "overlay"
        overlay.mkdir()
        with self.assertRaisesRegex(data.DataError, "every old source"):
            data.reindex(source, overlay / "new", [self.reference(
                old_source={"path": str(overlay), "model": "halo"})], NativeFixture)
        self.assertFalse((overlay / "new").exists())

    def test_private_unsupported_reason_is_redacted_and_preparer_closed(self):
        closed = []
        class Unsupported(NativeFixture):
            def index(self, row):
                return {"id": row["id"], "source_sha256": row["content_sha256"],
                        "supported": False, "reason": "SECRET TEXT", "total_tokens": None}

            def close(self):
                closed.append(True)
        result = self.prepare(5, Unsupported)
        self.assertEqual(closed, [True])
        file = self.manifest(result)["indices"][self.key]["index_file"]
        self.assertNotIn("SECRET", (Path(result["path"]) / file).read_text())
        with self.assertRaisesRegex(data.DataError, "empty buckets"):
            data.load_speed(result["path"], self.spec, BOUNDS, "smoke")


if __name__ == "__main__":
    unittest.main()
