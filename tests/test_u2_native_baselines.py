import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import BadZipFile, ZipFile

from u1_benchmark.core import ROOT, read
from u1_benchmark.v4_config import PROTOCOL_VERSION, RESULT_VERSION, validate_config
from u1_benchmark.v4_data import prepare, verified
from u2_quant import native_baselines as native


class NativeBaselines(TestCase):
    def test_presets_preserve_variants_and_cover_all_cells(self):
        speed = native.speed_configs(environment="/env", speed_data={"path": "/speed"})
        quality = native.quality_configs(environment="/env", quality_data={"path": "/quality"},
                                         gliner_data={"path": "/normalized"})
        cells = 0
        for model in native.MODELS:
            original = read(ROOT / f"configs/v4/{model}.json")
            self.assertEqual(speed[model]["baseline"], original["baseline"])
            self.assertEqual(quality[model]["baseline"], original["baseline"])
            for config in (speed[model], quality[model]):
                self.assertEqual(validate_config(config)["baseline"]["execution"]["runtime"], "eager")
            cells += sum(map(len, speed[model]["batch_sizes_by_bucket"].values()))
            self.assertEqual(quality[model]["quality_batch_sizes"], [1])
            self.assertEqual(quality[model]["quality_data"]["path"],
                             "/normalized" if model == "gliner" else "/quality")
        self.assertEqual(cells, 32)
        self.assertEqual(set(native.VERDICTS["koala"]), {"OK", "H", "H2", "HR", "S", "S3", "SH", "V", "V2"})
        self.assertEqual(native.VERDICTS["koala"]["OK"], "safe")
        self.assertEqual(native.VERDICTS["laya"]["A"], "unsafe")

    def test_smoke_corpus_is_reusable_and_balanced(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [{"id": f"{label}-{i}", "messages": [{"role": "user", "content": "Example."}],
                     "target": label, "target_role": "user", "target_index": 0,
                     "slices": {"technique": "fixture"}}
                    for label in ("safe", "unsafe") for i in range(6)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            prepare({"schema_version": 4, "kind": "quality", "source": str(source),
                     "fields": {k: k for k in rows[0]}, "version": "fixture",
                     "provenance": {}, "include_texts": False}, root / "raw")
            raw = native.data_ref(root / "raw")
            sample = native.smoke_corpus(raw, root / "smoke")
            self.assertEqual(sample, native.smoke_corpus(raw, root / "smoke"))
            selected, _ = verified(sample["path"], sample["manifest_sha256"])
            self.assertEqual([row["target"] for row in selected], ["safe"] * 4 + ["unsafe"] * 4)

    def test_archive_rejects_partial_and_wrong_protocol(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.zip"
            path.write_bytes(b"PK")
            with self.assertRaises(BadZipFile):
                native.archive_report(path, "baseline")
            for status, protocol, valid in (
                ("incomplete", PROTOCOL_VERSION, False),
                ("baseline_complete", "other", False),
                ("baseline_complete", PROTOCOL_VERSION, True),
            ):
                with ZipFile(path, "w") as archive:
                    archive.writestr("summary.json", json.dumps({"status": status, "complete": valid or status == "baseline_complete"}))
                    archive.writestr("resolved_run.json", json.dumps({"protocol_version": protocol, "result_version": RESULT_VERSION}))
                if valid:
                    self.assertEqual(native.archive_report(path, "baseline")[0]["status"], status)
                else:
                    with self.assertRaises(ValueError):
                        native.archive_report(path, "baseline")
