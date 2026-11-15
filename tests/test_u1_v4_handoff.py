"""Preset and documented import/plan checks, without model downloads or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from test_u1_v4_data import NativeFixture
from u1_benchmark.core import ROOT, digest
from u1_benchmark.v4_config import validate_config
from u1_benchmark.v4_data import prepare
from u1_benchmark.v4_planning import cells, schedule, workload
from u1_benchmark.v4_runner import plan


def preset(name):
    return json.loads((ROOT / "configs/v4" / (name + ".json")).read_text())


class HandoffTests(unittest.TestCase):
    def test_sparse_presets_and_comparisons(self):
        encoder = {"B1": [1, 256]}
        decoder = {"B1": [1, 64], "B2": [1, 32], "B3": [1, 16], "B4": [1, 8]}
        for name in ("gliner", "koala", "llama", "laya", "halo", "hive", "open", "shield"):
            cfg = validate_config(preset(name))
            expected = encoder if name in ("gliner", "koala", "llama", "laya") else decoder
            self.assertEqual(cfg["batch_sizes_by_bucket"], expected)
            self.assertEqual(cfg["quality_batch_sizes"], [1])
        for name, base in (("gliner_compile", "gliner"), ("gliner_ort", "gliner"), ("halo_fla", "halo")):
            cfg, reference = preset(name), preset(base)
            self.assertEqual(cfg["baseline"], reference["baseline"])
            self.assertEqual(cfg["batch_sizes_by_bucket"], reference["batch_sizes_by_bucket"])
            self.assertEqual(cfg["data"], reference["data"])

    def test_import_manifest_to_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = preset("koala")
            config_path = root / "model.json"
            config_path.write_text(json.dumps(cfg))
            source = root / "corpus.jsonl"
            source.write_text("".join(json.dumps({
                "id": f"row-{i}", "messages": [{"role": "user", "content": f"fixture {i}"}],
                "language": "en", "source": "fixture", "provenance": {"kind": "test"},
            }) + "\n" for i in range(512)))
            spec = json.loads((ROOT / "examples/u1_v4_speed_import.json").read_text())
            spec["source"] = str(source)
            spec["configs"][0]["config"] = str(config_path)
            result = prepare(spec, root / "data", NativeFixture)
            self.assertEqual(result["status"], "ready")
            cfg["data"] = {key: result[key] for key in ("path", "manifest_sha256")}
            actual = plan(cfg)
            self.assertEqual(actual["status"], "ready")
            self.assertEqual(actual["preflight_workers"], 2)
            self.assertEqual(actual["latency_workers"], 14)
            self.assertFalse(actual["gpu_checked"])
            self.assertEqual(actual["memory_workers"], 0)

    def test_sparse_workload_and_reference_grid(self):
        for name, old_grid, old_jobs, new_jobs, old_batches, new_batches in (
            ("koala", {"B1": [1, 16, 64, 256]}, 28, 14, 2580, 1740),
            ("open", {"B1": [1, 8, 32, 64], "B2": [1, 4, 16, 32],
                      "B3": [1, 4, 8, 16], "B4": [1, 4, 8]},
             105, 56, 9900, 6960),
        ):
            cfg = validate_config(preset(name))
            rows = {bucket: [{"id": f"{bucket}-{i}",
                              "index": {"total_tokens": bounds["min"] + i,
                                        "input_sha256": digest([bucket, i])}}
                             for i in range(400)]
                    for bucket, bounds in cfg["buckets"].items()}
            old = copy.deepcopy(cfg)
            old["batch_sizes_by_bucket"] = old_grid
            before, after = workload(old, rows), workload(cfg, rows)
            self.assertEqual((before["jobs"], after["jobs"]), (old_jobs, new_jobs))
            self.assertEqual((before["batches"], after["batches"]), (old_batches, new_batches))
            search = copy.deepcopy(cfg)
            search.update(mode="search", candidate=copy.deepcopy(cfg["baseline"]))
            with self.assertRaises(ValueError):
                schedule(search, rows, schedule(old, rows))

    def test_nonpreset_sizes_remain_supported(self):
        cfg = preset("koala")
        cfg["batch_sizes_by_bucket"] = {"B1": [2, 3, 24, 128, 257, 400]}
        checked = validate_config(cfg)
        self.assertEqual(cells(checked), [("B1", n) for n in (2, 3, 24, 128, 257, 400)])
