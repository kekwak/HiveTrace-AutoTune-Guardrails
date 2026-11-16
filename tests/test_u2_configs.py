"""Generated U2 configs are valid U1 v4 configs that differ from configs/v4 only where intended."""
import json
import unittest

from u1_benchmark.core import ROOT, sha
from u1_benchmark.v4_config import validate_config, variant_identity
from u2_quant import configs

GRIDS = {"gliner": {"B1": [1, 256]},
         "hive": {"B1": [1, 64], "B2": [1, 32], "B3": [1, 16], "B4": [1, 8]}}
DATA = {"path": "/workspace/u2-data/speed", "manifest_sha256": "a" * 64}
QUALITY = {"path": "/workspace/u2-data/quality", "manifest_sha256": "b" * 64}
QUALITY_GLINER = {"path": "/workspace/u2-data/quality-gliner", "manifest_sha256": "c" * 64}


def every():
    for model in ("gliner", "hive"):
        yield model, "bf16c", "baseline"
        yield model, "bf16c", "quality"
        for mode in ("search", "confirm", "quality"):
            yield model, "fp8", mode


def make(model, variant, mode):
    return configs.build(model, variant, mode, environment="/workspace/u2-env", data=DATA,
                         quality_data=QUALITY_GLINER if model == "gliner" else QUALITY,
                         baseline_ref="/workspace/u2-runs/x" if mode == "search" else None)


class Build(unittest.TestCase):
    def test_all_configs_validate(self):
        for args in every():
            with self.subTest(args=args):
                validate_config(make(*args))

    def test_candidate_identifies_exact_baseline(self):
        for model in ("gliner", "hive"):
            cfg = make(model, "fp8", "confirm")
            self.assertEqual(cfg["comparison"], "quantization")
            self.assertEqual(cfg["candidate"]["quantization"]["baseline_identity"], variant_identity(cfg["baseline"]))
            self.assertEqual(cfg["candidate"]["quantization"]["method"], "fp8")
            self.assertEqual(cfg["baseline"], make(model, "bf16c", "baseline")["baseline"])

    def test_preparation_and_scoring_equal_v4(self):
        for model in ("gliner", "hive"):
            v4 = json.loads((ROOT / f"configs/v4/{model}.json").read_text())["baseline"]
            cfg = make(model, "fp8", "confirm")
            for side in ("baseline", "candidate"):
                for key in ("weights", "tokenizer", "template", "preparation", "scoring"):
                    self.assertEqual(cfg[side][key], v4[key], (model, side, key))
                self.assertEqual(cfg[side]["execution"]["dtype"], "bfloat16")
                self.assertEqual(cfg[side]["execution"]["runtime"], "compile")

    def test_grids_equal_u1_presets(self):
        for model, grid in GRIDS.items():
            self.assertEqual(make(model, "bf16c", "baseline")["batch_sizes_by_bucket"], grid)
            v4 = json.loads((ROOT / f"configs/v4/{model}.json").read_text())
            self.assertEqual(make(model, "bf16c", "baseline")["buckets"], v4["buckets"])

    def test_quality_config_mapping(self):
        cfg = make("hive", "fp8", "quality")
        self.assertEqual(cfg["quality_batch_sizes"], [1])
        self.assertEqual(cfg["label_mapping"]["positive_label"], "unsafe")
        self.assertNotIn("data", cfg)

    def test_adapter_hash_tracks_source(self):
        cfg = make("gliner", "bf16c", "baseline")
        self.assertEqual(cfg["baseline"]["adapter"]["module"], "u2_quant/adapter.py")
        self.assertEqual(cfg["baseline"]["adapter"]["source_sha256"], sha(ROOT / "u2_quant/adapter.py"))
        self.assertEqual(cfg["baseline"]["adapter"]["dependencies"], {"torchao": "0.17.0"})

    def test_main_points_gliner_to_normalized_corpus(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root, patch.object(configs, "ROOT", ROOT), \
                patch.object(configs, "write", lambda path, value: written.__setitem__(path.name, value)):
            written = {}
            configs.main(["--quality-manifest", "q" * 64, "--quality-manifest-gliner", "g" * 64])
        self.assertEqual(written["gliner_fp8_quality.json"]["quality_data"],
                         {"path": "/workspace/u2-data/quality-gliner", "manifest_sha256": "g" * 64})
        self.assertEqual(written["hive_fp8_quality.json"]["quality_data"],
                         {"path": "/workspace/u2-data/quality", "manifest_sha256": "q" * 64})

    def test_fp8_identity_covers_quantize_source(self):
        cfg = make("gliner", "fp8", "confirm")
        self.assertEqual(cfg["candidate"]["quantization"]["quantize_sha256"], sha(ROOT / "u2_quant/quantize.py"))

    def test_speed_import_references_both_models(self):
        spec = configs.speed_import("/workspace/u2-data/role-corpus/corpus.jsonl", {
            "gliner": "configs/u2/gliner_bf16c_baseline.json", "hive": "configs/u2/hive_bf16c_baseline.json"})
        self.assertEqual(spec["kind"], "speed")
        self.assertEqual([c["name"] for c in spec["configs"]], ["u2-gliner", "u2-hive"])
        self.assertEqual(spec["provenance"]["revision"], "7324b8823ded971c05076e0c7f4ace5d9b867863")


if __name__ == "__main__":
    unittest.main()
