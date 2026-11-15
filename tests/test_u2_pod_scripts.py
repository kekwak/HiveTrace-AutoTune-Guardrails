"""Pod scripts: syntax, staged order via --dry-run, and isolation from colleagues' paths."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest

from u1_benchmark.core import ROOT

SETUP, RUN = ROOT / "scripts/u2_pod_setup.sh", ROOT / "scripts/u2_pod_run.sh"
STAGES = ("preflight", "baseline", "quality-bf16c", "search-fp8", "quality-fp8", "confirm", "pack")


def dry(stage, model="gliner"):
    return subprocess.run(["bash", str(RUN), stage, model, "--dry-run"], capture_output=True, text=True,
                          check=True, cwd=ROOT).stdout


class Scripts(unittest.TestCase):
    def test_scripts_parse(self):
        for path in (SETUP, RUN):
            self.assertEqual(subprocess.run(["bash", "-n", str(path)]).returncode, 0, path)

    def test_pod_run_dry_run_order(self):
        expected = {"baseline": ("run", "configs/u2/gliner_bf16c_baseline.json"),
                    "quality-bf16c": ("quality", "configs/u2/gliner_bf16c_quality.json"),
                    "search-fp8": ("run", "configs/u2/gliner_fp8_search.json"),
                    "quality-fp8": ("quality", "configs/u2/gliner_fp8_quality.json"),
                    "confirm": ("run", "configs/u2/gliner_fp8_confirm.json")}
        for stage, (command, config) in expected.items():
            with self.subTest(stage=stage):
                out = dry(stage)
                self.assertIn(f"tools/benchmark.py {command}", out)
                self.assertIn(config, out)
        self.assertIn("--baseline-ref /workspace/u2-runs/gliner-baseline", dry("search-fp8"))
        preflight = dry("preflight")
        self.assertIn("u2_quant.preflight", preflight)
        self.assertNotRegex(preflight, r"tools/benchmark.py (run|quality)")
        self.assertRegex(dry("pack"), r"tar czf /workspace/u2-runs/gliner-results-[0-9-]+\.tgz -C /workspace/u2-runs --exclude=\*\.tgz gliner-\*")

    def test_unknown_stage_or_model_fails(self):
        for args in (["nope", "gliner"], ["baseline", "koala"]):
            result = subprocess.run(["bash", str(RUN), *args, "--dry-run"], capture_output=True, cwd=ROOT)
            self.assertNotEqual(result.returncode, 0, args)

    def test_scripts_touch_only_u2_paths(self):
        for path in (SETUP, RUN):
            text = path.read_text()
            for forbidden in ("/workspace/u1-env", "u1-shared", "/workspace/hf", "runpodctl", "/workspace/guardrails"):
                self.assertNotIn(forbidden, text, (path, forbidden))
            for target in re.findall(r"/workspace/[\w.-]+", text):
                self.assertTrue(target.startswith("/workspace/u2-"), (path, target))

    def test_setup_rebuilds_unready_steps(self):
        text = SETUP.read_text()
        self.assertIn('ready() {', text)
        for target in ('"$DATA/quality"', '"$DATA/speed"'):
            self.assertIn(f"ready {target} ||", text)
        self.assertIn('rm -rf "$DATA/quality-src" "$DATA/quality"', text)
        self.assertIn('rm -rf "$DATA/speed"', text)
        self.assertIn('rm -rf "$ENV"', text)

    def test_pack_archives_only_this_model(self):
        with tempfile.TemporaryDirectory() as runs:
            for name in ("gliner-baseline", "gliner-quality-bf16c", "hive-baseline"):
                (Path(runs) / name).mkdir()
                (Path(runs) / name / "result.json").write_text("{}")
            subprocess.run(["bash", str(RUN), "pack", "gliner"], check=True, cwd=ROOT,
                           env={**os.environ, "U2_RUNS": runs}, capture_output=True)
            archives = list(Path(runs).glob("gliner-results-*.tgz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0]) as tar:
                tops = {name.split("/")[0] for name in tar.getnames()}
        self.assertEqual(tops, {"gliner-baseline", "gliner-quality-bf16c"})

    def test_preflight_refuses_without_cuda(self):
        result = subprocess.run([sys.executable, "-m", "u2_quant.preflight"], capture_output=True, text=True, cwd=ROOT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
