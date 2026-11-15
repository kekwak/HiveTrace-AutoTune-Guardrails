"""U2 profile team-rtx-quant: the U1 RTX lock plus exactly torchao; other profiles unchanged."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from u1_benchmark import environment
from u1_benchmark.core import ROOT

BASE = json.loads((ROOT / "env/team_rtx.lock.json").read_text())
QUANT_PATH = ROOT / "env/team_rtx_quant.lock.json"


def saved_profile(directory, name):
    (Path(directory) / "setup").mkdir(parents=True)
    (Path(directory) / "setup/profile.json").write_text(json.dumps({"profile": name}))


class QuantLock(unittest.TestCase):
    def test_quant_lock_is_base_plus_torchao(self):
        quant = json.loads(QUANT_PATH.read_text())
        extra = [p for p in quant["packages"] if p["name"] == "torchao"]
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]["version"], "0.17.0")
        rest = [p for p in quant["packages"] if p["name"] != "torchao"]
        self.assertEqual(rest, BASE["packages"])
        self.assertEqual(quant["package_count"], BASE["package_count"] + 1)
        self.assertEqual(quant["profile"], "team-rtx-quant")

    def test_quant_lock_text_matches_requirements(self):
        quant = json.loads(QUANT_PATH.read_text())
        self.assertEqual(environment.rtx_requirements(quant),
                         (ROOT / "env/team_rtx_quant.lock.txt").read_text())


class QuantProfile(unittest.TestCase):
    def test_profile_accepts_quant_and_keeps_others(self):
        with tempfile.TemporaryDirectory() as quant, tempfile.TemporaryDirectory() as rtx:
            saved_profile(quant, "team-rtx-quant")
            saved_profile(rtx, "team-rtx")
            self.assertEqual(environment.profile(quant), "team-rtx-quant")
            self.assertEqual(environment.expected_lock(quant), json.loads(QUANT_PATH.read_text()))
            self.assertEqual(environment.expected_lock(rtx), BASE)

    def test_quant_profile_gpu_gate(self):
        self.assertTrue(environment.extended_gpu("NVIDIA GeForce RTX 5090", "team-rtx-quant"))
        self.assertFalse(environment.extended_gpu("Tesla T4", "team-rtx-quant"))

    def test_setup_rtx_compares_profile_specific_text(self):
        calls = []
        with tempfile.TemporaryDirectory() as root, \
                patch.object(environment, "precheck_rtx", return_value={}), \
                patch("u1_benchmark.process.invoke", side_effect=lambda *a, **k: calls.append(a)), \
                patch("u1_benchmark.process.finish_archive", return_value=True):
            state = environment.setup_rtx(Path(root) / "env", "team-rtx-quant")
            written = (Path(root) / "env/setup/requirements.txt").read_text()
        self.assertEqual(state["status"], "passed", state)
        self.assertEqual(written, (ROOT / "env/team_rtx_quant.lock.txt").read_text())
        self.assertIn("torchao", written)


if __name__ == "__main__":
    unittest.main()
