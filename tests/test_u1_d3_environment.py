"""CPU-only checks of the separate d3 profile and its dependency provenance."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.team_d3_resolve import ROOTS, lock_text, target_requirements, validate_report
from u1_benchmark.core import ROOT
from u1_benchmark.environment import expected_lock, extended_gpu, profile, rtx_requirements
from u1_benchmark.orchestrator import setup


class D3EnvironmentTests(unittest.TestCase):
    def test_profile_routes_without_changing_existing_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            setup_dir = Path(folder) / "setup"
            setup_dir.mkdir()
            for name, filename in (("team-d3", "team_d3.lock.json"),
                                   ("team-rtx", "team_rtx.lock.json"),
                                   ("team-t4", "team_environment.lock.json"),
                                   ("team-t4-laya", "team_rtx.lock.json")):
                (setup_dir / "profile.json").write_text(json.dumps({"profile": name}))
                self.assertEqual(profile(folder), name)
                self.assertEqual(expected_lock(folder), json.loads((ROOT / "env" / filename).read_text()))
        with patch("u1_benchmark.environment.setup_rtx", return_value={"status": "test"}) as install:
            self.assertEqual(setup("unused", "team-d3"), {"status": "test"})
            install.assert_called_once_with("unused", "team-d3")

    def test_d3_gpu_policy(self):
        for name in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"):
            self.assertTrue(extended_gpu(name, "team-d3"))
        for name in ("Tesla T4", "NVIDIA A100", "RTX 5090 virtual"):
            self.assertFalse(extended_gpu(name, "team-d3"))

    def test_resolved_lock_and_preserved_artifacts(self):
        lock = json.loads((ROOT / "env/team_d3.lock.json").read_text())
        old = json.loads((ROOT / "env/team_rtx.lock.json").read_text())
        packages = {p["name"]: p for p in lock["packages"]}
        previous = {p["name"]: p for p in old["packages"]}
        self.assertEqual(len(packages), lock["package_count"])
        self.assertEqual(packages["torch"]["version"], "2.11.0+cu130")
        self.assertEqual(packages["transformers"]["version"], "5.17.0")
        self.assertEqual(packages["flash-linear-attention"]["version"], "0.5.2")
        self.assertEqual(packages["fla-core"]["version"], "0.5.2")
        for name in packages.keys() & previous.keys() - {"transformers", "flash-linear-attention", "fla-core"}:
            self.assertEqual((packages[name]["version"], packages[name]["sha256"]),
                             (previous[name]["version"], previous[name]["sha256"]))
        self.assertIn("nvidia-cudnn-cu13", packages)
        self.assertIn("nvidia-cublas", packages)
        self.assertIn("triton", packages)
        self.assertTrue(lock["target_marker_closure_verified"])
        self.assertFalse(lock["source_smoke_passed"])
        self.assertFalse(lock["lock_redeployment_verified"])
        self.assertFalse(lock["gpu_performance_verified"])
        text = (ROOT / "env/team_d3.lock.txt").read_text()
        self.assertEqual(lock_text(lock), text)
        self.assertEqual(rtx_requirements(lock), text)
        report_path = ROOT / "env/team_d3.resolve.json"
        self.assertEqual(hashlib.sha256(report_path.read_bytes()).hexdigest(),
                         lock["source_install_report_sha256"])
        report = json.loads(report_path.read_text())
        validate_report(report, target_requirements(report, ROOTS), lock["computed_download_sha256"])

    def test_missing_hash_and_incompatible_python_rejected(self):
        row = {"metadata": {"name": "demo", "version": "1.0", "requires_python": ">=3.13"},
               "download_info": {"url": "https://files.pythonhosted.org/demo-1.0-py3-none-any.whl",
                                 "archive_info": {"hashes": {}}}}
        with self.assertRaisesRegex(ValueError, "artifact"):
            validate_report({"install": [row]}, ["demo==1.0"])
        row["metadata"]["requires_python"] = ">=3.14"
        with self.assertRaisesRegex(ValueError, "Python 3.13 incompatible"):
            validate_report({"install": [row]}, ["demo==1.0"])

    def test_linux_marker_and_extra_closure(self):
        report = {"install": [{"metadata": {"name": "torch", "requires_dist": [
            'cuda-toolkit[cublas]==13.0.2; platform_system == "Linux"',
            'windows-only; sys_platform == "win32"']}},
            {"metadata": {"name": "cuda-toolkit", "requires_dist": [
                'nvidia-cublas==13.0.0; extra == "cublas"',
                'unused; extra == "other"']}}]}
        required = target_requirements(report, ["torch==2.11.0+cu130"])
        self.assertIn("cuda-toolkit[cublas]==13.0.2", required)
        self.assertIn("nvidia-cublas==13.0.0", required)
        self.assertFalse(any("windows-only" in r or "unused" in r for r in required))
        with self.assertRaisesRegex(ValueError, "unsatisfied"):
            validate_report({"install": []}, ROOTS)

    def test_protectai_requested_shape(self):
        config = json.loads((ROOT / "configs/v4/protectai.json").read_text())
        model = config["baseline"]
        self.assertEqual(model["adapter"], "hf_sequence_classification")
        self.assertEqual(model["preparation"], {"limit": 512, "text_selection": "assistant_else_user",
                                               "add_special_tokens": True})
        self.assertEqual(model["scoring"], {"method": "softmax_argmax", "labels": ["SAFE", "INJECTION"]})
        self.assertEqual(model["sdk_assets"], {})
        self.assertIsNone(model["template"])
        self.assertEqual(config["batch_sizes_by_bucket"], {"B1": [1, 256]})

    def test_accuknox_configs_share_native_baseline(self):
        from u1_benchmark.d3_family import BLOCK_QUESTION, CODE_HASHES, RESPONSE_QUESTION, validate_d3
        from u1_benchmark.v4_config import validate_config
        baseline = json.loads((ROOT / "configs/v4/accuknox.json").read_text())
        spec = baseline["baseline"]
        self.assertEqual(spec["preparation"]["questions"], {
            "user": BLOCK_QUESTION, "user_assistant": RESPONSE_QUESTION, "assistant": RESPONSE_QUESTION})
        self.assertEqual(spec["execution"]["linear_attention"], "native")
        self.assertEqual(spec["execution"]["dtype"], "native_sdk")
        manifest = spec["sdk_assets"]["manifest"]
        self.assertTrue(all(manifest.get(name) == digest for name, digest in CODE_HASHES.items()))
        self.assertFalse(any(name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in manifest))
        self.assertEqual(baseline["batch_sizes_by_bucket"], {
            "B1": [1, 64], "B2": [1, 32], "B3": [1, 16], "B4": [1, 8]})
        for path in (ROOT / "configs/v4").glob("accuknox*.json"):
            config = json.loads(path.read_text())
            self.assertEqual(config["baseline"], spec)
            if config["mode"] == "search":
                config["baseline_ref"] = "runs/accuknox-baseline"
                validate_d3(config["candidate"])
            validate_config(config)
            validate_d3(config["baseline"])


if __name__ == "__main__":
    unittest.main()
