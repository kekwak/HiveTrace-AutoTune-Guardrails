"""CPU orchestration fixtures; timings and GPU passports here are not evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.u1_external_adapter import Fixture
from test_u1_roles import EXTERNAL, fixture_data
from u1_benchmark import batch_runner, batching
from u1_benchmark.contracts import model_contract, parse_config
from u1_benchmark.core import read, sha, write
from u1_benchmark.datasets import load_speed


class BatchFixture(Fixture):
    def __init__(self, contract, spec, model="gliner"):
        super().__init__(contract, spec, model)
        self.metadata["batch_policy"] = {"padding": "longest", "padding_side": "right", "pad_token_id": 0}
        self.calls = []

    def request_batch(self, inputs):
        self.calls.append(("request", len(inputs)))
        return [self.request(value) for value in inputs]

    def verify_batch(self, inputs, indices):
        self.calls.append(("verify", len(inputs)))
        return [self.verify(value, index) for value, index in zip(inputs, indices, strict=True)]

    def prepare_batch_backend(self, batches):
        self.calls.append(("prepare", len(batches)))

    def prove_batch_backend(self, inputs):
        self.calls.append(("proof", len(inputs)))

    def reject_batch_over_limit(self, inputs):
        self.calls.append(("reject", len(inputs)))
        if not any(self.index(value)["total_tokens"] > self.limit for value in inputs):
            raise AssertionError("Missing over-limit fixture")


class FakeSampler:
    def __init__(self, *args):
        pass

    def start(self):
        pass

    def inference(self):
        pass

    def stop(self, path):
        value = {"windows": {w: {"status": "unavailable"} for w in ("setup", "inference")}}
        write(path, value)
        return value


def fake_probe(environment, directory):
    write(Path(directory) / "runtime.json", {k: "CPU-fixture" for k in (
        "python", "torch", "cuda", "cudnn", "gpu", "seed", "cpu_threads", "tf32", "hub_kernels")})
    write(Path(directory) / "packages.json", {})


class BatchExecution(unittest.TestCase):
    def test_decoder_boundary_uses_bucket_not_global_limit(self):
        from u1_benchmark.batch_worker import cell_boundary
        adapter = BatchFixture(model_contract("halo", 3), {})
        for bucket, expected in (("B1", 512), ("B2", 1024), ("B3", 2048), ("B4", 4096)):
            edge, above = cell_boundary(adapter, bucket)
            self.assertEqual(adapter.index(edge)["total_tokens"], expected)
            self.assertEqual(adapter.index(above)["total_tokens"], 4097)
        self.assertFalse(adapter.calls)

    def config(self, root, mode="pilot", sizes=None):
        data = fixture_data(root)
        value = {"schema_version": 3, "model": "gliner", "mode": mode, "seed": 17,
                 "environment": str(root / "env"), "comparison": "engineering",
                 "baseline": {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}},
                 "data": {k: data[k] for k in ("path", "manifest_sha256")},
                 "batch_sizes_by_bucket": {"B1": sizes or [1, 8]}}
        return parse_config(value)

    def test_matrix_diagnostics_once_and_pilot_has_no_memory_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root)
            data, _ = load_speed(cfg["data"]["path"], "gliner", "pilot")
            calls = []

            def launch(phase, variant, directory, measurement):
                calls.append((phase, variant, str(directory), measurement["batch_size"]))
                write(directory / "result.json", {"status": "passed"})

            result = batch_runner.execute_matrix(cfg, data, root / "out", launch)
            self.assertTrue(all(v["status"] == "completed" for v in result.values()))
            self.assertEqual([c[0] for c in calls].count("batch_preflight"), 2)
            self.assertEqual([c[0] for c in calls].count("batch_latency"), 2)
            self.assertEqual(len({c[2] for c in calls}), len(calls))
            self.assertFalse(any(c[0] in ("batch_memory", "batch_profile") for c in calls))

    def test_failed_cell_keeps_other_sizes_and_timeout_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root, sizes=[1, 8, 16])
            data, _ = load_speed(cfg["data"]["path"], "gliner", "pilot")
            calls = []

            def launch(phase, variant, directory, measurement):
                calls.append((phase, measurement["batch_size"]))
                if measurement["batch_size"] == 8:
                    write(directory / "result.json", {"status": "oom", "error_type": "OutOfMemoryError"})
                    raise RuntimeError("fixture")
                write(directory / "result.json", {"status": "passed"})

            result = batch_runner.execute_matrix(cfg, data, root / "out", launch)
            self.assertEqual(result["B1/8"]["status"], "oom")
            self.assertEqual(result["B1/16"]["status"], "completed")
            self.assertNotIn(("batch_latency", 8), calls)
            with self.assertRaises(TimeoutError):
                batch_runner.execute_matrix(cfg, data, root / "timeout", lambda *args: (_ for _ in ()).throw(TimeoutError()))
            self.assertEqual(read(root / "timeout/matrix_status.json")["B1/1"]["status"], "timeout")

    def test_actual_worker_uses_batch_and_reuses_preflight_without_replaying_diagnostics(self):
        import torch
        from u1_benchmark.batch_worker import execute
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root, sizes=[8])
            data, _ = load_speed(cfg["data"]["path"], "gliner", "pilot")
            base = {"schema_version": 3, "batch_protocol_version": 1, "config": cfg,
                    "variant": "baseline", "dataset": cfg["data"]["path"], "private": True,
                    "inputs_sha256": "fixture", "sources_sha256": "fixture"}
            adapter = BatchFixture(model_contract("gliner", 3), cfg["baseline"])
            pf = root / "pf"
            job = {**base, "phase": "batch_preflight", "directory": str(pf), "measurement": {
                "bucket": "B1", "batch_size": 8, "batches": batching.smoke_plans(data["B1"], 8)}}
            write(root / "job.json", job)
            with patch("u1_benchmark.environment.probe", side_effect=fake_probe), \
                    patch("u1_benchmark.worker.telemetry", return_value={"uuid": "fixture", "driver": "fixture"}), \
                    patch("u1_benchmark.adapters.load", return_value=adapter), \
                    patch("u1_benchmark.batch_adapters.wrap", side_effect=lambda a: a), \
                    patch("u1_benchmark.memory.Sampler", FakeSampler), patch.object(torch.cuda, "synchronize"):
                result = execute(root / "job.json")
                self.assertEqual(result["status"], "passed", result)
                self.assertEqual(len(result["timings"]), 3)
                self.assertTrue(all(len(r["scores"]) == 8 for r in result["timings"]))
                adapter.calls.clear()
                measurement = batching.schedule(cfg, data)[0]
                job = {**base, "phase": "batch_latency", "directory": str(root / "latency"),
                       "measurement": measurement, "preflight": {"path": str(pf), "sha256": sha(pf / "result.json")}}
                write(root / "job.json", job)
                result = execute(root / "job.json")
                self.assertEqual(result["status"], "passed", result)
                self.assertEqual(len(result["rows"]), 12)
                self.assertEqual(adapter.calls.count(("request", 8)), 32)
                self.assertFalse(any(kind in ("verify", "proof", "reject") for kind, _ in adapter.calls))
                self.assertNotIn("Assistant", (root / "latency/result.json").read_text())
                job["preflight"]["sha256"] = "wrong"
                job["directory"] = str(root / "stale")
                write(root / "job.json", job)
                result = execute(root / "job.json")
                self.assertEqual(result["status"], "failed")

    def test_existing_configs_remain_singleton_and_new_presets_match_grid(self):
        from u1_benchmark.core import ROOT
        from u1_benchmark.contracts import MODEL_SPECS
        for path in (ROOT / "configs/benchmark-v3").glob("*.json"):
            self.assertNotIn("batch_sizes_by_bucket", read(path))
        configs = list((ROOT / "configs/benchmark-batch").glob("*.json"))
        self.assertEqual(len(configs), 11)
        self.assertEqual({read(p)["model"] for p in configs}, set(MODEL_SPECS))
        for path in configs:
            cfg = read(path)
            self.assertEqual(cfg["batch_sizes_by_bucket"], batching.default_grid(cfg["model"]))
        for path in (ROOT / "configs/smoke-batch").glob("*.json"):
            cfg = read(path)
            self.assertEqual(cfg["mode"], "smoke")
            self.assertEqual(cfg["batch_sizes_by_bucket"], batching.default_grid(cfg["model"]))


if __name__ == "__main__":
    unittest.main()
