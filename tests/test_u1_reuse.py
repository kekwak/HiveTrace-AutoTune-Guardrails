"""Reuse lifecycle and v4 reference regression tests; no GPU or network."""
from copy import deepcopy
import importlib
import io
import itertools
import os
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_u1_v4_execution as execution
import test_u1_v4_report as historical
from u1_benchmark.core import digest, read, write
from u1_benchmark import v4_config as config, v4_planning as planning, v4_runner as runner
from u1_benchmark.v4_evidence import read_reference, check_reference, ReferenceMismatch
from u1_benchmark.v4_memory import Sampler
from u1_benchmark.v4_lifecycle import StageSampler
from u1_benchmark.v4_report import build, table
from u1_benchmark.statistics import assess_reference, unpaired


def reuse_format(archive):
    """Create current-format synthetic evidence from the historical fixture."""
    records = [read(path / "result.json") for path in archive.workers]
    plans = planning.schedule(archive.cfg, archive.data)
    groups = planning.execution_groups(archive.cfg, plans)
    archive.resolved.update(result_version=config.RESULT_VERSION, protocol_version=config.PROTOCOL_VERSION,
                            cache_policy=config.CACHE_POLICY, schedule=plans,
                            execution_groups=[planning.group_identity(g) for g in groups])
    shutil.rmtree(archive.root / "jobs")
    archive.workers, archive.state["stages"] = [], []
    for index, group in enumerate(groups):
        for order, stage in enumerate(group["stages"]):
            raw = next(deepcopy(value) for value in records
                       if all(value["job"].get(k) == v for k, v in stage.items()))
            job = {**raw.pop("job"), **stage, "lifecycle_id": f"worker-{index:05d}", "stage_order": order}
            raw.update(result_version=5, lifecycle_id=job["lifecycle_id"], stage_order=order,
                       process_id=index + 1, cache_policy=config.CACHE_POLICY, load_owner=order == 0)
            if order:
                raw["setup_seconds"] = 0
                raw.pop("torch_setup", None)
            archive.add(job, raw)
    archive.refreeze()


class ReferenceStatisticsTests(unittest.TestCase):
    def test_independent_block_bootstrap_is_reproducible_not_paired(self):
        a = [[8.] * 30, [10.] * 30, [12.] * 30]
        b = [[x * .7 for x in block] for block in a]
        actual = unpaired(a, b)
        self.assertEqual(actual, unpaired(a, b))
        self.assertLess(actual["ratio_interval_95"][0], .7)
        self.assertGreater(actual["ratio_interval_95"][1], .7)

    def test_threshold_noise_and_slowdown(self):
        a = [[10.] * 30] * 3
        aa = [(a[0], a[0])] * 2
        for value, threshold, status in ((7., .2, "pass"), (8., .2, "below_target"),
                                         (9., .1, "below_target"), (12., .2, "below_target")):
            with self.subTest(value=value):
                result = assess_reference(a, [[value] * 30] * 3, aa, threshold)
                self.assertEqual(result["status"], status)
                self.assertFalse(result["between_session_drift_checked"])
        noisy = assess_reference(a, [[6.] * 30, [9.] * 30, [7.] * 30], aa)
        self.assertEqual(noisy["status"], "inconclusive")
        self.assertIn("effect_not_above_observed_drift", noisy["reasons"])

    def test_invalid_or_empty_series_rejected(self):
        for bad in ([], [[]], [[0]], [[float("nan")]], [[float("inf")]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                unpaired([[10.]], bad)


class ReferenceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_v4_baseline_is_reused_without_editing_and_without_monitor_gate(self):
        baseline = historical.Archive(self.root / "base", monitoring=False)
        baseline.state["status"] = "baseline_complete"
        baseline.refreeze()
        old = build(baseline.root)
        before = {str(p.relative_to(baseline.root)): p.read_bytes() for p in baseline.root.rglob("*") if p.is_file()}
        reference = read_reference(baseline.root)
        candidate = historical.Archive(self.root / "candidate", mode="search", monitoring=False)
        reuse_format(candidate)
        candidate.resolved["source_inventory"] = {"new/implementation.py": digest("new source")}
        candidate.refreeze()
        compatibility = check_reference(candidate.cfg, reference, candidate.resolved["source_inventory"],
                                        candidate.resolved["input_indices"], candidate.resolved["environment"])
        self.assertTrue(compatibility["lifecycle_differs"])
        candidate.reference(baseline)
        value = build(candidate.root)
        self.assertEqual(value["status"], "reference_pass", value["errors"])
        self.assertEqual(value["comparison_design"], "unpaired_reference")
        self.assertFalse(value["monitoring_is_acceptance_gate"])
        self.assertEqual(value["cells"]["custom/4"]["p95"]["ratio"], .7)
        self.assertEqual(build(baseline.root, save=False), old)
        self.assertEqual(before, {str(p.relative_to(baseline.root)): p.read_bytes()
                                  for p in baseline.root.rglob("*") if p.is_file()})
        self.assertIn("lifecycle", (candidate.root / "summary.md").read_text())

    def test_new_confirm_monitoring_is_informational_and_design_is_paired(self):
        archive = historical.Archive(self.root, mode="confirm", monitoring=False)
        reuse_format(archive)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "confirm_pass", result["errors"])
        self.assertEqual(result["comparison_design"], "paired")
        self.assertTrue(result["all_cells_faster"])
        self.assertEqual(result["model_loads"], 4)

    def test_legacy_confirm_status_is_not_upgraded_retroactively(self):
        historical.Archive(self.root, mode="confirm", monitoring=False)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("monitoring_validation_not_passed", result["errors"])

    def test_historical_monitoring_is_not_invalidated_or_reapplied(self):
        archive = historical.Archive(self.root, mode="confirm", monitoring=True)
        reuse_format(archive)
        archive.resolved["source_inventory"] = {"changed.py": digest("new sampler")}
        archive.refreeze()
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "confirm_pass")
        self.assertEqual(result["monitoring_status"], "historical_only")
        self.assertEqual(result["monitoring_evidence"]["recorded_status"], "passed")
        self.assertFalse(result["monitoring_evidence"]["current_implementation_validated"])

    def test_lifecycle_tampering_cannot_pass(self):
        archive = historical.Archive(self.root, mode="confirm", monitoring=False)
        reuse_format(archive)
        path = archive.worker()
        archive.mutate(path, lambda value: value.update(lifecycle_id="other-worker"))
        result = build(self.root, save=False)
        self.assertNotEqual(result["status"], "confirm_pass")
        self.assertIn("worker_lifecycle_mismatch", result["errors"])

    def test_missing_reference_has_safe_actionable_diagnostic(self):
        with self.assertRaises(ReferenceMismatch) as error:
            read_reference(self.root / "SECRET-missing.zip")
        self.assertEqual(error.exception.code, "reference_invalid")
        self.assertNotIn("SECRET", str(error.exception))

    def test_table_keeps_paired_and_reference_designs_separate(self):
        baseline = historical.Archive(self.root / "base", monitoring=False)
        baseline.state["status"] = "baseline_complete"
        baseline.refreeze()
        candidate = historical.Archive(self.root / "search", mode="search", monitoring=False)
        reuse_format(candidate)
        candidate.reference(baseline)
        confirm = historical.Archive(self.root / "confirm", mode="confirm", monitoring=False)
        reuse_format(confirm)
        result = table([candidate.root, confirm.root], self.root / "table.json")
        self.assertEqual(result["rows"], 2)
        self.assertEqual({g["identity"]["comparison_design"] for g in result["groups"].values()},
                         {"paired", "unpaired_reference"})


class ReuseExecutionTests(execution.Fixture):
    run_real = execution.RealReportRunnerTests.run_real

    def setUp(self):
        super().setUp()
        self.report = importlib.import_module("u1_benchmark.v4_report")
        clock = itertools.count(1_000_000, 1_000_000)
        self.enterContext(patch.object(execution.worker.time, "perf_counter_ns", side_effect=lambda: next(clock)))

    def test_baseline_main_precedes_aa_and_load_is_recorded_once(self):
        target, summary = self.run_real(self.cfg("baseline", pilot=False), "baseline", execution.input_rows(400))
        self.assertEqual(summary["status"], "baseline_complete", summary["errors"])
        self.assertEqual(len(self.adapters), 1)
        self.assertEqual([job["phase"] for job in self.launched if job["stage"] == "latency"],
                         ["baseline"] * 3 + ["aa"] * 4)
        results = [read(p) for p in sorted((target / "jobs").glob("*/result.json"))]
        self.assertEqual(sum(r["load_owner"] for r in results), 1)
        self.assertTrue(all(r["setup_seconds"] == 0 for r in results[1:]))
        self.assertEqual(len(self.samplers), 1)
        self.assertEqual(self.samplers[0].events.count("start"), 1)
        self.assertEqual(self.samplers[0].events.count("stop"), 1)
        self.assertEqual(summary["model_loads"], 1)

    def test_quality_rng_restored_and_model_reused_for_all_sizes(self):
        original = random.getstate()
        self.addCleanup(random.setstate, original)
        random.seed(123)
        def hook(adapter):
            base = adapter.request_batch
            def request(group):
                base(group)
                return [{"label": "safe" if random.random() < .5 else "unsafe"} for _ in group]
            adapter.request_batch = request
        self.adapter_hook = hook
        cfg = self.cfg("quality")
        cfg["quality_batch_sizes"] = [1, 3, 2]
        target, summary = self.run_real(cfg, "quality", execution.input_rows(5))
        self.assertEqual(summary["status"], "quality_complete", summary["errors"])
        self.assertEqual(len(self.adapters), 1)
        predictions = [p.read_text() for p in sorted((target / "jobs").glob("*/predictions.jsonl"))]
        self.assertEqual(predictions, [predictions[0]] * 3)
        self.assertEqual(summary["quality_protocol_version"], 2)

    def test_quality_failure_does_not_reload_or_continue_same_worker(self):
        self.adapter_hook = lambda adapter: setattr(adapter, "fail_at", 7)
        cfg = self.cfg("quality")
        cfg["quality_batch_sizes"] = [1, 3, 4]
        target, summary = self.run_real(cfg, "failed-quality", execution.input_rows(5))
        statuses = [read(p)["status"] for p in sorted((target / "jobs").glob("*/result.json"))]
        self.assertEqual(statuses, ["passed", "failed", "not_run"])
        self.assertEqual(len(self.adapters), 1)
        self.assertTrue(self.adapters[0].closed)
        self.assertFalse(summary["complete"])

    def test_cache_environment_is_private_and_restored(self):
        keys = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH")
        before = {key: os.environ.get(key) for key in keys}
        seen = []
        self.adapter_hook = lambda adapter: seen.append({k: os.environ.get(k) for k in keys})
        self.run_real(self.cfg("smoke", pilot=False), "cache-smoke", execution.input_rows())
        self.assertEqual({key: os.environ.get(key) for key in keys}, before)
        self.assertTrue(all(".private" in value for value in seen[0].values()))

    def test_manifest_tampering_rejected_before_model_load(self):
        def launch(args, log, timeout):
            if "_worker_v4" in [str(a) for a in args]:
                path = Path(args[-1])
                value = read(path)
                value["main_batches"] = []
                write(path, value)
            return self.launcher(args, log, timeout)
        cfg = self.cfg()
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg)):
            runner.run(cfg, self.root / "tampered", launcher=launch)
        self.assertEqual(self.adapters, [])


class MemoryWindowTests(unittest.TestCase):
    def test_checkpoint_after_stop_preserves_final_stage_file(self):
        for value, expected in (("120", "measured"), ("N/A", "unavailable")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                raw = Sampler("GPU-test")
                stage = StageSampler(SimpleNamespace(sampler=raw, adapter=None), "0", "latency")
                path = Path(directory) / "memory.json"
                with patch("u1_benchmark.v4_memory.time.monotonic", return_value=1):
                    stage.start()
                raw._receive(f"GPU-test,{raw.pid},{value}", 1.05)
                with patch("u1_benchmark.v4_memory.time.monotonic", return_value=1.1):
                    stage.inference()
                raw._receive(f"GPU-test,{raw.pid},{value}", 1.15)
                partial = stage.checkpoint(path)
                self.assertTrue(partial["checkpoint"])
                final = stage.stop(path)
                self.assertEqual(final["status"], expected)
                original = path.read_bytes()
                raw.begin_window("1", "setup")
                for snapshot in (stage.checkpoint(path), stage.stop(path)):
                    self.assertEqual(snapshot, final)
                    self.assertNotIn("checkpoint", snapshot)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(read(path), snapshot)
                self.assertIsNone(raw.stage_windows[-1]["end"])

    def test_windows_are_disjoint_and_do_not_inherit_previous_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler = Sampler("GPU-test")
            sampler._started = True
            with patch("u1_benchmark.v4_memory.time.monotonic", return_value=1):
                sampler.begin_window("0", "setup")
            sampler._receive(f"GPU-test,{sampler.pid},100", 1.5)
            with patch("u1_benchmark.v4_memory.time.monotonic", return_value=2):
                sampler.begin_window("0", "latency", "inference")
            sampler._receive(f"GPU-test,{sampler.pid},120", 2.5)
            with patch("u1_benchmark.v4_memory.time.monotonic", return_value=3):
                sampler.end_window()
            a = sampler.stage_snapshot(Path(directory) / "a.json", "0", complete=True)
            with patch("u1_benchmark.v4_memory.time.monotonic", return_value=4):
                sampler.begin_window("1", "latency", "inference")
            sampler._receive(f"GPU-test,{sampler.pid},90", 4.5)
            sampler._receive(f"GPU-test,{sampler.pid},N/A", 4.6)
            sampler.end_window()
            b = sampler.stage_snapshot(Path(directory) / "b.json", "1", complete=True)
            self.assertEqual(a["windows"]["inference"]["sampled_gpu_peak_mib"], 120)
            self.assertEqual(b["windows"]["inference"]["sampled_gpu_peak_mib"], 90)
            self.assertEqual(b["windows"]["inference"]["status"], "partial")
            self.assertEqual(len(a["samples"]), 2)
            self.assertEqual(a["stage_windows"][0]["end"], a["stage_windows"][1]["start"])


class ConsoleTests(unittest.TestCase):
    def test_explanations_do_not_print_arbitrary_values(self):
        from tools.benchmark import explain
        stream = io.StringIO()
        with patch("sys.stderr", stream):
            explain({"diagnostic_code": "reference_gpu_differs", "error": "SECRET", "model_loads": "SECRET",
                     "reference_compatibility": {"status": "compatible", "reference_result_version": "SECRET"}})
            explain({"model_loads": 1, "latency_stages": "SECRET"})
        self.assertNotIn("SECRET", stream.getvalue())
        self.assertIn("GPU", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
