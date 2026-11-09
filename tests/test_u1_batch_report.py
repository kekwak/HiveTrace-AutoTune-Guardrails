"""Synthetic archived workers exercise reports without model or GPU imports."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from u1_benchmark import batch_report
from u1_benchmark.batching import cells, memory_plans, schedule, smoke_plans
from u1_benchmark.contracts import BUCKETS, model_contract
from u1_benchmark.core import digest, read, sha, write


def answer(label="safe"):
    return {"label": label, "scores": {"safe": .9 if label == "safe" else .1}, "score_type": "independent_probabilities"}


class Archive:
    def __init__(self, root, mode="screening", candidate=True, comparison="engineering", grid=None, model="gliner", fixture=False):
        self.root = root
        self.cfg = {"schema_version": 3, "model": model, "seed": 17, "mode": mode,
                    "comparison": comparison, "baseline": {"backend": "eager", "dtype": "float32"},
                    "candidate": {"backend": "compile", "dtype": "float32"} if candidate else None,
                    "batch_sizes_by_bucket": grid or {"B1": [1, 8]}}
        self.variants = ["baseline"] + (["candidate"] if candidate else [])
        self.fixture = fixture
        self.data = {b: [{"id": f"{b}-{i:03d}", "bucket": b,
                          "index": {"total_tokens": BUCKETS[b]["min"] + 19 + i, "input_sha256": digest([b, i])}}
                         for i in range(400)] for b in self.cfg["batch_sizes_by_bucket"]}
        write(root / "config.json", self.cfg)
        write(root / "status.json", {"status": "completed"})
        write(root / "inputs.json", [r for rows in self.data.values() for r in rows])
        write(root / "sources" / "worker.json", {"synthetic_test_artifact": True})
        write(root / "source_files.json", {"sources/worker.json": sha(root / "sources/worker.json")})
        write(root / "dataset_identity.json", {"manifest_sha256": digest(self.data), "model": model})
        write(root / "lock.json", {"test": True})
        self.jobs = schedule(self.cfg, self.data)
        write(root / "schedule.json", self.jobs)
        states = {}
        for bucket, size in cells(self.cfg):
            key = f"{bucket}/{size}"
            states[key] = {"status": "preflight_passed" if mode == "smoke" else "completed", "bucket": bucket, "batch_size": size}
            base = root / "batches" / bucket / str(size)
            plans = smoke_plans(self.data[bucket], size, 17)
            measure = {"bucket": bucket, "batch_size": size, "batches": plans}
            for variant in self.variants:
                value = self.worker(base / "preflight" / variant, variant, "batch_preflight", measure)
                # Deliberately opposite single-preflight verdict: never join it to measured responses.
                value.update(verified_batches=self.rows(plans, label="unsafe"),
                             timings=self.rows(plans, label="unsafe"), aba_passed=True,
                             boundary={"tokens": BUCKETS[bucket]["max"], "model_limit": value["contract"]["limit"],
                                       "over_limit_rejected": True}, memory=self.memory())
                write(base / "preflight" / variant / "result.json", value)
                if mode == "screening":
                    mem = {**measure, "batches": memory_plans(self.data[bucket], size, 17)}
                    for repeat in range(3):
                        path = base / "memory" / f"{variant}_{repeat}"
                        value = self.worker(path, variant, "batch_memory", mem)
                        value.update(memory=self.memory(), torch_setup={"allocated": 10, "reserved": 20},
                                     torch_inference={"allocated": 11, "reserved": 21})
                        write(path / "result.json", value)
                        write(path / "memory.json", value["memory"])
                    path = base / "profiles" / variant
                    self.worker(path, variant, "batch_profile", {**measure, "batches": plans[:2]})
                    for label in ("short", "long"):
                        with gzip.open(path / (label + ".trace.json.gz"), "wt") as f:
                            json.dump({"traceEvents": [{"cat": "kernel", "name": "synthetic"}]}, f)
        write(root / "matrix_status.json", states)
        for i, job in enumerate(self.jobs):
            path = root / "latency" / f"{i:03d}"
            value = self.worker(path, job["variant"], "batch_latency", job)
            value["rows"] = self.rows(job["batches"], latency=7 if job["variant"] == "candidate" else 10)
            self.save_latency(path, value)

    @staticmethod
    def memory():
        return {"windows": {w: {"status": "measured", "samples": 1, "valid_gpu_samples": 1,
                                "sampled_gpu_peak_mib": 123} for w in ("setup", "inference")}}

    @staticmethod
    def rows(plans, label="safe", latency=10):
        return [{"batch": p, "scores": [answer(label) for _ in p["ids"]], "latency_ms": latency,
                 "started_unix": 100 + i, "finished_unix": 100 + i + latency / 1000}
                for i, p in enumerate(plans)]

    def worker(self, path, variant, phase, measurement):
        value = {"schema_version": 3, "batch_protocol_version": 1, "status": "passed", "phase": phase,
                 "variant": variant, "job": measurement, "bucket": measurement["bucket"], "batch_size": measurement["batch_size"],
                 "config_sha256": digest(self.cfg), "inputs_sha256": sha(self.root / "inputs.json"),
                 "sources_sha256": sha(self.root / "source_files.json"), "contract": model_contract(self.cfg["model"], 3),
                 "adapter_spec": self.cfg[variant], "metadata": {"batch_policy": {
                     "padding": "longest", "padding_side": "right", "truncation": False, "sort_by_length": False},
                     "batch_backend_proof": {"kernels": ["synthetic"]}},
                 "prepared_shapes": sorted([list(s) for s in {(measurement["batch_size"], p["padded_length"]) for p in measurement["batches"]}]),
                 "prepared_compile_count": 0, "final_compile_count": 0, "size": {"bytes": 100}, "pid": 123,
                 "load_seconds": .1, "setup_seconds": .2, "backend_preparation_seconds": .05, "warmup_seconds": .1}
        if self.fixture:
            value["metadata"]["test_fixture"] = True
        write(path / "result.json", value)
        write(path / "runtime.json", dict(python="3.13", torch="test", cuda="test", cudnn=1,
              gpu="test", seed=42, cpu_threads=1, tf32=False, hub_kernels=False))
        write(path / "telemetry_before.json", {"uuid": "same-test-gpu", "driver": "test"})
        write(path / "packages.json", {"synthetic": "1"})
        return value

    @staticmethod
    def save_latency(path, value):
        write(path / "result.json", value)
        (path / "latencies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in value["rows"]))

    def change_candidate_latency(self, latency):
        for i, job in enumerate(self.jobs):
            if job["variant"] == "candidate":
                path = self.root / "latency" / f"{i:03d}"
                value = read(path / "result.json")
                value["rows"] = self.rows(job["batches"], latency=latency)
                self.save_latency(path, value)


class BatchReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_screening_does_not_mix_preflight_and_batch_answers(self):
        Archive(self.root)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "screening_pass", report)
        self.assertTrue(report["all_cells_faster"])
        self.assertTrue(report["all_cells_meet_threshold"])
        cell = report["cells"]["B1/8"]
        self.assertEqual(cell["verdicts"]["n"], 400)
        self.assertEqual(cell["verdicts"]["unstable_ids"], [])
        self.assertEqual(cell["throughput"]["baseline"]["texts"], 2400)
        self.assertEqual(cell["throughput"]["baseline"]["padded_tokens"],
                         cell["throughput"]["baseline"]["useful_tokens"] + cell["throughput"]["baseline"]["padding_tokens"])
        rows = batch_report.table_rows(self.root, report, "archive.zip")
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["comparison_group"], rows[1]["comparison_group"])
        self.assertTrue((self.root / "REPORT.md").exists())

    def test_positive_matrix_is_not_twenty_percent_acceptance(self):
        archive = Archive(self.root, grid={"B1": [8]})
        archive.change_candidate_latency(9)
        report = batch_report.build(self.root)
        self.assertTrue(report["all_cells_faster"], report)
        self.assertFalse(report["all_cells_meet_threshold"])
        self.assertEqual(report["status"], "rejected")

    def test_parameter_threshold_and_quantization_flips(self):
        archive = Archive(self.root, comparison="quantization", grid={"B1": [8]})
        for i, job in enumerate(archive.jobs):
            if job["variant"] != "candidate":
                continue
            path = self.root / "latency" / f"{i:03d}"
            value = read(path / "result.json")
            for row in value["rows"]:
                for n, key in enumerate(row["batch"]["ids"]):
                    if key in ("B1-001", "B1-002"):
                        row["scores"][n] = answer("unsafe")
            archive.save_latency(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "quantization_measured", report)
        self.assertEqual(report["cells"]["B1/8"]["verdicts"]["changed_count"], 2)
        self.assertEqual(report["macro_verdict_change_fraction_by_batch_size"]["8"], 2 / 400)
        self.assertFalse(report["quality_evaluated"])
        self.assertTrue(report["all_cells_faster"])
        self.assertFalse(report["all_cells_meet_threshold"])

    def test_missing_memory_does_not_erase_healthy_cell(self):
        Archive(self.root)
        (self.root / "batches/B1/8/memory/candidate_2/result.json").unlink()
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["cells"]["B1/1"]["status"], "screening_pass")
        self.assertEqual(report["cells"]["B1/8"]["completeness"], "incomplete")
        self.assertFalse(report["all_cells_faster"])

    def test_oom_and_private_exception_are_sanitized(self):
        Archive(self.root)
        path = self.root / "matrix_status.json"
        state = read(path)
        state["B1/8"].update(status="oom", error_type="private-prompt-secret")
        write(path, state)
        report = batch_report.build(self.root)
        self.assertEqual(report["cells"]["B1/8"]["status"], "oom")
        self.assertNotIn("private-prompt-secret", json.dumps(report))

    def test_source_path_and_worker_hash_fail_closed(self):
        Archive(self.root, mode="smoke")
        write(self.root / "source_files.json", {"../private": "bad"})
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("unsafe_artifact_path", report["errors"])

    def test_runtime_and_contract_changes_fail_closed(self):
        Archive(self.root, mode="smoke")
        path = self.root / "batches/B1/8/preflight/baseline/runtime.json"
        value = read(path)
        value["torch"] = "different"
        write(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("runtime_identity_mismatch", report["errors"])
        self.assertEqual(batch_report.table_rows(self.root, report, "x"), [])

    def test_raw_row_mismatch_and_compile_drift(self):
        Archive(self.root, grid={"B1": [8]})
        path = self.root / "latency/000/result.json"
        value = read(path)
        value["rows"][0]["latency_ms"] = 99
        write(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["cells"]["B1/8"]["status"], "incomplete")
        self.assertIn("raw_rows_mismatch", report["cells"]["B1/8"]["errors"])
        value["rows"][0]["latency_ms"] = 10
        value["final_compile_count"] = 1
        write(path, value)
        report = batch_report.build(self.root)
        self.assertIn("compilation_during_measurement", report["cells"]["B1/8"]["errors"])

    def test_baseline_only_and_fixture_never_screening_pass(self):
        Archive(self.root, candidate=False, fixture=True, grid={"B1": [8]})
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "fixture_complete", report)
        self.assertFalse(report["all_cells_faster"])
        self.assertEqual(report["cells"]["B1/8"]["status"], "baseline_complete")
        self.assertEqual(len(report["cells"]["B1/8"]["baseline_series_p95_ms"]), 3)
        self.assertEqual(batch_report.table_rows(self.root, report, "fixture"), [])

    def test_pilot_and_smoke_do_not_get_acceptance(self):
        Archive(self.root, mode="pilot", grid={"B1": [8]})
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "pilot_complete", report)
        self.assertFalse(report["all_cells_faster"])
        self.assertIsNone(report["cells"]["B1/8"]["p95"]["ratio_interval_95"])

    def test_staircase_not_scheduled(self):
        Archive(self.root, model="halo", mode="smoke", grid={"B1": [1, 8], "B2": [1], "B3": [1], "B4": [1]})
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "smoke_passed", report)
        self.assertEqual(report["matrix"]["B2"]["8"], "not_scheduled")
        self.assertNotIn("B2/8", report["cells"])

    def test_profile_requires_real_kernel_event(self):
        Archive(self.root, grid={"B1": [8]})
        path = self.root / "batches/B1/8/profiles/candidate/long.trace.json.gz"
        with gzip.open(path, "wt") as stream:
            json.dump({"traceEvents": [{"cat": "cpu_op"}]}, stream)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("profile_without_kernels", report["cells"]["B1/8"]["errors"])

    def test_single_to_batch_differences_are_not_within_cell_instability(self):
        archive = Archive(self.root)
        for i, job in enumerate(archive.jobs):
            if job["batch_size"] != 8:
                continue
            path = self.root / "latency" / f"{i:03d}"
            value = read(path / "result.json")
            for row in value["rows"]:
                row["scores"] = [answer("unsafe") for _ in row["scores"]]
            archive.save_latency(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "screening_pass", report)
        self.assertEqual(report["single_to_batch"]["B1/8"]["baseline"]["changed_count"], 400)
        self.assertEqual(report["cells"]["B1/8"]["verdict_status"], "unchanged")

    def test_instability_rejects_cell_but_not_speed_flag(self):
        archive = Archive(self.root, grid={"B1": [8]})
        i = next(i for i, job in enumerate(archive.jobs) if job["variant"] == "candidate")
        path = self.root / "latency" / f"{i:03d}"
        value = read(path / "result.json")
        value["rows"][0]["scores"][0] = answer("unsafe")
        archive.save_latency(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "rejected", report)
        self.assertEqual(report["verdict_status"], "unstable")
        self.assertTrue(report["all_cells_faster"])

    def test_precision_mismatch_is_global_not_only_cell_local(self):
        Archive(self.root, mode="smoke")
        passport = {"parameter_dtypes": ["float32"], "autocast_enabled": True,
                    "autocast_dtype": "float16", "policy": "native_sdk"}
        for path in self.root.glob("batches/*/*/preflight/*/result.json"):
            value = read(path)
            value["precision_policy_version"] = 1
            value["metadata"]["precision"] = dict(passport)
            if "/8/" in str(path) and value["variant"] == "candidate":
                value["metadata"]["precision"]["autocast_dtype"] = "bfloat16"
            write(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("worker_integrity_failure", report["errors"])

    def test_pair_padding_semantics_not_backend_description(self):
        Archive(self.root, mode="smoke", grid={"B1": [8]})
        path = self.root / "batches/B1/8/preflight/candidate/result.json"
        value = read(path)
        value["metadata"]["batch_policy"]["implementation"] = "different-quantized-runtime"
        write(path, value)
        self.assertEqual(batch_report.build(self.root)["status"], "smoke_passed")
        value["metadata"]["batch_policy"]["padding_side"] = "left"
        write(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("paired_padding_policy_mismatch", report["errors"])

    def test_worker_source_hash_and_contract_are_strict(self):
        Archive(self.root, mode="smoke", grid={"B1": [8]})
        path = self.root / "batches/B1/8/preflight/candidate/result.json"
        value = read(path)
        value["sources_sha256"] = "bad"
        write(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("worker_integrity_failure", report["errors"])
        value["sources_sha256"] = sha(self.root / "source_files.json")
        value["contract"]["rules"] = "different-rule"
        write(path, value)
        report = batch_report.build(self.root)
        self.assertIn("inference_contract_mismatch", report["errors"])

    def test_parameter_ten_percent_and_noisy_comparison(self):
        archive = Archive(self.root, comparison="parameter", grid={"B1": [8]})
        archive.change_candidate_latency(8.5)
        self.assertEqual(batch_report.build(self.root)["status"], "screening_pass")
        for i, job in enumerate(archive.jobs):
            if job["phase"] != "aa" or job["side"] != "b":
                continue
            path = self.root / "latency" / f"{i:03d}"
            value = read(path / "result.json")
            value["rows"] = archive.rows(job["batches"], latency=13)
            archive.save_latency(path, value)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "inconclusive")
        self.assertFalse(report["all_cells_faster"])

    def test_stable_slowdown_not_inconclusive(self):
        archive = Archive(self.root, grid={"B1": [8]})
        archive.change_candidate_latency(12)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["cells"]["B1/8"]["reasons"], ["consistent_slowdown"])

    def test_quality_is_single_only_and_recomputed_from_predictions(self):
        from u1_benchmark.quality import metrics
        archive = Archive(self.root, mode="smoke", grid={"B1": [8]})
        archive.cfg.update(quality_data={"path": "unused-during-report"}, label_mapping={
            "labels": ["safe", "unsafe"], "targets": {"safe": "safe", "unsafe": "unsafe"},
            "verdicts": {"safe": "safe", "unsafe": "unsafe"}})
        write(self.root / "config.json", archive.cfg)
        for path in self.root.glob("batches/*/*/preflight/*/result.json"):
            value = read(path)
            value["config_sha256"] = digest(archive.cfg)
            write(path, value)
        qrows = [{"id": "q1", "target": "safe"}]
        predictions = [{"id": "q1", "status": "passed", "score": answer()}]
        write(self.root / "quality_inputs.json", qrows)
        measurement = {"bucket": "B1", "batch_size": 1, "batches": smoke_plans(archive.data["B1"], 1)}
        for variant in archive.variants:
            path = self.root / "quality" / variant
            value = archive.worker(path, variant, "quality", measurement)
            value.update(quality_batch_size=1, quality=metrics(qrows, predictions, archive.cfg["label_mapping"]))
            write(path / "result.json", value)
            (path / "predictions.jsonl").write_text(json.dumps(predictions[0]) + "\n")
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "smoke_passed", report)
        self.assertTrue(report["quality_evaluated"])
        self.assertEqual(report["quality_batch_size"], 1)
        self.assertEqual(report["quality_status"], "evaluated_batch_1_only")
        path = self.root / "quality/candidate/result.json"
        value = read(path)
        value["quality_batch_size"] = 8
        write(path, value)
        self.assertEqual(batch_report.build(self.root)["status"], "incomplete")

    def test_missing_unique_inputs_and_missing_raw_side(self):
        Archive(self.root, grid={"B1": [8]})
        path = self.root / "latency/000/latencies.jsonl"
        path.unlink()
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        inputs = read(self.root / "inputs.json")[:-1]
        write(self.root / "inputs.json", inputs)
        report = batch_report.build(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["all_cells_faster"])


if __name__ == "__main__":
    unittest.main()
