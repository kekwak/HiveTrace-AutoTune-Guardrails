"""Real quality worker/report paths with an external CPU-only adapter fixture."""
import copy
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from examples.u1_external_adapter import Fixture
from test_u1_quality_batches import MAPPING
from u1_benchmark import contracts, datasets, quality_batches, quality_report
from u1_benchmark.adapters import check_index
from u1_benchmark.batch_adapters import UnsupportedBatch
from u1_benchmark.core import digest, lines, read, sha, write
from u1_benchmark.messages import private_strings, select
from u1_benchmark.runner_v2 import public_rows
from u1_benchmark.worker_v2 import execute


PRECISION = {"parameter_dtypes": ["float32"], "autocast_enabled": False,
             "autocast_dtype": None, "policy": "native_sdk"}


class OutOfMemoryError(RuntimeError):
    pass


class WorkerBatchFixture(Fixture):
    def __init__(self, contract, spec, model):
        super().__init__(contract, spec, model)
        self.options = spec.get("options", {})
        self.metadata.update(precision=copy.deepcopy(PRECISION), calls=[], close_count=0)
        if self.options.get("quantized"):
            self.metadata["precision"]["parameter_dtypes"] = ["int8"]
        self.metadata["batch_policy"] = {"padding": "longest", "padding_side": "right",
                                         "pad_token_id": 0, "truncation": False,
                                         "sort_by_length": False}
        if self.options.get("unsupported"):
            self.metadata["batch_policy"]["truncation"] = True
        self.batch_count = 0

    def record(self, name, size):
        self.metadata["calls"].append([name, size])

    def index(self, value):
        if self.options.get("index_oom"):
            raise OutOfMemoryError(private_strings(value)[-1])
        return super().index(value)

    def prepare_backend(self, inputs):
        self.record("prepare", len(inputs))

    def prove_backend(self, value):
        self.record("proof", 1)
        return {"test_fixture": True}

    def prepare_batch_backend(self, groups):
        self.record("prepare_batch", len(groups))
        self.metadata["prepared_batch_sizes"] = [len(group) for group in groups]
        self.metadata["private_echo"] = [text for group in groups for value in group
                                         for text in private_strings(value)]

    def prove_batch_backend(self, inputs):
        self.record("proof_batch", len(inputs))
        return {"test_fixture": True, "backend": "CPU fixture only"}

    def score(self, value, size):
        text = select(self.name, value)["effective_messages"][-1]["content"]
        unsafe = text.startswith("harmful")
        if self.options.get("quantized") and size > 1:
            unsafe = not unsafe
        return {"label": "unsafe" if unsafe else "safe"}

    def verify(self, value, expected):
        self.record("verify", 1)
        check_index(self.index(value), expected)
        return self.score(value, 1)

    def request_batch(self, inputs):
        return [self.score(value, len(inputs)) for value in inputs]

    def verify_batch(self, inputs, indices):
        self.record("verify_batch", len(inputs))
        self.batch_count += 1
        if self.batch_count == self.options.get("stall_batch"):
            write(self.options["stall_marker"], {"entered": True})
            time.sleep(120)
        for value, expected in zip(inputs, indices, strict=True):
            check_index(self.index(value), expected)
        if self.batch_count == self.options.get("oom_batch"):
            raise OutOfMemoryError(private_strings(inputs[0])[-1])
        if self.batch_count == self.options.get("unsupported_batch"):
            raise UnsupportedBatch(private_strings(inputs[0])[-1])
        if self.options.get("change_precision"):
            self.metadata["precision"]["parameter_dtypes"] = ["float16"]
        if self.batch_count == self.options.get("unmapped_batch"):
            return [{"label": "unmapped-native-verdict"} for _ in inputs]
        return [self.score(value, len(inputs)) for value in inputs]

    def close(self):
        self.metadata["close_count"] += 1
        self.metadata["cleanup"] = {"precision": copy.deepcopy(self.metadata["precision"])}


class ScalarWorkerFixture(WorkerBatchFixture):
    request_batch = None
    prepare_batch_backend = None
    prove_batch_backend = None
    verify_batch = None


def create(*, model, spec, directory, contract):
    """Loaded by the real source-pinned external adapter loader, not patched in."""
    fixture = ScalarWorkerFixture if spec.get("options", {}).get("scalar_only") else WorkerBatchFixture
    return fixture(contract, spec, model)


def fake_probe(environment, directory):
    write(directory / "runtime.json", {key: "CPU-fixture" for key in (
        "python", "torch", "cuda", "cudnn", "gpu", "seed", "cpu_threads", "tf32", "hub_kernels")})
    write(directory / "packages.json", {})


def killable_worker(job_path):
    """Real child/timeout path, with GPU work replaced by the CPU fixture."""
    import torch
    with patch("u1_benchmark.environment.probe", side_effect=fake_probe), \
            patch("u1_benchmark.worker.telemetry", return_value={"uuid": "CPU-fixture", "driver": "fixture"}), \
            patch.object(torch.cuda, "synchronize"):
        execute(job_path)


class QualityWorkerCase(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable; worker requires torch but no GPU")
        self.torch = torch
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        source_rows = [{"id": f"q{i}", "messages": [
            {"role": "user", "content": f"PRIVATE context {i}"},
            {"role": "assistant", "content": (
                "float32" if i == 0 else "native_sdk" if i == 2 else
                f"{'harmful' if i % 2 else 'benign'} PRIVATE answer {i}")}],
            "target": "harmful" if i % 2 else "benign", "target_role": "assistant"}
            for i in range(7)]
        self.prepare_rows(source_rows)

    def prepare_rows(self, source_rows, name="data"):
        source = self.root / f"{name}-source.jsonl"
        datasets.jsonl(source, source_rows)
        self.data = datasets.prepare({"schema_version": 3, "kind": "quality", "source": str(source),
            "fields": {k: k for k in ("id", "messages", "target", "target_role")},
            "version": "cpu-fixture", "provenance": "test"}, self.root / name)
        self.rows, _ = datasets.verified(self.data["path"], self.data["manifest_sha256"])
        self.output = self.root / "output"
        write(self.output / "inputs.json", [])
        write(self.output / "quality_inputs.json", public_rows(self.rows))
        write(self.output / "source_files.json", {name: sha(self.output / name)
            for name in ("inputs.json", "quality_inputs.json")})

    def config(self, sizes=(1, 3), baseline_options=None, candidate_options=None):
        spec = {"module": str(Path(__file__).resolve()), "source_sha256": sha(__file__),
                "dependencies": {}, "options": baseline_options or {}}
        value = {"schema_version": 3, "model": "gliner", "mode": "quality", "seed": 17,
                 "environment": str(self.root / "environment"), "baseline": spec,
                 "quality_data": {k: self.data[k] for k in ("path", "manifest_sha256")},
                 "label_mapping": MAPPING, "quality_batch_sizes": list(sizes)}
        if candidate_options is not None:
            artifact = self.root / "toy-artifact.json"
            write(artifact, {"fixture": "not real quantized weights"})
            model = contracts.MODEL_SPECS["gliner"]
            value.update(comparison="quantization", candidate={**spec, "options": candidate_options,
                "artifact": {"path": str(artifact), "sha256": digest(contracts.file_inventory(artifact)),
                             "base_checkpoint": model["repo"], "base_revision": model["revision"],
                             "method": "test-fixture", "bits": 8, "runtime": "cpu-fixture"}})
        return contracts.parse_config(value)

    def run_worker(self, cfg, size, variant="baseline", **job_overrides):
        directory = self.output / "quality" / str(size) / variant
        job = {"quality_protocol_version": 1, "config": cfg, "phase": "quality",
               "variant": variant, "directory": str(directory), "private": True,
               "measurement": {"quality_batch_size": size},
               "inputs_sha256": sha(self.output / "inputs.json"),
               "quality_inputs_sha256": digest(public_rows(self.rows)), **job_overrides}
        job_path = self.root / "job.json"
        write(job_path, job)
        with patch("u1_benchmark.environment.probe", side_effect=fake_probe), \
                patch("u1_benchmark.worker.telemetry", return_value={"uuid": "CPU-fixture", "driver": "fixture"}), \
                patch.object(self.torch.cuda, "synchronize"):
            returned = execute(job_path)
        saved = read(directory / "result.json")
        self.assertEqual(returned["status"], saved["status"])
        ledger_path = self.output / "quality_status.json"
        ledger = read(ledger_path) if ledger_path.exists() else {}
        ledger[f"{size}/{variant}"] = {"status": saved["status"]}
        write(ledger_path, ledger)
        return directory, saved

    def run_matrix(self, cfg):
        results = {}
        for size in cfg["quality_batch_sizes"]:
            for variant in ("baseline", "candidate") if cfg.get("candidate") else ("baseline",):
                results[size, variant] = self.run_worker(cfg, size, variant)
        return results

    def assert_closed_with_precision(self, result, precision=PRECISION):
        self.assertEqual(result["metadata"]["close_count"], 1)
        self.assertEqual(result["metadata"]["precision"], precision)
        self.assertEqual(result["metadata"]["cleanup"]["precision"], precision)

    def assert_private_output(self, directory):
        for path in directory.rglob("*"):
            if path.is_file() and ".private" not in path.parts:
                self.assertNotIn("PRIVATE", path.read_text(), str(path))
        self.assertFalse((directory / "error.txt").exists())


class QualityWorkerTests(QualityWorkerCase):
    def test_hard_timeout_checkpoint_and_archive_never_contain_private_metadata(self):
        from u1_benchmark.process import archive, invoke
        marker = self.root / "stalled.json"
        cfg = self.config(sizes=(3,), baseline_options={"stall_batch": 2, "stall_marker": str(marker)})
        directory = self.output / "quality/3/baseline"
        job_path = self.root / "killable-job.json"
        write(job_path, {"quality_protocol_version": 1, "config": cfg, "phase": "quality", "variant": "baseline",
                        "directory": str(directory), "private": True, "measurement": {"quality_batch_size": 3},
                        "inputs_sha256": sha(self.output / "inputs.json"),
                        "quality_inputs_sha256": digest(public_rows(self.rows))})
        code = ("import sys; sys.path.insert(0, 'tests'); "
                "from test_u1_quality_worker_report import killable_worker; killable_worker(sys.argv[1])")
        with self.assertRaises(TimeoutError):
            invoke([sys.executable, "-B", "-c", code, job_path], directory / "stdout.log", 10)
        self.assertTrue(marker.exists(), "Worker must reach the stall after its first checkpoint")
        result = read(directory / "result.json")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["metadata"]["close_count"], 0, "SIGKILL must bypass final redaction")
        self.assertEqual(result["quality"]["overall"]["evaluated"], 3)
        self.assertEqual(sum(p["status"] == "passed" for p in lines(directory / "predictions.jsonl")), 3)
        self.assert_private_output(directory)
        with zipfile.ZipFile(archive(self.output)) as saved:
            self.assertIsNone(saved.testzip())
            for name in saved.namelist():
                self.assertNotIn(b"PRIVATE", saved.read(name), name)

    def test_checkpoint_sanitizes_copy_without_changing_adapter_precision(self):
        row = self.rows[1]
        text = private_strings(row["messages"])[-1]
        result = {"metadata": {"echo": {text: [text, (text,)]}, "label": text,
                               "precision": copy.deepcopy(PRECISION),
                               "batch_policy": {"padding": "longest", "padding_side": "right"}}}
        original = copy.deepcopy(result)
        saved = quality_batches.public_checkpoint(result, self.rows)
        self.assertEqual(result, original)
        self.assertEqual(saved["metadata"]["precision"], PRECISION)
        self.assertEqual(saved["metadata"]["batch_policy"], original["metadata"]["batch_policy"])
        self.assertNotIn(text, str(saved))
        self.assertEqual(quality_batches.public_checkpoint(result, self.rows, private=False), original)

    def test_scalar_only_external_rejects_requested_batch_with_one_row(self):
        self.prepare_rows(self.rows[:1], "singleton")
        directory, result = self.run_worker(self.config(sizes=(3,), baseline_options={"scalar_only": True}), 3)
        self.assertEqual(result["status"], "unsupported", result)
        self.assertEqual(result["error_type"], "UnsupportedBatch")
        self.assertFalse(any(name == "verify" for name, _ in result["metadata"]["calls"]))
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)

    def test_indexing_oom_is_terminal_without_plan_and_report_keeps_status(self):
        cfg = self.config(sizes=(3,), baseline_options={"index_oom": True})
        directory, result = self.run_worker(cfg, 3)
        self.assertEqual(result["status"], "oom", result)
        self.assertFalse((directory / "quality_plan.json").exists())
        self.assertFalse((directory / "predictions.jsonl").exists())
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)
        report = quality_report.collect(self.output, cfg)
        self.assertEqual(report["quality_by_batch_size"]["3"]["variants"]["baseline"]["status"], "oom")
        self.assertFalse(report["quality_evaluated"])
        self.assertFalse(report["quality_complete"])

    def test_unmapped_verdict_keeps_earlier_batches_valid_for_report(self):
        cfg = self.config(sizes=(3,), baseline_options={"unmapped_batch": 2})
        directory, result = self.run_worker(cfg, 3)
        self.assertEqual(result["status"], "failed", result)
        predictions = lines(directory / "predictions.jsonl")
        self.assertEqual([p["status"] for p in predictions], ["passed"] * 3 + ["error"] * 3 + ["not_run"])
        self.assertTrue(all("score" not in p for p in predictions[3:]))
        self.assertEqual(result["quality"]["overall"]["evaluated"], 3)
        report = quality_report.collect(self.output, cfg)
        item = report["quality_by_batch_size"]["3"]["variants"]["baseline"]
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["quality"]["overall"]["evaluated"], 3)
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)

    def test_private_user_string_does_not_corrupt_role_evidence_or_plan_hash(self):
        self.prepare_rows([{"id": "q0", "messages": [{"role": "user", "content": "user"}],
                            "target": "benign", "target_role": "user"}], "role-collision")
        cfg = self.config(sizes=(3,))
        directory, result = self.run_worker(cfg, 3)
        self.assertEqual(result["status"], "passed", result)
        plan = read(directory / "quality_plan.json")
        self.assertEqual(plan["indices"]["q0"]["target_role"], "user")
        self.assertEqual(plan["indices"]["q0"]["source_mode"], "user-only")
        self.assertEqual(digest(plan), result["quality_plan_sha256"])
        self.assertEqual(result["metadata"]["private_echo"], ["[redacted-input]"])
        self.assertTrue(quality_report.collect(self.output, cfg)["quality_complete"])

    def test_real_batch_wrapper_tail_close_precision_and_private_output(self):
        directory, result = self.run_worker(self.config(), 3)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["quality_protocol_version"], 1)
        self.assertEqual(result["quality_batch_size"], 3)
        self.assertEqual(result["metadata"]["calls"], [
            ["prepare_batch", 3], ["proof_batch", 3], ["verify_batch", 3],
            ["verify_batch", 3], ["verify_batch", 1]])
        self.assertEqual(result["metadata"]["prepared_batch_sizes"], [3, 3, 1])
        self.assertIn("batch_backend_proof", result["metadata"])
        self.assertEqual(result["quality"]["overall"]["accuracy"], 1)
        plan = read(directory / "quality_plan.json")
        self.assertEqual([p["id"] for p in lines(directory / "predictions.jsonl")], plan["ordered_ids"])
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)

    def test_size_one_retains_single_adapter_path(self):
        directory, result = self.run_worker(self.config(), 1)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["metadata"]["calls"], [["prepare", 7], ["proof", 1]] + [["verify", 1]] * 7)
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)

    def test_oom_preserves_partial_predictions_and_closes(self):
        directory, result = self.run_worker(self.config(baseline_options={"oom_batch": 2}), 3)
        self.assertEqual(result["status"], "oom", result)
        self.assertEqual(result["error_type"], "OutOfMemoryError")
        self.assertEqual([p["status"] for p in lines(directory / "predictions.jsonl")],
                         ["passed"] * 3 + ["oom"] * 3 + ["not_run"])
        self.assertEqual(result["quality"]["overall"]["evaluated"], 3)
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)

    def test_invalid_batch_policy_reports_unsupported_without_single_fallback(self):
        directory, result = self.run_worker(self.config(baseline_options={"unsupported": True}), 3)
        self.assertEqual(result["status"], "unsupported", result)
        self.assertEqual(result["error_type"], "UnsupportedBatch")
        self.assertEqual(result["metadata"]["calls"], [])
        self.assertEqual([p["status"] for p in lines(directory / "predictions.jsonl")], ["not_run"] * 7)
        self.assertFalse(result["quality_evaluated"])
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)

    def test_unsupported_during_inference_keeps_partial_rows_private(self):
        directory, result = self.run_worker(self.config(baseline_options={"unsupported_batch": 2}), 3)
        self.assertEqual(result["status"], "unsupported", result)
        self.assertEqual([p["status"] for p in lines(directory / "predictions.jsonl")],
                         ["passed"] * 3 + ["unsupported"] * 3 + ["not_run"])
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)

    def test_real_precision_guard_rejects_mutation_and_still_closes(self):
        directory, result = self.run_worker(self.config(baseline_options={"change_precision": True}), 3)
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["error_type"], "PrecisionError")
        self.assertEqual(result["metadata"]["close_count"], 1)
        self.assertEqual(result["quality"]["overall"]["evaluated"], 0)
        self.assert_private_output(directory)

    def test_input_snapshot_mismatch_rejected_before_inference(self):
        directory, result = self.run_worker(self.config(), 3, quality_inputs_sha256="tampered")
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["error_type"], "ValueError")
        self.assertEqual(result["metadata"]["calls"], [])
        self.assert_closed_with_precision(result)
        self.assert_private_output(directory)


class QualityReportTests(QualityWorkerCase):
    def nonfixture_evidence(self, result, size):
        value = copy.deepcopy(result)
        value["metadata"]["test_fixture"] = False
        value["metadata"]["batch_backend_proof" if size > 1 else "backend_proof"] = {
            "kernels": ["synthetic-validator-test-only-not-a-GPU-measurement"]}
        return value

    def test_nonfixture_requires_backend_padding_and_prepared_shapes(self):
        cfg = self.config(sizes=(3,))
        directory, original = self.run_worker(cfg, 3)
        valid = self.nonfixture_evidence(original, 3)
        write(directory / "result.json", valid)
        self.assertTrue(quality_report.collect(self.output, cfg)["quality_complete"])
        for missing in ("batch_backend_proof", "batch_policy", "kernels", "prepared_shapes",
                        "truncation", "sort_by_length", "pad_token_id", "padding_side"):
            with self.subTest(missing=missing):
                value = copy.deepcopy(valid)
                if missing in ("batch_backend_proof", "batch_policy"):
                    value["metadata"].pop(missing)
                elif missing == "kernels":
                    value["metadata"]["batch_backend_proof"]["kernels"] = []
                elif missing == "prepared_shapes":
                    value.pop(missing)
                else:
                    value["metadata"]["batch_policy"].pop(missing)
                write(directory / "result.json", value)
                report = quality_report.collect(self.output, cfg)
                self.assertFalse(report["quality_complete"])
                self.assertFalse(report["quality_evaluated"])
                self.assertEqual(report["quality_status"], "incomplete")

    def test_nonfixture_single_and_partial_workers_require_backend_proof(self):
        for size, options in ((1, {}), (3, {"oom_batch": 2})):
            with self.subTest(size=size):
                write(self.output / "quality_status.json", {})
                cfg = self.config(sizes=(size,), baseline_options=options)
                directory, result = self.run_worker(cfg, size)
                valid = self.nonfixture_evidence(result, size)
                write(directory / "result.json", valid)
                self.assertTrue(quality_report.collect(self.output, cfg)["quality_evaluated"])
                valid["metadata"].pop("backend_proof" if size == 1 else "batch_backend_proof")
                write(directory / "result.json", valid)
                report = quality_report.collect(self.output, cfg)
                self.assertFalse(report["quality_evaluated"])
                self.assertFalse(report["quality_complete"])

    def test_nonfixture_padding_mismatch_blocks_same_size_comparison(self):
        cfg = self.config(sizes=(3,), candidate_options={"quantized": True})
        outcomes = self.run_matrix(cfg)
        for (size, variant), (directory, result) in outcomes.items():
            value = self.nonfixture_evidence(result, size)
            if variant == "candidate":
                value["metadata"]["batch_policy"]["padding_side"] = "left"
            write(directory / "result.json", value)
        report = quality_report.collect(self.output, cfg)
        self.assertFalse(report["quality_complete"])
        self.assertEqual(report["quality_by_batch_size"]["3"]["baseline_to_candidate"]["status"], "blocked_padding_mismatch")

    def test_missing_execution_ledger_or_cell_is_incomplete(self):
        cfg = self.config(sizes=(3,))
        self.run_worker(cfg, 3)
        ledger = self.output / "quality_status.json"
        for missing in ("file", "cell"):
            with self.subTest(missing=missing):
                if missing == "file":
                    ledger.unlink()
                else:
                    write(ledger, {})
                report = quality_report.collect(self.output, cfg)
                self.assertFalse(report["quality_complete"])
                self.assertEqual(report["quality_status"], "incomplete")

    def test_partial_worker_contract_spec_and_precision_are_validated(self):
        cfg = self.config(baseline_options={"oom_batch": 2})
        results = self.run_matrix(cfg)
        directory, original = results[3, "baseline"]
        for tamper in ("contract", "adapter_spec", "missing_precision", "changed_precision", "precision_error"):
            with self.subTest(tamper=tamper):
                result = copy.deepcopy(original)
                if tamper in ("contract", "adapter_spec"):
                    result[tamper] = {}
                elif tamper == "missing_precision":
                    result["metadata"].pop("precision")
                elif tamper == "precision_error":
                    # A final, apparently valid passport cannot establish the
                    # precision used before a detected mid-worker mode change.
                    result["error_type"] = "PrecisionError"
                else:
                    result["metadata"]["precision"]["parameter_dtypes"] = ["float16"]
                write(directory / "result.json", result)
                report = quality_report.collect(self.output, cfg)
                item = report["quality_by_batch_size"]["3"]["variants"]["baseline"]
                self.assertEqual(item["status"], "incomplete")
                self.assertFalse(item["quality_evaluated"])
                self.assertEqual(report["quality_single_to_batch"]["3"]["baseline"]["status"], "unavailable")

    def test_prediction_content_identity_cannot_be_changed_or_removed(self):
        cfg = self.config(sizes=(3,))
        directory, _ = self.run_worker(cfg, 3)
        original = lines(directory / "predictions.jsonl")
        for missing in (False, True):
            with self.subTest(missing=missing):
                predictions = copy.deepcopy(original)
                if missing:
                    predictions[0].pop("content_sha256")
                else:
                    predictions[0]["content_sha256"] = digest("other source")
                datasets.jsonl(directory / "predictions.jsonl", predictions)
                report = quality_report.collect(self.output, cfg)
                self.assertEqual(report["quality_by_batch_size"]["3"]["variants"]["baseline"]["status"], "incomplete")
                self.assertFalse(report["quality_evaluated"])

    def test_partial_worker_runtime_mismatch_blocks_comparison(self):
        cfg = self.config(candidate_options={"oom_batch": 2})
        results = self.run_matrix(cfg)
        directory, _ = results[3, "candidate"]
        write(directory / "telemetry_before.json", {"uuid": "different-runtime", "driver": "fixture"})
        report = quality_report.collect(self.output, cfg)
        self.assertEqual(report["quality_by_batch_size"]["3"]["baseline_to_candidate"]["status"], "blocked_runtime_mismatch")
        self.assertEqual(report["quality_single_to_batch"]["3"]["candidate"]["status"], "blocked_runtime_mismatch")

    def test_same_size_and_single_to_batch_changed_quant_labels_never_approve(self):
        cfg = self.config(candidate_options={"quantized": True})
        results = self.run_matrix(cfg)
        checked = []
        report = quality_report.collect(self.output, cfg, lambda path, variant: checked.append((path, variant)))
        self.assertEqual(len(checked), 4)
        self.assertTrue(report["quality_complete"])
        self.assertTrue(report["quality_fixture"])
        self.assertEqual(report["quality_status"], "evaluated")
        self.assertIsNone(report["quality_preserved"])
        cells = report["quality_by_batch_size"]
        self.assertEqual(cells["1"]["baseline_to_candidate"]["changed_count"], 0)
        changed = cells["3"]["baseline_to_candidate"]
        self.assertEqual(changed["status"], "complete")
        self.assertEqual(changed["changed_count"], 6)
        self.assertEqual(changed["n"], 7)
        self.assertNotIn("quality_preserved", changed)
        self.assertEqual(changed["left_quality"]["overall"]["accuracy"], 1)
        self.assertAlmostEqual(changed["right_quality"]["overall"]["accuracy"], 1 / 7)
        single = report["quality_single_to_batch"]["3"]
        self.assertEqual(single["baseline"]["changed_count"], 0)
        self.assertEqual(single["candidate"]["changed_count"], 6)
        self.assertEqual(cells["3"]["variants"]["candidate"]["actual_batch_sizes"], [3, 3, 1])
        precision = {**PRECISION, "parameter_dtypes": ["int8"]}
        self.assert_closed_with_precision(results[3, "candidate"][1], precision)

    def test_failed_worker_partial_metrics_use_only_common_predictions(self):
        cfg = self.config(candidate_options={"quantized": True, "oom_batch": 2})
        self.run_matrix(cfg)
        report = quality_report.collect(self.output, cfg)
        cell = report["quality_by_batch_size"]["3"]
        self.assertFalse(report["quality_complete"])
        self.assertEqual(report["quality_status"], "incomplete")
        self.assertIsNone(report["quality_preserved"])
        candidate = cell["variants"]["candidate"]
        self.assertEqual(candidate["status"], "oom")
        self.assertTrue(candidate["quality_evaluated"])
        self.assertEqual(candidate["quality"]["overall"]["failures"], {"oom": 3, "not_run": 1})
        comparison = cell["baseline_to_candidate"]
        self.assertEqual(comparison["status"], "partial")
        self.assertEqual(comparison["n"], 3)
        self.assertEqual(comparison["changed_count"], 3)
        self.assertAlmostEqual(comparison["coverage"], 3 / 7)
        self.assertEqual(comparison["left_quality"]["overall"]["total"], 3)
        self.assertEqual(comparison["right_quality"]["overall"]["accuracy"], 0)
        self.assertEqual(report["quality_single_to_batch"]["3"]["candidate"]["n"], 3)

    def test_unsupported_worker_has_no_successful_comparison(self):
        cfg = self.config(candidate_options={"unsupported": True})
        self.run_matrix(cfg)
        report = quality_report.collect(self.output, cfg)
        cell = report["quality_by_batch_size"]["3"]
        self.assertEqual(cell["variants"]["candidate"]["status"], "unsupported")
        self.assertFalse(cell["variants"]["candidate"]["quality_evaluated"])
        self.assertEqual(cell["baseline_to_candidate"]["status"], "no_common_predictions")
        self.assertFalse(report["quality_complete"])
        self.assertIsNone(report["quality_preserved"])

    def test_tampered_plan_and_indices_are_incomplete(self):
        cfg = self.config(sizes=(3,))
        directory, result = self.run_worker(cfg, 3)
        original = read(directory / "quality_plan.json")
        for tamper in ("order", "index", "summary", "prediction_order"):
            with self.subTest(tamper=tamper):
                plan, value = copy.deepcopy(original), copy.deepcopy(result)
                predictions = lines(directory / "predictions.jsonl")
                if tamper == "order":
                    plan["ordered_ids"].reverse()
                    value["quality_plan_sha256"] = digest(plan)
                elif tamper == "index":
                    plan["indices"]["q0"]["input_sha256"] = "changed"
                elif tamper == "summary":
                    value["quality"]["overall"]["accuracy"] = 0
                else:
                    predictions.reverse()
                write(directory / "quality_plan.json", plan)
                write(directory / "result.json", value)
                datasets.jsonl(directory / "predictions.jsonl", predictions)
                report = quality_report.collect(self.output, cfg)
                self.assertFalse(report["quality_complete"])
                self.assertEqual(report["quality_by_batch_size"]["3"]["variants"]["baseline"]["status"], "incomplete")

    def test_consistently_rehashed_candidate_indices_block_comparisons(self):
        cfg = self.config(candidate_options={"quantized": True})
        results = self.run_matrix(cfg)
        directory, result = results[3, "candidate"]
        plan = read(directory / "quality_plan.json")
        plan["indices"]["q0"]["ids_sha256"] = digest("different tokenizer input")
        plan = quality_batches.make_plan(public_rows(self.rows), plan["indices"], plan["excluded"], 3, cfg["seed"])
        write(directory / "quality_plan.json", plan)
        write(directory / "result.json", {**result, "quality_plan_sha256": digest(plan)})
        report = quality_report.collect(self.output, cfg)
        self.assertEqual(report["quality_by_batch_size"]["3"]["baseline_to_candidate"]["status"], "blocked_input_mismatch")
        self.assertEqual(report["quality_single_to_batch"]["3"]["candidate"]["status"], "blocked_input_mismatch")
        self.assertFalse(report["quality_complete"])
        self.assertIsNone(report["quality_preserved"])

    def test_missing_predictions_are_incomplete_and_source_tampering_rejected(self):
        cfg = self.config(sizes=(3,))
        directory, _ = self.run_worker(cfg, 3)
        (directory / "predictions.jsonl").unlink()
        report = quality_report.collect(self.output, cfg)
        self.assertFalse(report["quality_complete"])
        self.assertFalse(report["quality_evaluated"])
        write(self.output / "quality_inputs.json", [])
        with self.assertRaisesRegex(ValueError, "quality_source_hash_mismatch"):
            quality_report.collect(self.output, cfg)


if __name__ == "__main__":
    unittest.main()
