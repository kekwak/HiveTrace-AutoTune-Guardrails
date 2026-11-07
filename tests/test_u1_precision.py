"""Precision policy and historical input compatibility; no GPU acceptance."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from u1_benchmark import contracts, datasets, distribution, report_v2, role_data, runner_v2
from u1_benchmark.core import ROOT, read, sha, write
from u1_benchmark.precision import PrecisionError, PrecisionGuard, PrecisionTracker, validate_precision
from test_u1_roles import EXTERNAL, ToyRolePreparer, fixture_data, simulated_role_worker


def passport(dtype="bfloat16", policy="native_sdk"):
    return {"parameter_dtypes": ["float32"], "autocast_enabled": True,
            "autocast_dtype": dtype, "policy": policy}


def recorded(value=None):
    return {"precision_policy_version": 1, "metadata": {"precision": value or passport()}}


class PrecisionPolicy(unittest.TestCase):
    def test_validation_and_snapshot_are_independent(self):
        value = passport()
        snapshot = validate_precision(value)
        value["parameter_dtypes"].append("float16")
        self.assertEqual(snapshot, passport())
        for value in (None, {}, {**passport(), "extra": True},
                      {**passport(), "parameter_dtypes": []},
                      {**passport(), "parameter_dtypes": ["private_payload"]},
                      {**passport(), "parameter_dtypes": ["float32", "float16"]},
                      {**passport(), "autocast_enabled": 1},
                      {**passport(), "autocast_enabled": False},
                      {**passport(), "autocast_dtype": "float32"},
                      {**passport(), "policy": "unverified arbitrary text"}):
            with self.subTest(value=value), self.assertRaises(PrecisionError):
                validate_precision(value)
        explicit = {"parameter_dtypes": ["float16", "uint8"], "autocast_enabled": False,
                    "autocast_dtype": None, "policy": "explicit"}
        self.assertEqual(validate_precision(explicit), explicit)

    def test_guard_requires_metadata_and_catches_in_place_drift_on_error(self):
        adapter = SimpleNamespace(metadata={})
        with self.assertRaises(PrecisionError):
            PrecisionGuard(adapter, required=True)
        PrecisionGuard(adapter).call(lambda: None)
        adapter.metadata["precision"] = passport()
        guard = PrecisionGuard(adapter, required=True)
        self.assertEqual(guard.call(lambda: "ok"), "ok")
        def change():
            adapter.metadata["precision"]["parameter_dtypes"].append("float16")
            raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(PrecisionError, "during worker"):
            guard.call(change)

    def test_workers_of_variant_must_keep_precision_for_all_phases(self):
        for phase in ("latency", "memory", "profile", "quality"):
            with self.subTest(phase=phase):
                tracker = PrecisionTracker("parameter")
                tracker.add("baseline", recorded())
                with self.assertRaisesRegex(PrecisionError, "between workers"):
                    tracker.add("baseline", {**recorded(passport("float16")), "phase": phase})

    def test_engineering_pair_checks_mode_not_policy_description(self):
        tracker = PrecisionTracker("engineering")
        tracker.add("baseline", recorded())
        tracker.add("candidate", recorded(passport(policy="explicit")))
        self.assertFalse(tracker.summary()["precision_changed"])
        for change in (passport("float16"), {**passport(), "parameter_dtypes": ["bfloat16"]},
                       {**passport(), "autocast_enabled": False, "autocast_dtype": None}):
            with self.subTest(change=change):
                tracker = PrecisionTracker("engineering")
                tracker.add("baseline", recorded())
                with self.assertRaisesRegex(PrecisionError, "Engineering"):
                    tracker.add("candidate", recorded(change))

    def test_parameter_and_quantization_record_explicit_differences(self):
        for kind in ("parameter", "quantization"):
            with self.subTest(kind=kind):
                tracker = PrecisionTracker(kind)
                tracker.add("baseline", recorded())
                tracker.add("candidate", recorded(passport("float16", "explicit")))
                self.assertTrue(tracker.summary()["precision_changed"])
                self.assertEqual(tracker.summary()["precision_status"], "recorded_consistent")

    def test_legacy_passports_are_not_invented_or_mixed_with_new(self):
        tracker = PrecisionTracker("engineering")
        tracker.add("baseline", {"metadata": {}})
        tracker.add("candidate", {"metadata": {}})
        self.assertEqual(tracker.summary(), {"precision_status": "historical_unrecorded"})
        for variant in ("baseline", "candidate"):
            with self.subTest(variant=variant), self.assertRaises(PrecisionError):
                tracker.add(variant, recorded())
        tracker = PrecisionTracker("parameter")
        tracker.add("baseline", recorded())
        with self.assertRaises(PrecisionError):
            tracker.add("candidate", {"metadata": {}})
        with self.assertRaises(PrecisionError):
            tracker.add("baseline", {"precision_policy_version": 1, "metadata": {}})

    def test_validated_precision_is_not_redacted_as_private_prose(self):
        from u1_benchmark.worker_v2 import redact_files
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            write(path, {"metadata": {"precision": passport(), "note": "float32"}})
            redact_files(tmp, ["float32", "bfloat16"])
            self.assertEqual(read(path)["metadata"], {"precision": passport(), "note": "[redacted-input]"})
            write(path, {"metadata": {"precision": {"invalid": "private payload"}}})
            redact_files(tmp, ["private payload"])
            self.assertEqual(read(path)["metadata"]["precision"], {"invalid": "[redacted-input]"})
            write(path, {"metadata": {"precision": {**passport(), "parameter_dtypes": ["private_payload"]}}})
            redact_files(tmp, ["private_payload"])
            self.assertEqual(read(path)["metadata"]["precision"]["parameter_dtypes"], ["[redacted-input]"])


class InputCompatibility(unittest.TestCase):
    def test_old_laya_indices_survive_execution_rule_changes_only(self):
        for version in (2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source.jsonl"
                datasets.jsonl(source, [{"id": str(i), "text": "input " + str(i)} for i in range(400)])
                fields = {"id": "id", "text": "text"}
                spec = {"kind": "speed", "source": str(source), "fields": fields,
                        "models": ["laya"], "version": "old-laya", "provenance": "unit-test"}
                if version == 3:
                    spec["schema_version"] = 3
                    preparer = ToyRolePreparer
                else:
                    from test_u1_handoff import ToyPreparer
                    preparer = ToyPreparer
                old = contracts.RULES["laya"].replace("native SDK autocast", "FP16 autocast")
                with patch.dict(contracts.RULES, {"laya": old}):
                    result = datasets.prepare(spec, root / "data", preparer)
                data, manifest = datasets.load_speed(result["path"], "laya", "screening", result["manifest_sha256"])
                self.assertEqual(len(data["B1"]), 400)
                self.assertIn("FP16 autocast", manifest["indices"]["laya"]["contract"]["rules"])
                self.assertEqual(sha(root / "data/manifest.json"), result["manifest_sha256"])
                if version == 3:
                    rows, _ = datasets.verified(result["path"])
                    contract = manifest["indices"]["laya"]["contract"]
                    for key in set(contract) - {"rules"}:
                        changed = copy.deepcopy(manifest)
                        changed["indices"]["laya"]["contract"][key] = {"changed": contract[key]}
                        with self.subTest(field=key), self.assertRaisesRegex(ValueError, "Stale role"):
                            role_data.load_speed(root / "data", "laya", "screening", changed, rows)

    def test_fetch_compares_preparation_not_runtime_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture_data(root)
            release = {"manifest_sha256": data["manifest_sha256"], "revision": None}
            changed = {**contracts.model_contract("gliner", 3), "rules": "execution rules changed"}
            with patch.object(distribution, "read", return_value=release), \
                    patch.object(distribution, "MODEL_SPECS", {"gliner": {}}), \
                    patch.object(distribution, "model_contract", return_value=changed):
                result = distribution.fetch(root / "installed", source=data["path"])
            self.assertEqual(result["model_contract_status"], {"gliner": "current"})

    def test_reindex_does_not_rebuild_other_models_for_precision_only(self):
        from tools.reindex_role_model import rebuild
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            datasets.jsonl(source, [{"id": str(i), "text": "input " + str(i)} for i in range(400)])
            spec = {"schema_version": 3, "kind": "speed", "source": str(source),
                    "fields": {"id": "id", "text": "text"}, "models": ["laya", "gliner"],
                    "version": "old", "provenance": "unit-test"}
            old = contracts.RULES["laya"].replace("native SDK autocast", "FP16 autocast")
            with patch.dict(contracts.RULES, {"laya": old}):
                datasets.prepare(spec, root / "base", ToyRolePreparer)
            result = rebuild(root / "base", root / "new", ["gliner"], "new", ToyRolePreparer)
            self.assertEqual(result["status"], "ready")
            for name in ("corpus.jsonl", "indices/laya.jsonl", "buckets/laya/B1.json"):
                self.assertEqual((root / "base" / name).read_bytes(), (root / "new" / name).read_bytes())


class WorkerAndReport(unittest.TestCase):
    def test_new_external_laya_requires_passport_and_quality_drift_is_fatal(self):
        import torch
        from u1_benchmark.adapters import load
        from u1_benchmark.worker_v2 import execute

        for case in ("missing", "stable", "drift"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "quality.jsonl"
                datasets.jsonl(source, [{"id": "q", "messages": [{"role": "user", "content": "library"}],
                                        "target": "good", "target_role": "user"}])
                data = datasets.prepare({"schema_version": 3, "source": str(source), "kind": "quality",
                    "fields": {k: k for k in ("id", "messages", "target", "target_role")},
                    "version": "fixture", "provenance": "unit-test"}, root / "data")
                spec = {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}}
                cfg = contracts.parse_config({"schema_version": 3, "model": "laya", "mode": "quality",
                    "baseline": spec, "environment": str(root / "env"),
                    "quality_data": {k: data[k] for k in ("path", "manifest_sha256")},
                    "label_mapping": {"labels": ["good", "bad"], "targets": {"good": "good"},
                                      "verdicts": {"safe": "good", "unsafe": "bad"}}})
                adapter = load("laya", spec, root, 3)
                if case != "missing":
                    adapter.metadata["precision"] = passport(policy="explicit")
                if case == "drift":
                    verify = adapter.verify
                    def drifting(*args):
                        value = verify(*args)
                        adapter.metadata["precision"]["autocast_dtype"] = "float16"
                        return value
                    adapter.verify = drifting
                write(root / "job.json", {"config": cfg, "directory": str(root / "worker"),
                    "variant": "baseline", "phase": "quality", "inputs_sha256": "fixture", "private": True})
                with patch("u1_benchmark.environment.probe"), \
                        patch("u1_benchmark.worker.telemetry", return_value={"uuid": "fixture"}), \
                        patch("u1_benchmark.adapters.load", return_value=adapter), patch.object(torch.cuda, "synchronize"):
                    outcome = execute(root / "job.json")
                self.assertEqual(outcome["precision_policy_version"], 1)
                if case == "stable":
                    self.assertEqual(outcome["status"], "passed", outcome)
                    self.assertEqual(outcome["quality"]["overall"]["coverage"], 1)
                else:
                    self.assertEqual(outcome["status"], "failed", outcome)
                    self.assertEqual(outcome["error_type"], "PrecisionError")
                    self.assertFalse(outcome["quality_evaluated"])

    def baseline(self, root, invoke):
        data = fixture_data(root)
        env = root / "env"
        write(env / "setup/status.json", {"status": "passed"})
        write(env / "setup/lock.json", read(ROOT / "env/team_rtx.lock.json"))
        write(env / "setup/profile.json", {"profile": "team-rtx"})
        spec = {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}}
        cfg = {"schema_version": 3, "model": "gliner", "baseline": spec, "environment": str(env),
               "data": {k: data[k] for k in ("path", "manifest_sha256")}}
        with patch.object(runner_v2, "invoke", side_effect=invoke):
            return runner_v2.run(cfg, root / "out")

    @staticmethod
    def recorded_worker(args, *rest):
        simulated_role_worker(args[-1])
        job = read(args[-1])
        path = Path(job["directory"]) / "result.json"
        value = read(path)
        value.update(precision_policy_version=1)
        value["metadata"]["precision"] = passport()
        write(path, value)

    def test_runner_stops_on_memory_drift_and_offline_report_rejects_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.baseline(root, self.recorded_worker)
            self.assertEqual(result["status"], "fixture_complete", result)
            self.assertEqual(result["precision_status"], "recorded_consistent")
            path = root / "out/memory/baseline_1/result.json"
            value = read(path)
            value["metadata"]["precision"] = passport("float16")
            write(path, value)
            rebuilt = report_v2.build(root / "out")
            self.assertEqual(rebuilt["status"], "incomplete")
            self.assertIn("precision differs between workers", rebuilt["errors"][0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def drift(args, *rest):
                self.recorded_worker(args)
                job = read(args[-1])
                if job["phase"] == "memory":
                    path = Path(job["directory"]) / "result.json"
                    value = read(path)
                    value["metadata"]["precision"] = passport("float16")
                    write(path, value)
            result = self.baseline(root, drift)
            self.assertEqual(result["status"], "incomplete", result)
            self.assertEqual(read(root / "out/status.json")["error_type"], "PrecisionError")
            self.assertFalse((root / "out/memory/baseline_1").exists())
            self.assertTrue((root / "out.zip").is_file())

    def test_worker_inference_contract_remains_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.baseline(root, self.recorded_worker)
            path = root / "out/latency/000/result.json"
            value = read(path)
            value["contract"]["rules"] = "altered scoring"
            write(path, value)
            result = report_v2.build(root / "out")
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("Inference contract changed", result["errors"])

    def test_table_separates_precision_and_historical_report_keeps_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.baseline(root, self.recorded_worker)
            expected = result["buckets"]
            groups = []
            for dtype in ("float16", "bfloat16"):
                path = root / "out/preflight/baseline/result.json"
                value = read(path)
                value["metadata"]["precision"] = passport(dtype)
                write(path, value)
                # Table projection only; these deliberately fake CPU timings cannot gain GPU approval.
                with patch.object(report_v2, "build", return_value={**result, "status": "baseline_complete"}):
                    report_v2.table([root / "out"], root / "table.json")
                row = read(root / "table.json")["rows"][0]
                groups.append(row["comparison_group"])
                self.assertEqual(row["precision"]["autocast_dtype"], dtype)
            self.assertNotEqual(*groups)
            for path in (root / "out").glob("**/result.json"):
                value = read(path)
                value.pop("precision_policy_version", None)
                value.get("metadata", {}).pop("precision", None)
                write(path, value)
            rebuilt = report_v2.build(root / "out")
            self.assertEqual(rebuilt["buckets"], expected)
            self.assertEqual(rebuilt["status"], "fixture_complete")


if __name__ == "__main__":
    unittest.main()
