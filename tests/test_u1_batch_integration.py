"""Real runner/process/archive integration with explicitly simulated CPU workers.

These tests exercise orchestration and evidence validation, NOT batched GPU
adapters. Latency and trace events are synthetic, memory is unavailable, and
every worker is marked test_fixture so no GPU acceptance may be issued.
"""
import gzip
import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from u1_benchmark import contracts, datasets, process, quality, runner_v2
from u1_benchmark.adapters import load
from u1_benchmark.batching import materialize
from u1_benchmark.core import ROOT, digest, lines, read, sha, write


EXTERNAL = ROOT / "examples/u1_external_adapter.py"


def fixture_spec(change=False):
    return {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL),
            "dependencies": {}, "options": {"change_verdict": change}}


class InputFixture:
    def __init__(self, model):
        self.name, self.assets, self.versions = model, {"test_fixture": True}, {}
        self.adapter = load(model, fixture_spec(), ".")

    def parity(self):
        return {"status": "passed", "scope": "CPU input fixture only"}

    def index(self, row):
        value = self.adapter.index(row["text"])
        return {**value, "id": row["id"], "bucket": "B1", "supported": True}


def make_inputs(root):
    source = root / "source.jsonl"
    datasets.jsonl(source, [{"id": str(i), "text": ("change " if i < 4 else "keep ") + str(i)}
                           for i in range(400)])
    result = datasets.prepare({"source": str(source), "kind": "speed",
                               "fields": {"id": "id", "text": "text"},
                               "version": "batch-cpu-fixture", "models": ["gliner"],
                               "provenance": "unit-test"}, root / "data", InputFixture)
    return {key: result[key] for key in ("path", "manifest_sha256")}


def simulated_batch_worker(job_path, hang=False):
    """Called in a fresh interpreter. Never import torch or touch a GPU."""
    job = read(job_path)
    cfg, phase, variant = job["config"], job["phase"], job["variant"]
    directory = Path(job["directory"])
    adapter = load(cfg["model"], cfg[variant], directory)
    adapter.metadata.update(batch_policy={"padding": "longest", "padding_side": "right",
                                          "truncation": False, "sort_by_length": False,
                                          "pad_token_id": 0, "test_fixture": True},
                            synthetic_timings=True)
    runtime = {key: "cpu_fixture" for key in
               ("python", "torch", "cuda", "cudnn", "gpu", "seed", "cpu_threads", "tf32", "hub_kernels")}
    write(directory / "runtime.json", runtime)
    write(directory / "packages.json", {})
    write(directory / "telemetry_before.json", {"uuid": "cpu_fixture", "driver": "none"})
    measurement = job.get("measurement")
    result = {"schema_version": cfg["schema_version"], "status": "running", "phase": phase,
              "variant": variant, "config_sha256": digest(cfg), "contract": adapter.contract,
              "adapter_spec": cfg[variant], "inputs_sha256": job["inputs_sha256"],
              "metadata": adapter.metadata, "pid": os.getpid(), "size": adapter.size(),
              "job": measurement, "quality_evaluated": False, "setup_seconds": 0.0}
    write(directory / "result.json", result)
    if hang:
        time.sleep(30)
        raise AssertionError("Timeout test failed to terminate its child")
    try:
        if phase == "quality":
            assert measurement is None and "batch_protocol_version" not in job
            q = cfg["quality_data"]
            rows, _ = datasets.verified(q["path"], q["manifest_sha256"])
            predictions = [{"id": row["id"], "status": "passed", "score": adapter.request(row["text"])}
                           for row in rows]
            datasets.jsonl(directory / "predictions.jsonl", predictions)
            result.update(quality_batch_size=1, quality_evaluated=True,
                          quality=quality.metrics(rows, predictions, cfg["label_mapping"]))
        else:
            assert job["batch_protocol_version"] == 1
            data, _ = datasets.load_speed(job["dataset"], cfg["model"], cfg["mode"],
                                         cfg["data"]["manifest_sha256"])
            rows = data[measurement["bucket"]]
            plans = measurement["batches"]
            members = [materialize(plan, rows) for plan in plans]
            result.update(batch_protocol_version=1, sources_sha256=job["sources_sha256"],
                          bucket=measurement["bucket"], batch_size=measurement["batch_size"],
                          prepared_shapes=sorted({(measurement["batch_size"], p["padded_length"]) for p in plans}),
                          prepared_compile_count=0, final_compile_count=0,
                          load_seconds=0.0, backend_preparation_seconds=0.0, warmup_seconds=0.0)
            if phase != "batch_preflight":
                from u1_benchmark.batch_worker import verify_evidence
                verify_evidence(job, directory, adapter)
            elapsed = 60.0 if variant == "candidate" else 100.0

            def records(diagnostic=False):
                output = []
                for i, (plan, group) in enumerate(zip(plans, members, strict=True)):
                    # This is a fixture simulation, not a claimed batch runtime.
                    answers = [adapter.verify(row["text"], row["index"]) for row in group]
                    if diagnostic:
                        answers = [{"label": "diagnostic-only"} for _ in group]
                    output.append({"batch": plan, "scores": answers, "latency_ms": elapsed,
                                   "started_unix": 1000.0 + i, "finished_unix": 1000.0 + i + elapsed / 1000})
                return output

            memory = {"test_fixture": True, "windows": {window: {"status": "unavailable", "reason": "CPU fixture"}
                                                         for window in ("setup", "inference")}}
            if phase == "batch_preflight":
                result.update(aba_passed=True, boundary={"tokens": 512, "model_limit": 512, "over_limit_rejected": True},
                              verified_batches=records(diagnostic=True), timings=records(diagnostic=True), memory=memory)
            elif phase == "batch_latency":
                result["rows"] = records()
                datasets.jsonl(directory / "latencies.jsonl", result["rows"])
            elif phase == "batch_memory":
                result.update(memory=memory, torch_setup={"allocated": 0, "reserved": 0},
                              torch_inference={"allocated": 0, "reserved": 0})
                write(directory / "memory.json", memory)
            elif phase == "batch_profile":
                for label in ("short", "long"):
                    with gzip.open(directory / (label + ".trace.json.gz"), "wt") as stream:
                        json.dump({"test_fixture": True, "traceEvents": [{"cat": "kernel", "name": "synthetic_CPU_fixture_not_GPU"}]}, stream)
            else:
                raise AssertionError("Unexpected phase: " + phase)
        result["status"] = "passed"
        write(directory / "result.json", result)
    finally:
        adapter.close()


class BatchIntegrationTests(unittest.TestCase):
    def setup_run(self, root, *, mode="pilot", sizes=(1, 8), comparison="engineering", candidate=True, flip=False):
        env = root / "env"
        write(env / "setup/status.json", {"status": "passed"})
        write(env / "setup/lock.json", read(ROOT / "env/team_environment.lock.json"))
        cfg = {"schema_version": 2, "model": "gliner", "baseline": fixture_spec(),
               "mode": mode, "comparison": comparison, "environment": str(env),
               "data": make_inputs(root), "batch_sizes_by_bucket": {"B1": list(sizes)}}
        if candidate:
            cfg["candidate"] = fixture_spec(flip)
            if comparison == "quantization":
                artifact = root / "fixture.weights"
                artifact.write_bytes(b"CPU interface fixture, not quantized weights")
                spec = contracts.MODEL_SPECS["gliner"]
                cfg["candidate"]["artifact"] = {"path": str(artifact),
                    "sha256": digest(contracts.file_inventory(artifact)), "base_checkpoint": spec["repo"],
                    "base_revision": spec["revision"], "method": "test_fixture", "bits": 4, "runtime": "cpu_fixture"}
        return cfg

    def run_children(self, cfg, output, *, timeout_first_latency=False):
        def invoke(args, log, remaining):
            self.assertGreater(remaining, 0)
            self.assertIn("_worker_v2", [str(arg) for arg in args])
            job = read(args[-1])
            hang = timeout_first_latency and job["phase"] == "batch_latency"
            code = ("import sys; sys.path.insert(0, 'tests'); "
                    "from test_u1_batch_integration import simulated_batch_worker; "
                    "simulated_batch_worker(sys.argv[1], hang=sys.argv[2] == '1')")
            process.invoke([sys.executable, "-B", "-c", code, args[-1], "1" if hang else "0"],
                           log, min(1 if hang else 30, remaining))
        with patch.object(runner_v2, "invoke", side_effect=invoke):
            return runner_v2.run(cfg, output)

    def check_artifacts(self, output, result, *, complete=True):
        from u1_benchmark.report_v2 import build
        self.assertEqual(build(output), result)
        self.assertNotEqual(result["status"], "screening_pass")
        self.assertFalse(result["all_cells_faster"])
        self.assertFalse(result["all_cells_meet_threshold"])
        if complete:
            diagnostics = {"summary": result}
            if result["status"] != "fixture_complete":
                diagnostics["state"] = read(output / "status.json")
                diagnostics["logs"] = {str(path.relative_to(output)): path.read_text()
                                       for path in output.rglob("stdout.log") if path.stat().st_size}
                error = output / ".private/error.txt"
                if error.exists():
                    diagnostics["error"] = error.read_text()
            self.assertEqual(result["status"], "fixture_complete", diagnostics)
            self.assertFalse(result["gpu_acceptance"])
            self.assertEqual(result["acceptance"], "interface_test_only")
        outcomes = [read(path) for path in output.rglob("result.json")]
        pids = [value["pid"] for value in outcomes]
        self.assertEqual(len(pids), len(set(pids)))
        self.assertNotIn(os.getpid(), pids)
        for path in output.rglob("stdout.process.json"):
            worker = read(path.parent / "result.json")
            self.assertEqual(read(path)["pid"], worker["pid"])
        for name, checksum in read(output / "source_files.json").items():
            self.assertEqual(sha(output / name), checksum)
        self.assertTrue((output / "sources/u1_benchmark/batching.py").is_file())
        self.assertTrue((output / "sources/u1_benchmark/batch_worker.py").is_file())
        with zipfile.ZipFile(str(output) + ".zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(json.loads(archive.read("summary.json")), result)
            self.assertIn("REPORT.md", archive.namelist())
            self.assertFalse(any(".private" in name or name == "corpus.jsonl" for name in archive.namelist()))
            self.assertFalse(any(b"change 0" in archive.read(name) for name in archive.namelist()
                                 if name.endswith((".json", ".jsonl", ".log"))))

    def test_two_cell_pilot_quality_is_single_and_preflight_is_not_instability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_run(root)
            source = root / "quality-source.jsonl"
            datasets.jsonl(source, [{"id": "q1", "text": "keep private quality text", "target": "good"},
                                   {"id": "q2", "text": "change private quality text", "target": "bad"}])
            q = datasets.prepare({"source": str(source), "kind": "quality", "version": "batch-quality-fixture",
                                  "provenance": "unit-test", "fields": {key: key for key in ("id", "text", "target")}}, root / "quality-data")
            cfg["quality_data"] = {key: q[key] for key in ("path", "manifest_sha256")}
            cfg["label_mapping"] = {"labels": ["0", "1"], "targets": {"good": "0", "bad": "1"},
                                    "verdicts": {"safe": "0", "unsafe": "1"}, "positive_label": "1"}
            output = root / "pilot.v1"
            result = self.run_children(cfg, output)
            self.check_artifacts(output, result)
            self.assertTrue(result["quality_evaluated"])
            self.assertEqual(result["quality_batch_size"], 1)
            self.assertEqual(result["quality_status"], "evaluated_batch_1_only")
            self.assertIsNone(result["quality_preserved"])
            self.assertEqual(len(list((output / "quality").glob("*/result.json"))), 2)
            jobs = read(output / "schedule.json")
            self.assertEqual(len(jobs), 4)
            self.assertEqual({job["phase"] for job in jobs}, {"ab"})
            self.assertTrue(all(len(job["batches"]) == 12 for job in jobs))
            preflights = list((output / "batches").glob("*/*/preflight/*/result.json"))
            self.assertEqual(len(preflights), 4)
            for item in result["cells"].values():
                self.assertEqual(item["status"], "pilot_complete")
                self.assertEqual(item["verdicts"]["unstable_ids"], [])
                self.assertEqual(item["verdicts"]["changed_count"], 0)
            self.assertFalse(list(output.glob("batches/*/*/memory/*")))
            self.assertFalse(list(output.glob("batches/*/*/profiles/*")))
            self.assertIn("B1/8", result["single_to_batch"])
            with zipfile.ZipFile(str(output) + ".zip") as archive:
                self.assertFalse(any(b"private quality text" in archive.read(name)
                                     for name in archive.namelist() if name.endswith((".json", ".jsonl", ".log"))))

    def test_baseline_screening_all_stages_and_table_excludes_fixture(self):
        from u1_benchmark.report_v2 import table
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_run(root, mode="screening", sizes=(1,), candidate=False)
            output = root / "baseline"
            result = self.run_children(cfg, output)
            self.check_artifacts(output, result)
            item = result["cells"]["B1/1"]
            self.assertEqual(item["status"], "baseline_complete")
            self.assertEqual(item["p95"]["baseline_ms"], 100)
            self.assertEqual(item["verdicts"]["n"], 400)
            self.assertEqual(len(item["memory"]), 3)
            self.assertEqual(item["memory_status"], "partially_unavailable")
            self.assertEqual(Counter(job["phase"] for job in read(output / "schedule.json")), {"aa": 4, "baseline": 3})
            self.assertEqual(table([Path(str(output) + ".zip")], root / "table.json")["rows"], 0)

    def test_engineering_screening_rejects_four_unique_flips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_run(root, mode="screening", sizes=(1,), flip=True)
            output = root / "engineering"
            result = self.run_children(cfg, output)
            self.check_artifacts(output, result)
            item = result["cells"]["B1/1"]
            self.assertEqual(item["status"], "rejected")
            self.assertEqual(item["verdicts"]["n"], 400)
            self.assertEqual(item["verdicts"]["changed_count"], 4)
            self.assertEqual(set(item["verdicts"]["changed_ids"]), {"0", "1", "2", "3"})
            self.assertEqual(item["verdicts"]["unstable_ids"], [])

    def test_quantization_screening_unique_denominator_and_tamper_detection(self):
        from u1_benchmark.report_v2 import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_run(root, mode="screening", sizes=(1,), comparison="quantization", flip=True)
            output = root / "quantization"
            result = self.run_children(cfg, output)
            self.check_artifacts(output, result)
            item = result["cells"]["B1/1"]
            self.assertEqual(item["status"], "quantization_measured")
            self.assertEqual(item["acceptance"], "not_assessed_without_quality_rule")
            self.assertEqual(item["verdicts"]["changed_count"], 4)
            self.assertEqual(item["verdicts"]["changed_fraction"], .01)
            self.assertEqual(item["verdicts"]["directions"], [{"baseline": "safe", "candidate": "unsafe", "count": 4}])
            self.assertEqual(result["macro_verdict_change_fraction_by_batch_size"], {"1": .01})
            self.assertFalse(result["quality_evaluated"])
            raw = output / "latency/000/latencies.jsonl"
            before = lines(raw)
            datasets.jsonl(raw, before[:-1])
            broken = build(output)
            self.assertEqual(broken["status"], "incomplete")
            self.assertFalse(broken["all_cells_faster"])
            datasets.jsonl(raw, before)
            self.assertEqual(build(output), result)
            archived = output / "sources/u1_benchmark/batching.py"
            archived.write_text(archived.read_text() + "\n# altered fixture snapshot\n")
            self.assertIn("archived_source_hash_mismatch", build(output)["errors"])

    def test_timeout_kills_child_and_preserves_completed_preflights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_run(root)
            output = root / "timed-out"
            result = self.run_children(cfg, output, timeout_first_latency=True)
            self.check_artifacts(output, result, complete=False)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(read(output / "status.json")["status"], "timeout")
            matrix = read(output / "matrix_status.json")
            self.assertEqual(matrix["B1/1"]["status"], "timeout")
            self.assertEqual(matrix["B1/8"]["status"], "preflight_passed")
            process_state = read(output / "latency/000/stdout.process.json")
            self.assertEqual(process_state["status"], "timeout")
            self.assertIsNotNone(process_state["returncode"])
            self.assertFalse((output / "latency/001").exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(process_state["pid"], 0)


if __name__ == "__main__":
    unittest.main()
