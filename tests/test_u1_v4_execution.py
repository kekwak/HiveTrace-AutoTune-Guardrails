"""CPU integration contracts for v4 orchestration, workers and frozen evidence.

No real runtime, subprocess, model or GPU is used. Runner tests replace only
dataset loading and the independent report builder, not scheduling/archiving.
"""
from copy import deepcopy
import importlib
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from u1_benchmark.core import ROOT, digest, read, sha, write
from u1_benchmark import v4_config as config
from u1_benchmark import v4_evidence as evidence
from u1_benchmark import v4_planning as planning
from u1_benchmark import v4_privacy as privacy
from u1_benchmark import v4_runner as runner
from u1_benchmark import v4_worker as worker


SECRET = "PRIVATE-EXECUTION-INPUT-7f34"
PRIVATE_PROMPT = "CONFIDENTIAL-INLINE-POLICY-93b2"
PRECISION = {"parameter_dtypes": ["float32"], "autocast_enabled": False,
             "autocast_dtype": None, "policy": "explicit"}
RUNTIME = {"gpu": "CPU fixture only", "gpu_uuid": "GPU-fake", "driver": "fake",
           "python": "fixture", "monitor_method": config.MONITOR_VERSION,
           "pid": 123, "host_sha256": "a" * 64, "timestamp": 1}
MAPPING = {"labels": ["safe", "unsafe"], "targets": {"safe": "safe", "unsafe": "unsafe"},
           "verdicts": {"safe": "safe", "unsafe": "unsafe"}, "positive_label": "unsafe"}


class CPUAdapter:
    """Inline external-adapter fixture with observable loads and requests."""
    def __init__(self, spec, directory):
        self.spec, self.directory = deepcopy(spec), Path(directory)
        self.limit = spec["preparation"]["limit"]
        self.compile_count = 0
        self.requests, self.verified, self.prepared = [], [], []
        self.fail_at = None
        self.error = RuntimeError(SECRET)
        self.closed = False
        self.metadata = {"precision": deepcopy(PRECISION), "backend": spec["execution"]["runtime"],
                         "backend_proof": {"status": "passed", "implementation": "cpu_fixture"},
                         "debug_private": SECRET}

    def index(self, value):
        messages = [{"role": "user", "content": value}] if isinstance(value, str) else value
        return {"total_tokens": sum(len(m["content"].split()) for m in messages),
                "input_sha256": digest(messages)}

    def prepare_batch_backend(self, groups):
        self.prepared.extend(deepcopy(groups))

    def prove_batch_backend(self, group):
        return deepcopy(self.metadata["backend_proof"])

    def request_batch(self, group):
        self.requests.append(deepcopy(group))
        if self.fail_at == len(self.requests):
            raise self.error
        return [{"label": "safe"} for _ in group]

    def verify_batch(self, group, expected):
        if [self.index(value) for value in group] != expected:
            raise ValueError("Fixture index mismatch")
        self.verified.append(deepcopy(group))
        return self.request_batch(group)

    def boundary(self, length):
        return "edge " * length, "above " * (self.limit + 1)

    def reject_batch_over_limit(self, group):
        if not any(self.index(value)["total_tokens"] > self.limit for value in group):
            raise ValueError("Fixture expected an over-limit input")

    def size(self):
        return {"parameters": 4, "bytes": 16}

    def close(self):
        self.closed = True


def create(spec, directory):
    """Explicit external adapter factory, also usable by load_adapter."""
    return CPUAdapter(spec, directory)


class FakeSampler:
    def __init__(self, gpu_uuid):
        self.uuid, self.events = gpu_uuid, []

    def start(self):
        self.events.append("start")

    def inference(self):
        self.events.append("inference")

    def stop(self, path):
        self.events.append("stop")
        result = {"status": "measured", "method_id": config.MONITOR_VERSION,
                  "method_version": 2, "requested_interval_seconds": .1,
                  "windows": {phase: {"status": "measured", "sampled_gpu_peak_mib": 10,
                                      "valid_gpu_samples": 1}
                              for phase in ("setup", "inference")}}
        write(path, result)
        return result


def variant(runtime="eager", module=None):
    source = {"repo": "fixture/explicit-model", "revision": "a" * 40}
    module = Path(__file__ if module is None else module).resolve()
    return {"adapter": {"module": str(module),
                        "source_sha256": sha(module), "dependencies": {}},
            "weights": dict(source), "tokenizer": dict(source), "template": None,
            "preparation": {"limit": 512}, "scoring": {},
            "execution": {"runtime": runtime, "dtype": "float32", "attention": "eager",
                          "sdpa_backend": None, "linear_attention": "reference"}}


def input_rows(n=5):
    adapter = CPUAdapter(variant(), ".")
    result = []
    for i in range(n):
        messages = [{"role": "user", "content": f"{SECRET}-{i} " + "word " * (i % 8)}]
        result.append({"id": f"input-{i:04d}", "messages": messages,
                       "index": adapter.index(messages), "target": "safe",
                       "slices": {"role": "user", "language": "en"}})
    return result


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="u1-v4-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.external_module = self.root / "cpu_adapter.py"
        self.external_module.write_text(
            "from test_u1_v4_execution import CPUAdapter\n\n"
            "def create(spec, directory):\n"
            "    return CPUAdapter(spec, directory)\n")
        self.environment = self.root / "environment"
        python = self.environment / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("CPU fixture: never executed\n")
        self.adapters, self.samplers, self.launched = [], [], []
        self.adapter_hook = None
        self.counter = 0
        self.inventory = {"u1_benchmark/v4_config.py": sha(ROOT / "u1_benchmark/v4_config.py")}
        self.runtime = Mock(return_value=deepcopy(RUNTIME))
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            reset_peak_memory_stats=Mock(), synchronize=Mock(),
            max_memory_allocated=Mock(return_value=16), max_memory_reserved=Mock(return_value=32)))
        # Fail closed if an injection is accidentally lost. This never spawns.
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("Real subprocess forbidden")))
        self.enterContext(patch.object(worker, "configure_gpu", side_effect=AssertionError("Real GPU forbidden")))
        self.enterContext(patch.object(evidence, "source_inventory", return_value=self.inventory))
        self.enterContext(patch.object(runner, "source_inventory", return_value=self.inventory))

    def cfg(self, mode="baseline", pilot=True):
        value = {"schema_version": 4, "baseline": variant(module=self.external_module), "mode": mode,
                 "pilot": pilot, "seed": 17, "environment": str(self.environment),
                 "timeout_seconds": 60, "buckets": {"custom": {"min": 1, "max": 512}},
                 "batch_sizes_by_bucket": {"custom": [3]}, "data": {"path": "fixture-only", "manifest_sha256": "b" * 64},
                 "quality_batch_sizes": [3], "label_mapping": deepcopy(MAPPING)}
        if mode in ("search", "confirm"):
            value["candidate"] = variant("compile", module=self.external_module)
        if mode == "search":
            value["baseline_ref"] = str(self.root / "reference")
        if mode == "quality":
            value["pilot"] = False
            value.pop("data")
            value.pop("buckets")
            value.pop("batch_sizes_by_bucket")
            value["quality_data"] = {"path": "quality-fixture", "manifest_sha256": "c" * 64}
        return config.validate_config(value)

    def factory(self, spec, directory):
        adapter = CPUAdapter(spec, directory)
        if self.adapter_hook:
            self.adapter_hook(adapter)
        self.adapters.append(adapter)
        return adapter

    def sampler(self, gpu_uuid):
        value = FakeSampler(gpu_uuid)
        self.samplers.append(value)
        return value

    def execute(self, path):
        return worker.execute(path, adapter_factory=self.factory, runtime_factory=self.runtime,
                              torch_module=self.torch, sampler_factory=self.sampler)

    def worker_job(self, stage="latency", source=None):
        source = input_rows() if source is None else source
        directory = self.root / f"worker-{self.counter}"
        self.counter += 1
        directory.mkdir()
        cfg = self.cfg("quality" if stage == "quality" else "baseline")
        resolved = {"schema_version": 4, "result_version": 4, "protocol_version": config.PROTOCOL_VERSION,
                    "config": cfg, "source_inventory": self.inventory, "environment": RUNTIME}
        resolved_path = directory / "resolved_run.json"
        checksum = evidence.immutable(resolved_path, resolved)
        input_path = directory / ".private/inputs.json"
        write(input_path, source)
        plans = planning.smoke_plans(source, min(3, len(source)))
        job = {"stage": stage, "variant": "baseline", "seed": 17, "batch_size": 3,
               "bucket": "custom", "bucket_max": 512, "batches": plans,
               "plan_sha256": digest(plans), "quality_batch_size": 3,
               "label_mapping": MAPPING, "resolved_path": str(resolved_path),
               "resolved_sha256": checksum, "input_path": str(input_path),
               "input_file_sha256": sha(input_path)}
        proof_path = directory / "preflight.json"
        write(proof_path, {"status": "passed", "result_version": 4, "resolved_sha256": checksum,
                           "job": {**deepcopy(job), "stage": "preflight"},
                           "metadata": {"precision": PRECISION}, "backend_proof": {"status": "passed"}})
        job.update(preflight_path=str(proof_path), preflight_sha256=sha(proof_path))
        path = directory / "job.json"
        write(path, job)
        return path

    def private_config_job(self, stage="preflight", source=None):
        path = self.worker_job(stage, source)
        job = read(path)
        resolved = read(job["resolved_path"])
        full = resolved["config"]
        full["baseline"]["template"] = {"text": PRIVATE_PROMPT}
        private_path = path.parent / ".private/config.json"
        write(private_path, full)
        resolved.update(config=privacy.public_config(full), config_sha256=digest(full),
                        variant_identities={"baseline": digest(full["baseline"])})
        write(job["resolved_path"], resolved)
        job.update(resolved_sha256=sha(job["resolved_path"]), private_config_path=str(private_path),
                   private_config_sha256=sha(private_path))
        proof = read(job["preflight_path"])
        proof["resolved_sha256"] = job["resolved_sha256"]
        write(job["preflight_path"], proof)
        job["preflight_sha256"] = sha(job["preflight_path"])
        write(path, job)
        return path

    def launcher(self, args, log, timeout):
        self.assertGreater(timeout, 0)
        args = [str(arg) for arg in args]
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(SECRET + " private subprocess diagnostics")
        if "_runtime_v4" in args:
            write(args[args.index("--output") + 1], RUNTIME)
            return
        self.assertIn("_worker_v4", args)
        path = Path(args[args.index("--job") + 1])
        self.launched.append(read(path))
        self.execute(path)

    def fake_report(self, directory, save=True):
        cfg = read(Path(directory) / "resolved_run.json")["config"]
        value = {"status": "baseline_complete" if cfg["mode"] == "baseline" else "passed"}
        if save:
            write(Path(directory) / "summary.json", value)
        return value

    def report_stub(self):
        # Report validation is owned by a separate test suite. Keep imports
        # independent while that module is concurrently being implemented.
        return patch.dict("sys.modules", {"u1_benchmark.v4_report": SimpleNamespace(build=self.fake_report)})

    def runner_inputs(self, cfg, source=None):
        source = input_rows() if source is None else source
        if cfg["mode"] == "quality":
            return source, {"kind": "quality"}, {}
        data = {"custom": source}
        return data, {"kind": "speed"}, runner.input_indices(data)

    def reference(self, source=None):
        source = input_rows(400) if source is None else source
        cfg = self.cfg("baseline", pilot=False)
        data, manifest, indices = self.runner_inputs(cfg, source)
        root = self.root / "reference"
        root.mkdir()
        resolved = {"schema_version": 4, "result_version": 4, "protocol_version": config.PROTOCOL_VERSION,
                    "config": cfg, "source_inventory": self.inventory, "environment": deepcopy(RUNTIME),
                    "input_indices": indices, "data_manifest_sha256": digest(manifest),
                    "schedule": planning.schedule(cfg, data)}
        checksum = evidence.immutable(root / "resolved_run.json", resolved)
        write(root / "status.json", {"status": "baseline_complete", "resolved_sha256": checksum})
        return root

    def archive_text(self, directory):
        with zipfile.ZipFile(Path(directory).with_suffix(".zip")) as archive:
            self.assertFalse(any(part.startswith(".") for name in archive.namelist() for part in Path(name).parts))
            return "\n".join(archive.read(name).decode() for name in archive.namelist())


class ConfigurationTests(Fixture):
    def test_halo_cue_is_hashed_in_public_config_and_remains_private(self):
        cfg = self.cfg()
        cfg["baseline"]["adapter"] = "halo"
        cfg["baseline"]["preparation"]["cue"] = PRIVATE_PROMPT
        original = deepcopy(cfg)
        projected = privacy.public_config(cfg)
        self.assertEqual(projected["baseline"]["preparation"]["cue"],
                         "[private-sha256:" + digest(PRIVATE_PROMPT) + "]")
        self.assertFalse(PRIVATE_PROMPT in json.dumps(projected))
        self.assertIn(PRIVATE_PROMPT, privacy.private_values([], cfg))
        self.assertEqual(cfg, original)
        cfg["baseline"]["preparation"]["cue"] += " changed"
        self.assertNotEqual(privacy.public_config(cfg), projected)

    def test_pilot_rejected_for_nonspeed_modes(self):
        for mode in ("quality", "smoke"):
            cfg = self.cfg(mode, pilot=False)
            cfg["pilot"] = True
            with self.subTest(mode=mode), self.assertRaisesRegex(config.ConfigurationError, "Pilot applies only"):
                config.validate_config(cfg)

    def test_old_formats_and_incomplete_candidates_rejected(self):
        for old in ({"schema_version": 3}, {"preset": "gliner_compile"}, {"model": "halo"}):
            with self.subTest(old=old), self.assertRaises(ValueError):
                config.validate_config(old)
        cfg = self.cfg("confirm")
        cfg["candidate"] = {"execution": {"runtime": "compile"}}
        with self.assertRaises(ValueError):
            config.validate_config(cfg)

    def test_search_requires_reference_and_no_inheritance(self):
        cfg = self.cfg("search")
        original = deepcopy(cfg)
        resolved = config.validate_config(cfg)
        resolved["candidate"]["preparation"]["limit"] = 1
        self.assertEqual(cfg, original)
        cfg.pop("baseline_ref")
        with self.assertRaises(ValueError):
            config.validate_config(cfg)

    def test_comparison_preserves_preparation_and_engineering_precision(self):
        for path, value in ((("preparation", "limit"), 128), (("scoring", "threshold"), .9),
                            (("weights", "revision"), "b" * 40), (("execution", "dtype"), "float16")):
            cfg = self.cfg("confirm")
            cfg["candidate"][path[0]][path[1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                config.validate_config(cfg)

    def test_builtin_baseline_external_candidate_accepts_same_declared_inputs(self):
        for mode in ("search", "confirm"):
            with self.subTest(mode=mode):
                cfg = self.cfg(mode)
                cfg["baseline"]["adapter"] = "llama"
                original = deepcopy(cfg)
                self.assertNotEqual(config.preparation_identity(cfg["baseline"]),
                                    config.preparation_identity(cfg["candidate"]))
                self.assertEqual(config.declared_inputs(cfg["baseline"]),
                                 config.declared_inputs(cfg["candidate"]))
                resolved = config.validate_config(cfg)
                self.assertEqual(resolved["baseline"]["adapter"], "llama")
                self.assertEqual(resolved["candidate"]["adapter"], original["candidate"]["adapter"])
                self.assertEqual(cfg, original)

    def test_builtin_external_pair_rejects_changed_tokenizer(self):
        for mode in ("search", "confirm"):
            with self.subTest(mode=mode):
                cfg = self.cfg(mode)
                cfg["baseline"]["adapter"] = "llama"
                cfg["candidate"]["tokenizer"]["revision"] = "b" * 40
                self.assertNotEqual(config.declared_inputs(cfg["baseline"]),
                                    config.declared_inputs(cfg["candidate"]))
                with self.assertRaisesRegex(config.ConfigurationError, "identical preparation"):
                    config.validate_config(cfg)


class WorkerTests(Fixture):
    def test_private_config_checksum_digest_projection_and_variant_binding(self):
        for layer in ("file_hash", "config_digest", "projection", "variant_identity"):
            path = self.private_config_job()
            job = read(path)
            resolved = read(job["resolved_path"])
            full = read(job["private_config_path"])
            full["baseline"]["template"]["text"] += " changed"
            write(job["private_config_path"], full)
            if layer != "file_hash":
                job["private_config_sha256"] = sha(job["private_config_path"])
            if layer in ("projection", "variant_identity"):
                resolved["config_sha256"] = digest(full)
            if layer == "variant_identity":
                resolved["config"] = privacy.public_config(full)
            write(job["resolved_path"], resolved)
            job["resolved_sha256"] = sha(job["resolved_path"])
            write(path, job)
            with self.subTest(layer=layer):
                result = self.execute(path)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(self.adapters, [])
                self.runtime.assert_not_called()
                self.assertFalse(PRIVATE_PROMPT in (path.parent / "result.json").read_text())

    def test_top_level_backend_proof_redacts_row_and_inline_prompt_echoes(self):
        source = input_rows()
        echo = source[0]["messages"][0]["content"]
        def inject(adapter):
            adapter.prove_batch_backend = lambda group: {
                "status": "passed", "diagnostic": {"row": echo, "prompt": PRIVATE_PROMPT}}
        self.adapter_hook = inject
        path = self.private_config_job(source=source)
        result = self.execute(path)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["backend_proof"]["diagnostic"],
                         {"row": "[redacted-input]", "prompt": "[redacted-input]"})
        saved = (path.parent / "result.json").read_text()
        self.assertFalse(SECRET in saved or PRIVATE_PROMPT in saved)
        self.assertEqual(self.adapters[0].spec["template"]["text"], PRIVATE_PROMPT)

    def test_latency_score_echoes_fail_without_public_checkpoint_leaks(self):
        for kind in ("row", "prompt"):
            def inject(adapter):
                request = adapter.request_batch
                def echoed(group):
                    values = request(group)
                    for value, messages in zip(values, group):
                        value["category"] = messages[0]["content"] if kind == "row" else PRIVATE_PROMPT
                    return values
                adapter.request_batch = echoed
            self.adapter_hook = inject
            path = self.private_config_job("latency")
            with self.subTest(kind=kind), patch.object(worker, "write", wraps=write) as writes:
                result = self.execute(path)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "PrivateOutputError")
            self.assertTrue(result["rows"])
            self.assertTrue(all(score["category"] == "[redacted-input]"
                                for row in result["rows"] for score in row["scores"]))
            for call in writes.call_args_list:
                if Path(call.args[0]).name == "result.json":
                    encoded = json.dumps(call.args[1])
                    self.assertFalse(SECRET in encoded or PRIVATE_PROMPT in encoded)
            self.assertEqual(read(path.parent / "result.json"), result)

    def test_quality_echoes_are_excluded_from_public_predictions(self):
        for kind in ("row", "prompt"):
            def inject(adapter):
                request = adapter.request_batch
                def echoed(group):
                    values = request(group)
                    for value, messages in zip(values, group):
                        value["category"] = messages[0]["content"] if kind == "row" else PRIVATE_PROMPT
                    return values
                adapter.request_batch = echoed
            self.adapter_hook = inject
            path = self.private_config_job("quality", input_rows(5))
            with self.subTest(kind=kind):
                result = self.execute(path)
                text = (path.parent / "predictions.jsonl").read_text()
                self.assertFalse(SECRET in text or PRIVATE_PROMPT in text,
                                 "Private response text must not enter quality JSONL")
                predictions = [json.loads(line) for line in text.splitlines()]
                self.assertEqual(len(predictions), 5)
                self.assertTrue(all(p["status"] == "private_output_rejected" and "score" not in p for p in predictions))
                self.assertEqual(result["quality"]["overall"]["evaluated"], 0)
                self.assertEqual(result["quality"]["overall"]["failures"], {"private_output_rejected": 5})
                self.assertFalse(result["quality_evaluated"])
                self.assertFalse((path.parent / "predictions.jsonl.tmp").exists())

    def test_external_metadata_cannot_exempt_secret_without_adapter_spec(self):
        def inject(adapter):
            del adapter.spec
            adapter.metadata["labels"] = [PRIVATE_PROMPT]
            request = adapter.request_batch
            def echoed(group):
                values = request(group)
                for value in values:
                    value["category"] = PRIVATE_PROMPT
                return values
            adapter.request_batch = echoed
        self.adapter_hook = inject
        for stage in ("quality", "latency"):
            with self.subTest(stage=stage):
                path = self.private_config_job(stage)
                result = self.execute(path)
                self.assertFalse(hasattr(self.adapters[-1], "spec"))
                self.assertFalse(PRIVATE_PROMPT in (path.parent / "result.json").read_text())
                if stage == "quality":
                    text = (path.parent / "predictions.jsonl").read_text()
                    self.assertFalse(PRIVATE_PROMPT in text)
                    predictions = [json.loads(line) for line in text.splitlines()]
                    self.assertEqual(len(predictions), 5)
                    self.assertTrue(all(p["status"] == "private_output_rejected" for p in predictions))
                    self.assertEqual(result["quality"]["overall"]["evaluated"], 0)
                else:
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["error_type"], "PrivateOutputError")

    def test_preflight_prepares_boundary_shape_before_ort_request(self):
        prepared_shapes = set()
        def inject(adapter):
            prepare, request = adapter.prepare_batch_backend, adapter.request_batch
            def shape(group):
                return len(group), max(adapter.index(value)["total_tokens"] for value in group)
            def prepare_shapes(groups):
                prepared_shapes.update(shape(group) for group in groups)
                return prepare(groups)
            def request_shape(group):
                if shape(group) not in prepared_shapes:
                    raise ValueError("Unprepared ORT fixture shape")
                return request(group)
            adapter.prepare_batch_backend = prepare_shapes
            adapter.request_batch = request_shape
        self.adapter_hook = inject
        path = self.worker_job("preflight")
        job = read(path)
        resolved = read(job["resolved_path"])
        resolved["config"]["baseline"]["execution"]["runtime"] = "ort"
        write(job["resolved_path"], resolved)
        job["resolved_sha256"] = sha(job["resolved_path"])
        write(path, job)
        result = self.execute(path)
        self.assertEqual(result["status"], "passed")
        self.assertIn((3, 512), prepared_shapes)
        self.assertEqual(self.adapters[0].spec["execution"]["runtime"], "ort")
        self.assertTrue(result["over_limit_rejected"])

    def test_preflight_real_checks_and_monitor_callbacks(self):
        path = self.worker_job("preflight")
        result = self.execute(path)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["over_limit_rejected"])
        self.assertTrue(result["aba_passed"])
        self.assertEqual(result["boundary_checked"], 512)
        self.assertEqual(self.samplers[0].events, ["start", "inference", "stop"])
        self.assertTrue(self.adapters[0].closed)
        self.assertNotIn(SECRET, (path.parent / "result.json").read_text())

    def test_latency_warmup_exact_batches_and_monitor_callbacks(self):
        path = self.worker_job()
        result = self.execute(path)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(self.adapters[0].requests), 23)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["compile_before"], result["compile_after"])
        self.assertEqual(self.samplers[0].events, ["start", "inference", "stop"])
        self.assertTrue(all(r["latency_ms"] > 0 for r in result["rows"]))

    def test_quality_is_nonspeed_and_keeps_partial_tail(self):
        path = self.worker_job("quality", input_rows(5))
        result = self.execute(path)
        self.assertEqual(result["status"], "passed")
        self.assertEqual([len(g) for g in self.adapters[0].verified], [3, 2])
        self.assertEqual(len(self.adapters[0].requests), 2)
        self.assertEqual(result["rows"], [])
        self.assertNotIn("warmup_seconds", result)
        self.assertEqual(result["quality"]["overall"]["evaluated"], 5)
        predictions = [json.loads(line) for line in (path.parent / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len({p["id"] for p in predictions}), 5)
        self.assertNotIn(SECRET, json.dumps(predictions))

    def test_partial_latency_failure_checkpoint_and_private_errors(self):
        self.adapter_hook = lambda a: setattr(a, "fail_at", 22)
        path = self.worker_job()
        result = self.execute(path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(read(path.parent / "result.json"), result)
        self.assertNotIn(SECRET, (path.parent / "result.json").read_text())
        self.assertTrue(self.adapters[0].closed)
        self.assertEqual(self.samplers[0].events[-1], "stop")

    def test_quality_failure_retains_each_row_and_completed_tail_state(self):
        self.adapter_hook = lambda a: setattr(a, "fail_at", 2)
        path = self.worker_job("quality", input_rows(5))
        result = self.execute(path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["quality"]["overall"]["evaluated"], 3)
        predictions = [json.loads(line) for line in (path.parent / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(predictions), 5)
        self.assertEqual([p["status"] for p in predictions].count("failed"), 2)
        self.assertNotIn(SECRET, (path.parent / "result.json").read_text())

    def test_snapshot_and_private_input_tampering_rejected_before_load(self):
        for target in ("resolved_path", "input_path"):
            with self.subTest(target=target):
                path = self.worker_job()
                job = read(path)
                Path(job[target]).write_text("{}")
                result = self.execute(path)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(self.adapters, [])
        self.runtime.assert_not_called()

    def test_environment_mismatch_rejected_before_load(self):
        self.runtime.return_value["driver"] = "changed"
        result = self.execute(self.worker_job())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.adapters, [])

    def test_source_change_rejected_before_runtime_or_adapter(self):
        path = self.worker_job()
        with patch.object(evidence, "source_inventory", return_value={}):
            result = self.execute(path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.adapters, [])
        self.runtime.assert_not_called()

    def test_oom_unsupported_and_timeout_keep_status_and_cleanup(self):
        errors = [(type("OutOfMemoryError", (RuntimeError,), {})(SECRET), "oom"),
                  (NotImplementedError(SECRET), "unsupported"), (TimeoutError(SECRET), "timeout")]
        for error, expected in errors:
            def fail(adapter):
                adapter.fail_at, adapter.error = 1, error
            self.adapter_hook = fail
            with self.subTest(expected=expected):
                path = self.worker_job()
                result = self.execute(path)
                self.assertEqual(result["status"], expected)
                self.assertTrue(self.adapters[-1].closed)
                self.assertEqual(self.samplers[-1].events[-1], "stop")
                self.assertNotIn(SECRET, (path.parent / "result.json").read_text())

    def test_preflight_checksum_tampering_rejected_without_requests(self):
        path = self.worker_job()
        job = read(path)
        write(job["preflight_path"], {"status": "passed"})
        result = self.execute(path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.adapters[0].requests, [])

    def test_old_snapshot_version_rejected_even_with_valid_checksum(self):
        path = self.worker_job("preflight")
        job = read(path)
        value = read(job["resolved_path"])
        value["schema_version"] = value["result_version"] = 3
        write(job["resolved_path"], value)
        job["resolved_sha256"] = sha(job["resolved_path"])
        write(path, job)
        result = self.execute(path)
        self.assertEqual(result["status"], "failed", "Worker must reject old snapshot formats")
        self.assertEqual(self.adapters, [])

    def test_native_execution_cannot_change_after_preflight(self):
        for field, expected, changed in (("attention", "sdpa", "eager"),
                                         ("linear_attention", "native", "reference")):
            for actual in (expected, changed, None):
                with self.subTest(field=field, actual=actual):
                    path = self.worker_job()
                    job = read(path)
                    proof = read(job["preflight_path"])
                    proof["metadata"][field] = expected
                    write(job["preflight_path"], proof)
                    job["preflight_sha256"] = sha(job["preflight_path"])
                    write(path, job)
                    self.adapter_hook = lambda adapter: adapter.metadata.update({field: actual}) if actual else None
                    result = self.execute(path)
                    self.assertEqual(result["status"], "passed" if actual == expected else "failed")
                    if actual != expected:
                        self.assertEqual(self.adapters[-1].requests, [])

    def test_preflight_proof_cannot_be_borrowed_from_other_variant(self):
        path = self.worker_job()
        job = read(path)
        proof = read(job["preflight_path"])
        proof["job"]["variant"] = "candidate"
        write(job["preflight_path"], proof)
        job["preflight_sha256"] = sha(job["preflight_path"])
        write(path, job)
        result = self.execute(path)
        self.assertEqual(result["status"], "failed", "Preflight proof must bind variant/cell")
        self.assertFalse(any(adapter.requests for adapter in self.adapters))

    def test_nested_metadata_cannot_leak_input_on_failure(self):
        source = input_rows()
        def leak(adapter):
            adapter.metadata["preparation"] = {"rendered_prompt": source[0]["messages"][0]["content"]}
            adapter.fail_at = 1
        self.adapter_hook = leak
        path = self.worker_job(source=source)
        result = self.execute(path)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(SECRET in (path.parent / "result.json").read_text(),
                         "Allowed metadata containers must not bypass input redaction")

    def test_quality_monitor_has_inference_boundary(self):
        result = self.execute(self.worker_job("quality"))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.samplers[0].events, ["start", "inference", "stop"])


class MonitoringTests(Fixture):
    def measure(self, sampler_factory=None):
        from u1_benchmark.precision import PrecisionGuard
        from u1_benchmark import v4_monitoring as monitoring
        source = input_rows()
        adapter = self.factory(variant(), self.root)
        plans = planning.smoke_plans(source, 1)
        clock = itertools.count(1_000_000, 1_000_000)
        with patch("u1_benchmark.v4_memory.Sampler", side_effect=sampler_factory or self.sampler), \
                patch.object(monitoring.time, "perf_counter_ns", side_effect=lambda: next(clock)):
            return monitoring.measure(adapter, source, plans, PrecisionGuard(adapter, required=True),
                                      self.torch, "GPU-fake", self.root)

    def test_on_off_paired_callbacks_and_identical_requests(self):
        result = self.measure()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["blocks"]), 3)
        self.assertEqual(len(self.samplers), 3)
        self.assertTrue(all(s.events == ["start", "inference", "stop"] for s in self.samplers))
        self.assertEqual(len(self.adapters[0].requests), 6 * 23)
        for block in result["blocks"]:
            self.assertEqual(block["on"]["rows"], block["off"]["rows"])
            self.assertEqual(block["off"]["memory"]["status"], "disabled")
        self.assertNotIn(SECRET, (self.root / "monitoring.json").read_text())

    def test_unavailable_monitor_cannot_validate_overhead(self):
        class Unavailable(FakeSampler):
            def stop(self, path):
                self.events.append("stop")
                return {"status": "unavailable"}
        result = self.measure(Unavailable)
        self.assertEqual(result["status"], "unvalidated")
        self.assertFalse(result["memory_available"])

    def test_sampling_stops_when_monitored_request_fails(self):
        # The off side completes 20 warmups + 3 batches before on starts.
        self.adapter_hook = lambda adapter: setattr(adapter, "fail_at", 24)
        with self.assertRaises(RuntimeError):
            self.measure()
        self.assertEqual(len(self.samplers), 1)
        self.assertEqual(self.samplers[0].events, ["start", "stop"])


class EvidenceTests(Fixture):
    def test_immutable_snapshot_cannot_be_overwritten(self):
        path = self.root / "immutable.json"
        checksum = evidence.immutable(path, {"schema_version": 4})
        with self.assertRaises(FileExistsError):
            evidence.immutable(path, {"schema_version": 3})
        self.assertEqual(sha(path), checksum)
        self.assertEqual(read(path), {"schema_version": 4})

    def test_reference_checksum_and_old_formats_rejected_before_report(self):
        root = self.reference()
        with self.report_stub():
            reference = evidence.read_reference(root)
        self.assertEqual(reference["resolved"]["config"]["mode"], "baseline")
        original = read(root / "resolved_run.json")
        changed = deepcopy(original)
        changed["schema_version"] = 3
        write(root / "resolved_run.json", changed)
        with self.assertRaises(ValueError):
            evidence.read_reference(root)
        write(root / "resolved_run.json", {**original, "changed": True})
        with self.assertRaisesRegex(ValueError, "checksum"):
            evidence.read_reference(root)

    def test_reference_identity_covers_sources_inputs_seed_and_environment(self):
        root = self.reference()
        with self.report_stub():
            reference = evidence.read_reference(root)
        cfg = self.cfg("search", pilot=False)
        indices = reference["resolved"]["input_indices"]
        evidence.check_reference(cfg, reference, self.inventory, indices, RUNTIME)
        for kwargs in ({"sources": {}}, {"inputs": {}}, {"environment": {**RUNTIME, "driver": "changed"}}):
            values = {"sources": self.inventory, "inputs": indices, "environment": RUNTIME, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evidence.check_reference(cfg, reference, **values)
        with self.assertRaises(ValueError):
            evidence.check_reference({**cfg, "seed": 99}, reference, self.inventory, indices)
        evidence.assert_environment({**RUNTIME, "gpu_uuid": "GPU-other", "pid": 999}, RUNTIME)

    def test_reference_copy_is_exclusive_and_checksum_indexed(self):
        root = self.reference()
        with self.report_stub():
            reference = evidence.read_reference(root)
        target = self.root / "copied"
        evidence.copy_reference(reference, target)
        manifest = read(target / "reference.json")
        for name, expected in manifest["files"].items():
            self.assertEqual(sha(target / "reference" / name), expected)
        with self.assertRaises(FileExistsError):
            evidence.copy_reference(reference, target)

    def test_archive_reference_rejects_traversal_and_symlinks(self):
        for number, name in enumerate(("../escaped.json", "link.json")):
            archive = self.root / f"unsafe-{number}.zip"
            with zipfile.ZipFile(archive, "w") as z:
                member = zipfile.ZipInfo(name)
                if number:
                    member.create_system = 3
                    member.external_attr = 0o120777 << 16
                z.writestr(member, "{}")
            with self.subTest(name=name), self.assertRaises(ValueError):
                with evidence.reference_folder(archive):
                    self.fail("Unsafe reference accepted")
        self.assertFalse((self.root / "escaped.json").exists())


class RunnerTests(Fixture):
    def test_plan_missing_data_is_not_ready_but_invalid_config_raises(self):
        from u1_benchmark.v4_data import DataError
        cfg = self.cfg()
        for error in (DataError(SECRET), FileNotFoundError(SECRET)):
            with self.subTest(error=type(error).__name__), patch.object(runner, "inputs", side_effect=error):
                value = runner.plan(cfg)
            self.assertEqual(value, {"status": "not_ready", "phase": "data",
                                     "error_type": type(error).__name__, "schema_version": 4,
                                     "data_ready": False, "gpu_checked": False})
            self.assertNotIn(SECRET, json.dumps(value))
        with patch.object(runner, "inputs") as inputs:
            with self.assertRaises(config.ConfigurationError):
                runner.plan({**cfg, "schema_version": 3})
            inputs.assert_not_called()
        with patch.object(runner, "inputs", side_effect=ValueError("Invalid input contract")):
            with self.assertRaises(ValueError):
                runner.plan(cfg)
        self.assertEqual(self.adapters, [])
        self.runtime.assert_not_called()

    def test_plan_missing_or_stale_reference_is_not_ready(self):
        cfg = self.cfg("search", pilot=False)
        source = self.runner_inputs(cfg, input_rows(400))
        for error in (ValueError(SECRET), FileNotFoundError(SECRET)):
            with self.subTest(error=type(error).__name__), patch.object(runner, "inputs", return_value=source), \
                    patch.object(runner, "read_reference", side_effect=error):
                value = runner.plan(cfg)
            self.assertEqual(value, {"status": "not_ready", "phase": "reference",
                                     "error_type": type(error).__name__, "schema_version": 4,
                                     "data_ready": True, "reference_ready": False, "gpu_checked": False})
            self.assertNotIn(SECRET, json.dumps(value))
        with patch.object(runner, "inputs", return_value=source), \
                patch.object(runner, "read_reference", return_value={"resolved": {}}), \
                patch.object(runner, "check_reference", side_effect=ValueError(SECRET)):
            value = runner.plan(cfg)
        self.assertEqual((value["status"], value["phase"]), ("not_ready", "reference"))
        self.assertFalse(value["reference_ready"])
        self.assertNotIn(SECRET, json.dumps(value))
        self.assertEqual(self.adapters, [])
        self.runtime.assert_not_called()

    def test_plan_counts_warmup_and_does_not_launch(self):
        cfg = self.cfg()
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg)):
            result = runner.plan(cfg)
        self.assertEqual((result["preflight_workers"], result["latency_workers"]), (1, 1))
        self.assertEqual((result["warmup_batches"], result["timed_batches"], result["timed_input_occurrences"]), (20, 12, 36))
        self.assertFalse(result["gpu_checked"])
        self.assertEqual(self.adapters, [])

    def test_search_never_loads_or_requests_baseline_and_copies_reference(self):
        source = input_rows(400)
        reference = self.reference(source)
        before = {p.name: p.read_bytes() for p in reference.iterdir()}
        cfg, target = self.cfg("search", pilot=False), self.root / "search-run"
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg, source)), self.report_stub():
            result = runner.run(cfg, target, launcher=self.launcher)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(self.launched), 4)
        self.assertEqual({j["variant"] for j in self.launched}, {"candidate"})
        self.assertTrue(all(a.spec["execution"]["runtime"] == "compile" for a in self.adapters))
        self.assertTrue(all(a.requests and a.closed for a in self.adapters))
        self.assertEqual({p.name: p.read_bytes() for p in reference.iterdir()}, before)
        self.assertTrue((target / "reference.json").exists())
        self.assertNotIn(SECRET, self.archive_text(target))

    def test_confirm_launches_fresh_workers_with_alternating_pairs(self):
        cfg, target = self.cfg("confirm", pilot=False), self.root / "confirm-run"
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg, input_rows(400))), self.report_stub():
            result = runner.run(cfg, target, launcher=self.launcher)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(self.launched), 12)
        self.assertEqual(len({id(a) for a in self.adapters}), 12)
        self.assertEqual(len({j["input_path"] for j in self.launched}), 12)
        self.assertTrue(all(a.closed for a in self.adapters))
        latency = [j for j in self.launched if j["stage"] == "latency"]
        self.assertEqual([j["variant"] for j in latency[:4]], ["baseline"] * 4)
        self.assertEqual([j["variant"] for j in latency[4:]],
                         ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"])
        self.assertFalse((target / "reference.json").exists())

    def test_quality_launches_only_quality_workers_without_speed_inputs(self):
        cfg, target = self.cfg("quality"), self.root / "quality-run"
        cfg["quality_batch_sizes"] = [3, 4]
        with patch.object(runner, "verified", return_value=(input_rows(5), {"kind": "quality"})), \
                patch.object(runner, "load_speed", side_effect=AssertionError("Speed data requested")), self.report_stub():
            plan = runner.plan(cfg)
            result = runner.run(cfg, target, launcher=self.launcher)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(plan["quality_input_occurrences"], 10)
        self.assertEqual(plan["latency_workers"], 0)
        self.assertEqual(plan["warmup_batches"], 0)
        self.assertEqual([j["stage"] for j in self.launched], ["quality", "quality"])
        self.assertEqual([[len(g) for g in a.requests] for a in self.adapters], [[3, 2], [4, 1]])
        self.assertNotIn(SECRET, self.archive_text(target))

    def test_timeout_still_archives_and_excludes_private_logs_inputs(self):
        cfg, target = self.cfg(), self.root / "failed-run"
        def timeout(args, log, remaining):
            if "_runtime_v4" in [str(a) for a in args]:
                return self.launcher(args, log, remaining)
            Path(log).write_text(SECRET + " private failure")
            raise TimeoutError(SECRET)
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg)), self.report_stub():
            result = runner.run(cfg, target, launcher=timeout)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(read(target / "jobs/00000/result.json")["status"], "timeout")
        self.assertNotIn(SECRET, self.archive_text(target))
        self.assertEqual(self.adapters, [])

    def test_reference_mismatch_rejected_before_any_launcher_call(self):
        self.reference()
        cfg = self.cfg("search", pilot=False)
        cfg["seed"] = 99
        launch = Mock(side_effect=AssertionError("Must not launch mismatched reference"))
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg, input_rows(400))), self.report_stub():
            result = runner.run(cfg, self.root / "bad-reference", launcher=launch)
        self.assertEqual(result["status"], "failed")
        launch.assert_not_called()


class RealReportRunnerTests(Fixture):
    """Real runner, worker, report, reference validation and ZIP packaging.

    Only model/runtime execution, corpus loading and the request clock are
    fixtures. Statistics consume deterministic positive timings; these tests
    make no speed or hardware acceptance claim.
    """
    def setUp(self):
        super().setUp()
        self.report = importlib.import_module("u1_benchmark.v4_report")
        self.assertTrue(callable(self.report.build))
        clock = itertools.count(1_000_000, 1_000_000)
        self.enterContext(patch.object(worker.time, "perf_counter_ns", side_effect=lambda: next(clock)))

    def run_real(self, cfg, name, source):
        target = self.root / name
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg, source)):
            state = runner.run(cfg, target, launcher=self.launcher)
        self.assertNotIn("report_error_type", state)
        summary = read(target / "summary.json")
        self.assertEqual(state["status"], summary["status"])
        self.assertEqual(read(target / "status.json"), state)
        self.assertEqual(sha(target / "resolved_run.json"), state["resolved_sha256"])
        return target, summary

    def assert_archived_report(self, target, summary):
        archive_path = target.with_suffix(".zip")
        self.assertNotIn(SECRET, self.archive_text(target))
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(json.loads(archive.read("summary.json")), summary)
            self.assertIn("summary.md", archive.namelist())
            self.assertIn("resolved_run.json", archive.namelist())
            self.assertEqual(json.loads(archive.read("status.json"))["status"], summary["status"])
        # Archived jobs contain original absolute input paths. Make those paths
        # unavailable to prove the offline report really uses public evidence.
        private = target / ".private"
        hidden = target / ".private-hidden-for-offline-test"
        private.rename(hidden)
        try:
            with evidence.reference_folder(archive_path) as extracted:
                before = {str(p.relative_to(extracted)): sha(p) for p in extracted.rglob("*") if p.is_file()}
                with patch.object(runner, "inputs", side_effect=AssertionError("Offline corpus access")), \
                        patch.object(evidence, "source_inventory", side_effect=AssertionError("Offline live-source access")), \
                        patch.object(config, "validate_config", side_effect=AssertionError("Offline live config validation")):
                    rebuilt = self.report.build(extracted, save=False)
                self.assertEqual(rebuilt, summary)
                after = {str(p.relative_to(extracted)): sha(p) for p in extracted.rglob("*") if p.is_file()}
                self.assertEqual(after, before, "save=False must not modify archived evidence")
        finally:
            hidden.rename(private)

    def test_inline_prompt_stays_private_but_adapter_receives_full_config(self):
        cfg = self.cfg("smoke", pilot=False)
        cfg["baseline"]["template"] = {"text": PRIVATE_PROMPT}
        target, summary = self.run_real(cfg, "private-prompt-smoke", input_rows())
        self.assertEqual(summary["status"], "preflight_passed", summary["errors"])
        self.assertEqual(len(self.adapters), 1)
        self.assertEqual(self.adapters[0].spec["template"], {"text": PRIVATE_PROMPT})
        resolved = read(target / "resolved_run.json")
        full = read(target / ".private/config.json")
        self.assertEqual(full["baseline"]["template"]["text"], PRIVATE_PROMPT)
        self.assertEqual(resolved["config_sha256"], digest(full))
        self.assertEqual(resolved["variant_identities"]["baseline"], digest(full["baseline"]))
        self.assertEqual(resolved["config"]["baseline"]["template"]["text"],
                         "[private-sha256:" + digest(PRIVATE_PROMPT) + "]")
        job = self.launched[0]
        self.assertEqual(sha(job["private_config_path"]), job["private_config_sha256"])
        self.assertFalse(PRIVATE_PROMPT in (target / "resolved_run.json").read_text())
        self.assertFalse(PRIVATE_PROMPT in self.archive_text(target))
        self.assert_archived_report(target, summary)

    def test_full_baseline_zip_reference_search_and_tamper_detection(self):
        source = input_rows(400)
        baseline, base_summary = self.run_real(self.cfg("baseline", pilot=False), "real-baseline", source)
        self.assertEqual(base_summary["status"], "baseline_complete", base_summary["errors"])
        self.assertTrue(base_summary["complete"])
        self.assertEqual(base_summary["memory_status"], "measured")
        self.assert_archived_report(baseline, base_summary)
        archive = baseline.with_suffix(".zip")
        before = sha(archive)
        reference = evidence.read_reference(archive)
        self.assertEqual(reference["sha256"], before)
        previous_loads, previous_jobs = len(self.adapters), len(self.launched)
        cfg = self.cfg("search", pilot=False)
        cfg["baseline_ref"] = str(archive)
        target, summary = self.run_real(cfg, "real-search", source)
        self.assertEqual(summary["status"], "exploratory", summary["errors"])
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["independently_confirmed"])
        self.assertEqual({j["variant"] for j in self.launched[previous_jobs:]}, {"candidate"})
        self.assertEqual(len(self.adapters) - previous_loads, 4)
        self.assertTrue(all(a.spec["execution"]["runtime"] == "compile" for a in self.adapters[previous_loads:]))
        self.assertEqual(sha(archive), before)
        self.assertEqual(summary["cells"]["custom/3"]["throughput"]["candidate"]["texts"], 1206)
        self.assert_archived_report(target, summary)
        with evidence.reference_folder(target.with_suffix(".zip")) as extracted:
            result_path = extracted / "reference/jobs/00000/result.json"
            value = read(result_path)
            value["forged"] = True
            write(result_path, value)
            rejected = self.report.build(extracted, save=False)
            self.assertFalse(rejected["complete"])
            self.assertIn("reference_file_hash_mismatch", rejected["errors"])

    def test_full_confirm_real_report_and_fresh_workers(self):
        target, summary = self.run_real(self.cfg("confirm", pilot=False), "real-confirm", input_rows(400))
        self.assertTrue(summary["complete"], summary["errors"])
        self.assertEqual(summary["status"], "inconclusive")
        self.assertEqual(summary["verdict_status"], "unchanged")
        self.assertFalse(summary["independently_confirmed"])
        self.assertEqual(len({id(a) for a in self.adapters}), 12)
        self.assertTrue(all(a.closed for a in self.adapters))
        self.assert_archived_report(target, summary)

    def test_pilot_search_full_baseline_zip_prefix_and_offline_report(self):
        source = input_rows(400)
        baseline, base_summary = self.run_real(self.cfg("baseline", pilot=False), "pilot-reference", source)
        self.assertEqual(base_summary["status"], "baseline_complete", base_summary["errors"])
        full = read(baseline / "resolved_run.json")["schedule"]
        original = next(job for job in full if job["phase"] == "baseline" and job["block"] == 0)
        cfg = self.cfg("search", pilot=True)
        cfg["baseline_ref"] = str(baseline.with_suffix(".zip"))
        with patch.object(runner, "inputs", return_value=self.runner_inputs(cfg, source)):
            plan = runner.plan(cfg)
        self.assertEqual((plan["preflight_workers"], plan["latency_workers"], plan["timed_batches"]), (1, 1, 12))
        self.assertEqual(plan["timed_input_occurrences"], 36)
        first_job, first_adapter = len(self.launched), len(self.adapters)
        target, summary = self.run_real(cfg, "real-pilot-search", source)
        self.assertEqual(summary["status"], "pilot_complete", summary["errors"])
        self.assertEqual(summary["speed_status"], "not_evaluated_pilot")
        self.assertTrue(summary["complete"])
        jobs = self.launched[first_job:]
        self.assertEqual([(job["stage"], job["variant"]) for job in jobs],
                         [("preflight", "candidate"), ("latency", "candidate")])
        self.assertEqual(len(self.adapters) - first_adapter, 2)
        self.assertEqual(jobs[1]["batches"], original["batches"][:12])
        self.assertEqual(jobs[1]["ids"], original["ids"][:36])
        self.assertEqual(jobs[1]["plan_sha256"], digest(original["batches"][:12]))
        self.assert_archived_report(target, summary)

    def test_directory_reference_real_search_and_copied_evidence_tampering(self):
        source = input_rows(400)
        baseline, base_summary = self.run_real(self.cfg("baseline", pilot=False), "directory-baseline", source)
        self.assertEqual(base_summary["status"], "baseline_complete", base_summary["errors"])
        before = {str(p.relative_to(baseline)): sha(p) for p in baseline.rglob("*") if p.is_file()}
        cfg = self.cfg("search", pilot=False)
        cfg["baseline_ref"] = str(baseline)
        first_job, first_adapter = len(self.launched), len(self.adapters)
        target, summary = self.run_real(cfg, "directory-search", source)
        self.assertEqual(summary["status"], "exploratory", summary["errors"])
        self.assertEqual(len(self.adapters) - first_adapter, 4)
        self.assertEqual({j["variant"] for j in self.launched[first_job:]}, {"candidate"})
        self.assertEqual(summary["cells"]["custom/3"]["p95"]["ratio"], 1)
        self.assertEqual(self.report.build(target, save=False), summary)
        after = {str(p.relative_to(baseline)): sha(p) for p in baseline.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        self.assertNotIn(SECRET, self.archive_text(target))
        copied = target / "reference/jobs/00000/result.json"
        write(copied, {**read(copied), "forged": True})
        rejected = self.report.build(target, save=False)
        self.assertFalse(rejected["complete"])
        self.assertIn("reference_file_hash_mismatch", rejected["errors"])

    def test_paired_quality_real_report_public_rows_and_partial_tails(self):
        cfg = self.cfg("quality")
        cfg["candidate"] = variant("compile", module=self.external_module)
        cfg["quality_batch_sizes"] = [1, 3]
        source = input_rows(5)
        target, summary = self.run_real(cfg, "real-quality", source)
        self.assertEqual(summary["status"], "quality_complete", summary["quality"].get("errors"))
        self.assertEqual(summary["speed_status"], "not_measured")
        self.assertEqual(summary["cells"], {})
        frozen = read(target / "resolved_run.json")
        self.assertEqual(frozen["quality_rows"], [{k: row[k] for k in ("id", "target", "slices")} for row in source])
        batch = summary["quality"]["quality_by_batch_size"]["3"]
        for value in batch["variants"].values():
            self.assertEqual(value["actual_batch_sizes"], [3, 2])
            self.assertEqual(value["quality"]["overall"]["evaluated"], 5)
        self.assertEqual(batch["baseline_to_candidate"]["status"], "complete")
        self.assertTrue(all(j["stage"] == "quality" for j in self.launched))
        self.assert_archived_report(target, summary)

    def test_real_report_accepts_builtin_external_declared_input_pair(self):
        cfg = self.cfg("quality")
        cfg["baseline"]["adapter"] = "llama"
        cfg["candidate"] = variant("compile", module=self.external_module)
        target, summary = self.run_real(cfg, "real-external-quality", input_rows(5))
        self.assertEqual(summary["status"], "quality_complete", summary["errors"])
        self.assert_archived_report(target, summary)

    def test_failed_quality_archive_preserves_partial_predictions_privately(self):
        self.adapter_hook = lambda adapter: setattr(adapter, "fail_at", 2)
        target, summary = self.run_real(self.cfg("quality"), "real-failed-quality", input_rows(5))
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["status"], "incomplete")
        self.assertTrue(summary["quality"]["quality_evaluated"])
        cell = summary["quality"]["quality_by_batch_size"]["3"]["variants"]["baseline"]
        self.assertEqual(cell["quality"]["overall"]["evaluated"], 3)
        self.assertEqual(cell["quality"]["overall"]["failures"], {"failed": 2})
        self.assert_archived_report(target, summary)


if __name__ == "__main__":
    unittest.main()
