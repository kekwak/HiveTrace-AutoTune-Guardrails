import copy
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from u1_benchmark import core
from u1_benchmark.statistics import assess, compare_scores, paired, paired_rows, quantile
from u1_benchmark.models import quiet_dispatch, filtered, stable_call
from u1_benchmark.memory import Sampler, process_mib
from u1_benchmark.process import archive, invoke
from u1_benchmark.worker import preparation_rows


SCORE = {"label": "safe", "p_safe": .9, "p_unsafe": .1, "logit_margin": -2.2}


def dataset(root):
    root.mkdir(parents=True)
    records, indices = [], []
    specs = {}
    for name, lo, hi in (("B1", 1, 512), ("B2", 513, 1024), ("B3", 1025, 2048), ("B4", 2049, 4096)):
        ids = []
        for language in ("en", "ru"):
            for source, count in (("wikipedia", 150), ("synthetic", 50)):
                for i in range(count):
                    text = f"{name} {language} {source} {i}"
                    key = hashlib.sha256(text.encode()).hexdigest()
                    ids.append(key)
                    records.append({"id": key, "text_sha256": key, "text": text,
                                    "language": language, "source": source})
                    indices.append({"id": key, "supported": True, "bucket": name, "total_tokens": hi,
                                    "user_tokens": hi - 5, "input_sha256": "f" * 64})
        core.write(root / f"buckets/{name}.ids.json", ids)
        specs[name] = {"min": lo, "max": hi, "ids_file": f"buckets/{name}.ids.json"}
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (root / "indices").mkdir()
    for model in ("halo", "gliner"):
        rows = indices if model == "halo" else indices[:400]
        (root / f"indices/{model}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    core.write(root / "manifest.json", {"buckets": specs,
        "files": {str(p.relative_to(root)): core.sha(p) for p in root.rglob("*") if p.is_file()}})
    return core.sha(root / "manifest.json")


class CorpusAndStatistics(unittest.TestCase):
    def test_halo_preparation_covers_only_worker_shapes(self):
        rows = [{"id": str(i), "index": {"total_tokens": n}}
                for i, n in enumerate((408, 450, 900, 2048, 4075))]
        selected, scope = preparation_rows("halo", "latency", rows, {"ids": ["2", "0", "2"]})
        self.assertEqual(selected, [rows[2], rows[0], rows[2]])
        self.assertEqual(scope, "measurement_shapes")
        selected, scope = preparation_rows("halo", "profile", rows)
        self.assertEqual(selected, [rows[0], rows[-1]])
        self.assertEqual(scope, "profile_shapes")
        for model, phase in (("halo", "preflight"), ("halo", "memory"),
                             ("gliner", "latency"), ("gliner", "profile")):
            self.assertEqual(preparation_rows(model, phase, rows),
                             (rows, "full_corpus_and_limit"))
        with self.assertRaises(KeyError):
            preparation_rows("halo", "latency", rows, {"ids": ["missing"]})

    def test_pinned_data_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            checksum = dataset(root)
            with patch.object(core, "MANIFEST_SHA", checksum):
                halo, _ = core.load_data(root, "halo")
                gliner, _ = core.load_data(root, "gliner")
                self.assertEqual([len(x) for x in halo.values()], [400] * 4)
                self.assertEqual(list(gliner), ["B1"])
                (root / "corpus.jsonl").write_text("{}")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    core.load_data(root, "halo")

    def test_schedule_exact_pairs_and_pilot(self):
        data = {"B1": [{"id": str(i)} for i in range(400)]}
        cfg = {"seed": 17, "mode": "screening"}
        jobs = core.schedule(cfg, data)
        self.assertEqual(len(jobs), 10)
        self.assertEqual([len(j["ids"]) for j in jobs], [30] * 4 + [400] * 6)
        for a, b in zip(jobs[::2], jobs[1::2]):
            self.assertEqual(a["ids"], b["ids"])
        self.assertEqual(jobs[2]["side"], "b")
        self.assertEqual(jobs, core.schedule(cfg, data))
        self.assertEqual(len(core.schedule({**cfg, "mode": "pilot"}, data)), 6)

    def test_pooled_quantile_not_mean_of_quantiles(self):
        blocks = [([1] * 100, [1] * 100), ([100], [50])]
        result = paired(blocks)
        self.assertEqual(result["baseline_ms"], 1)
        self.assertNotEqual(result["baseline_ms"], 50.5)
        self.assertEqual(result, paired(blocks))

    def test_noise_and_strict_threshold(self):
        aa = [([100] * 30, [100] * 30)] * 2
        self.assertEqual(assess([([100] * 400, [70] * 400)] * 3, aa)["status"], "pass")
        self.assertEqual(assess([([100] * 400, [80] * 400)] * 3, aa)["status"], "below_target")
        slower = assess([([100] * 400, [125] * 400)] * 3, aa)
        self.assertEqual(slower["status"], "below_target")
        self.assertEqual(slower["reasons"], ["consistent_slowdown"])
        mixed = [([100] * 400, [value] * 400) for value in (90, 130, 130)]
        self.assertEqual(assess(mixed, aa)["status"], "inconclusive")
        noisy = [([100] * 30, [150] * 30)] * 2
        self.assertEqual(assess([([100] * 400, [70] * 400)] * 3, noisy)["status"], "inconclusive")
        self.assertEqual(assess([([100] * 400, [125] * 400)] * 3, noisy)["status"], "inconclusive")
        with self.assertRaises(ValueError):
            quantile([float("nan")], .95)

    def test_verdicts_unique_ids_and_instability(self):
        a = {"x": [SCORE] * 4}
        b = {"x": [SCORE] * 3}
        self.assertEqual(compare_scores(a, b)["n"], 1)
        b["x"] += [{**SCORE, "label": "unsafe", "p_unsafe": .8, "p_safe": .2}]
        result = compare_scores(a, b)
        self.assertEqual(result["status"], "not_ok")
        self.assertEqual(result["unstable_ids"], ["x"])
        with self.assertRaises(ValueError):
            compare_scores(a, {})
        with self.assertRaises(ValueError):
            compare_scores(a, {"x": [{**SCORE, "p_unsafe": float("inf")}]})

    def test_pair_order_and_hashes(self):
        job = {"bucket": "B1", "phase": "ab", "block": 0, "side": "a", "variant": "baseline", "ids": ["x"]}
        other = {**job, "side": "b", "variant": "candidate"}
        row = {"id": "x", "input_sha256": "a", "total_tokens": 20, "latency_ms": 1, "score": SCORE}
        outcomes = [{"job": j, "status": "passed", "rows": [dict(row)]} for j in (job, other)]
        paired_rows([job, other], outcomes, "B1")
        outcomes[1]["rows"][0]["input_sha256"] = "b"
        with self.assertRaisesRegex(ValueError, "identity"):
            paired_rows([job, other], outcomes, "B1")

    def test_config_limits(self):
        self.assertEqual(core.config({"preset": "halo_fla", "environment": "/tmp/e"})["mode"], "screening")
        for patch_cfg in ({"timeout_seconds": 40000}, {"mode": "confirmation"}, {"seed": 0}, {"hidden_fallback": True}):
            with self.assertRaises(ValueError):
                core.config({"preset": "halo_fla", "environment": "/tmp/e", **patch_cfg})


class ProcessesAndMemory(unittest.TestCase):
    def test_boundary_two_token_unit_does_not_skip_even_length(self):
        from u1_benchmark.models import boundary_text
        a = SimpleNamespace(limit=512)
        def prepare(texts):
            length = 17 + 2 * texts[0].count("a ") + texts[0].count("hello ")
            if length > a.limit:
                raise ValueError("Input exceeds limit")
            return {"input_ids": [0] * length}
        a.prepare = prepare
        with patch("u1_benchmark.models.normalized", side_effect=lambda x: x):
            for target in (512, 513):
                text = boundary_text(a, target)
                self.assertEqual(17 + 2 * text.count("a ") + text.count("hello "), target)
                self.assertEqual(a.limit, 512)

    def test_boundary_with_long_prompt_and_restore(self):
        from u1_benchmark.models import boundary_text
        a = SimpleNamespace(limit=512)
        def prepare(texts):
            length = texts[0].count("a ") + 150
            if length > a.limit:
                raise ValueError("Input exceeds limit")
            return {"input_ids": [0] * length}
        a.prepare = prepare
        with patch("u1_benchmark.models.normalized", side_effect=lambda x: x):
            text = boundary_text(a, 512)
            self.assertEqual(len(prepare([text])["input_ids"]), 512)
            self.assertEqual(a.limit, 512)
            with self.assertRaisesRegex(ValueError, "Cannot construct"):
                boundary_text(a, 100)
            self.assertEqual(a.limit, 512)

    def test_quiet_dispatch_restores_after_error(self):
        original = lambda a: a
        module = SimpleNamespace(fn=original)
        wrapped = filtered(lambda a, scale=1: a * scale)
        with self.assertRaises(RuntimeError):
            with quiet_dispatch(module, {"fn": wrapped}):
                self.assertEqual(module.fn(2, scale=3, ignored=True), 6)
                raise RuntimeError()
        self.assertIs(module.fn, original)

    def test_recompilation_not_silent(self):
        a = SimpleNamespace(compile_count=0)
        def scores(texts):
            a.compile_count += 1
            return [SCORE]
        a.scores = scores
        with self.assertRaisesRegex(RuntimeError, "compilation"):
            stable_call(a, "text")

    def test_actual_process_timeout_partial_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invoke([sys.executable, "-c", "print('ok')"], root / "ok.log", 10)
            with self.assertRaises(TimeoutError):
                invoke([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(20)"], root / "slow.log", .3)
            self.assertEqual(core.read(root / "slow.process.json")["status"], "timeout")
            self.assertIn("started", (root / "slow.log").read_text())
            first = core.read(root / "ok.process.json")["pid"]
            self.assertNotEqual(first, core.read(root / "slow.process.json")["pid"])

    def test_archive_no_weights_secrets_or_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            core.write(root / "status.json", {"status": "failed"})
            (root / "model.onnx").write_bytes(b"weights")
            (root / ".env").write_text("secret")
            (root / "venv").mkdir()
            (root / "venv/token.json").write_text("secret")
            (root / "link.json").symlink_to(root / ".env")
            with zipfile.ZipFile(archive(root)) as z:
                self.assertEqual(set(z.namelist()), {"status.json", "archive_contents.json"})

    def test_memory_pid_uuid_and_separate_windows(self):
        self.assertEqual(process_mib("GPU-a, 1, 10\nGPU-b, 1, 99\nGPU-a, 2, 88", 1, "GPU-a"), 10)
        self.assertIsNone(process_mib("GPU-a, 1, N/A", 1, "GPU-a"))
        with tempfile.TemporaryDirectory() as tmp:
            s = Sampler("GPU-a")
            s.thread = SimpleNamespace(join=lambda **kwargs: None)
            s.rows = [{"phase": "setup", "started": 1, "gpu_process_mib": 500, "host_rss_bytes": 1},
                      {"phase": "inference", "started": 2, "gpu_process_mib": 100, "host_rss_bytes": 2},
                      {"phase": "inference", "started": 2.1, "gpu_process_mib": 110, "host_rss_bytes": 2}]
            result = s.stop(Path(tmp) / "memory.json")
            self.assertEqual(result["windows"]["inference"]["sampled_gpu_peak_mib"], 110)
            self.assertEqual(result["windows"]["setup"]["sampled_gpu_peak_mib"], 500)

    def test_archive_keeps_required_results_beyond_optional_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            core.write(root / "preflight/candidate/result.json", {"evidence": "x" * 256})
            core.write(root / "preflight/candidate/backend.json", {"duplicate": "x" * 256})
            core.write(root / "dataset/manifest.json", {"corpus": "x" * 256})
            target = Path(tmp) / "repacked.zip"
            with patch("u1_benchmark.process.OPTIONAL_FILE_LIMIT", 64), patch("u1_benchmark.process.OPTIONAL_ARCHIVE_LIMIT", 64):
                archive(root, target=target)
            with zipfile.ZipFile(target) as z:
                self.assertIn("preflight/candidate/result.json", z.namelist())
                self.assertIn("dataset/manifest.json", z.namelist())
                self.assertNotIn("preflight/candidate/backend.json", z.namelist())
                manifest = json.loads(z.read("archive_contents.json"))
                self.assertGreater(manifest["required_bytes"], 64)
                self.assertEqual(manifest["optional_bytes"], 0)


class OfflineReport(unittest.TestCase):
    def make_run(self, root, mode="screening"):
        checksum = dataset(root / "dataset")
        cfg = {"preset": "gliner_compile", "seed": 17, "mode": mode, "environment": "unused"}
        core.write(root / "config.json", cfg)
        core.write(root / "status.json", {"status": "completed"})
        with patch.object(core, "MANIFEST_SHA", checksum):
            data, _ = core.load_data(root / "dataset", "gliner")
        jobs = core.schedule(cfg, data)
        core.write(root / "schedule.json", jobs)
        for variant in ("baseline", "candidate"):
            pf = root / "preflight" / variant
            core.write(pf / "result.json", {"status": "passed", "contract": {"same": True},
                       "variant": variant, "config_sha256": core.digest(cfg)})
            (pf / "scores.jsonl").write_text("\n".join(json.dumps({"id": r["id"], "score": SCORE}) for r in data["B1"]))
        for i, j in enumerate(jobs):
            p = root / "latency" / f"{i:03d}"
            core.write(p / "result.json", {"status": "passed", "job": j, "contract": {"same": True},
                "variant": j["variant"], "config_sha256": core.digest(cfg),
                "rows": [{"id": k, "input_sha256": "f" * 64, "total_tokens": 512,
                          "latency_ms": 70 if j["variant"] == "candidate" else 100, "score": SCORE} for k in j["ids"]]})
            core.write(p / "runtime.json", dict.fromkeys(("python", "torch", "cuda", "cudnn", "gpu", "seed",
                                                          "cpu_threads", "tf32", "hub_kernels"), "same"))
            core.write(p / "telemetry_before.json", {"uuid": "same", "driver": "same"})
        for v in ("baseline", "candidate"):
            for i in range(3):
                core.write(root / "memory" / f"{v}_{i}" / "result.json", {"status": "passed", "variant": v,
                    "config_sha256": core.digest(cfg),
                    "pid": i, "torch_setup": {}, "torch_at_inference_start": {}, "torch_inference": {}, "size": {},
                    "memory": {"windows": {w: {"status": "measured"} for w in ("setup", "inference")}}})
            core.write(root / "profiles" / v / "result.json", {"status": "passed"})
        return checksum

    def test_complete_report_and_missing_side(self):
        from u1_benchmark.report import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checksum = self.make_run(root)
            with patch.object(core, "MANIFEST_SHA", checksum):
                result = build(root)
                self.assertEqual(result["status"], "screening_pass")
                self.assertEqual(result["speed_status"], "screening_pass")
                self.assertEqual(result["verdict_status"], "ok")
                self.assertEqual(len(result["memory"]), 6)
                self.assertFalse(result["quality_evaluated"])
                (root / "latency/009/result.json").unlink()
                self.assertEqual(build(root)["status"], "incomplete")

    def test_pilot_never_passes_and_environment_mismatch(self):
        from u1_benchmark.report import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checksum = self.make_run(root, "pilot")
            with patch.object(core, "MANIFEST_SHA", checksum):
                result = build(root)
                self.assertEqual(result["status"], "pilot_complete")
                self.assertEqual(result["speed_and_verdict_status"], "not_evaluated_pilot")
                bucket = result["buckets"]["B1"]
                self.assertEqual(bucket["status"], "pilot_complete")
                self.assertEqual(bucket["speed_status"], "not_evaluated_pilot")
                self.assertIsNone(bucket["p95"]["ratio_interval_95"])
                self.assertIsNone(bucket["baseline_drift_p95"])
                core.write(root / "latency/001/telemetry_before.json", {"uuid": "other", "driver": "same"})
                self.assertEqual(build(root)["status"], "incomplete")

    def test_fast_candidate_with_unstable_verdict_cannot_pass(self):
        from u1_benchmark.report import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checksum = self.make_run(root)
            path = root / "latency/005/result.json"
            item = core.read(path)
            self.assertEqual(item["variant"], "candidate")
            item["rows"][0]["score"] = {**SCORE, "label": "unsafe", "p_safe": .1, "p_unsafe": .9}
            core.write(path, item)
            with patch.object(core, "MANIFEST_SHA", checksum):
                result = build(root)
            self.assertEqual(result["speed_status"], "screening_pass")
            self.assertEqual(result["verdict_status"], "not_ok")
            self.assertEqual(result["status"], "rejected")

    def test_preflight_failure_stops_before_latency_and_archives(self):
        from u1_benchmark.orchestrator import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, out = root / "env", root / "out"
            core.write(env / "setup/status.json", {"status": "passed"})
            core.write(env / "setup/lock.json", core.read(core.ROOT / "env/team_environment.lock.json"))
            calls = []
            checksum = None
            def fake(args, log, timeout):
                nonlocal checksum
                calls.append([str(a) for a in args])
                if "_dataset" in args:
                    checksum = dataset(out / "dataset")
                    core.MANIFEST_SHA = checksum
                else:
                    raise RuntimeError("preflight fixture failure")
            with patch.object(core, "MANIFEST_SHA", "fixture"), patch("u1_benchmark.orchestrator.invoke", side_effect=fake):
                result = run({"preset": "gliner_compile", "environment": str(env)}, out)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(calls), 2)
            self.assertFalse((out / "latency").exists())
            self.assertTrue(out.with_suffix(".zip").exists())


class AdapterRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from test_experiments import ExperimentGlinerCPU
            ExperimentGlinerCPU.setUpClass()
            cls.fixture = ExperimentGlinerCPU()
        except ImportError as e:
            raise unittest.SkipTest(str(e))

    def test_compile_scores_routing_and_restore(self):
        import torch
        from u1_benchmark.models import activate
        from u1_benchmark.worker import verify_actual_request
        from u1_benchmark.models import normalized
        a = self.fixture.adapter()
        a.compile_count = 0
        text = "The public library."
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            before = a.scores([text])
            def fake_compile(fn, **kwargs):
                self.assertTrue(kwargs["fullgraph"])
                self.assertTrue(kwargs["dynamic"])
                a.compile_count += 1
                return fn
            with patch("torch.compile", side_effect=fake_compile), patch("torch.cuda.synchronize"):
                activate(a, "compile", "candidate", [text], tmp)
            self.assertEqual(a.scores([text]), before)
            row = {"text": text, "id": "x", "index": {"input_sha256": core.digest(normalized(a.prepare([text])))}}
            self.assertEqual(verify_actual_request(a, row), before[0])
            a.restore_encoder()
            self.assertEqual(a.scores([text]), before)

    def test_size_does_not_require_unloaded_model_card_files(self):
        import torch
        from u1_benchmark.models import model_size
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, output = root / "snapshot", root / "output"
            snapshot.mkdir()
            output.mkdir()
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"fixture")
            a = SimpleNamespace(model=torch.nn.Linear(2, 2), contract={"model": "fixture", "revision": "pin"})
            with patch("huggingface_hub.hf_hub_download", return_value=str(snapshot / "config.json")) as download:
                result = model_size(a, output)
            download.assert_called_once_with("fixture", "config.json", revision="pin", local_files_only=True)
            self.assertEqual(result["checkpoint_files_bytes"], {"model.safetensors": 7})
            self.assertEqual(result["unique_parameter_buffer_storage_bytes"], 24)

    def test_ort_route_wrapper_restores_shared_validation_texts(self):
        import torch
        from u1_benchmark.models import activate
        from followup import onnx_service
        a = self.fixture.adapter()
        a.compile_count = 0
        old = onnx_service.VALIDATION_TEXTS
        visited = []
        def install(adapter, path):
            original = adapter.model.encoder.forward
            def forward(*args, **kwargs):
                visited.append(True)
                return original(*args, **kwargs)
            adapter.model.encoder.forward = forward
            adapter.restore_encoder = lambda: setattr(adapter.model.encoder, "forward", original)
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            with patch("followup.team_smoke.preload_ort"), patch.object(onnx_service, "install", side_effect=install), \
                 patch.object(onnx_service, "finish"), patch("torch.cuda.synchronize"):
                activate(a, "ort", "candidate", ["The public library."], tmp)
            self.assertTrue(visited)
            self.assertIs(onnx_service.VALIDATION_TEXTS, old)
            a.restore_encoder()


if __name__ == "__main__":
    unittest.main()
