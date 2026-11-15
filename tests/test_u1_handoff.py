import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from u1_benchmark import contracts, datasets, quality, runner_v2
from u1_benchmark.adapters import check_index, load, parse_generated
from u1_benchmark.core import ROOT, digest, lines, read, sha, write
from u1_benchmark.worker_v2 import current_rows, stable_request


EXTERNAL = ROOT / "examples/u1_external_adapter.py"


class ToyPreparer:
    def __init__(self, name):
        self.name, self.assets, self.versions = name, {"fixture": "test"}, {}
        self.a = load(name, {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}}, ".")

    def parity(self):
        return {"status": "passed", "scope": "test fixture"}

    def index(self, row):
        ix = self.a.index(row["text"])
        n = ix["total_tokens"]
        b = next((k for k, spec in contracts.BUCKETS.items() if spec["min"] <= n <= spec["max"]), None)
        return {**ix, "id": row["id"], "bucket": b, "supported": n <= self.a.limit}


def speed_data(root, n=400):
    source = root / "source.jsonl"
    datasets.jsonl(source, [{"id": str(i), "text": ("change " if i < 4 else "keep ") + str(i)} for i in range(n)])
    spec = {"kind": "speed", "source": str(source), "fields": {"id": "id", "text": "text"},
            "version": "fixture-v1", "models": ["gliner"], "provenance": "unit-test"}
    return datasets.prepare(spec, root / "data", ToyPreparer)


def fixture_spec(candidate=False):
    return {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {},
            "options": {"change_verdict": candidate}}


def simulated_worker(job_path):
    """Real child process and real adapter calls; all timings are explicitly fake."""
    job = read(job_path)
    cfg, phase, variant = job["config"], job["phase"], job["variant"]
    path = Path(job["directory"])
    data, _ = datasets.load_speed(job["dataset"], cfg["model"], cfg["mode"], cfg["data"]["manifest_sha256"])
    rows = current_rows(phase, data, job.get("measurement"))
    adapter = load(cfg["model"], cfg[variant], path)
    adapter.prepare_backend([r["text"] for r in rows])
    runtime = {k: "fixture" for k in ("python", "torch", "cuda", "cudnn", "gpu", "seed", "cpu_threads", "tf32", "hub_kernels")}
    write(path / "runtime.json", runtime)
    write(path / "packages.json", {})
    write(path / "telemetry_before.json", {"uuid": "fixture", "driver": "fixture"})
    result = {"status": "passed", "config_sha256": digest(cfg), "variant": variant,
              "inputs_sha256": job["inputs_sha256"], "contract": adapter.contract, "metadata": adapter.metadata,
              "job": job.get("measurement"), "pid": os.getpid(), "size": adapter.size(), "setup_seconds": 0}
    if phase == "preflight":
        datasets.jsonl(path / "scores.jsonl", [{"id": r["id"], "bucket": r["bucket"], "score": adapter.verify(r["text"], r["index"])} for r in rows])
        result.update(boundary={"over_limit_rejected": True}, aba=[{"label": "safe"}] * 2)
    elif phase == "latency":
        result["rows"] = [{"id": r["id"], "input_sha256": r["index"]["input_sha256"], "total_tokens": r["index"]["total_tokens"],
                           "score": stable_request(adapter, r["text"]), "latency_ms": 60 if variant == "candidate" else 100} for r in rows]
        datasets.jsonl(path / "latencies.jsonl", result["rows"])
    elif phase == "memory":
        result.update(memory={"windows": {w: {"status": "measured"} for w in ("setup", "inference")}},
                      torch_setup={}, torch_inference={})
    elif phase == "profile":
        for name in ("short", "long"):
            import gzip
            with gzip.open(path / (name + ".trace.json.gz"), "wt") as stream:
                json.dump({"traceEvents": [{"cat": "kernel", "name": "fixture"}]}, stream)
    else:
        raise AssertionError(phase)
    write(path / "result.json", result)
    adapter.close()


class Contracts(unittest.TestCase):
    def test_generate_elided_mask_still_checks_actual_tokens(self):
        import torch
        from types import SimpleNamespace
        from u1_benchmark.adapters import Builtin, input_index
        class Model(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                return input_ids
        model = Model()
        ids = torch.tensor([[1, 2, 3]])
        expected = input_index({'input_ids': ids, 'attention_mask': torch.ones_like(ids)}, 3)
        def request(text):
            model(input_ids=ids)
            return {'label': 'safe', 'generated_tokens': 1}
        fake = SimpleNamespace(index=lambda text: expected, name='hive', target=model, forward_calls=0,
                               metadata={}, request=request, check_device=lambda: None)
        self.assertEqual(Builtin.verify(fake, 'abc', expected)['label'], 'safe')
        self.assertTrue(fake.metadata['generation_implicit_all_ones_mask'])
        ids[0, 1] = 4
        with self.assertRaises(ValueError):
            Builtin.verify(fake, 'abc', expected)

    def test_size_uses_loaded_files_not_unrelated_exports(self):
        from types import SimpleNamespace
        from u1_benchmark.adapters import Builtin
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'weights')
            fake = SimpleNamespace(legacy=None, name='koala', model=SimpleNamespace(parameters=lambda: [], buffers=lambda: []))
            with patch('huggingface_hub.try_to_load_from_cache', return_value=str(root / 'config.json')):
                result = Builtin.size(fake)
            self.assertEqual(result['checkpoint_files_bytes'], {'model.safetensors': 7})
        self.assertIn('unsafe>=0.65', contracts.RULES['halo'])

    def test_eight_models_and_limits(self):
        self.assertEqual(len(contracts.MODEL_SPECS), 8)
        for model in contracts.MODEL_SPECS:
            c = contracts.model_contract(model)
            self.assertEqual(len(c["revision"]), 40)
            self.assertEqual(len(contracts.required_buckets(model)), 1 if c["limit"] == 512 else 4)
        self.assertIn("hipocap", contracts.model_contract("llama")["model"])

    def test_responses_are_native(self):
        for value in ({"label": "S"}, {"label": "class9", "scores": {"class9": 1.}, "score_type": "probabilities"},
                      {"label": "safe", "p_safe": .9, "p_unsafe": .1, "public_confidence": .9, "logit_margin": -2.197},
                      {"label": "unsafe", "category": "S5", "generated_tokens": 4}):
            self.assertIs(contracts.validate_response(value), value)
        for value in ({"label": ""}, {"label": "safe", "scores": {"safe": float("nan")}, "score_type": "logits"},
                      {"label": "safe", "scores": {"safe": .2}, "score_type": "probabilities"},
                      {"label": "safe", "text": "private input"}):
            with self.assertRaises(ValueError):
                contracts.validate_response(value)
        self.assertEqual(parse_generated("unsafe\nS5", "open", 4)["category"], "S5")
        with self.assertRaises(ValueError):
            parse_generated("maybe safe", "hive", 1)

    def test_external_source_and_contract_swap(self):
        with self.assertRaises(ValueError):
            load("gliner", {**fixture_spec(), "source_sha256": "wrong"}, ".")
        a = load("gliner", fixture_spec(), ".")
        ix = a.index("Unicode e\u0301 日本語")
        a.verify("Unicode e\u0301 日本語", ix)
        with self.assertRaises(ValueError):
            check_index({**ix, "input_sha256": "changed"}, ix)
        a.reject_over_limit("x" * 513)


class Data(unittest.TestCase):
    def test_repeatability_private_archive_and_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = speed_data(root, 12)
            self.assertEqual(a["status"], "not_ready")
            rows, manifest = datasets.load_speed(a["path"], "gliner", "pilot", a["manifest_sha256"])
            self.assertEqual(len(rows["B1"]), 12)
            with self.assertRaises(ValueError):
                datasets.load_speed(a["path"], "gliner", "screening", a["manifest_sha256"])
            spec = {"kind": "speed", "source": str(root / "source.jsonl"), "fields": {"id": "id", "text": "text"},
                    "version": "fixture-v1", "models": ["gliner"], "provenance": "unit-test"}
            b = datasets.prepare(spec, root / "data2", ToyPreparer)
            self.assertEqual(a["manifest_sha256"], b["manifest_sha256"])
            (Path(a["path"]) / "corpus.jsonl").write_text("{}")
            with self.assertRaises(ValueError):
                datasets.load_speed(a["path"], "gliner", "pilot", a["manifest_sha256"])

    def test_boundaries_empty_unicode_special_tokens(self):
        p = ToyPreparer("halo")
        for length, bucket in ((512, "B1"), (513, "B2"), (1024, "B2"), (1025, "B3"), (2048, "B3"), (2049, "B4"), (4096, "B4")):
            ix = p.index({"id": "x", "text": "x" * length})
            self.assertEqual(ix["bucket"], bucket)
            self.assertEqual(ix["total_tokens"], length)
        for text in ("", "\n", "日本語🙂", "[MASK] <|im_end|>"):
            ix = p.index({"id": "x", "text": text})
            p.a.verify(text, ix)
        self.assertFalse(p.index({"id": "x", "text": "x" * 4097})["supported"])

    def test_quality_import_rejects_duplicate_ids_and_partial_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rows.jsonl"
            rows = [{"id": "a", "text": "x", "target": "yes", "slices": {"lang": "en"}},
                    {"id": "b", "text": "y", "target": "no", "slices": {}}]
            datasets.jsonl(p, rows)
            with self.assertRaises(ValueError):
                datasets.import_rows(p, {k: k for k in rows[0]}, "quality")
            rows[1] = rows[0]
            datasets.jsonl(p, rows)
            with self.assertRaises(ValueError):
                datasets.import_rows(p, {k: k for k in rows[0]}, "quality")


class Quality(unittest.TestCase):
    def test_metrics_coverage_and_undefined_denominators(self):
        rows = [{"id": "a", "target": "good", "slices": {"lang": "en"}},
                {"id": "b", "target": "bad", "slices": {"lang": "ru"}}]
        mapping = {"targets": {"good": "0", "bad": "1"}, "verdicts": {"safe": "0", "unsafe": "1"},
                   "labels": ["0", "1"], "positive_label": "1"}
        predictions = [{"id": "a", "status": "passed", "score": {"label": "safe"}}, {"id": "b", "status": "over_limit"}]
        result = quality.metrics(rows, predictions, mapping)
        self.assertEqual(result["overall"]["coverage"], .5)
        self.assertIsNone(result["overall"]["fnr"])
        self.assertIsNone(result["overall"]["macro_f1"])
        self.assertIsNone(result["quality_preserved"])
        with self.assertRaises(ValueError):
            quality.metrics(rows, predictions[:1], mapping)
        predictions[0]["score"]["label"] = "unknown"
        with self.assertRaises(ValueError):
            quality.metrics(rows, predictions, mapping)

    def test_unique_ids_and_instability(self):
        a = {"x": [{"label": "a"}] * 3, "y": [{"label": "b"}]}
        b = {"x": [{"label": "c"}] * 8, "y": [{"label": "b"}]}
        result = quality.agreement(a, b)
        self.assertEqual(result["changed_fraction"], .5)
        self.assertEqual(result["n"], 2)
        b["y"].append({"label": "d"})
        self.assertEqual(quality.agreement(a, b)["unstable_ids"], ["y"])


class EndToEnd(unittest.TestCase):
    def test_redaction_preserves_json_with_quotes(self):
        from u1_benchmark.worker_v2 import redact_files
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            secret = 'private "quoted"\ninput'
            write(path, {'metadata': secret, 'score': {'label': 'safe'}})
            redact_files(tmp, [secret, 'safe'])
            self.assertEqual(read(path), {'metadata': '[redacted-input]', 'score': {'label': 'safe'}})

    def test_baseline_only_complete_and_table_rejects_fixture(self):
        from u1_benchmark.report_v2 import table
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = speed_data(root)
            env = root / 'env'
            write(env / 'setup/status.json', {'status': 'passed'})
            write(env / 'setup/lock.json', read(ROOT / 'env/team_environment.lock.json'))
            cfg = {'schema_version': 2, 'model': 'gliner', 'baseline': fixture_spec(),
                   'data': {k: data[k] for k in ('path', 'manifest_sha256')}, 'environment': str(env)}
            with patch.object(runner_v2, 'invoke', side_effect=lambda args, *rest: simulated_worker(args[-1])):
                result = runner_v2.run(cfg, root / 'out')
            self.assertEqual(result['status'], 'fixture_complete', result)
            self.assertEqual(result['speed_status'], 'measured_baseline')
            self.assertEqual(result['buckets']['B1']['p95']['baseline_ms'], 100)
            self.assertEqual(len(result['memory']), 3)
            self.assertEqual(table([root / 'out.zip'], root / 'table.json')['rows'], 0)

    def test_real_quality_worker_with_cpu_fixture_and_private_texts(self):
        from u1_benchmark.worker_v2 import execute
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "quality.jsonl"
            datasets.jsonl(source, [{"id": "a", "text": "change private request alpha", "target": "bad", "slices": {"group": "tail"}},
                                    {"id": "b", "text": "keep private request beta", "target": "good", "slices": {"group": "main"}},
                                    {"id": "c", "text": "x" * 513, "target": "bad", "slices": {"group": "tail"}}])
            data = datasets.prepare({"source": str(source), "kind": "quality", "version": "fixture-q", "provenance": "unit-test",
                                     "fields": {k: k for k in ("id", "text", "target", "slices")}}, root / "data")
            cfg = contracts.parse_config({"schema_version": 2, "model": "gliner", "environment": str(root / "env"),
                "baseline": fixture_spec(True), "quality_data": {k: data[k] for k in ("path", "manifest_sha256")}, "mode": "quality",
                "label_mapping": {"targets": {"bad": "1", "good": "0"}, "verdicts": {"safe": "0", "unsafe": "1"},
                                  "labels": ["0", "1"], "positive_label": "1"}})
            job = {"config": cfg, "directory": str(root / "worker"), "variant": "baseline", "phase": "quality", "inputs_sha256": "test", "private": True}
            write(root / "job.json", job)
            with patch("followup.team_skeleton.probe"), patch("u1_benchmark.worker.telemetry", return_value={"uuid": "test"}), patch.object(torch.cuda, "synchronize"):
                outcome = execute(root / "job.json")
            self.assertEqual(outcome["status"], "passed", outcome)
            self.assertEqual(outcome["quality"]["overall"]["coverage"], 2 / 3)
            self.assertEqual(outcome["quality"]["overall"]["accuracy"], 1)
            self.assertEqual(outcome["quality"]["overall"]["failures"], {"over_limit": 1})
            from u1_benchmark.process import archive
            with zipfile.ZipFile(archive(root / "worker")) as z:
                self.assertFalse(any(b"private request" in z.read(n) for n in z.namelist()))

    def test_timeout_and_partial_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = speed_data(root, 12)
            env = root / "env"
            write(env / "setup/status.json", {"status": "passed"})
            write(env / "setup/lock.json", read(ROOT / "env/team_environment.lock.json"))
            cfg = {"schema_version": 2, "model": "gliner", "baseline": fixture_spec(), "mode": "pilot", "environment": str(env),
                   "data": {k: data[k] for k in ("path", "manifest_sha256")}}
            with patch.object(runner_v2, "invoke", side_effect=TimeoutError("fixture")):
                result = runner_v2.run(cfg, root / "out")
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(read(root / "out/status.json")["status"], "timeout")
            self.assertTrue((root / "out.zip").exists())

    def test_quantized_fixture_fresh_processes_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = speed_data(root)
            env = root / "env"
            write(env / "setup/status.json", {"status": "passed"})
            write(env / "setup/lock.json", read(ROOT / "env/team_environment.lock.json"))
            artifact = root / "toy.weights"
            artifact.write_bytes(b"not a quantized model")
            candidate = fixture_spec(True)
            base = contracts.MODEL_SPECS["gliner"]
            candidate["artifact"] = {"path": str(artifact), "sha256": digest(contracts.file_inventory(artifact)),
                "base_checkpoint": base["repo"], "base_revision": base["revision"], "method": "fixture", "bits": 4, "runtime": "cpu_fixture"}
            cfg = {"schema_version": 2, "model": "gliner", "baseline": fixture_spec(), "candidate": candidate,
                   "comparison": "quantization", "data": {k: data[k] for k in ("path", "manifest_sha256")}, "environment": str(env)}
            def invoke(args, log, timeout):
                self.assertGreater(timeout, 0)
                self.assertIn("_worker_v2", [str(x) for x in args])
                with Path(log).open("w") as stream:
                    subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0,'tests'); from test_u1_handoff import simulated_worker; simulated_worker(sys.argv[1])", str(args[-1])],
                                   cwd=ROOT, check=True, stdout=stream, stderr=stream, timeout=30)
            with patch.object(runner_v2, "invoke", side_effect=invoke):
                result = runner_v2.run(cfg, root / "out")
            self.assertEqual(result["status"], "fixture_complete", result)
            self.assertEqual(result["speed_status"], "measured_no_acceptance_threshold")
            self.assertEqual(result["buckets"]["B1"]["verdicts"]["changed_count"], 4)
            self.assertEqual(result["macro_verdict_change_fraction"], .01)
            pids = [read(p)["pid"] for p in (root / "out").glob("*/**/result.json")]
            self.assertEqual(len(pids), len(set(pids)))
            with zipfile.ZipFile(root / "out.zip") as z:
                self.assertFalse(any(".private" in n or n == "corpus.jsonl" for n in z.namelist()))
                self.assertFalse(any(b"change 0" in z.read(n) for n in z.namelist() if n.endswith((".json", ".jsonl", ".log"))))
            from u1_benchmark.report_v2 import build
            before = copy.deepcopy(result)
            self.assertEqual(build(root / "out"), before)
            outcome = root / "out/latency/004/result.json"
            out = read(outcome)
            out["rows"].pop()
            write(outcome, out)
            self.assertEqual(build(root / "out")["status"], "incomplete")

    def test_baseline_schedule_and_current_shapes(self):
        data = {"B1": [{"id": str(i), "index": {"total_tokens": i + 1}} for i in range(400)]}
        cfg = {"mode": "screening", "seed": 17}
        jobs = runner_v2.schedule(cfg, data)
        self.assertEqual(len(jobs), 7)
        self.assertTrue(all(j["variant"] == "baseline" for j in jobs))
        self.assertEqual([len(j["ids"]) for j in jobs], [30] * 4 + [400] * 3)
        self.assertEqual(len(current_rows("latency", data, jobs[0])), 30)
        cfg["candidate"] = {"backend": "compile"}
        jobs = runner_v2.schedule(cfg, data)
        self.assertEqual(len(jobs), 10)
        for a, b in zip(jobs[::2], jobs[1::2]):
            self.assertEqual(a["ids"], b["ids"])


if __name__ == "__main__":
    unittest.main()
