import copy
import json
import tempfile
import unittest
import os
import sys
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

from u1_benchmark import contracts, datasets, quality, role_data, runner_v2
from u1_benchmark.adapters import check_index, load
from u1_benchmark.core import ROOT, digest, read, sha, write
from u1_benchmark.environment import auth_check, expected_lock, rtx_name, rtx_requirements
from u1_benchmark.messages import UnsupportedInput, content_hash, messages, payload, select
from u1_benchmark.statistics import baseline_diagnostics

U = {"role": "user", "content": "User question"}
A = {"role": "assistant", "content": "Assistant answer"}
EXTERNAL = ROOT / "examples/u1_external_adapter.py"


class ToyRolePreparer:
    def __init__(self, name):
        self.name, self.assets, self.versions = name, {"fixture": "test-only"}, {}
        self.adapter = load(name, {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}}, ".", 3)

    def parity(self):
        return {"status": "passed", "scope": "CPU test fixture"}

    def index(self, row):
        ix = self.adapter.index(payload(row))
        n = ix["total_tokens"]
        bucket = next((k for k, b in contracts.BUCKETS.items() if b["min"] <= n <= b["max"]), None)
        return {**ix, "id": row["id"], "supported": n <= self.adapter.limit, "bucket": bucket, "reason": None}


def fixture_data(root, count=400):
    source = root / "messages.jsonl"
    datasets.jsonl(source, [{"id": str(i), "messages": [{"role": "assistant", "content":
        ("change " if i < 4 else "keep ") + str(i)}], "language": "ru" if i % 2 else "en"} for i in range(count)])
    spec = {"schema_version": 3, "kind": "speed", "source": str(source),
            "fields": {k: k for k in ("id", "messages", "language")}, "models": ["gliner"],
            "version": "fixture-roles", "provenance": "test-only"}
    return datasets.prepare(spec, root / "data", ToyRolePreparer)


def simulated_role_worker(job_path):
    """Actual adapter and files; explicitly fake CPU timings, never a GPU result."""
    from u1_benchmark.worker_v2 import current_rows
    job = read(job_path)
    cfg, phase, variant = job["config"], job["phase"], job["variant"]
    directory = Path(job["directory"])
    data, _ = datasets.load_speed(job["dataset"], cfg["model"], cfg["mode"], cfg["data"]["manifest_sha256"])
    rows = current_rows(phase, data, job.get("measurement"))
    adapter = load(cfg["model"], cfg[variant], directory, 3)
    adapter.prepare_backend([payload(r) for r in rows])
    write(directory / "runtime.json", {k: "fixture" for k in ("python", "torch", "cuda", "cudnn", "gpu", "seed", "cpu_threads", "tf32", "hub_kernels")})
    write(directory / "packages.json", {})
    write(directory / "telemetry_before.json", {"uuid": "fixture", "driver": "fixture"})
    result = {"status": "passed", "variant": variant, "pid": os.getpid(), "contract": adapter.contract,
              "config_sha256": digest(cfg), "inputs_sha256": job["inputs_sha256"], "job": job.get("measurement"),
              "metadata": adapter.metadata, "size": adapter.size(), "setup_seconds": 0}
    if phase == "preflight":
        datasets.jsonl(directory / "scores.jsonl", [{"id": r["id"], "bucket": r["bucket"], "score": adapter.verify(payload(r), r["index"])} for r in rows])
        result.update(boundary={"over_limit_rejected": True}, aba=[{"label": "safe"}] * 2)
    elif phase == "latency":
        result["rows"] = [{"id": r["id"], "input_sha256": r["index"]["input_sha256"], "total_tokens": r["index"]["total_tokens"],
                           "score": adapter.request(payload(r)), "latency_ms": 60 if variant == "candidate" else 100} for r in rows]
        datasets.jsonl(directory / "latencies.jsonl", result["rows"])
    elif phase == "memory":
        result.update(memory={"windows": {w: {"status": "measured"} for w in ("setup", "inference")}}, torch_setup={}, torch_inference={})
    elif phase == "profile":
        import gzip
        for name in ("short", "long"):
            with gzip.open(directory / (name + ".trace.json.gz"), "wt") as stream:
                json.dump({"traceEvents": [{"cat": "kernel", "name": "fixture"}]}, stream)
    write(directory / "result.json", result)


class Roles(unittest.TestCase):
    def test_canonical_validation_and_hash(self):
        self.assertEqual(messages("hello"), [{"role": "user", "content": "hello"}])
        self.assertNotEqual(content_hash([U]), content_hash([{**U, "role": "assistant"}]))
        for value in ([], [A, U], [U, U], [{"role": "tool", "content": "x"}], [{**U, "secret": 1}]):
            with self.assertRaises(ValueError):
                messages(value)

    def test_eight_role_contracts(self):
        for model in ("gliner", "koala", "llama", "laya"):
            self.assertEqual(select(model, [U, A])["effective_messages"], [A])
            self.assertEqual(select(model, [U, A])["context_indices"], [])
        self.assertEqual(select("halo", [U, A])["effective_messages"], [U])
        self.assertEqual(select("hive", [A])["target_role"], "assistant")
        self.assertEqual(select("hive", [U, A])["context_indices"], [0])
        self.assertEqual(select("shield", [U, A])["target_index"], 1)
        for model, value in (("halo", [A]), ("shield", [A]), ("open", [A]), ("open", [U, A])):
            with self.assertRaises(UnsupportedInput):
                select(model, value)

    def test_new_contract_does_not_replace_old_open(self):
        old, new = contracts.model_contract("open"), contracts.model_contract("open", 3)
        self.assertIn("2510", old["rules"])
        self.assertIn("native 0124", new["rules"])
        self.assertNotIn("template_repo", new["preparation_spec"])
        self.assertEqual(old["revision"], new["revision"])
        self.assertNotEqual(old["preparation_sha256"], new["preparation_sha256"])

    def test_target_change_cannot_pass_index_check(self):
        a = load("gliner", {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}}, ".", 3)
        ix = a.index([U, A])
        with self.assertRaises(ValueError):
            check_index({**ix, "target_role": "user"}, ix)
        a.verify([U, A], ix)

    def test_boundaries_and_unicode(self):
        for n, bucket in ((512, "B1"), (513, "B2"), (1024, "B2"), (1025, "B3"), (2048, "B3"),
                          (2049, "B4"), (4096, "B4")):
            b = contracts.BUCKETS[bucket]
            self.assertTrue(0 <= role_data.stratum(n, b) <= 7)
        for n in (0, 513):
            with self.assertRaises(ValueError):
                role_data.stratum(n, contracts.BUCKETS["B1"])
        for text in ("", "ё e\u0301 🙂\n<|im_end|>"):
            self.assertEqual(messages([{"role": "assistant", "content": text}])[0]["content"], text)

    def test_model_buckets_reproducible_and_stale_contract_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = fixture_data(root)
            rows, manifest = datasets.load_speed(first["path"], "gliner", "screening", first["manifest_sha256"])
            self.assertEqual(len(rows["B1"]), 400)
            spec = {"schema_version": 3, "kind": "speed", "source": str(root / "messages.jsonl"),
                    "fields": {k: k for k in ("id", "messages", "language")}, "models": ["gliner"],
                    "version": "fixture-roles", "provenance": "test-only"}
            second = datasets.prepare(spec, root / "again", ToyRolePreparer)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            with patch("u1_benchmark.role_data.model_contract", return_value={}):
                with self.assertRaises(ValueError):
                    datasets.load_speed(first["path"], "gliner", "screening")
            schedule = runner_v2.schedule({"mode": "screening", "seed": 17, "candidate": {}}, rows)
            pairs = {}
            for job in schedule:
                key = (job["phase"], job["block"], job["bucket"])
                pairs.setdefault(key, []).append(job["ids"])
            self.assertTrue(all(a == b for a, b in pairs.values()))

    def test_depletion_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = fixture_data(Path(tmp), 12)
            self.assertEqual(result["status"], "not_ready")
            with self.assertRaises(ValueError):
                datasets.load_speed(result["path"], "gliner", "screening")
            self.assertEqual(len(datasets.load_speed(result["path"], "gliner", "pilot")[0]["B1"]), 12)

    def test_effective_input_deduplication_and_length_strata(self):
        rows, index = [], {}
        for n in range(1, 513):
            for j in range(2):
                key = f"{n}:{j}"
                rows.append({"id": key, "language": "en"})
                index[key] = {"supported": True, "total_tokens": n, "input_sha256": str(n), "source_mode": "user-only"}
        ids, eligible, histogram = role_data.select_bucket(rows, index, contracts.BUCKETS["B1"], "test")
        self.assertEqual(eligible, 512)
        self.assertEqual(histogram, [50] * 8)
        self.assertEqual(len({index[key]["input_sha256"] for key in ids}), 400)

    def test_native_templates_local_cache(self):
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from jinja2 import TemplateError
        except ImportError:
            self.skipTest("Jinja2 unavailable")
        cache = Path.home() / ".cache/huggingface/hub"
        hive = cache / "models--hivetrace--HiveTraceGuard-Pro/snapshots/ed008276c96253e14885e6ade3b7b6d58daa2480/chat_template.jinja"
        shield = cache / "models--google--shieldgemma-2b/snapshots/d1dffc9c8c9237a90aab09c61383791e718ef9e8/tokenizer_config.json"
        if not hive.is_file() or not shield.is_file():
            self.skipTest("Pinned tokenizer assets not cached; no network in tests")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        rendered = env.from_string(hive.read_text()).render(messages=[U, A])
        self.assertIn("Target: last assistant message", rendered)
        template = env.from_string(read(shield)["chat_template"])
        rendered = template.render(messages=[U, A], guideline="response-policy", bos_token="<bos>")
        self.assertIn("Does the Chatbot Response", rendered)
        with self.assertRaises(Exception):
            template.render(messages=[A], guideline="response-policy", bos_token="<bos>")

    def test_quality_target_coverage_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "quality.jsonl"
            datasets.jsonl(source, [{"id": "q", "messages": [U, A], "target": "bad", "target_role": "assistant", "slices": {"language": "en"}}])
            spec = {"schema_version": 3, "kind": "quality", "source": str(source),
                    "fields": {k: k for k in ("id", "messages", "target", "target_role", "slices")},
                    "version": "test", "provenance": "fixture"}
            result = datasets.prepare(spec, root / "quality")
            rows, _ = datasets.verified(result["path"])
            self.assertEqual(rows[0]["target_index"], 1)
            mapping = {"labels": ["negative", "positive"], "targets": {"bad": "positive"},
                       "verdicts": {"safe": "negative", "unsafe": "positive"}, "positive_label": "positive"}
            metrics = quality.metrics(rows, [{"id": "q", "status": "target_mismatch"}], mapping)
            self.assertEqual(metrics["overall"]["coverage"], 0)
            self.assertIsNone(metrics["overall"]["fnr"])
            public = runner_v2.public_rows(rows)
            self.assertNotIn("messages", public[0])
            self.assertNotIn("Assistant answer", json.dumps(public))

    def test_quantized_fixture_unique_flips(self):
        spec = {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}}
        a, b = load("gliner", spec, ".", 3), load("gliner", {**spec, "options": {"change_verdict": True}}, ".", 3)
        baseline, candidate = {}, {}
        for i in range(400):
            value = [{"role": "assistant", "content": ("change " if i < 4 else "keep ") + str(i)}]
            baseline[str(i)] = [a.request(value)] * 3
            candidate[str(i)] = [b.request(value)] * 3
        result = quality.agreement(baseline, candidate)
        self.assertEqual((result["n"], result["changed_count"], result["changed_fraction"]), (400, 4, .01))
        self.assertEqual(result["unstable_ids"], [])

    def test_role_quantization_full_fresh_process_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture_data(root)
            env = root / "env"
            write(env / "setup/status.json", {"status": "passed"})
            write(env / "setup/lock.json", read(ROOT / "env/team_rtx.lock.json"))
            write(env / "setup/profile.json", {"profile": "team-rtx"})
            artifact = root / "fixture.weights"
            artifact.write_bytes(b"test only; not quantized weights")
            spec = {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}}
            base = contracts.MODEL_SPECS["gliner"]
            candidate = {**spec, "options": {"change_verdict": True}, "artifact": {"path": str(artifact),
                "sha256": digest(contracts.file_inventory(artifact)), "base_checkpoint": base["repo"],
                "base_revision": base["revision"], "method": "fixture", "bits": 4, "runtime": "cpu_fixture"}}
            cfg = {"schema_version": 3, "model": "gliner", "baseline": spec, "candidate": candidate,
                   "comparison": "quantization", "environment": str(env), "data": {k: data[k] for k in ("path", "manifest_sha256")}}
            def invoke(args, log, timeout):
                with Path(log).open("w") as stream:
                    subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0,\"tests\"); from test_u1_roles import simulated_role_worker; simulated_role_worker(sys.argv[1])", str(args[-1])],
                                   cwd=ROOT, check=True, stdout=stream, stderr=stream, timeout=30)
            with patch.object(runner_v2, "invoke", side_effect=invoke):
                result = runner_v2.run(cfg, root / "out")
            self.assertEqual(result["status"], "fixture_complete", result)
            self.assertEqual(result["macro_verdict_change_fraction"], .01)
            pids = [read(path)["pid"] for path in (root / "out").glob("*/**/result.json")]
            self.assertEqual(len(pids), len(set(pids)))
            with zipfile.ZipFile(root / "out.zip") as archive:
                self.assertFalse(any(b"change 0" in archive.read(name) for name in archive.namelist()
                                     if name.endswith((".json", ".jsonl", ".log"))))

    def test_real_role_quality_worker_target_mismatch_and_privacy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from u1_benchmark.worker_v2 import execute
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            datasets.jsonl(source, [{"id": "a", "messages": [U, A], "target": "bad", "target_role": "assistant"},
                {"id": "b", "messages": [U, A], "target": "bad", "target_role": "user"}])
            data = datasets.prepare({"schema_version": 3, "kind": "quality", "source": str(source),
                "fields": {k:k for k in ("id", "messages", "target", "target_role")}, "version": "fixture", "provenance": "test"}, root / "data")
            cfg = contracts.parse_config({"schema_version": 3, "model": "gliner", "environment": str(root / "env"),
                "baseline": {"module": str(EXTERNAL), "source_sha256": sha(EXTERNAL), "dependencies": {}},
                "quality_data": {k:data[k] for k in ("path", "manifest_sha256")}, "mode": "quality",
                "label_mapping": {"targets": {"bad":"1"}, "verdicts": {"safe":"0", "unsafe":"1"}, "labels":["0","1"]}})
            write(root / "job.json", {"config": cfg, "directory": str(root / "worker"), "variant": "baseline", "phase": "quality", "inputs_sha256": "fixture", "private": True})
            with patch("u1_benchmark.environment.probe"), patch("u1_benchmark.worker.telemetry", return_value={"uuid":"fixture"}), patch.object(torch.cuda,"synchronize"):
                result = execute(root / "job.json")
            self.assertEqual(result["status"], "passed", result)
            self.assertEqual(result["quality"]["overall"]["failures"], {"target_mismatch":1})
            self.assertEqual(result["quality"]["overall"]["coverage"], .5)
            self.assertNotIn("Assistant answer", (root / "worker/result.json").read_text())


class EnvironmentAndNoise(unittest.TestCase):
    def test_auth_failure_never_exposes_token_or_exception_text(self):
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest("huggingface_hub unavailable")
        with patch.object(huggingface_hub, "get_token", return_value="secret-token"), patch.object(
                huggingface_hub.HfApi, "list_repo_files", side_effect=RuntimeError("secret-token in url")):
            result = auth_check("shield")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertNotIn("secret-token", json.dumps(result))
        self.assertFalse(result["weights_downloaded"])

    def test_auth_check_only_fetches_metadata(self):
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest("huggingface_hub unavailable")
        with patch.object(huggingface_hub, "get_token", return_value="test-token"), patch.object(
                huggingface_hub.HfApi, "list_repo_files", return_value=["config.json", "model.safetensors"]), patch.object(
                huggingface_hub, "get_hf_file_metadata") as head:
            result = auth_check("shield")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(head.call_count, 2)

    def test_rtx_lock_only_adds_laya(self):
        base = read(ROOT / "env/team_environment.lock.json")
        new = read(ROOT / "env/team_rtx.lock.json")
        self.assertEqual([p for p in new["packages"] if p["name"] != "laya"], base["packages"])
        self.assertEqual(new["package_count"], 97)
        self.assertFalse(new["source_smoke_passed"])
        self.assertEqual(rtx_requirements(new), (ROOT / "env/team_rtx.lock.txt").read_text())
        try:
            from packaging.requirements import Requirement
        except ImportError:
            self.skipTest("packaging unavailable")
        versions = {p["name"]: p["version"] for p in new["packages"]}
        for raw in new["laya_requires_dist"]:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({"extra": "", "python_version": "3.13"}):
                self.assertIn(req.name.replace("_", "-"), versions)
                self.assertIn(versions[req.name.replace("_", "-")], req.specifier)

    def test_gpu_names_no_fallback(self):
        for name in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090", "RTX 4090"):
            self.assertTrue(rtx_name(name))
        for name in ("Tesla T4", "RTX 4090 D", "RTX 5090 D", "CPU", "RTX 3090"):
            self.assertFalse(rtx_name(name))

    def test_old_profile_default_and_new_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(expected_lock(root), read(ROOT / "env/team_environment.lock.json"))
            write(root / "setup/profile.json", {"profile": "team-rtx"})
            self.assertEqual(expected_lock(root)["profile"], "team-rtx")

    def test_baseline_noise_is_diagnostic_not_acceptance(self):
        blocks = [[100.] * 20, [200.] * 20, [300.] * 20]
        result = baseline_diagnostics(blocks, [([100.] * 30, [300.] * 30)] * 2)
        self.assertEqual(result["baseline_drift_p95"], 2)
        self.assertEqual(result["aa_envelope_p95"], 2)
        self.assertFalse(result["stability_assessed"])
        self.assertEqual(result["baseline_series_p95_ms"], [100, 200, 300])
        self.assertEqual(result, baseline_diagnostics(blocks, [([100.] * 30, [300.] * 30)] * 2))
        with self.assertRaises(ValueError):
            baseline_diagnostics([[]], [([1], [1])])


if __name__ == "__main__":
    unittest.main()
