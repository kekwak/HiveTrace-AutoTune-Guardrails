"""CPU-only saved artifacts; no model, dataset, GPU or network access."""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from u1_benchmark.core import digest, read, sha, write
from u1_benchmark.v4_config import PROTOCOL_VERSION, validate_config
from u1_benchmark.v4_evidence import environment_identity
from u1_benchmark.v4_planning import schedule, smoke_plans
from u1_benchmark.v4_report import _quantization, agreement, build, frozen_config, metrics, table


def answer(label="class_a"):
    return {"label": label, "scores": {"class_a": .6, "class_b": .3, "class_c": .1},
            "score_type": "probabilities"}


PRECISION = {"parameter_dtypes": ["float32"], "autocast_enabled": False,
             "autocast_dtype": None, "policy": "explicit"}
ENV = {"gpu": "synthetic", "packages": {"engine": "1"}, "monitor_method": "synthetic-v2",
       "gpu_uuid": "test-gpu", "pid": 1, "timestamp": 1}
OBSERVED = {"artifacts": {"weight_files": {"model.safetensors": digest("weights")},
                          "weight_inventory_sha256": digest({"model.safetensors": digest("weights")})},
            "preparation": {"assets": {"effective_tokenizer": digest("tokenizer"),
                                       "effective_chat_template": digest("template")}},
            "batch_policy": {"padding": "longest", "padding_side": "right", "truncation": False,
                             "sort_by_length": False, "pad_token_id": 0},
            "attention": "sdpa", "linear_attention": "native"}


class Archive:
    def __init__(self, root, mode="baseline", pilot=False, gain=.30, monitoring=True,
                 comparison="engineering", count=400, env=None, corpus="corpus", buckets=None, sizes=(4,)):
        self.root = root
        spec = {"adapter": "koala", "weights": {"path": "/missing/weights", "sha256": "a" * 64},
                "tokenizer": {"path": "/missing/tokenizer", "sha256": "b" * 64}, "template": None,
                "preparation": {"limit": 64}, "scoring": {},
                "execution": {"runtime": "eager", "dtype": "float32"}}
        cfg = {"schema_version": 4, "baseline": spec, "mode": mode, "pilot": pilot,
               "buckets": buckets or {"custom": {"min": 1, "max": 64}}, "comparison": comparison,
               "data": {"path": "/missing/corpus", "manifest_sha256": digest(corpus)}}
        cfg["batch_sizes_by_bucket"] = {b: list(sizes) for b in cfg["buckets"]}
        if mode in ("search", "confirm", "quality"):
            cfg["candidate"] = deepcopy(spec)
            cfg["candidate"]["execution"]["attention"] = "sdpa"
        if mode == "search":
            cfg["baseline_ref"] = "/missing/reference.zip"
        if mode == "quality":
            cfg.pop("data")
            cfg.pop("buckets")
            cfg.pop("batch_sizes_by_bucket")
            cfg["quality_data"] = {"path": "/missing/quality", "manifest_sha256": digest(corpus)}
            cfg["quality_batch_sizes"] = [1, 2]
            cfg["label_mapping"] = {"labels": ["a", "b", "c"], "targets": {"a": "a", "b": "b", "c": "c"},
                                    "verdicts": {"class_a": "a", "class_b": "b", "class_c": "c"}}
        if comparison == "quantization":
            base_cfg = {k: v for k, v in cfg.items() if k not in ("candidate", "baseline_ref")}
            base_cfg.update(mode="baseline", comparison="engineering")
            cfg["candidate"]["adapter"] = {"module": "/missing/quantized.py", "source_sha256": digest("quantized"), "dependencies": {}}
            cfg["candidate"]["quantization"] = {"baseline_identity": digest(validate_config(base_cfg)["baseline"])}
        self.cfg = validate_config(cfg)
        self.data = {b: [{"id": f"{b}-{i}", "index": {"input_sha256": digest([b, i]), "total_tokens": 8 + i % 10}}
                         for i in range(count)] for b in self.cfg.get("buckets", {})}
        plans = schedule(self.cfg, self.data) if mode != "quality" else []
        environment = env or ENV
        self.resolved = {"schema_version": 4, "result_version": 4, "protocol_version": PROTOCOL_VERSION,
                         "config": self.cfg, "source_inventory": {"old/absent.py": digest("source")},
                         "environment": environment, "data_manifest_sha256": digest(corpus),
                         "input_indices": {b: [{"id": r["id"], **r["index"]} for r in rows]
                                           for b, rows in self.data.items()}, "schedule": plans,
                         "monitoring_validation": {"status": "passed", "monitor_method": environment["monitor_method"],
                            "environment": environment_identity(environment),
                            "source_inventory_sha256": digest({"old/absent.py": digest("source")}),
                            "checks": {name: {"status": "passed", "p50": {"ratio_interval_95": [.99, 1.02]},
                                             "p95": {"ratio_interval_95": [.99, 1.03]}} for name in ("gliner", "open")}}
                            if monitoring else None}
        if mode == "quality":
            self.resolved.pop("input_indices")
            self.resolved["quality_rows"] = [{"id": f"q{i}", "target": label, "slices": {"language": "en"}}
                                             for i, label in enumerate(("a", "b", "c"))]
        write(root / "resolved_run.json", self.resolved)
        self.state = {"status": "completed", "resolved_sha256": sha(root / "resolved_run.json"), "stages": []}
        self.workers = []
        variants = ["candidate"] if mode == "search" else ["baseline"] + (["candidate"] if "candidate" in self.cfg else [])
        if mode == "quality":
            for size in self.cfg["quality_batch_sizes"]:
                for variant in variants:
                    self.add_quality(size, variant)
        else:
            for bucket in sorted(self.data):
                rows = self.data[bucket]
                for size in sorted(sizes):
                    preflights = {}
                    for variant in variants:
                        job = {"stage": "preflight", "variant": variant, "bucket": bucket, "batch_size": size, "bucket_max": 64}
                        plan = smoke_plans(rows, size, self.cfg["seed"])
                        path = self.add(job, {"plans": plan, "backend_proof": {"kernels": ["native"]},
                            "aba_passed": True, "over_limit_rejected": True, "boundary_checked": 64,
                            "responses": [[answer() for _ in batch["ids"]] for batch in plan]})
                        preflights[variant] = path
                    for job in plans:
                        if (job["bucket"], job["batch_size"]) != (bucket, size):
                            continue
                        latency = 10 * (1 - gain) if job["variant"] == "candidate" else 10
                        value = {"rows": [{"batch_id": p["id"], **{k: p[k] for k in
                            ("ids", "input_sha256", "total_tokens", "padded_tokens", "useful_tokens")},
                            "latency_ms": latency, "scores": [answer() for _ in p["ids"]], "timestamp": i + 1}
                            for i, p in enumerate(job["batches"])], "setup_seconds": 1, "warmup_seconds": 1,
                            "compile_before": 0, "compile_after": 0}
                        pf = preflights[job["variant"]]
                        self.add({**job, "stage": "latency", "preflight_path": str(pf / "result.json"),
                                  "preflight_sha256": sha(pf / "result.json")}, value)
        write(root / "status.json", self.state)

    def add(self, job, extra):
        path = self.root / "jobs" / f"{len(self.workers):05d}"
        job = {**job, "resolved_sha256": self.state["resolved_sha256"], "seed": self.cfg["seed"],
               "resolved_path": "/relocated/resolved_run.json", "input_path": "/deleted/.private/inputs.json",
               "input_file_sha256": digest("private input")}
        value = {"result_version": 4, "job": job, "status": "passed", "resolved_sha256": self.state["resolved_sha256"],
                 "metadata": {"precision": deepcopy(PRECISION)}, "runtime": deepcopy(self.resolved["environment"]), **extra}
        write(path / "job.json", job)
        write(path / "result.json", value)
        self.state["stages"].append({"path": str(path.relative_to(self.root)), "stage": job["stage"],
                                     "variant": job["variant"], "status": value["status"]})
        self.workers.append(path)
        return path

    def add_quality(self, size, variant):
        rows = self.resolved["quality_rows"]
        ids = [r["id"] for r in rows]
        random.Random(self.cfg["seed"]).shuffle(ids)
        indices = {key: {"input_sha256": digest(key), "total_tokens": 8} for key in ids}
        batches = []
        for start in range(0, len(ids), size):
            members = ids[start:start + size]
            batches.append({"ids": members, "actual_batch_size": len(members), "lengths": [8] * len(members),
                            "padded_length": 8, "inputs_sha256": digest([indices[key] for key in members])})
        plan = {"quality_protocol_version": 1, "batch_size": size, "seed": self.cfg["seed"], "ordered_ids": ids,
                "indices": indices, "excluded": {}, "batches": batches}
        predictions = [{"id": key, "status": "passed", "score": answer("class_" + rows[int(key[1:])]["target"])} for key in ids]
        path = self.add({"stage": "quality", "variant": variant, "quality_batch_size": size,
                         "label_mapping": self.cfg["label_mapping"]},
                        {"quality_batch_size": size, "quality_plan_sha256": digest(plan),
                         "backend_proof": {"kernels": ["native"]}, "quality": metrics(rows, predictions, self.cfg["label_mapping"])})
        write(path / "quality_plan.json", plan)
        self.predictions(path, predictions)
        return path

    @staticmethod
    def predictions(path, predictions):
        (path / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))

    def worker(self, stage="latency", variant=None, phase=None):
        return next(p for p in self.workers if (j := read(p / "job.json"))["stage"] == stage
                    and (variant is None or j["variant"] == variant) and (phase is None or j.get("phase") == phase))

    def mutate(self, path, change):
        value = read(path / "result.json")
        change(value)
        write(path / "result.json", value)

    def refreeze(self):
        write(self.root / "resolved_run.json", self.resolved)
        checksum = self.state["resolved_sha256"] = sha(self.root / "resolved_run.json")
        preflights = {}
        for path in self.workers:
            job = read(path / "job.json")
            job["resolved_sha256"] = checksum
            if job["stage"] == "latency":
                job["preflight_sha256"] = preflights[job["bucket"], job["batch_size"], job["variant"]]
            value = read(path / "result.json")
            value.update(job=job, resolved_sha256=checksum)
            write(path / "job.json", job)
            write(path / "result.json", value)
            if job["stage"] == "preflight":
                preflights[job["bucket"], job["batch_size"], job["variant"]] = sha(path / "result.json")
        write(self.root / "status.json", self.state)

    def observations(self, variants=("baseline", "candidate")):
        for path in self.workers:
            if read(path / "job.json")["variant"] in variants:
                self.mutate(path, lambda v: v["metadata"].update(deepcopy(OBSERVED)))
        self.refreeze()

    def reference(self, baseline):
        build(baseline.root)
        shutil.copytree(baseline.root, self.root / "reference")
        root = self.root / "reference"
        files = {str(p.relative_to(root)): sha(p) for p in root.rglob("*.json")}
        write(self.root / "reference.json", {"sha256": digest(files), "files": files})


class V4ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def archive_and_remove(self, source, name):
        target = self.root / name
        with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(source)))
            archive.writestr("sources/u1_benchmark/v4_report.py", "raise RuntimeError('archived code must not execute')\n")
        shutil.rmtree(source)
        return target

    def cli(self, *arguments):
        tool = Path(__file__).resolve().parents[1] / "tools" / "benchmark.py"
        result = subprocess.run([sys.executable, "-B", str(tool), *map(str, arguments)], capture_output=True, text=True)
        self.assertNotIn("Traceback", result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_baseline_complete_without_profiles_or_memory(self):
        archive = Archive(self.root)
        result = build(self.root)
        self.assertEqual(result["status"], "baseline_complete")
        cell = result["cells"]["custom/4"]
        self.assertEqual(cell["p50"]["baseline_ms"], 10)
        self.assertEqual(cell["p95"]["baseline_interval_95_ms"], [10, 10])
        self.assertEqual(len(cell["baseline_series_p95_ms"]), 3)
        self.assertEqual(len(cell["aa"]["block_ratios"]), 2)
        self.assertEqual(result["memory_status"], "unavailable")
        self.assertEqual(read(self.root / "summary.json"), result)
        self.assertTrue((self.root / "summary.md").exists())
        self.assertEqual(read(self.root / "status.json"), archive.state)

    def test_read_only_build_has_no_writes(self):
        Archive(self.root)
        before = {str(p): (sha(p), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(build(self.root, save=False)["status"], "baseline_complete")
        after = {str(p): (sha(p), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_confirm_requires_monitoring_validation(self):
        Archive(self.root, "confirm", monitoring=False)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["speed_status"], "pass")
        self.assertIn("monitoring_validation_not_passed", result["errors"])
        self.assertFalse(result["all_cells_faster"])

    def test_monitoring_requires_both_actual_bounds_and_matching_sources(self):
        for name in ("missing", "bound", "sources"):
            archive = Archive(self.root / name, "confirm")
            monitoring = archive.resolved["monitoring_validation"]
            if name == "missing":
                monitoring["checks"].pop("open")
            elif name == "bound":
                monitoring["checks"]["gliner"]["p50"]["ratio_interval_95"][1] = 1.051
            else:
                monitoring["source_inventory_sha256"] = "0" * 64
            archive.refreeze()
            result = build(archive.root, save=False)
            self.assertEqual(result["status"], "inconclusive")
            self.assertFalse(result["all_cells_faster"])

    def test_confirm_pass_and_threshold_is_separate_from_faster(self):
        Archive(self.root / "strong", "confirm")
        self.assertEqual(build(self.root / "strong", save=False)["status"], "confirm_pass")
        Archive(self.root / "weak", "confirm", gain=.15)
        result = build(self.root / "weak", save=False)
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["all_cells_faster"])
        self.assertFalse(result["all_cells_meet_threshold"])
        Archive(self.root / "parameter", "confirm", gain=.15, comparison="parameter")
        self.assertEqual(build(self.root / "parameter", save=False)["status"], "confirm_pass")

    def test_thresholds_are_strict(self):
        for comparison, gain in (("engineering", .20), ("parameter", .10)):
            root = self.root / comparison
            Archive(root, "confirm", gain=gain, comparison=comparison)
            result = build(root, save=False)
            self.assertNotEqual(result["status"], "confirm_pass")
            self.assertTrue(result["all_cells_faster"])

    def test_all_cells_faster_requires_every_cell(self):
        archive = Archive(self.root, "confirm", buckets={"first": {"min": 1, "max": 64}, "second": {"min": 1, "max": 64}})
        for path in archive.workers:
            job = read(path / "job.json")
            if job["stage"] == "latency" and job["variant"] == "candidate" and job["bucket"] == "second":
                archive.mutate(path, lambda v: [r.update(latency_ms=10) for r in v["rows"]])
        result = build(self.root, save=False)
        self.assertTrue(result["cells"]["first/4"]["faster"])
        self.assertFalse(result["all_cells_faster"])
        self.assertEqual(result["status"], "inconclusive")

    def test_external_candidate_can_share_builtin_input_contract(self):
        archive = Archive(self.root, "confirm")
        archive.resolved["config"]["candidate"]["adapter"] = {
            "module": "/missing/external.py", "source_sha256": digest("external"), "dependencies": {}}
        archive.refreeze()
        self.assertEqual(build(self.root, save=False)["status"], "confirm_pass")

    def test_failed_preflight_and_skipped_latency_preserve_other_cells(self):
        archive = Archive(self.root, buckets={"first": {"min": 1, "max": 64}, "second": {"min": 1, "max": 64}})
        kept = []
        for path, stage in zip(archive.workers, archive.state["stages"]):
            job = read(path / "job.json")
            if job["bucket"] == "first" and job["stage"] == "latency":
                shutil.rmtree(path)
                continue
            if job["bucket"] == "first":
                archive.mutate(path, lambda v: v.update(status="oom"))
                stage["status"] = "oom"
            kept.append(stage)
        archive.state["stages"] = kept
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["cells"]["first/4"]["complete"])
        self.assertEqual(result["cells"]["second/4"]["status"], "baseline_complete")
        self.assertEqual(result["cells"]["second/4"]["p95"]["baseline_ms"], 10)

    def test_noise_prevents_confirmation(self):
        archive = Archive(self.root, "confirm")
        paths = [p for p in archive.workers if (j := read(p / "job.json")).get("phase") == "aa" and j["side"] == "b"]
        for path in paths:
            archive.mutate(path, lambda v: [r.update(latency_ms=15) for r in v["rows"]])
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["all_cells_faster"])

    def test_native_multiclass_flip_and_repeat_instability(self):
        archive = Archive(self.root, "confirm")
        path = archive.worker(variant="candidate", phase="ab")
        archive.mutate(path, lambda v: v["rows"][0]["scores"][0].update(label="class_c"))
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["verdict_status"], "unstable")
        self.assertTrue(result["cells"]["custom/4"]["verdicts"]["unstable_ids"])

    def test_score_delta_extrema_preserve_cartesian_metric_with_linear_access(self):
        class CountedScores(dict):
            accesses = 0

            def __getitem__(self, key):
                type(self).accesses += 1
                return super().__getitem__(key)

        rng = random.Random(7)
        names = [f"class_{i}" for i in range(9)]
        left, right, expected = {}, {}, dict.fromkeys(names, 0)
        repetitions = 0
        for key, counts in (("single", (1, 1)), ("unequal", (3, 7)), ("batch256", (192, 193))):
            a = [{name: rng.uniform(-4, 3) for name in names} for _ in range(counts[0])]
            b = [{name: rng.uniform(-2, 5) for name in names} for _ in range(counts[1])]
            for name in names:
                expected[name] = max(expected[name], max(abs(x[name] - y[name]) for x in a for y in b))
            left[key] = [{"label": "native", "score_type": "logits", "scores": CountedScores(v)} for v in a]
            right[key] = [{"label": "native", "score_type": "logits", "scores": CountedScores(v)} for v in b]
            repetitions += sum(counts)
        result = agreement(left, right)
        self.assertEqual(result["max_abs_score_delta"], expected)
        self.assertLessEqual(CountedScores.accesses, 4 * len(names) * repetitions)

    def test_unstable_full_baseline_still_complete_and_reusable(self):
        baseline = Archive(self.root / "base")
        baseline.mutate(baseline.worker(phase="baseline"), lambda v: v["rows"][0]["scores"][0].update(label="class_b"))
        result = build(baseline.root, save=False)
        self.assertEqual(result["status"], "baseline_complete")
        self.assertEqual(result["verdict_status"], "unstable")
        search = Archive(self.root / "search", "search")
        search.reference(baseline)
        result = build(search.root, save=False)
        self.assertEqual(result["status"], "exploratory")
        self.assertEqual(result["verdict_status"], "unstable")

    def test_saved_schedule_cannot_omit_expected_main_block(self):
        archive = Archive(self.root)
        path = archive.workers[-1]
        shutil.rmtree(path)
        archive.state["stages"].pop()
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("latency_schedule_mismatch", result["errors"])

    def test_failed_run_and_missing_preflight_cannot_pass(self):
        archive = Archive(self.root)
        archive.state["status"] = "failed"
        write(self.root / "status.json", archive.state)
        self.assertEqual(build(self.root, save=False)["status"], "partial")
        archive.state["status"] = "completed"
        archive.state["stages"].pop(0)
        shutil.rmtree(archive.workers[0])
        write(self.root / "status.json", archive.state)
        self.assertIn("missing_preflight", build(self.root, save=False)["errors"])

    def test_optional_missing_memory_does_not_veto_latency(self):
        archive = Archive(self.root)
        path = archive.add({"stage": "memory", "variant": "baseline"}, {})
        (path / "result.json").unlink()
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "baseline_complete")
        self.assertEqual(result["memory_status"], "unavailable")

    def test_profile_failure_is_reported_without_vetoing_speed(self):
        archive = Archive(self.root, "confirm")
        archive.add({"stage": "profile", "variant": "baseline", "bucket": "custom", "batch_size": 4},
                    {"status": "failed"})
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "confirm_pass")
        self.assertEqual(result["profile_status"], "failed")
        self.assertIn("stage_not_passed", result["cells"]["custom/4"]["profiles"]["variants"]["baseline"]["errors"])
        self.assertNotIn("stage_not_passed", result["errors"])

    def test_missing_requested_profiles_do_not_veto_speed(self):
        archive = Archive(self.root, "confirm")
        archive.resolved["config"]["profile_trace"] = True
        archive.refreeze()
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "confirm_pass")
        self.assertEqual(result["profile_status"], "unavailable")
        self.assertTrue(result["all_cells_faster"])

    def test_invalid_profile_precision_does_not_poison_other_cells(self):
        archive = Archive(self.root, buckets={"first": {"min": 1, "max": 64}, "second": {"min": 1, "max": 64}})
        path = archive.add({"stage": "profile", "variant": "baseline", "bucket": "first", "batch_size": 4},
                           {"profiles": ["short.trace.json.gz", "long.trace.json.gz"]})
        archive.mutate(path, lambda v: v["metadata"]["precision"].update(parameter_dtypes=["float16"]))
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "baseline_complete")
        self.assertEqual(result["profile_status"], "failed")
        self.assertTrue(all(c["complete"] for c in result["cells"].values()))

    def test_per_series_preparation_memory_and_size_evidence(self):
        archive = Archive(self.root)
        for path in archive.workers:
            job = read(path / "job.json")
            if job["stage"] != "latency":
                continue
            def resources(value):
                value.update(setup_seconds=job["block"] + 1, warmup_seconds=.3, backend_preparation_seconds=.2,
                    memory={"status": "measured", "windows": {
                        "setup": {"status": "measured", "sampled_gpu_peak_mib": 100, "valid_gpu_samples": 4, "host_rss_peak_bytes": 1234},
                        "inference": {"status": "measured", "sampled_gpu_peak_mib": 200, "valid_gpu_samples": 6}}},
                    torch_setup={"allocated_bytes": 11, "reserved_bytes": 22},
                    torch_inference={"allocated_bytes": 33, "reserved_bytes": 44},
                    size={"unique_parameter_buffer_storage_bytes": 80, "checkpoint_files_bytes": {"one.bin": 50, "two.bin": 70},
                          "export_files_bytes": {"encoder.onnx": 2}, "dtypes": ["torch.float32"], "original_encoder_retained": True})
            archive.mutate(path, resources)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "baseline_complete")
        cell = result["cells"]["custom/4"]
        self.assertEqual((cell["preparation_status"], cell["memory_status"], cell["size_status"]), ("measured",) * 3)
        self.assertEqual(len(cell["series_evidence"]), 7)
        series = next(v for v in cell["series_evidence"] if v["phase"] == "baseline" and v["block"] == 2)
        self.assertEqual(series["preparation"]["setup_seconds"]["value"], 3)
        self.assertEqual(series["preparation"]["warmup_seconds"]["value"], .3)
        self.assertEqual(series["preparation"]["backend_preparation_seconds"]["value"], .2)
        self.assertEqual(series["memory"]["sampler"]["setup"]["sampled_gpu_peak_mib"], 100)
        self.assertEqual(series["memory"]["sampler"]["inference"]["sampled_gpu_peak_mib"], 200)
        self.assertTrue(series["memory"]["sampler"]["sampling_may_miss_short_peaks"])
        self.assertEqual(series["memory"]["torch_setup"]["allocated_bytes"]["value"], 11)
        self.assertEqual(series["memory"]["torch_inference"]["reserved_bytes"]["value"], 44)
        self.assertEqual(series["size"]["checkpoint_files_bytes"]["total_bytes"], 120)
        self.assertEqual(series["size"]["export_files_bytes"]["total_bytes"], 2)
        self.assertEqual(series["size"]["unique_parameter_buffer_storage_bytes"]["value"], 80)

    def test_optional_resource_failures_have_separate_statuses(self):
        archive = Archive(self.root, "confirm")
        path = archive.worker(variant="candidate")
        archive.mutate(path, lambda v: v.update(
            backend_preparation_seconds=-1, memory={"status": "unavailable"},
            torch_inference={"allocated_bytes": 10, "reserved_bytes": 20},
            size={"checkpoint_files_bytes": {"bad.bin": -1}}))
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "confirm_pass")
        records = result["cells"]["custom/4"]["series_evidence"]
        changed = next(r for r in records if r["variant"] == "candidate" and r["block"] == 0)
        self.assertEqual(changed["preparation"]["backend_preparation_seconds"]["status"], "invalid")
        self.assertEqual(changed["memory"]["sampler"]["status"], "unavailable")
        self.assertEqual(changed["memory"]["torch_inference"]["status"], "measured")
        self.assertEqual(changed["size"]["checkpoint_files_bytes"]["status"], "invalid")
        self.assertIsNone(changed["size"]["checkpoint_files_bytes"]["total_bytes"])

    def test_single_to_batch_changes_use_unique_ids_per_variant(self):
        archive = Archive(self.root, "confirm", sizes=(1, 4))
        for path in archive.workers:
            job = read(path / "job.json")
            if job["stage"] == "latency" and job["batch_size"] == 4 and job["variant"] == "candidate":
                def flip(value):
                    for row in value["rows"]:
                        for key, score in zip(row["ids"], row["scores"]):
                            if int(key.rsplit("-", 1)[1]) < 10:
                                score["label"] = "class_b"
                archive.mutate(path, flip)
        result = build(self.root, save=False)
        comparison = result["single_to_batch"]["custom/4"]
        self.assertEqual(comparison["baseline"]["changed_count"], 0)
        candidate = comparison["candidate"]
        self.assertEqual((candidate["n"], candidate["common_n"], candidate["total"]), (400, 400, 400))
        self.assertEqual(candidate["changed_count"], 10)
        self.assertEqual(candidate["changed_fraction"], 10 / 400)
        self.assertEqual(candidate["unstable_ids"], [])

    def test_pilot_single_to_batch_reports_common_coverage(self):
        Archive(self.root, pilot=True, count=100, sizes=(1, 4))
        result = build(self.root, save=False)
        comparison = result["single_to_batch"]["custom/4"]["baseline"]
        self.assertEqual(comparison["status"], "partial")
        self.assertEqual((comparison["single_unique_n"], comparison["batch_unique_n"], comparison["n"]), (12, 48, 12))
        self.assertEqual(comparison["total"], 100)
        self.assertEqual(comparison["coverage"], .12)
        self.assertEqual(result["status"], "pilot_complete")

    def test_quantization_macro_changes_are_per_batch_without_quality_approval(self):
        archive = Archive(self.root, "confirm", comparison="quantization",
                          buckets={"first": {"min": 1, "max": 64}, "second": {"min": 1, "max": 64}})
        for path in archive.workers:
            job = read(path / "job.json")
            if job["stage"] != "latency" or job["variant"] != "candidate":
                continue
            limit = 100 if job["bucket"] == "first" else 200
            def flip(value):
                for row in value["rows"]:
                    for key, score in zip(row["ids"], row["scores"]):
                        if int(key.rsplit("-", 1)[1]) < limit:
                            score["label"] = "class_b"
            archive.mutate(path, flip)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "quantization_measured")
        self.assertEqual(result["macro_verdict_change_fraction_by_batch_size"], {"4": .375})
        macro = result["quantization"]["by_batch_size"]["4"]
        self.assertEqual(macro["buckets"]["first"]["n"], 400)
        self.assertEqual(macro["macro_unstable_fraction"], 0)
        self.assertIsNone(result["quantization"]["quality_preserved"])
        self.assertFalse(result["all_cells_meet_threshold"])

    def test_redacted_quantization_replays_full_identities_without_private_config(self):
        from u1_benchmark.v4_privacy import public_config
        archive = Archive(self.root, "confirm", comparison="quantization")
        original = deepcopy(archive.cfg)
        secrets = ("PRIVATE TEMPLATE SENTINEL", "PRIVATE SYSTEM SENTINEL")
        original["baseline"]["adapter"] = {"module": "/missing/baseline.py", "source_sha256": digest("baseline"), "dependencies": {}}
        for variant in ("baseline", "candidate"):
            original[variant]["template"] = {"text": secrets[0]}
            original[variant]["preparation"]["system_prompt"] = secrets[1]
        original["candidate"]["quantization"]["baseline_identity"] = digest(original["baseline"])
        identities = {v: digest(original[v]) for v in ("baseline", "candidate")}
        archive.resolved.update(config=public_config(original), config_sha256=digest(original), variant_identities=identities)
        for path in archive.workers:
            archive.mutate(path, lambda value: value["metadata"].update(variant_sha256=identities[value["job"]["variant"]]))
        archive.refreeze()
        self.assertNotEqual(digest(archive.resolved["config"]["baseline"]), identities["baseline"])
        self.assertNotEqual(digest(archive.resolved["config"]), archive.resolved["config_sha256"])
        self.assertFalse((self.root / ".private" / "config.json").exists())
        result = build(self.root)
        self.assertEqual(result["status"], "quantization_measured")
        for filename in ("resolved_run.json", "summary.json", "summary.md"):
            for secret in secrets:
                self.assertNotIn(secret, (self.root / filename).read_text())
        archive.resolved["variant_identities"]["baseline"] = "0" * 64
        archive.refreeze()
        self.assertIn("frozen_quantization_mismatch", build(self.root, save=False)["errors"])

    def test_saved_variant_identity_shape_is_checked_without_normalization(self):
        archive = Archive(self.root, "confirm")
        identities = {v: digest(archive.cfg[v]) for v in ("baseline", "candidate")}
        snapshot = deepcopy(archive.cfg)
        self.assertIs(frozen_config(archive.cfg, identities), archive.cfg)
        self.assertEqual(archive.cfg, snapshot)
        for invalid in ({}, {"baseline": identities["baseline"]},
                        {**identities, "candidate": "bad hash"}, {**identities, "unexpected": "a" * 64}):
            with self.subTest(identities=invalid):
                with self.assertRaisesRegex(ValueError, "invalid_variant_identities"):
                    frozen_config(archive.cfg, invalid)

    def test_quantization_macro_is_not_pooled_and_does_not_hide_missing_buckets(self):
        archive = Archive(self.root, "confirm", comparison="quantization",
                          buckets={"first": {"min": 1, "max": 64}, "second": {"min": 1, "max": 64}})
        cells = {"first/4": {"complete": True, "verdicts": {"n": 20, "changed_count": 10, "changed_fraction": .5, "unstable_ids": []}},
                 "second/4": {"complete": True, "verdicts": {"n": 10, "changed_count": 1, "changed_fraction": .1, "unstable_ids": []}}}
        result = _quantization(archive.cfg, cells)
        self.assertEqual(result["by_batch_size"]["4"]["macro_changed_fraction"], .3)
        cells["second/4"]["complete"] = False
        result = _quantization(archive.cfg, cells)
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["by_batch_size"]["4"]["macro_changed_fraction"])

    def test_bad_rows_and_scores_never_pass(self):
        mutations = {
            "count": lambda v: v["rows"].pop(),
            "id": lambda v: v["rows"][0]["ids"].__setitem__(0, "other"),
            "latency": lambda v: v["rows"][0].update(latency_ms=0),
            "scores": lambda v: v["rows"][0]["scores"].pop(),
            "score": lambda v: v["rows"][0]["scores"][0]["scores"].update(class_a="not numeric"),
            "compile": lambda v: v.update(compile_after=1),
            "timestamp": lambda v: v["rows"][1].update(timestamp=0),
            "hash": lambda v: v.update(resolved_sha256="0" * 64),
            "overflow": lambda v: [r.update(latency_ms=1e308) for r in v["rows"]],
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                root = self.root / name
                archive = Archive(root)
                archive.mutate(archive.worker(phase="baseline" if name == "overflow" else None), mutation)
                self.assertEqual(build(root, save=False)["status"], "incomplete")

    def test_frozen_hash_and_nonfinite_json(self):
        archive = Archive(self.root)
        path = self.root / "resolved_run.json"
        path.write_text(path.read_text() + " ")
        self.assertIn("resolved_hash_mismatch", build(self.root, save=False)["errors"])
        write(path, archive.resolved)
        worker = archive.worker() / "result.json"
        worker.write_text(worker.read_text().replace('"latency_ms": 10', '"latency_ms": NaN', 1))
        self.assertEqual(build(self.root, save=False)["status"], "incomplete")

    def test_runtime_and_engineering_precision_drift(self):
        for name in ("runtime", "precision", "missing_precision"):
            with self.subTest(name=name):
                root = self.root / name
                archive = Archive(root, "confirm")
                path = archive.worker(variant="candidate")
                def change(value):
                    if name == "runtime":
                        value["runtime"]["packages"]["engine"] = "2"
                    elif name == "precision":
                        value["metadata"]["precision"]["parameter_dtypes"] = ["float16"]
                    else:
                        value["metadata"].pop("precision")
                archive.mutate(path, change)
                self.assertEqual(build(root, save=False)["status"], "incomplete")

    def test_observed_metadata_consistent_workers_pass(self):
        archive = Archive(self.root)
        archive.observations()
        self.assertEqual(build(self.root, save=False)["status"], "baseline_complete")

    def test_observed_metadata_mutation_or_disappearance_cannot_pass(self):
        for field in OBSERVED:
            for missing in (False, True):
                with self.subTest(field=field, missing=missing):
                    root = self.root / (field + str(missing))
                    archive = Archive(root)
                    archive.observations()
                    def change(value):
                        if missing:
                            value["metadata"].pop(field)
                        elif field == "artifacts":
                            weights = {"model.safetensors": digest("changed weights")}
                            value["metadata"][field] = {"weight_files": weights, "weight_inventory_sha256": digest(weights)}
                        elif field == "preparation":
                            value["metadata"][field]["assets"]["effective_tokenizer"] = digest("changed tokenizer")
                        elif field in ("attention", "linear_attention"):
                            value["metadata"][field] = "eager" if field == "attention" else "reference"
                        else:
                            value["metadata"][field]["padding_side"] = "left"
                    archive.mutate(archive.worker(phase="baseline"), change)
                    result = build(root, save=False)
                    self.assertEqual(result["status"], "incomplete")
                    self.assertIn(f"worker_{field}_mismatch", result["errors"])

    def test_external_variant_can_omit_observed_metadata(self):
        archive = Archive(self.root, "confirm")
        archive.resolved["config"]["candidate"]["adapter"] = {
            "module": "/missing/external.py", "source_sha256": digest("external"), "dependencies": {}}
        archive.observations(variants=("baseline",))
        self.assertEqual(build(self.root, save=False)["status"], "confirm_pass")

    def test_observed_metadata_is_per_variant(self):
        archive = Archive(self.root, "confirm", comparison="parameter")
        archive.observations()
        for path in archive.workers:
            if read(path / "job.json")["variant"] == "candidate":
                archive.mutate(path, lambda v: v["metadata"]["artifacts"].update(weight_files={"candidate.bin": digest("candidate")}))
        archive.refreeze()
        self.assertEqual(build(self.root, save=False)["status"], "confirm_pass")

    def test_quality_workers_check_observed_preparation_across_batch_sizes(self):
        archive = Archive(self.root, "quality")
        archive.observations()
        path = next(p for p in archive.workers if (j := read(p / "job.json"))["variant"] == "baseline"
                    and j["quality_batch_size"] == 2)
        archive.mutate(path, lambda v: v["metadata"]["preparation"]["assets"].update(effective_chat_template=digest("other template")))
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("worker_preparation_mismatch", result["quality"]["errors"])

    def test_observed_metadata_checked_across_cells(self):
        archive = Archive(self.root, buckets={"first": {"min": 1, "max": 64}, "second": {"min": 1, "max": 64}})
        archive.observations()
        for path in archive.workers:
            if read(path / "job.json")["bucket"] == "second":
                archive.mutate(path, lambda v: v["metadata"]["batch_policy"].update(pad_token_id=5))
        archive.refreeze()
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["cells"]["first/4"]["status"], "baseline_complete")
        self.assertIn("worker_batch_policy_mismatch", result["cells"]["second/4"]["errors"])

    def test_nonidentity_runtime_fields_may_differ(self):
        archive = Archive(self.root)
        archive.mutate(archive.worker(), lambda v: v["runtime"].update(pid=5678, timestamp=123, gpu_uuid="different"))
        self.assertEqual(build(self.root, save=False)["status"], "baseline_complete")

    def test_smoke_and_pilot_never_pass_final(self):
        for mode, pilot, expected in (("smoke", False, "preflight_passed"), ("baseline", True, "pilot_complete"),
                                     ("confirm", True, "pilot_complete")):
            root = self.root / mode
            Archive(root, mode, pilot=pilot, count=12)
            result = build(root, save=False)
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["all_cells_meet_threshold"])
            if mode == "smoke":
                self.assertTrue(all("p95" not in v for v in result["cells"].values()))
                self.assertEqual(result["speed_status"], "not_measured")

    def test_full_speed_requires_400_per_bucket(self):
        Archive(self.root, count=399)
        self.assertIn("full_speed_requires_400_unique_ids", build(self.root, save=False)["errors"])

    def test_search_reuses_embedded_reference_without_paired_interval(self):
        baseline = Archive(self.root / "base")
        search = Archive(self.root / "search", "search")
        search.reference(baseline)
        result = build(search.root, save=False)
        self.assertEqual(result["status"], "exploratory")
        cell = result["cells"]["custom/4"]
        self.assertIsNone(cell["p95"]["ratio_interval_95"])
        self.assertFalse(cell["paired"])
        self.assertEqual(cell["reference_diagnostics"]["status"], "baseline_complete")
        self.assertFalse(result["all_cells_meet_threshold"])
        path = search.root / "reference" / "jobs" / "00001" / "result.json"
        path.write_text(path.read_text() + " ")
        self.assertIn("reference_file_hash_mismatch", build(search.root, save=False)["errors"])

    def test_real_reference_copy_and_pilot_reuse(self):
        from u1_benchmark.v4_evidence import copy_reference, read_reference
        baseline = Archive(self.root / "base")
        summary = build(baseline.root)
        baseline.state["status"] = summary["status"]
        write(baseline.root / "status.json", baseline.state)
        ref = read_reference(baseline.root)
        search = Archive(self.root / "search", "search", pilot=True)
        copy_reference(ref, search.root)
        self.assertEqual(sha(search.root / "reference" / "resolved_run.json"), baseline.state["resolved_sha256"])
        result = build(search.root, save=False)
        self.assertEqual(result["status"], "pilot_complete")
        self.assertFalse(result["all_cells_meet_threshold"])

    def test_unknown_protocol_is_explicitly_rejected(self):
        archive = Archive(self.root)
        archive.resolved["protocol_version"] = "not-supported"
        write(self.root / "resolved_run.json", archive.resolved)
        archive.state["resolved_sha256"] = sha(self.root / "resolved_run.json")
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertIn("unsupported_protocol", result["errors"])

    def test_quality_only_metrics_and_partial_batch(self):
        Archive(self.root, "quality")
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "quality_complete")
        self.assertEqual(result["speed_status"], "not_measured")
        quality = result["quality"]
        self.assertEqual(quality["quality_status"], "evaluated")
        size = quality["quality_by_batch_size"]["2"]
        self.assertEqual(size["variants"]["baseline"]["actual_batch_sizes"], [2, 1])
        self.assertEqual(size["baseline_to_candidate"]["n"], 3)
        self.assertEqual(size["baseline_to_candidate"]["left_quality"]["overall"]["accuracy"], 1)
        self.assertIsNone(quality["quality_preserved"])

    def test_quality_hashed_public_ids_without_source_texts(self):
        archive = Archive(self.root, "quality")
        archive.resolved["quality_rows"] = [{**{k: v for k, v in row.items() if k != "id"},
                                            "id_sha256": digest(row["id"])} for row in archive.resolved["quality_rows"]]
        archive.refreeze()
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "quality_complete")
        self.assertEqual(result["quality"]["id_encoding"], "sha256")
        self.assertNotIn('"q0"', json.dumps(result))

    def test_quality_partial_common_id_denominator(self):
        archive = Archive(self.root, "quality")
        path = archive.worker("quality", "candidate")
        predictions = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        predictions[0] = {"id": predictions[0]["id"], "status": "oom"}
        archive.predictions(path, predictions)
        archive.mutate(path, lambda v: v.update(status="oom", quality=metrics(archive.resolved["quality_rows"], predictions,
                                                                            archive.cfg["label_mapping"])))
        archive.state["stages"][archive.workers.index(path)]["status"] = "oom"
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "incomplete")
        pair = result["quality"]["quality_by_batch_size"]["1"]["baseline_to_candidate"]
        self.assertEqual((pair["n"], pair["total"]), (2, 3))
        self.assertEqual(pair["coverage"], 2 / 3)
        self.assertEqual(pair["left_quality"]["overall"]["total"], 2)

    def test_quality_repeats_report_instability(self):
        archive = Archive(self.root, "quality")
        path = archive.add_quality(1, "baseline")
        predictions = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        predictions[0]["score"]["label"] = "class_c" if predictions[0]["score"]["label"] != "class_c" else "class_a"
        archive.predictions(path, predictions)
        archive.mutate(path, lambda v: v.update(quality=metrics(archive.resolved["quality_rows"], predictions, archive.cfg["label_mapping"])))
        write(self.root / "status.json", archive.state)
        quality = build(self.root, save=False)["quality"]
        repeat = quality["quality_by_batch_size"]["1"]["variants"]["baseline"]["repeat_stability"]
        self.assertEqual(repeat["status"], "unstable")
        self.assertEqual(repeat["n"], 3)

    def test_quality_tampering_fails_closed(self):
        archive = Archive(self.root, "quality")
        archive.mutate(archive.worker("quality"), lambda v: v["quality"]["overall"].update(accuracy=.123))
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("quality_summary_mismatch", result["quality"]["errors"])

    def test_quality_precision_failure_invalidates_partial_predictions(self):
        archive = Archive(self.root, "quality")
        path = archive.worker("quality")
        archive.mutate(path, lambda v: v.update(status="failed", error_type="PrecisionError"))
        archive.state["stages"][archive.workers.index(path)]["status"] = "failed"
        write(self.root / "status.json", archive.state)
        result = build(self.root, save=False)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("quality_precision_failure", result["quality"]["errors"])
        self.assertFalse(result["quality"]["quality_by_batch_size"]["1"]["variants"]["baseline"]["quality_evaluated"])

    def test_table_keeps_distinct_environments_and_corpora(self):
        roots = [self.root / str(i) for i in range(3)]
        Archive(roots[0])
        Archive(roots[1], env={**ENV, "gpu": "other"})
        Archive(roots[2], corpus="other")
        result = table(roots, self.root / "table.json")
        self.assertEqual(len(result["groups"]), 3)
        self.assertFalse(result["rejected"])
        self.assertEqual((result["status"], result["rows"], result["omitted"]), ("completed", 3, 0))
        self.assertFalse(any((r / "summary.json").exists() for r in roots))

    def test_zip_build_without_saving_leaves_archive_and_siblings_unchanged(self):
        source = self.root / "deleted-run"
        Archive(source)
        expected = build(source, save=False)
        archive = self.archive_and_remove(source, "baseline.zip")
        before = {str(p): (sha(p), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(build(archive, save=False), expected)
        after = {str(p): (sha(p), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(source.exists())

    def test_zip_build_saves_sibling_reports_without_overwriting_zip(self):
        source = self.root / "deleted-run"
        Archive(source)
        archive = self.archive_and_remove(source, "baseline.zip")
        before = sha(archive), archive.stat().st_mtime_ns
        result = build(archive)
        self.assertEqual(result["status"], "baseline_complete")
        self.assertEqual(read(self.root / "baseline.zip.report.json"), result)
        self.assertIn("Status: baseline_complete", (self.root / "baseline.zip.report.md").read_text())
        self.assertEqual((sha(archive), archive.stat().st_mtime_ns), before)
        self.assertFalse(source.exists())

    def test_zip_search_replays_embedded_reference_after_source_deletion(self):
        baseline = Archive(self.root / "baseline")
        search = Archive(self.root / "search", "search")
        search.reference(baseline)
        expected = build(search.root, save=False)
        archive = self.archive_and_remove(search.root, "search.zip")
        shutil.rmtree(baseline.root)
        self.assertEqual(build(archive, save=False), expected)
        self.assertEqual(expected["status"], "exploratory")

    def test_zip_quality_replays_jsonl_after_source_deletion(self):
        source = self.root / "quality"
        Archive(source, "quality")
        archive = self.archive_and_remove(source, "quality.zip")
        result = build(archive, save=False)
        self.assertEqual(result["status"], "quality_complete")
        self.assertEqual(result["quality"]["quality_by_batch_size"]["2"]["baseline_to_candidate"]["n"], 3)

    def test_zip_table_cli_has_completed_status_and_counts(self):
        archives = []
        for name, corpus in (("first", "one"), ("second", "two")):
            source = self.root / name
            Archive(source, corpus=corpus)
            archives.append(self.archive_and_remove(source, name + ".zip"))
        before = {str(path.resolve()): sha(path) for path in archives}
        output = self.root / "table.json"
        code, public = self.cli("table", "--inputs", *archives, "--output", output)
        self.assertEqual(code, 0)
        self.assertEqual((public["status"], public["rows"], public["omitted"]), ("completed", 2, 0))
        self.assertNotIn("groups", public)
        saved = read(output)
        self.assertEqual(len(saved["groups"]), 2)
        self.assertEqual({run["source"] for group in saved["groups"].values() for run in group["runs"]}, set(before))
        self.assertEqual({str(path.resolve()): sha(path) for path in archives}, before)
        self.assertNotIn("u1-reference-", json.dumps(saved))

    def test_table_partial_status_counts_rejected_archives(self):
        good, bad = self.root / "good", self.root / "bad"
        Archive(good)
        failed = Archive(bad)
        failed.mutate(failed.worker(), lambda v: v.update(status="oom"))
        archives = [self.archive_and_remove(source, source.name + ".zip") for source in (good, bad)]
        corrupt = self.root / "corrupt.zip"
        corrupt.write_bytes(b"not a zip")
        code, public = self.cli("table", "--inputs", *archives, corrupt, "--output", self.root / "table.json")
        self.assertEqual(code, 1)
        self.assertEqual((public["status"], public["rows"], public["omitted"]), ("partial", 1, 2))
        saved = read(self.root / "table.json")
        self.assertEqual(len(saved["rejected"]), 2)
        self.assertEqual({r["source"] for r in saved["rejected"]}, {str(archives[1].resolve()), str(corrupt.resolve())})

    def test_zip_report_cli_succeeds_after_source_deletion(self):
        source = self.root / "run"
        Archive(source)
        archive = self.archive_and_remove(source, "run.zip")
        before = sha(archive)
        code, public = self.cli("report", "--output", archive)
        self.assertEqual(code, 0)
        self.assertEqual(public["status"], "baseline_complete")
        self.assertEqual(read(self.root / "run.zip.report.json")["status"], "baseline_complete")
        self.assertEqual(sha(archive), before)

    def test_zip_report_refuses_output_symlink_to_archive(self):
        source = self.root / "run"
        Archive(source)
        archive = self.archive_and_remove(source, "run.zip")
        before = sha(archive)
        (self.root / "run.zip.report.md").symlink_to(archive)
        with self.assertRaisesRegex(ValueError, "unsafe_report_output"):
            build(archive)
        self.assertEqual(sha(archive), before)

    def test_old_format_explicitly_unsupported(self):
        write(self.root / "config.json", {"schema_version": 3})
        self.assertIn("unsupported_old_format_explicit_migration_required", build(self.root, save=False)["errors"])

    def test_import_does_not_load_model_catalog_or_runtime(self):
        Archive(self.root)
        script = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'datasets', 'huggingface_hub', 'speed_dataset'} or name in {'contracts', 'v4_data', 'v4_worker', 'v4_preparation'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import u1_benchmark.v4_report
assert u1_benchmark.v4_report.build(sys.argv[1], save=False)['status'] == 'baseline_complete'
"""
        subprocess.run([sys.executable, "-B", "-c", script, str(self.root)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
