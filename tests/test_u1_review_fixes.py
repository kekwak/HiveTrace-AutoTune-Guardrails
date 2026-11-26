import ast
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from tools import benchmark, diagnose_open_precision as diagnostic, package_current_smoke as launcher
from u1_benchmark import adapters, contracts, datasets, environment, orchestrator, process, runner_v2
from u1_benchmark.core import ROOT, read, sha, write


class Archives(unittest.TestCase):
    def test_full_basename_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gliner.fp32", "gliner.int8"):
                directory = root / name
                write(directory / "status.json", {"status": "failed"})
                target = process.archive(directory)
                self.assertEqual(target.name, name + ".zip")
                before = target.read_bytes()
                with self.assertRaises(FileExistsError):
                    process.archive(directory)
                self.assertEqual(target.read_bytes(), before)
            explicit = root / "explicit.zip"
            explicit.write_bytes(b"previous result")
            with self.assertRaises(FileExistsError):
                process.archive(root / "gliner.fp32", target=explicit)
            self.assertEqual(explicit.read_bytes(), b"previous result")

    def test_exclusive_creation_handles_check_create_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write(root / "status.json", {})
            target = process.archive_path(root)
            target.write_bytes(b"racing writer")
            with patch.object(process, "check_archive_target", return_value=target):
                with self.assertRaises(FileExistsError):
                    process.archive(root)
            self.assertEqual(target.read_bytes(), b"racing writer")

    def test_partial_zip_is_not_mistaken_for_a_complete_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write(root / "status.json", {"status": "completed"})
            with patch.object(zipfile.ZipFile, "write", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    process.archive(root)
            self.assertFalse(process.archive_path(root).exists())

    def test_conflict_stops_before_setup_or_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "env.fp32"
            process.archive_path(root).write_bytes(b"previous")
            with patch.object(environment, "precheck_rtx") as precheck:
                with self.assertRaises(FileExistsError):
                    environment.setup_rtx(root)
                precheck.assert_not_called()
            with patch("followup.team_skeleton.precheck") as precheck:
                with self.assertRaises(FileExistsError):
                    orchestrator.setup(root)
                precheck.assert_not_called()
            cfg = {"schema_version": 3, "model": "gliner", "data": {"profile": "published"},
                   "environment": str(root / "missing")}
            with patch.object(runner_v2, "invoke") as invoke:
                with self.assertRaises(FileExistsError):
                    runner_v2.run(cfg, root)
                invoke.assert_not_called()
            self.assertFalse(root.exists())

    def test_packaging_failure_keeps_original_error_and_no_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"schema_version": 3, "model": "gliner", "data": {"profile": "published"},
                   "environment": str(root / "missing")}
            with patch.object(process, "archive", side_effect=OSError("private packaging detail")):
                result = runner_v2.run(cfg, root / "run")
            state = read(root / "run/status.json")
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(state["error_type"], "FileNotFoundError")
            self.assertEqual(state["archive_error_type"], "OSError")
            self.assertNotIn("private packaging detail", (root / "run/archive_status.json").read_text())
            self.assertTrue((root / "run/.private/error.txt").is_file())
            completed = {"status": "completed"}
            with patch.object(process, "archive", side_effect=OSError):
                self.assertFalse(process.finish_archive(root / "done", completed, root / "done/status.json"))
            self.assertEqual(completed["status"], "failed")


class ExternalImports(unittest.TestCase):
    def test_postponed_dataclass_annotations_and_reuse(self):
        code = ("from __future__ import annotations\nfrom dataclasses import dataclass\n"
                "from examples.u1_external_adapter import Fixture\n"
                "@dataclass\nclass Options:\n    name: str = 'adapter'\n"
                "def create(*, model, spec, directory, contract):\n"
                "    a = Fixture(contract, spec, model)\n    a.options = Options()\n    return a\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.py"
            path.write_text(code)
            spec = {"module": str(path), "source_sha256": sha(path), "dependencies": {}}
            name = "u1_external_" + spec["source_sha256"]
            try:
                first = adapters.load("gliner", spec, tmp, 3)
                second = adapters.load("gliner", spec, tmp, 3)
                self.assertEqual(first.options.name, "adapter")
                self.assertIs(type(first.options), type(second.options))
                self.assertIsNot(first, second)
                first.close()
                second.close()
            finally:
                sys.modules.pop(name, None)

    def test_failed_import_cleans_only_its_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text("raise RuntimeError('intentional import failure')\n")
            spec = {"module": str(path), "source_sha256": sha(path), "dependencies": {}}
            name = "u1_external_" + spec["source_sha256"]
            with self.assertRaises(RuntimeError):
                adapters.load("gliner", spec, tmp)
            self.assertNotIn(name, sys.modules)
            original = SimpleNamespace(__file__=str(Path(tmp) / "other.py"))
            with patch.dict(sys.modules, {name: original}):
                with self.assertRaisesRegex(ValueError, "collision"):
                    adapters.load("gliner", spec, tmp)
                self.assertIs(sys.modules[name], original)


class Console(unittest.TestCase):
    def call(self, arguments):
        stream = io.StringIO()
        with patch.object(sys, "argv", ["benchmark.py", *arguments]), contextlib.redirect_stdout(stream):
            code = benchmark.main()
        return code, json.loads(stream.getvalue())

    def test_auth_reason_is_visible_but_token_and_exception_are_not(self):
        result = {"status": "failed", "model": "shield", "error_type": "GatedRepoError",
                  "credential_available": True, "token": "hf_private_token", "error": "private input"}
        with patch.object(environment, "auth_check", return_value=result):
            code, summary = self.call(["auth-check"])
        self.assertEqual(code, 1)
        self.assertEqual(summary["error_type"], "GatedRepoError")
        self.assertTrue(summary["credential_available"])
        self.assertNotIn("private input", json.dumps(summary))
        self.assertNotIn("hf_private_token", json.dumps(summary))

    def test_fetch_and_prepare_show_identity_and_compatibility(self):
        from u1_benchmark import distribution
        result = {"status": "ready", "path": "/fixture/corpus", "manifest_sha256": "a" * 64,
                  "scope": "pinned_source_integrity", "model_contract_status": {"open": "requires_reindex"},
                  "messages": "private input"}
        with patch.object(distribution, "fetch", return_value=result):
            code, summary = self.call(["fetch-data"])
        self.assertEqual(code, 0)
        self.assertEqual(summary["model_contract_status"], {"open": "requires_reindex"})
        self.assertEqual(summary["manifest_sha256"], "a" * 64)
        self.assertNotIn("messages", summary)
        with patch.object(benchmark, "execute", return_value=result):
            code, summary = self.call(["prepare-data", "--spec", "fixture.json", "--output", "fixture"])
        self.assertEqual(code, 0)
        self.assertEqual(summary["path"], "/fixture/corpus")

    def test_run_failure_points_to_results_without_exception_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run.fp32"
            write(root / "status.json", {"status": "failed", "phase": "preflight", "error_type": "ValueError"})
            write(root / "summary.json", {"status": "incomplete"})
            process.archive(root)
            with patch.object(benchmark, "execute", return_value={"status": "incomplete", "error": "private input"}):
                code, summary = self.call(["run", "--config", "fixture", "--environment", "fixture", "--output", str(root)])
            self.assertEqual(code, 1)
            self.assertEqual(summary["phase"], "preflight")
            self.assertEqual(summary["archive_path"], str(process.archive_path(root)))
            self.assertEqual(summary["report_path"], str(root.resolve() / "summary.json"))
            self.assertNotIn("private input", json.dumps(summary))
        with patch.object(benchmark, "execute", side_effect=ValueError("hf_private_token")):
            code, summary = self.call(["auth-check"])
        self.assertEqual((code, summary), (1, {"status": "failed", "error_type": "ValueError"}))


class QualityConfig(unittest.TestCase):
    def test_only_quality_data_is_required_in_quality_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "quality/manifest.json", {"kind": "quality"})
            cfg = {"schema_version": 3, "model": "gliner", "environment": str(root / "env"), "mode": "quality",
                   "data": {"path": str(root / "missing-speed"), "manifest_sha256": "b" * 64},
                   "quality_data": {"path": str(root / "quality"), "manifest_sha256": sha(root / "quality/manifest.json")}}
            self.assertEqual(contracts.parse_config(cfg)["mode"], "quality")
            with self.assertRaises(FileNotFoundError):
                contracts.parse_config({**cfg, "mode": "screening"})
            with self.assertRaisesRegex(ValueError, "hash"):
                contracts.parse_config({**cfg, "quality_data": {**cfg["quality_data"], "manifest_sha256": "bad"}})
            without_quality = {key: value for key, value in cfg.items() if key != "quality_data"}
            with self.assertRaisesRegex(ValueError, "separate labeled"):
                contracts.parse_config(without_quality)


class SmokeLauncher(unittest.TestCase):
    def check_bundle(self, folder):
        import nbformat
        notebook = nbformat.read(folder / "current-smoke.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        assignments = {node.targets[0].id: ast.literal_eval(node.value)
                       for node in ast.parse(notebook.cells[1].source).body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in ("PAYLOAD", "SHA256")}
        raw = base64.b64decode(assignments["PAYLOAD"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), assignments["SHA256"])
        self.assertEqual(raw, (folder / "sources.zip").read_bytes())
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name, checksum in json.loads(archive.read("bundle_manifest.json")).items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), checksum)
        return notebook

    def test_missing_and_stale_overlay_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "launcher"
            with self.assertRaisesRegex(ValueError, "dataset-overlay"):
                launcher.build(ROOT, output)
            self.assertFalse(output.exists())
            with patch.object(datasets, "load_speed", side_effect=ValueError("Wrong dataset manifest hash")):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    launcher.build(ROOT, output, models=["open"], dataset_overlay=Path(tmp) / "old")
            self.assertFalse(output.exists())

    def test_launcher_without_open_and_diagnostic_need_no_overlay(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp) / "launcher"
            launcher.build(ROOT, output, models=["gliner", "shield"])
            notebook = self.check_bundle(output)
            self.assertNotIn("ENV.with_suffix", notebook.cells[2].source)
            diagnostic_output = Path(tmp) / "diagnostic"
            launcher.build(ROOT, diagnostic_output, open_diagnostic=True)
            notebook = self.check_bundle(diagnostic_output)
            self.assertIn("'--input-profile', 'native'", notebook.cells[-1].source)

    def test_valid_overlay_is_checked_and_embedded(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            folder = root / "overlay"
            manifest = {"status": "ready", "provenance": {"reindexed_models": ["open"],
                        "parent_manifest_sha256": read(ROOT / "configs/data/role_speed_v4.json")["manifest_sha256"]},
                        "indices": {"open": {"contract": contracts.model_contract("open", 3)}},
                        "files": {"corpus.jsonl": "a" * 64}}
            write(folder / "manifest.json", manifest)
            (folder / "REPORT.md").write_text("fixture")
            write(folder / "indices/open.jsonl", {})
            for bucket in range(1, 5):
                write(folder / f"buckets/open/B{bucket}.json", [])
            with patch.object(datasets, "load_speed", return_value=({}, manifest)) as load:
                launcher.build(ROOT, root / "launcher", models=["open"], dataset_overlay=folder)
            self.assertEqual(load.call_args.args[2], "screening")
            self.assertEqual(load.call_args.args[3], read(ROOT / "configs/benchmark-v3/open_baseline.json")["data"]["manifest_sha256"])
            notebook = self.check_bundle(root / "launcher")
            self.assertIn("DATA_OVERRIDE = {'path'", notebook.cells[3].source)


class OpenDiagnostic(unittest.TestCase):
    def test_explicit_profiles_and_legacy_aliases(self):
        self.assertEqual(diagnostic.resolve_profile(), "native")
        for version, profile in ((2, "task"), (3, "native")):
            with patch.object(diagnostic.warnings, "warn") as warn:
                self.assertEqual(diagnostic.resolve_profile(input_version=version), profile)
                self.assertIs(warn.call_args.args[1], FutureWarning)
        with self.assertRaises(ValueError):
            diagnostic.resolve_profile("task", 3)
        with patch("u1_benchmark.native_preparation.Preparer", return_value="native") as native, \
                patch("u1_benchmark.role_preparation.Preparer", return_value="task") as task:
            self.assertEqual(diagnostic.diagnostic_preparer("native"), "native")
            task.assert_not_called()
            self.assertEqual(diagnostic.diagnostic_preparer("task"), "task")
            native.assert_called_once_with("open")
            task.assert_called_once_with("open")

    def test_cases_use_full_verified_corpus_not_bucket_subset(self):
        from u1_benchmark.messages import content_hash
        rows = [{"id": "ignored", "messages": [{"role": "user", "content": "not selected"}]}]
        for index, key in enumerate(diagnostic.PINNED_IDS):
            value = [{"role": "user", "content": str(index)}]
            rows.append({"id": key, "messages": value, "content_sha256": content_hash(value)})
        manifest = {"files": {"corpus.jsonl": diagnostic.CORPUS_SHA256}}
        release_hash = read(ROOT / "configs/data/role_speed_v4.json")["manifest_sha256"]
        hashes = {str(index): key for index, key in enumerate(diagnostic.PINNED_IDS)}
        for row in rows[1:]:
            row["content_sha256"] = row["id"]
        with patch("u1_benchmark.core.sha", return_value=release_hash), \
                patch.object(datasets, "verified", return_value=(rows, manifest)), \
                patch("u1_benchmark.messages.content_hash", side_effect=lambda value: hashes[value[0]["content"]]):
            cases = diagnostic.diagnostic_cases("fixture")
            self.assertEqual([key for key, _ in cases[:3]], list(diagnostic.PINNED_IDS))
            self.assertEqual(len(cases), 4)
            rows[-1]["content_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "hash"):
                diagnostic.diagnostic_cases("fixture")

    def test_unpinned_corpus_is_rejected_before_loading_records(self):
        with patch("u1_benchmark.core.sha", return_value="changed"), patch.object(datasets, "verified") as verified:
            with self.assertRaisesRegex(ValueError, "pinned common corpus"):
                diagnostic.diagnostic_cases("fixture")
            verified.assert_not_called()


if __name__ == "__main__":
    unittest.main()
