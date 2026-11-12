import json
import os
import subprocess
import sys
from pathlib import Path

from guardrail_tuner.manifest import ordered_digest, tree_digest


def test_ordered_hash_depends_on_records_and_order_without_exposing_them():
    rows = [{"id": 1, "text": "private-a"}, {"id": 2, "text": "private-b"}]
    first = ordered_digest(rows)
    assert first == ordered_digest(rows)
    assert first["sha256"] != ordered_digest(list(reversed(rows)))["sha256"]
    assert "private-a" not in json.dumps(first)


def test_local_model_tree_hash_detects_file_changes(tmp_path: Path):
    (tmp_path / "weights.bin").write_bytes(b"one")
    first = tree_digest(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"two")
    assert first["sha256"] != tree_digest(tmp_path)["sha256"]
    assert "weights.bin" not in json.dumps(first)


def test_application_manifest_omits_input_and_command_secrets(tmp_path: Path):
    quality = tmp_path / "quality.jsonl"
    quality.write_text('{"text":"DO_NOT_SAVE_PRIVATE_TEXT"}\n')
    speed = tmp_path / "speed.jsonl"
    speed.write_text('{"text":"DO_NOT_SAVE_SPEED_TEXT"}\n')
    script = tmp_path / "runner.py"
    script.write_text(
        'import json,os,sys\nassert os.environ["GUARDRAIL_TUNER_SEED"] == "37"\n'
        "for line in sys.stdin:\n"
        ' print(json.dumps({"label":"safe"}), flush=True)\n'
    )
    command = json.dumps([sys.executable, "-u", str(script), "DO_NOT_SAVE_TOKEN"])
    env = os.environ.copy()
    env["HF_HOME"] = str(tmp_path / "hf")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrail_tuner.cli",
            "compare-app",
            "--reference-command",
            command,
            "--candidate-command",
            command,
            "--model",
            "org/model",
            "--model-revision",
            "commit",
            "--reference-runtime",
            "reference",
            "--candidate-runtime",
            "candidate",
            "--quality-file",
            str(quality),
            "--performance-file",
            str(speed),
            "--exact-field",
            "label",
            "--seed",
            "37",
            "--warmup",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert process.returncode == 0, process.stderr
    manifest_text = (tmp_path / "out" / "run_manifest.json").read_text()
    assert "DO_NOT_SAVE_PRIVATE_TEXT" not in manifest_text
    assert "DO_NOT_SAVE_SPEED_TEXT" not in manifest_text
    assert "DO_NOT_SAVE_TOKEN" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["seed"]["value"] == 37
    assert manifest["datasets"]["quality"]["identity"]["ordered_rows"]["rows"] == 1
    assert manifest["datasets"]["performance"]["identity"]["ordered_rows"]["rows"] == 1
    assert manifest["model"]["source"] == "user_declared"
    assert manifest["model"]["verified"] is False
    assert "reference_command_sha256" in manifest["cli_options"]


def test_explicit_manifest_is_not_overwritten(tmp_path: Path):
    target = tmp_path / "manifest.json"
    target.write_text("existing")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrail_tuner.cli",
            "check-generation",
            "org/model",
            "--device",
            "cpu",
            "--quality-file",
            "unused.jsonl",
            "--manifest",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert "Manifest path already exists" in process.stderr
    assert target.read_text() == "existing"


def test_dataset_digest_fails_closed_for_unsupported_data():
    digest = ordered_digest([{"number": float("nan")}])
    assert digest["status"] == "unavailable"
    assert "sha256" not in digest


def test_failed_run_can_write_sanitized_explicit_manifest(tmp_path: Path):
    quality = tmp_path / "q.jsonl"
    quality.write_text('{"text":"DO_NOT_STORE"}\n')
    runner = json.dumps([sys.executable, "-c", "import sys; sys.exit(3)"])
    target = tmp_path / "failed.json"
    env = os.environ.copy()
    env["HF_HOME"] = str(tmp_path / "hf")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrail_tuner.cli",
            "compare-app",
            "--reference-command",
            runner,
            "--candidate-command",
            runner,
            "--model",
            "org/model",
            "--model-revision",
            "sha",
            "--reference-runtime",
            "a",
            "--candidate-runtime",
            "b",
            "--quality-file",
            str(quality),
            "--performance-file",
            str(quality),
            "--exact-field",
            "label",
            "--output-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(target),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 1
    text = target.read_text()
    manifest = json.loads(text)
    assert manifest["outcome"]["status"] == "failed"
    assert manifest["error_type"] == "ValueError"
    assert "DO_NOT_STORE" not in text


def test_rejected_nonempty_output_does_not_get_manifest(tmp_path: Path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("untouched")
    runner = json.dumps([sys.executable, "-c", 'print("{}")'])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrail_tuner.cli",
            "compare-app",
            "--reference-command",
            runner,
            "--candidate-command",
            runner,
            "--model",
            "org/model",
            "--model-revision",
            "sha",
            "--reference-runtime",
            "a",
            "--candidate-runtime",
            "b",
            "--quality-file",
            "unused.jsonl",
            "--performance-file",
            "unused.jsonl",
            "--exact-field",
            "label",
            "--output-dir",
            str(existing),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert sorted(p.name for p in existing.iterdir()) == ["keep.txt"]
