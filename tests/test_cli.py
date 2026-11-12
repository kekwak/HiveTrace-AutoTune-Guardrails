import json
import os
import subprocess
import sys

import pytest


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "guardrail_tuner.cli", *map(str, args)],
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )


@pytest.mark.parametrize("value", [False, 0, "", {}])
def test_inspect_rejects_invalid_architectures(tmp_path, value):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": value}))
    result = run_cli("inspect", tmp_path)
    assert result.returncode == 1
    assert "architectures must be a list of strings" in result.stderr


@pytest.mark.parametrize("value", [None, []])
def test_inspect_allows_missing_architecture_list(tmp_path, value):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": value}))
    result = run_cli("inspect", tmp_path)
    assert result.returncode == 0
    assert json.loads(result.stdout)["architectures"] == []


def test_plan_classification_candidates_are_unverified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DebertaV2ForSequenceClassification"]})
    )
    result = run_cli("plan", tmp_path)
    assert result.returncode == 0
    plan = json.loads(result.stdout)
    assert plan["status"] == "unverified"
    assert {candidate["runtime"] for candidate in plan["candidates"]} == {
        "onnxruntime",
        "vllm",
        "torch_compile",
        "tensorrt",
    }


def test_plan_conditional_generation_is_unverified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]})
    )
    result = run_cli("plan", tmp_path)
    assert result.returncode == 0
    plan = json.loads(result.stdout)
    assert plan["status"] == "unverified"
    assert all(candidate["mode"] == "generation" for candidate in plan["candidates"])
