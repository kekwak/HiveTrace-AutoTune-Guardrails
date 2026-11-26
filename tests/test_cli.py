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


def test_plan_classification_hides_candidates_when_executable(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DebertaV2ForSequenceClassification"]})
    )
    result = run_cli("plan", tmp_path)
    assert result.returncode == 0
    plan = json.loads(result.stdout)
    assert plan["status"] == "unverified"
    assert plan["execution"]["task"] == "sequence_classification"
    assert "candidates" not in plan


def test_plan_conditional_generation_is_unverified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]})
    )
    result = run_cli("plan", tmp_path)
    assert result.returncode == 0
    plan = json.loads(result.stdout)
    assert plan["status"] == "unverified"
    assert plan["execution"]["task"] == "generation"
    assert plan["execution"]["runtime"] == "transformers"
    assert "candidates" not in plan


def test_plan_preserves_candidates_without_execution(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["CustomForSequenceClassification"],
                "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            }
        )
    )
    result = run_cli("plan", tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["execution"] is None
    assert {candidate["runtime"] for candidate in plan["candidates"]} == {
        "onnxruntime",
        "torch_compile",
        "tensorrt",
    }


@pytest.mark.parametrize(
    "auto_model",
    ["modeling_d3.D3Model", "org/runtime--modeling_d3.D3Model"],
)
def test_plan_detects_d3_contract_without_importing_model_code(tmp_path, auto_model):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5Model"],
                "auto_map": {"AutoModel": auto_model},
            }
        )
    )
    result = run_cli("plan", tmp_path, "--device", "cuda")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["inference_contract"] == "system_one"
    assert plan["status"] == "needs_inference_contract"
    assert plan["execution"] is None
    assert [candidate["runtime"] for candidate in plan["candidates"]] == ["d3"]
    assert "system_one" in plan["candidates"][0]["next_check"]


@pytest.mark.parametrize(
    "version,metadata",
    [
        (1, {"execution": {"version": 1}, "environment": {"gpu": None}, "rules": {"applied": []}}),
        (2, {"metadata": {"runtime": "native"}, "quality_evaluated": False}),
    ],
)
@pytest.mark.parametrize("debug", [False, True])
def test_run_outputs_metadata_only_in_debug(
    tmp_path, monkeypatch, capsys, version, metadata, debug
):
    from guardrail_tuner import cli, u1_bridge

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"execution": {"version": version}}))
    outputs = [{"label": "OK", "logits": [1.0, 0.0]}]
    result = {"outputs": outputs, **metadata}
    calls = []

    def infer(config, inputs):
        calls.append(inputs)
        return result

    monkeypatch.setattr(cli, "read_execution", lambda path: {"version": 1})
    monkeypatch.setattr(cli, "run_inference", infer)
    monkeypatch.setattr(u1_bridge, "execution", lambda *args: {"version": 2})
    monkeypatch.setattr(u1_bridge, "run_native", infer)
    args = ["guardrail-tuner", "run", "--config", str(plan_path), "--text", "test"]
    monkeypatch.setattr(sys, "argv", args + (["--debug"] if debug else []))

    assert cli.main() == 0
    assert calls == [["test"]]
    assert json.loads(capsys.readouterr().out) == (result if debug else {"outputs": outputs})
