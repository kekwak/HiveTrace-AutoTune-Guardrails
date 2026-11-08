import json
from unittest.mock import patch

import pytest

from guardrail_tuner.datasets import DatasetSource
from guardrail_tuner.optimize import optimize


def run_result(level, threads, latency, passed=True):
    return {
        "model": "org/model",
        "model_revision": "model-sha",
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "versions": {"torch": "test"},
        "datasets": {
            "quality": {"source": "huggingface", "resolved_revision": "quality-sha"},
            "performance": {"source": "huggingface", "resolved_revision": "speed-sha"},
        },
        "settings": {"ort_optimization": level, "ort_intra_op_threads": threads},
        "quality": {"passed": passed},
        "latency": {"pytorch": {"p95_ms": 10.0}, "onnxruntime": {"p95_ms": latency}},
    }


def invoke(output_dir):
    quality = DatasetSource("hub", "org/quality")
    speed = DatasetSource("hub", "org/speed")
    return optimize(
        "org/model",
        None,
        "cpu",
        quality,
        speed,
        "text",
        "text",
        512,
        1,
        0.001,
        "multiclass",
        0.5,
        str(output_dir),
    )


def test_selects_fastest_passing_candidate_and_pins_datasets(tmp_path):
    yaml = pytest.importorskip("yaml")
    reports = [
        run_result("all", 0, 3.0, False),
        run_result("all", 1, 6.0),
        run_result("extended", 0, 5.0),
        run_result("extended", 1, 7.0),
    ]
    with patch("guardrail_tuner.optimize.run_benchmark", side_effect=reports) as runner:
        result = invoke(tmp_path / "out")

    assert result["selected"]["graph_optimization"] == "extended"
    assert result["selected"]["intra_op_threads"] == 0
    assert runner.call_args_list[1].args[1] == "model-sha"
    assert runner.call_args_list[1].args[3].revision == "quality-sha"
    config = yaml.safe_load((tmp_path / "out" / "config.yaml").read_text())
    assert config["settings"]["ort_optimization"] == "extended"
    assert config["datasets"]["quality"]["resolved_revision"] == "quality-sha"
    assert "fail" in (tmp_path / "out" / "report.md").read_text()


def test_baseline_wins_if_candidates_fail_quality(tmp_path):
    pytest.importorskip("yaml")
    result = run_result("all", 0, 2.0, False)
    with patch("guardrail_tuner.optimize.run_benchmark", return_value=result):
        selection = invoke(tmp_path / "out")
    assert selection["selected"]["runtime"] == "pytorch"


def test_no_results_produces_report_without_config(tmp_path):
    pytest.importorskip("yaml")
    with patch("guardrail_tuner.optimize.run_benchmark", side_effect=RuntimeError("secret")):
        result = invoke(tmp_path / "out")
    assert result["status"] == "no_valid_candidate"
    assert not (tmp_path / "out" / "config.yaml").exists()
    report = (tmp_path / "out" / "report.md").read_text()
    assert "RuntimeError" in report
    assert "secret" not in report


def test_refuses_to_overwrite_existing_output(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "config.yaml").write_text(json.dumps({"existing": True}))
    with pytest.raises(ValueError, match="empty"):
        invoke(tmp_path)
