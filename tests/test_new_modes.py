import json
import sys
from pathlib import Path

import pytest

from guardrail_tuner.application import compare_applications
from guardrail_tuner.datasets import DatasetSource
from guardrail_tuner.model_tuning import tune_model


def datasets(tmp_path: Path) -> tuple[DatasetSource, DatasetSource]:
    paths = [tmp_path / "quality.jsonl", tmp_path / "speed.jsonl"]
    for path in paths:
        path.write_text('{"text":"hello world"}\n{"text":"hello"}\n', encoding="utf-8")
    return tuple(DatasetSource(kind="file", path=str(path)) for path in paths)  # type: ignore[return-value]


def test_jsonl_application_compares_fields_and_rejects_changed_decision(tmp_path: Path) -> None:
    quality, speed = datasets(tmp_path)
    script = tmp_path / "runner.py"
    script.write_text(
        "import json,sys\n"
        "row=json.loads(sys.stdin.readline())\n"
        'print(json.dumps({"decision":sys.argv[1],"scores":{"risk":0.25}}))\n',
        encoding="utf-8",
    )
    ref = json.dumps([sys.executable, str(script), "safe"])
    candidate = json.dumps([sys.executable, str(script), "unsafe"])
    result = compare_applications(
        ref,
        candidate,
        quality,
        speed,
        ["decision"],
        ["scores.risk:0.01"],
        0,
        10,
        str(tmp_path / "out"),
    )
    assert not result["quality"]["passed"]
    assert result["quality"]["exact_disagreements"] == {"decision": 2}
    assert result["selected"] == "reference"
    assert "hello world" not in (tmp_path / "out" / "report.md").read_text()


def test_jsonl_runner_requires_fields(tmp_path: Path) -> None:
    quality, speed = datasets(tmp_path)
    command = json.dumps([sys.executable, "-c", 'print("{}")'])
    with pytest.raises(ValueError, match="missing field"):
        compare_applications(
            command, command, quality, speed, ["decision"], [], 0, 10, str(tmp_path / "out")
        )


def test_tune_model_real_forward_with_identity_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vocab = model_dir / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_dir)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=7,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).save_pretrained(model_dir)
    monkeypatch.setattr(torch, "compile", lambda model: model)
    quality, speed = datasets(tmp_path)
    result = tune_model(
        str(model_dir), None, "cpu", quality, speed, "text", 16, 0, 1e-5, str(tmp_path / "out")
    )
    assert result["quality"]["passed"]
    assert result["quality"]["aa"]["passed"]
    assert result["selection"]["reason"] == "insufficient_aa_measurements"
    assert result["quality"]["max_abs_logit_error"] == 0
    assert result["scope"] == "model_forward_logits"
    by_batch = {item["batch_size"]: item for item in result["workload"]["results"]}
    assert by_batch[1]["status"] == "preliminary"
    assert by_batch[2]["status"] == "preliminary"
    assert by_batch[4]["reason"] == "insufficient_quality_samples_for_batch"


def test_json_equality_distinguishes_booleans_recursively() -> None:
    from guardrail_tuner.application import _json_equal

    assert not _json_equal({"value": [True]}, {"value": [1]})
    assert not _json_equal({"value": None}, {"value": False})
    assert _json_equal({"value": [1]}, {"value": [1.0]})


def test_jsonl_boolean_number_differences_fail_parity(tmp_path: Path) -> None:
    quality, speed = datasets(tmp_path)
    ref = json.dumps([sys.executable, "-c", "print('{\"result\":true}')"])
    candidate = json.dumps([sys.executable, "-c", "print('{\"result\":1}')"])
    result = compare_applications(
        ref, candidate, quality, speed, ["result"], [], 0, 10, str(tmp_path / "out")
    )
    assert result["quality"]["exact_disagreements"]["result"] == 2
    assert result["status"] == "rejected"


def test_numeric_tolerance_failure_rejects_candidate(tmp_path: Path) -> None:
    quality, speed = datasets(tmp_path)
    ref = json.dumps([sys.executable, "-c", "print('{\"score\":0.2}')"])
    candidate = json.dumps([sys.executable, "-c", "print('{\"score\":0.3}')"])
    result = compare_applications(
        ref, candidate, quality, speed, [], ["score:0.01"], 0, 10, str(tmp_path / "out")
    )
    assert result["quality"]["max_absolute_errors"]["score"] == pytest.approx(0.1)
    assert result["selected"] == "reference"
    assert result["selection"]["reason"] == "quality_failed"


def test_command_timeout_does_not_expose_stderr() -> None:
    from guardrail_tuner.application import _run

    with pytest.raises(ValueError, match="TimeoutExpired") as error:
        _run([sys.executable, "-c", "import time; time.sleep(2)"], {"text": "private"}, 0.02)
    assert "private" not in str(error.value)


def test_selection_requires_consistent_gain_and_stable_reference() -> None:
    from guardrail_tuner.measurement import selection

    def runs(values: list[float], count: int = 100) -> list[dict[str, float]]:
        return [{"requests": count, "p99_ms": value} for value in values]

    assert (
        selection({"reference": runs([10]), "candidate": runs([1])}, True)["status"]
        == "preliminary"
    )
    assert (
        selection({"reference": runs([10, 10.1, 10]), "candidate": runs([9, 9, 9])}, True)[
            "selected"
        ]
        == "candidate"
    )
    assert (
        selection({"reference": runs([10, 11, 10]), "candidate": runs([8, 8, 8])}, True)["reason"]
        == "reference_unstable"
    )
    assert (
        selection({"reference": runs([10, 10, 10]), "candidate": runs([9, 9.8, 9])}, True)[
            "selected"
        ]
        == "reference"
    )
    assert (
        selection({"reference": runs([10, 10, 10]), "candidate": runs([1, 1, 1])}, False)["reason"]
        == "quality_failed"
    )


def test_compile_failure_is_reported_and_not_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vocab = model_dir / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_dir)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=7,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).save_pretrained(model_dir)

    def broken_compile(model: object) -> object:
        raise RuntimeError("private implementation detail")

    monkeypatch.setattr(torch, "compile", broken_compile)
    quality, speed = datasets(tmp_path)
    result = tune_model(
        str(model_dir), None, "cpu", quality, speed, "text", 16, 0, 1e-5, str(tmp_path / "out")
    )
    assert result["candidate_error"] == "RuntimeError"
    assert result["status"] == "rejected"
    assert result["selected"] == "torch_eager"
    assert "private implementation detail" not in (tmp_path / "out" / "report.md").read_text()


def test_candidate_runtime_failure_is_sanitized_and_programming_errors_escape() -> None:
    from guardrail_tuner.model_tuning import CandidateRunner, CandidateRuntimeFailure

    class Broken:
        def run(self, text: str) -> None:
            raise RuntimeError("secret data")

    class Bug:
        def run(self, text: str) -> None:
            raise AssertionError("programming bug")

    with pytest.raises(CandidateRuntimeFailure, match="^RuntimeError$"):
        CandidateRunner(Broken()).run("private")  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="programming bug"):
        CandidateRunner(Bug()).run("private")  # type: ignore[arg-type]


def test_command_rejects_nonzero_and_invalid_json_without_echoing_output() -> None:
    from guardrail_tuner.application import _run

    with pytest.raises(ValueError, match="status 3"):
        _run([sys.executable, "-c", "import sys; sys.exit(3)"], {}, 2)
    with pytest.raises(ValueError, match="invalid JSON") as error:
        _run([sys.executable, "-c", 'print("private-data")'], {}, 2)
    assert "private-data" not in str(error.value)


def test_measurement_alternates_order_between_rounds() -> None:
    from guardrail_tuner.measurement import measure_pair

    events: list[str] = []

    class FakeRunner:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, item: object) -> None:
            events.append(self.name)

    result = measure_pair(FakeRunner("a"), FakeRunner("b"), list(range(100)), 0)
    assert len(result["reference"]) == len(result["candidate"]) == 3
    assert [events[index] for index in (0, 100, 200, 300, 400, 500)] == [
        "a",
        "b",
        "b",
        "a",
        "a",
        "b",
    ]


def test_persistent_runner_reuses_process_and_warmup() -> None:
    from guardrail_tuner.jsonl_process import PersistentJSONLRunner

    code = (
        "import json,sys\n"
        "n=0\n"
        "for line in sys.stdin:\n"
        " n+=1\n"
        ' print(json.dumps({"calls":n}),flush=True)\n'
    )
    with PersistentJSONLRunner([sys.executable, "-u", "-c", code], 2) as runner:
        assert runner.run({"text": "warmup"}) == {"calls": 1}
        assert runner.run({"text": "measured"}) == {"calls": 2}
        proc = runner.process
    assert proc.poll() is not None


def test_persistent_runner_timeout_closes_process() -> None:
    from guardrail_tuner.jsonl_process import PersistentJSONLRunner

    runner = PersistentJSONLRunner([sys.executable, "-u", "-c", "import time; time.sleep(3)"], 0.05)
    with pytest.raises(ValueError, match="TimeoutExpired"):
        runner.run({"private": "do not log"})
    assert runner.process.poll() is not None
    runner.close()


def test_persistent_runner_rejects_bad_json_and_early_exit() -> None:
    from guardrail_tuner.jsonl_process import PersistentJSONLRunner

    with PersistentJSONLRunner(
        [sys.executable, "-u", "-c", 'input(); print("private", flush=True)'], 2
    ) as runner:
        with pytest.raises(ValueError, match="invalid JSON") as error:
            runner.run({"text": "input"})
        assert "private" not in str(error.value)
    with PersistentJSONLRunner([sys.executable, "-u", "-c", "input()"], 2) as runner:
        with pytest.raises(ValueError, match="closed stdout"):
            runner.run({"text": "input"})


def test_persistent_compare_runs_aa_and_reports_warmed_boundary(tmp_path: Path) -> None:
    quality, speed = datasets(tmp_path)
    code = (
        "import json,sys\nfor line in sys.stdin:\n"
        ' print(json.dumps({"decision":"safe","score":0.3}),flush=True)\n'
    )
    command = json.dumps([sys.executable, "-u", "-c", code])
    result = compare_applications(
        command,
        command,
        quality,
        speed,
        ["decision"],
        ["score:0.01"],
        2,
        5,
        str(tmp_path / "out"),
        process_mode="persistent",
    )
    assert result["quality"]["aa"]["passed"]
    assert result["selection"]["reason"] == "insufficient_aa_measurements"
    assert result["settings"]["process_mode"] == "persistent"
    assert len(result["latency"]["reference_a"]) == len(result["latency"]["reference_b"]) == 1
    assert "excludes startup" in result["settings"]["measurement"]


def test_aa_disagreement_rejects_candidate(tmp_path: Path) -> None:
    quality, speed = datasets(tmp_path)
    code = (
        "import json,os,sys\nfor line in sys.stdin:\n"
        ' print(json.dumps({"pid":os.getpid()}),flush=True)\n'
    )
    command = json.dumps([sys.executable, "-u", "-c", code])
    result = compare_applications(
        command,
        command,
        quality,
        speed,
        ["pid"],
        [],
        0,
        5,
        str(tmp_path / "out"),
        process_mode="persistent",
    )
    assert not result["quality"]["aa"]["passed"]
    assert result["selection"]["reason"] == "aa_quality_failed"
    assert result["selected"] == "reference"


def test_aa_latency_gate_and_eligibility() -> None:
    from guardrail_tuner.measurement import aa_selection

    def runs(values: list[float]) -> list[dict[str, float]]:
        return [{"requests": 100, "p99_ms": value} for value in values]

    good = {
        "reference_a": runs([10, 10, 10]),
        "reference_b": runs([10.1, 10, 10]),
        "candidate": runs([8, 8, 8]),
    }
    assert aa_selection(good, True, True)["selected"] == "candidate"
    good["reference_b"] = runs([10, 12, 10])
    assert aa_selection(good, True, True)["reason"] == "aa_latency_unstable"
    assert aa_selection(good, False, True)["reason"] == "aa_quality_failed"
    assert aa_selection(good, True, False)["reason"] == "quality_failed"
