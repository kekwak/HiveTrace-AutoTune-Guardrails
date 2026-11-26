"""No weights, network, or GPU required for the shared CLI orchestration contract."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from guardrail_tuner.rules import compare_outputs
from guardrail_tuner.u1_bridge import execution, make_plan, validate_plan
from u1_benchmark import cli_integration
from u1_benchmark.core import read, write
from u1_benchmark.v4_config import preparation_identity, validate_config

ROOT = Path(__file__).resolve().parents[1]


def preset(name="protectai"):
    return read(ROOT / "configs/v4" / (name + ".json"))


def plan(name="protectai"):
    return make_plan(
        preset(name),
        {
            "rule_features": {
                "relative_positions": name == "protectai",
                "linear_attention": name == "accuknox",
            }
        },
    )


@pytest.mark.parametrize(
    "name,expected", [("protectai", ["c12", "c8", "u4", "u1"]), ("accuknox", ["c7", "c12"])]
)
def test_plan_keeps_full_native_contract(name, expected):
    original = validate_config(preset(name))["baseline"]
    value = plan(name)
    assert value["plan_version"] == 2
    assert value["config"]["baseline"] == original
    candidate = value["config"]["candidate"]
    assert candidate["execution"]["rules"] == expected
    assert preparation_identity(candidate) == preparation_identity(original)
    assert candidate["scoring"] == original["scoring"]
    assert candidate["weights"] == original["weights"]
    assert validate_plan(value)["candidate"] == candidate
    assert execution(value, "candidate")["variant"] == candidate


@pytest.mark.parametrize(
    "candidate",
    [
        [],
        [{"label": "a", "logits": []}],
        [{"label": "a", "logits": [float("nan")]}],
        [{"label": "a", "logits": [1, 2]}],
        [{"logits": [1]}],
    ],
)
def test_bad_outputs_cannot_pass_parity(candidate):
    assert not compare_outputs([{"label": "a", "logits": [1]}], candidate)["passed"]


def test_saved_parity_failure_blocks_inference_but_speed_does_not():
    value = plan()
    value["evaluation"] = {"verdict_status": "changed", "status": "rejected"}
    with pytest.raises(ValueError, match="agreement"):
        execution(value, "candidate")
    assert execution(value, "baseline")["variant"] == value["config"]["baseline"]
    value["evaluation"] = {"verdict_status": "unchanged", "status": "inconclusive"}
    assert execution(value, "candidate")["variant"] == value["config"]["candidate"]


def fake_report(directory, save=True):
    state = read(Path(directory) / "status.json")
    return {"status": state["status"], "complete": state["status"] == "completed"}


def test_eval_runs_one_baseline_then_one_candidate(tmp_path):
    calls = []

    def runner(cfg, output):
        calls.append(deepcopy(cfg))
        return {"status": "baseline_complete" if cfg["mode"] == "baseline" else "inconclusive"}

    with patch.object(cli_integration, "report", fake_report):
        result = cli_integration.evaluate(plan(), tmp_path / "eval", runner=runner)
    assert result["complete"]
    assert [c["mode"] for c in calls] == ["baseline", "search"]
    assert "candidate" not in calls[0]
    assert calls[1]["baseline_ref"] == str(tmp_path / "eval/baseline")
    assert calls[0]["baseline"] == calls[1]["baseline"]
    assert calls[0]["batch_sizes_by_bucket"] == calls[1]["batch_sizes_by_bucket"]


def test_reference_never_runs_baseline(tmp_path):
    calls = []
    with patch.object(cli_integration, "report", fake_report):
        cli_integration.evaluate(
            plan(),
            tmp_path / "eval",
            "/saved/reference.zip",
            runner=lambda cfg, out: calls.append(cfg) or {"status": "inconclusive"},
        )
    assert len(calls) == 1
    assert calls[0]["mode"] == "search"
    assert calls[0]["baseline_ref"] == "/saved/reference.zip"


def test_failed_baseline_keeps_partial_archive_and_skips_candidate(tmp_path):
    calls = []
    with patch.object(cli_integration, "report", fake_report):
        result = cli_integration.evaluate(
            plan(),
            tmp_path / "eval",
            runner=lambda cfg, out: calls.append(cfg) or {"status": "failed"},
        )
    assert len(calls) == 1
    assert result["status"] == "failed"
    assert (tmp_path / "eval.zip").is_file()
    assert (tmp_path / "eval/status.json").is_file()


def test_quality_does_not_read_speed_data_and_runs_once(tmp_path):
    value = plan()
    value["config"]["quality_data"] = {"path": "/private/quality", "manifest_sha256": "a" * 64}
    value["config"]["label_mapping"] = {
        "labels": ["safe", "unsafe"],
        "targets": {"0": "safe", "1": "unsafe"},
        "verdicts": {"SAFE": "safe", "INJECTION": "unsafe"},
    }
    calls = []
    with patch.object(cli_integration, "report", fake_report):
        cli_integration.evaluate(
            value,
            tmp_path / "eval",
            "/saved/baseline",
            runner=lambda cfg, out: calls.append(cfg) or {"status": "completed"},
        )
    assert [c["mode"] for c in calls] == ["search", "quality"]
    assert "data" not in calls[-1]
    assert "baseline_ref" not in calls[-1]


def test_archive_does_not_contain_private_plan(tmp_path):
    import zipfile

    value = plan("accuknox")
    value["config"]["baseline"]["preparation"]["questions"]["user"]["instructions"] = (
        "private_instruction_123"
    )
    value["config"]["candidate"]["preparation"] = deepcopy(
        value["config"]["baseline"]["preparation"]
    )
    with patch.object(cli_integration, "report", fake_report):
        cli_integration.evaluate(
            value, tmp_path / "eval", runner=lambda cfg, out: {"status": "failed"}
        )
    with zipfile.ZipFile(tmp_path / "eval.zip") as archive:
        assert not any(".private" in name for name in archive.namelist())
        assert all(
            b"private_instruction_123" not in archive.read(name) for name in archive.namelist()
        )


def test_external_adapter_no_invented_rules():
    cfg = preset()
    cfg["baseline"]["adapter"] = {
        "module": "/tmp/custom.py",
        "source_sha256": "a" * 64,
        "dependencies": {},
    }
    value = make_plan(cfg, {"rule_features": {}})
    assert value["status"] == "needs_inference_contract"


def test_invalid_rule_combinations_rejected():
    value = plan()
    value["config"]["candidate"]["execution"]["rules"] = ["not_a_rule"]
    with pytest.raises(ValueError):
        validate_plan(value)
    value = plan("accuknox")
    value["config"]["candidate"]["execution"]["rules"] = ["u1"]
    with pytest.raises(ValueError):
        validate_plan(value)


def test_cli_plan_and_eval_dispatch(tmp_path, monkeypatch, capsys):
    from guardrail_tuner import cli, u1_bridge

    cfg = tmp_path / "base.json"
    write(cfg, preset())
    monkeypatch.setattr(
        cli, "inspect_model", lambda *args: {"rule_features": {"relative_positions": True}}
    )
    monkeypatch.setattr(
        "sys.argv",
        ["guardrail-tuner", "plan", "--config", str(cfg), "--output", str(tmp_path / "plan.json")],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["plan_version"] == 2
    monkeypatch.setattr(
        u1_bridge, "evaluate", lambda *args: {"status": "inconclusive", "complete": True}
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "guardrail-tuner",
            "eval",
            "--config",
            str(tmp_path / "plan.json"),
            "--output",
            str(tmp_path / "eval"),
        ],
    )
    assert cli.main() == 0


def test_native_classifier_rules_keep_scores_and_release_activations(tmp_path):
    from test_u1_classifier_family import adapter

    from u1_benchmark.team_rules import RuleAdapter

    native = adapter(tmp_path, family="deberta", pos_att_type=["p2c", "c2p"], share_att_key=True)
    values = ["library", "The public library."]
    reference = native.request_batch(values)
    spec = deepcopy(native.spec)
    spec["execution"]["rules"] = ["c12", "c8"]
    ruled = RuleAdapter(native, spec, ["c12", "c8"])
    original_forward = native.model.forward
    try:
        ruled.prepare_batch_backend([values])
        answers = ruled.request_batch(values)
        assert [v["label"] for v in answers] == [v["label"] for v in reference]
        for a, b in zip(answers, reference):
            assert a["scores"] == pytest.approx(b["scores"], abs=1e-5)
        ruled.request_batch(["different words", "library"])
        assert ruled.request_batch(values) == answers
        assert all(getattr(m, "x", None) is None for m in native.model.modules())
    finally:
        model = native.model
        ruled.close()
        assert model.forward == original_forward


def test_fused_forward_does_not_reuse_same_input_buffer(tmp_path):
    import torch
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

    from guardrail_tuner.rules import RuledModel

    model = DebertaV2ForSequenceClassification(
        DebertaV2Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    ).eval()
    reference = deepcopy(model)
    ruled = RuledModel(model, ["c12"], 0, strict=True)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        ruled(input_ids=ids, attention_mask=mask)
        ids.fill_(4)
        actual = ruled(input_ids=ids, attention_mask=mask).logits
        expected = reference(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(actual, expected)


def test_fullgraph_compiles_relative_projection_rules(tmp_path, monkeypatch):
    import torch
    from test_u1_classifier_family import adapter
    from torch._dynamo.backends import registry

    from u1_benchmark.team_rules import RuleAdapter

    torch.set_num_threads(1)
    native = adapter(tmp_path, family="deberta", pos_att_type=["p2c", "c2p"], share_att_key=True)
    payloads = ["library.", "The public library."]
    expected = native.request_batch(payloads)
    original_lookup = registry.lookup_backend
    monkeypatch.setattr(
        registry,
        "lookup_backend",
        lambda backend: original_lookup("eager" if backend == "inductor" else backend),
    )
    spec = deepcopy(native.spec)
    spec["execution"]["rules"] = ["c12", "c8", "u4"]
    wrapped = RuleAdapter(native, spec, spec["execution"]["rules"])
    try:
        wrapped.prepare_batch_backend([payloads, [payloads[0]]])
        assert wrapped.compile_count > 0
        for actual, reference in zip(wrapped.request_batch(payloads), expected):
            assert actual["label"] == reference["label"]
            assert actual["scores"] == pytest.approx(reference["scores"], abs=1e-6)
    finally:
        wrapped.close()


def test_cli_error_never_prints_arbitrary_exception(tmp_path, monkeypatch, capsys):
    from guardrail_tuner import cli, u1_bridge

    def fail(*args):
        raise ValueError("PRIVATE-TEXT-AND-TOKEN")

    monkeypatch.setattr(u1_bridge, "evaluate", fail)
    monkeypatch.setattr(
        "sys.argv",
        ["guardrail-tuner", "eval", "--config", "missing", "--output", str(tmp_path / "result")],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    output = capsys.readouterr()
    assert stopped.value.code == 1
    assert json.loads(output.out)["error_type"] == "ValueError"
    assert "PRIVATE-TEXT-AND-TOKEN" not in output.out + output.err


def test_native_worker_reuses_model_but_not_requests(tmp_path):
    from io import StringIO
    from unittest.mock import Mock

    from test_u1_v4_execution import CPUAdapter, variant

    spec = variant()
    model = CPUAdapter(spec, tmp_path)
    model.prove_batch_backend = Mock(wraps=model.prove_batch_backend)
    requests = [
        {"operation": "load"},
        {"operation": "predict", "inputs": ["one", "two"]},
        {"operation": "predict", "inputs": ["three", "four"]},
    ]
    output = StringIO()
    with (
        patch("u1_benchmark.v4_evidence.configure_gpu"),
        patch("u1_benchmark.v4_adapters.load_adapter", return_value=model) as load,
    ):
        cli_integration.infer_session(
            {"variant": spec, "environment": "/fixture", "batch_size": 2},
            StringIO("".join(json.dumps(r) + "\n" for r in requests)),
            output,
            tmp_path,
        )
    values = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [v["status"] for v in values] == ["ready", "passed", "passed"]
    load.assert_called_once()
    model.prove_batch_backend.assert_called_once()
    assert model.requests == [["one", "two"], ["three", "four"]]
    assert model.closed


def test_outer_zip_report_survives_original_path_deletion(tmp_path):
    import shutil

    from test_u1_v4_report import Archive

    from guardrail_tuner.u1_bridge import offline_report
    from u1_benchmark.process import archive

    root = tmp_path / "eval"
    baseline = Archive(tmp_path / "external-baseline")
    candidate = Archive(root / "candidate", mode="search")
    candidate.reference(baseline)
    write(root / "requested_plan.json", {"plan_version": 2})
    write(
        root / "status.json",
        {
            "status": "completed",
            "baseline_reused": True,
            "stages": [],
            "requested_config_sha256": "a" * 64,
        },
    )
    before = offline_report(root)
    path = archive(root)
    shutil.rmtree(root)
    shutil.rmtree(baseline.root)
    after = offline_report(path)
    assert before == after
    assert after["complete"]


def test_team_quality_uses_saved_predictions_only(tmp_path):
    from test_u1_v4_report import Archive

    from u1_benchmark.v4_report import build

    quality = Archive(tmp_path / "quality", mode="quality")
    summary = build(quality.root, save=False)["quality"]
    assert cli_integration.team_quality(quality.root, summary)["status"] == "not_applicable"


def test_explicit_binary_quality_aggregation_and_mismatched_inputs(tmp_path):
    root = tmp_path / "quality"
    write(
        root / "resolved_run.json",
        {
            "config": {
                "label_mapping": {
                    "labels": ["safe", "unsafe"],
                    "targets": {"bad": "unsafe"},
                    "verdicts": {"ALLOW": "safe", "BLOCK": "unsafe"},
                }
            },
            "quality_rows": [{"id": "q1", "target": "bad", "slices": {"language": "ru"}}],
        },
    )
    for variant, label in [("baseline", "BLOCK"), ("candidate", "ALLOW")]:
        folder = root / "jobs" / variant
        write(
            folder / "job.json", {"stage": "quality", "variant": variant, "quality_batch_size": 1}
        )
        write(folder / "result.json", {"status": "passed"})
        (folder / "predictions.jsonl").write_text(
            json.dumps({"id": "q1", "status": "passed", "score": {"label": label}}) + "\n"
        )
    summary = {"quality_by_batch_size": {"1": {"baseline_to_candidate": {"status": "complete"}}}}
    value = cli_integration.team_quality(root, summary)
    assert value["status"] == "evaluated"
    assert value["by_batch_size"]["1"]["dataset"]["n_total"] == 1
    assert value["quality_preserved"] is None
    summary["quality_by_batch_size"]["1"]["baseline_to_candidate"]["status"] = (
        "blocked_input_mismatch"
    )
    assert (
        cli_integration.team_quality(root, summary)["by_batch_size"]["1"]["status"] == "unavailable"
    )


def test_historical_run_does_not_return_baseline_after_parity_failure(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from guardrail_tuner import execution as legacy

    runtime = legacy.Inference.__new__(legacy.Inference)
    runtime.revision, runtime.environment = "fixture", {}
    runtime.predict = Mock(
        side_effect=[[{"label": 0, "logits": [1.0, 0.0]}], [{"label": 1, "logits": [0.0, 1.0]}]]
    )
    runtime.enable_rules = Mock(return_value=SimpleNamespace(applied=["c12"], skipped={}))
    monkeypatch.setattr(legacy, "load_inference", lambda config: runtime)
    with pytest.raises(ValueError, match="parity failed"):
        legacy.predict_inputs(
            {"runtime": "transformers", "rules": ["c12"], "task": "sequence_classification"},
            ["hello"],
        )


def test_serve_loads_once_and_releases_runtime(monkeypatch):
    import queue
    import threading
    import urllib.request
    from unittest.mock import Mock

    from guardrail_tuner import u1_bridge

    runtime = Mock()
    runtime.predict.side_effect = lambda rows: [{"label": "safe"} for _ in rows]
    load = Mock(return_value=runtime)
    monkeypatch.setattr(u1_bridge, "load_native", load)
    servers = queue.Queue()
    http_server = u1_bridge.HTTPServer

    def create(address, handler):
        server = http_server(address, handler)
        servers.put(server)
        return server

    monkeypatch.setattr(u1_bridge, "HTTPServer", create)
    thread = threading.Thread(
        target=u1_bridge.serve_native, args=({"fixture": True}, 0), daemon=True
    )
    thread.start()
    server = servers.get(timeout=5)
    try:
        for body in ({"inputs": ["first"]}, {"messages": [{"role": "user", "content": "second"}]}):
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/infer",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert json.load(response)["outputs"] == [{"label": "safe"}]
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert not thread.is_alive()
    load.assert_called_once()
    assert runtime.predict.call_count == 2
    runtime.close.assert_called_once()
