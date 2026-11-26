import json
from pathlib import Path

import pytest

from guardrail_tuner.cli import inspect_model, plan_checks
from guardrail_tuner.execution import read_execution, run_inference
from guardrail_tuner.rules import rule_features, select_rules

DEBERTA = {"architectures": ["DebertaV2ForSequenceClassification"], "relative_attention": True}
LINEAR = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {"layer_types": ["linear_attention", "full_attention"]},
}


def execution(task: str, device: str = "cuda", runtime: str = "transformers") -> dict:
    return {"task": task, "device": device, "runtime": runtime}


def test_features_come_from_config() -> None:
    assert rule_features(DEBERTA) == {"relative_positions": True, "linear_attention": False}
    assert rule_features({"model_type": "deberta"})["relative_positions"]
    assert rule_features(LINEAR) == {"relative_positions": False, "linear_attention": True}
    assert rule_features({"architectures": ["BertForSequenceClassification"]}) == {
        "relative_positions": False,
        "linear_attention": False,
    }


def test_classification_on_cuda_selects_graphs_and_compile() -> None:
    plan = select_rules(rule_features(DEBERTA), execution("sequence_classification"))
    assert plan["rules"] == ["c12", "c8", "u4", "u1"]


def test_cpu_and_generation_do_not_fall_back_to_compile() -> None:
    cpu = select_rules(rule_features(DEBERTA), execution("sequence_classification", "cpu"))
    assert cpu["rules"] == ["c12", "c8"]
    generation = select_rules(rule_features(LINEAR), execution("generation"))
    assert generation["rules"] == ["c7"]
    assert generation["requires"] == ["flash-linear-attention"]
    assert "u4" in generation["skipped"]


def test_external_runtime_gets_no_rules() -> None:
    plan = select_rules(rule_features(DEBERTA), execution("generation", runtime="onnxruntime"))
    assert plan["rules"] == []


def test_backbone_without_task_gets_module_rules(tmp_path: Path) -> None:
    backbone = {**LINEAR, "architectures": ["Qwen3_5Model"], "auto_map": {"AutoModel": "m.Model"}}
    (tmp_path / "config.json").write_text(json.dumps(backbone))
    plan = plan_checks(inspect_model(str(tmp_path), None), "cuda")
    assert plan["execution"] is None
    assert plan["rules"]["rules"] == ["c7", "c12"]
    assert plan["rules"]["requires"] == ["flash-linear-attention"]
    assert plan["rules"]["skipped"].keys() == {"u4", "u1"}


def test_plan_writes_rules_into_execution(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(DEBERTA))
    plan = plan_checks(inspect_model(str(tmp_path), None), "cpu")
    assert plan["execution"]["rules"] == ["c12", "c8"]
    assert plan["rules"]["skipped"]["u4"] == "measured only on CUDA"


def test_unknown_rules_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(DEBERTA))
    plan = plan_checks(inspect_model(str(tmp_path), None), "cpu")
    plan["execution"]["rules"] = ["c2"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Invalid rules"):
        read_execution(str(path))


def tiny_deberta(model_dir: Path, version: int) -> None:
    import torch
    from transformers import (
        BertTokenizerFast,
        DebertaConfig,
        DebertaForSequenceClassification,
        DebertaV2Config,
        DebertaV2ForSequenceClassification,
    )

    model_dir.mkdir()
    vocab = model_dir / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_dir)
    options = {
        "vocab_size": 7,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "relative_attention": True,
        "max_relative_positions": 16,
        "pos_att_type": ["p2c", "c2p"],
        "num_labels": 3,
    }
    torch.manual_seed(0)
    if version == 1:
        DebertaForSequenceClassification(DebertaConfig(**options)).save_pretrained(model_dir)
    else:
        DebertaV2ForSequenceClassification(DebertaV2Config(**options)).save_pretrained(model_dir)


@pytest.mark.parametrize(("version", "fused"), [(1, False), (2, True)])
def test_rules_keep_outputs_and_skip_what_does_not_apply(
    tmp_path: Path, version: int, fused: bool
) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from guardrail_tuner.rules import RuledModel

    model_dir = tmp_path / "model"
    tiny_deberta(model_dir, version)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    inputs = tokenizer(["hello world", "world"], return_tensors="pt", padding=True)
    with torch.inference_mode():
        reference = model(**inputs).logits
        ruled = RuledModel(model, ["c12", "c8"], tokenizer.pad_token_id)
        candidate = ruled(**inputs).logits
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)
    assert ruled.applied == (["c12", "c8"] if fused else ["c8"])
    assert fused or ruled.skipped["c12"] == "projections are already fused"


def test_linear_attention_projections_are_fused() -> None:
    import torch

    from guardrail_tuner.rules import RuledModel

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj_qkv = torch.nn.Linear(8, 12, bias=False)
            self.in_proj_z = torch.nn.Linear(8, 4, bias=False)
            self.in_proj_b = torch.nn.Linear(8, 2, bias=False)
            self.in_proj_a = torch.nn.Linear(8, 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            parts = (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a)
            return torch.cat([part(x) for part in parts], -1)

    torch.manual_seed(0)
    layer = Layer().eval()
    x = torch.randn(2, 5, 8)
    with torch.inference_mode():
        reference = layer(x)
        ruled = RuledModel(layer, ["c12"], None)
        candidate = layer(x)
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)
    assert ruled.applied == ["c12"]


def test_linear_attention_kernels_must_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import torch

    from guardrail_tuner.rules import RuledModel

    monkeypatch.setitem(sys.modules, "fla", None)
    ruled = RuledModel(torch.nn.Linear(2, 2), ["c7"], None)
    assert ruled.applied == []
    assert ruled.skipped["c7"] == "flash-linear-attention does not import: ModuleNotFoundError"


def test_compiled_rules_keep_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from guardrail_tuner.rules import RuledModel

    model_dir = tmp_path / "model"
    tiny_deberta(model_dir, 2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(kw) or fn)
    inputs = tokenizer(["hello world"], return_tensors="pt")
    with torch.inference_mode():
        reference = model(**inputs).logits
        ruled = RuledModel(model, ["c12", "c8", "u4", "u1"], tokenizer.pad_token_id)
        candidate = ruled(**inputs).logits
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)
    assert calls == [{"dynamic": False}]
    if not torch.cuda.is_available():
        assert ruled.skipped["u1"] == "CUDA is unavailable"


def test_run_compares_rules_with_eager(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    tiny_deberta(model_dir, 2)
    config = json.loads((model_dir / "config.json").read_text())
    plan = plan_checks(
        {**inspect_model(str(model_dir), None), "rule_features": rule_features(config)}, "cpu"
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    result = run_inference(read_execution(str(path)), ["hello world", "world hello world"])
    assert result["rules"]["applied"] == ["c12", "c8"]
    assert result["rules"]["parity"]["passed"]
    assert result["rules"]["parity"]["label_changes"] == 0
    assert len(result["outputs"]) == 2
