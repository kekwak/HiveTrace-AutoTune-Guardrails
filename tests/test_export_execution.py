from pathlib import Path

import pytest


def test_selected_onnx_is_reused_and_inventory_is_enforced(tmp_path: Path) -> None:
    import torch
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import BertConfig, BertForSequenceClassification
    from u1_fixtures import tiny_tokenizer

    from guardrail_tuner.exported_classifier import ExportedClassifier, save_artifacts
    from guardrail_tuner.u1_bridge import execution

    tokenizer = tiny_tokenizer()
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    ).eval()
    source = tmp_path / "source"
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    ort = ORTModelForSequenceClassification.from_pretrained(
        source, export=True, provider="CPUExecutionProvider"
    )
    saved = save_artifacts(model, ort, tokenizer, tmp_path / "artifacts")["onnxruntime"]
    saved["path"] = str(Path(saved["path"]).relative_to(tmp_path))
    document = {
        "execution": {
            "version": 2,
            "batch_size": 2,
            "variant": {
                "adapter": "exported_sequence_classification",
                "weights": saved,
                "tokenizer": dict(saved),
                "preparation": {"limit": 512, "truncation": True},
                "scoring": {"method": "multiclass", "threshold": 0.5},
                "execution": {
                    "runtime": "onnxruntime",
                    "device": "cpu",
                    "providers": ["CPUExecutionProvider"],
                    "provider_options": {"CPUExecutionProvider": {}},
                    "graph_optimization": "all",
                    "intra_op_threads": 1,
                },
            },
        }
    }
    runtime = ExportedClassifier(execution(document, "candidate", tmp_path))
    try:
        actual = runtime.predict(["library", "The public library."])
        with torch.inference_mode():
            expected = model(
                **tokenizer(["library", "The public library."], padding=True, return_tensors="pt")
            ).logits
        torch.testing.assert_close(
            torch.tensor([r["logits"] for r in actual]), expected, atol=1e-5, rtol=1e-5
        )
    finally:
        runtime.close()
    (tmp_path / saved["path"] / "unexpected.data").write_bytes(b"changed external data")
    with pytest.raises(ValueError, match="inventory"):
        ExportedClassifier(execution(document, "candidate", tmp_path))
