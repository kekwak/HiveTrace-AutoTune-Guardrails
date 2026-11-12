import json
import os
import subprocess
import sys

import pytest

from guardrail_tuner.benchmark import percentile, texts


def test_percentile_and_dataset_validation():
    assert percentile([10, 20, 30], 0.95) == 29

    class FakeDataset:
        column_names = ["text"]

        def __getitem__(self, key):
            return [None]

    with pytest.raises(ValueError, match="strings"):
        texts(FakeDataset(), "text", "quality")


def test_cpu_benchmark_with_real_tiny_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("optimum.onnxruntime")
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    model_path = tmp_path / "model"
    model_path.mkdir()
    vocab = model_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_path)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=7,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=2,
        )
    ).save_pretrained(model_path)
    quality = tmp_path / "quality.jsonl"
    quality.write_text('{"text":"hello","label":0}\n{"text":"hello world","label":1}\n')
    speed = tmp_path / "speed.jsonl"
    speed.write_text('{"text":"world"}\n')
    env = os.environ.copy()
    env["HF_HOME"] = str(tmp_path / "hf-cache")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrail_tuner.cli",
            "benchmark",
            str(model_path),
            "--device",
            "cpu",
            "--output-mode",
            "multiclass",
            "--quality-label-column",
            "label",
            "--quality-file",
            str(quality),
            "--performance-file",
            str(speed),
            "--warmup",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["quality"]["passed"]
    assert report["quality"]["labels"]["samples"] == 2
    assert (
        report["quality"]["labels"]["pytorch_correct"]
        == report["quality"]["labels"]["onnxruntime_correct"]
    )
    for counts in report["quality"]["labels"]["per_class"]:
        assert sum(counts["pytorch"].values()) == 2
        assert counts["pytorch"]["tp"] + counts["pytorch"]["fn"] == 1
        assert counts["pytorch"]["tn"] + counts["pytorch"]["fp"] == 1
        assert counts["pytorch"] == counts["onnxruntime"]
    assert report["quality"]["decision_disagreements"] == 0
    assert report["latency"]["onnxruntime"]["requests"] == 1
    assert report["providers"][0] == "CPUExecutionProvider"
    assert report["workload"]["results"][0]["batch_size"] == 1
    assert report["workload"]["results"][0]["quality"]["passed"]


def test_cuda_fails_if_pytorch_has_no_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("optimum.onnxruntime")
    from guardrail_tuner.benchmark import run_benchmark
    from guardrail_tuner.datasets import DatasetSource

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source = DatasetSource("file", "unused.jsonl")
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        run_benchmark(
            "unused",
            None,
            "cuda",
            source,
            source,
            "text",
            "text",
            512,
            1,
            0.001,
            "multiclass",
            0.5,
        )


def test_cli_rejects_labels_with_logits(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrail_tuner.cli",
            "benchmark",
            str(tmp_path),
            "--device",
            "cpu",
            "--output-mode",
            "logits",
            "--quality-label-column",
            "label",
            "--quality-file",
            str(tmp_path / "quality.jsonl"),
            "--performance-file",
            str(tmp_path / "speed.jsonl"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--quality-label-column requires" in result.stderr


def test_cli_rejects_tensorrt_on_cpu(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrail_tuner.cli",
            "benchmark",
            str(tmp_path),
            "--device",
            "cpu",
            "--ort-provider",
            "tensorrt",
            "--output-mode",
            "logits",
            "--quality-file",
            str(tmp_path / "q.jsonl"),
            "--performance-file",
            str(tmp_path / "p.jsonl"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "TensorRT execution provider requires CUDA" in result.stderr
