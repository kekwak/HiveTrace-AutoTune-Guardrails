from guardrail_tuner.datasets import DatasetSource
from guardrail_tuner.generation import check_generation


def test_greedy_generation_with_cache_matches_full_forward(tmp_path, monkeypatch):
    import torch
    from transformers import BertTokenizerFast, GPT2Config, GPT2LMHeadModel

    directory = tmp_path / "model"
    directory.mkdir()
    vocab = directory / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(directory)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    ).save_pretrained(directory)
    data = tmp_path / "quality.jsonl"
    data.write_text('{"text":"hello world"}\n')
    result = check_generation(
        str(directory), None, "cpu", DatasetSource("file", str(data)), "text", 16, 3, 1e-4, False
    )
    assert result["cache"]["passed"]
    assert result["cache"]["sequence_mismatches"] == 0
    assert result["candidate"]["passed"] is None
    assert result["settings"]["reference_dtype"] == "float32"
    monkeypatch.setattr(torch, "compile", lambda forward: forward)
    compiled = check_generation(
        str(directory), None, "cpu", DatasetSource("file", str(data)), "text", 16, 2, 1e-4, True
    )
    assert compiled["candidate"]["passed"]


def test_vllm_probe_without_cuda_is_unavailable(monkeypatch):
    import torch

    from guardrail_tuner.vllm_probe import probe_vllm

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = probe_vllm("org/model", None, DatasetSource("file", "unused"), "text", 16, 3, 60)
    assert result["status"] == "unavailable"
    assert result["reason"] == "cuda_unavailable"


def test_vllm_probe_starts_worker_and_never_recommends_without_quality(monkeypatch):
    import json

    import torch

    from guardrail_tuner import vllm_probe

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vllm_probe.importlib.util, "find_spec", lambda name: object())

    class Dataset:
        column_names = ["text"]

        def __getitem__(self, name):
            return ["private"]

    monkeypatch.setattr(vllm_probe, "load_source", lambda source: (Dataset(), {"rows": 1}))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))

        class Process:
            returncode = 0

            def communicate(self, **kwargs):
                return json.dumps({"status": "tokens_match", "recommended": False}), None

        return Process()

    monkeypatch.setattr(vllm_probe.subprocess, "Popen", run)
    result = vllm_probe.probe_vllm(
        "org/model", "sha", DatasetSource("file", "unused"), "text", 16, 2, 5
    )
    assert result["status"] == "tokens_match"
    assert result["recommended"] is False
    assert calls[0][0][-1] == "guardrail_tuner.vllm_worker"
    assert "private" not in json.dumps(result)
