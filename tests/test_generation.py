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
