from types import SimpleNamespace

import pytest

from guardrail_tuner.protocol import APIRequestError, serve_request


def test_classify_uses_model_labels_and_probabilities():
    import torch

    forward_calls = []

    def forward(encoded):
        forward_calls.append(encoded)
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    inference = SimpleNamespace(
        config={"task": "sequence_classification", "model": "fixture", "batch_size": 2},
        model=SimpleNamespace(config=SimpleNamespace(problem_type=None)),
        labels={0: "class a", 1: "class b"},
        runner=SimpleNamespace(
            encode=lambda inputs: SimpleNamespace(attention_mask=torch.tensor([[1, 1], [1, 0]])),
            forward=forward,
        ),
    )
    result = serve_request(inference, "/classify", {"input": ["one two", "one"]})
    assert result["object"] == "list"
    assert result["model"] == "fixture"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}
    assert [row["index"] for row in result["data"]] == [0, 1]
    assert [row["label"] for row in result["data"]] == ["class b", "class a"]
    assert result["data"][0]["probs"] == pytest.approx([0.26894142, 0.73105858])
    assert result["data"][0]["num_classes"] == 2
    assert len(forward_calls) == 1


def test_generation_formats_chat_and_raw_text():
    calls = []

    def generate(prompt, limit, *, add_special_tokens):
        calls.append((prompt, limit, add_special_tokens))
        return [7, 2], 5

    inference = SimpleNamespace(
        config={"task": "generation", "model": "fixture", "max_new_tokens": 4},
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),
        tokenizer=SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: "rendered chat",
            decode=lambda ids, **kwargs: "verdict",
        ),
        generate=generate,
    )
    messages = [{"role": "user", "content": "prompt"}]
    result = serve_request(
        inference, "/v1/chat/completions", {"messages": messages, "max_tokens": 2}
    )
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "verdict"}
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 7
    result = serve_request(inference, "/v1/completions", {"prompt": ["raw", "second"]})
    assert result["object"] == "text_completion"
    assert result["choices"][1]["text"] == "verdict"
    assert result["usage"]["total_tokens"] == 14
    assert calls == [("rendered chat", 2, False), ("raw", 4, True), ("second", 4, True)]


@pytest.mark.parametrize(
    "path,body,reason",
    [
        ("/classify", {"input": "x"}, "does not support"),
        ("/v1/completions", {"prompt": "x", "model": "wrong"}, "does not match"),
        ("/v1/completions", {"prompt": "x", "stream": True}, "stream=false"),
        ("/v1/completions", {"prompt": "x", "max_tokens": 5}, "between 1 and 4"),
        ("/v1/chat/completions", {"messages": []}, "Expected text messages"),
    ],
)
def test_rejects_unsupported_requests(path, body, reason):
    inference = SimpleNamespace(
        config={"task": "generation", "model": "fixture", "max_new_tokens": 4},
    )
    with pytest.raises(APIRequestError, match=reason):
        serve_request(inference, path, body)
