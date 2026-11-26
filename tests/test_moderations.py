import pytest

from guardrail_tuner.moderations import Moderation


def test_classifier_maps_decisions_and_scores():
    adapter = Moderation(
        {"outputs": {"OK": [], "A": ["violence"], "B": ["violence"]}},
        "local-classifier",
        "sequence_classification",
    )
    assert adapter.prepare({"input": ["one", "two"]}) == ["one", "two"]
    response = adapter.response(
        [{"label": "A", "logits": [0, 0, 0]}, {"label": "OK", "logits": [1, 0, 0]}],
        {0: "OK", 1: "A", 2: "B"},
    )
    assert response["id"].startswith("modr-")
    assert response["model"] == "local-classifier"
    first, second = response["results"]
    assert first == {
        "flagged": True,
        "categories": {"violence": True},
        "category_scores": {"violence": pytest.approx(2 / 3)},
        "category_applied_input_types": {"violence": ["text"]},
    }
    assert second["flagged"] is False
    assert second["categories"] == {"violence": False}
    with pytest.raises(ValueError, match="cover every classification label"):
        adapter.response([{"label": "OK", "logits": [1, 0]}], {0: "OK", 1: "UNKNOWN"})


def test_generator_formats_input_and_matches_exact_verdict():
    adapter = Moderation(
        {
            "prompt_template": "Text: {input}\nVerdict:",
            "outputs": {"allow": [], "block": ["harassment"]},
        },
        "local-generator",
        "generation",
    )
    assert adapter.prepare({"input": "{input}"}) == ["Text: {input}\nVerdict:"]
    result = adapter.response([{"text": " block\n"}], {})["results"][0]
    assert result["flagged"] is True
    assert result["categories"] == {"harassment": True}
    assert result["category_scores"] == {}
    with pytest.raises(ValueError, match="Model verdict"):
        adapter.response([{"text": "not a valid verdict"}], {})


@pytest.mark.parametrize(
    "body", [{"input": []}, {"input": [1]}, {"input": "   "}, {"input": "x", "model": "other"}]
)
def test_invalid_moderation_request(body):
    adapter = Moderation({"outputs": {"OK": []}}, "local", "sequence_classification")
    with pytest.raises(ValueError):
        adapter.prepare(body)


def test_generation_requires_one_input_placeholder():
    with pytest.raises(ValueError, match="exactly one"):
        Moderation(
            {"prompt_template": "{input}{input}", "outputs": {"allow": []}},
            "local",
            "generation",
        )
