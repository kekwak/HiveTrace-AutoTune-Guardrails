import json

import pytest

from guardrail_tuner.quality import Prediction, QualityRow, compare_predictions, evaluate_quality


def test_paired_metrics_with_different_coverage():
    rows = [
        QualityRow("1", "a", "unsafe", {"technique": "sample"}),
        QualityRow("2", "b", "safe", {"technique": "sample"}),
        QualityRow("3", "c", "safe"),
    ]
    base = [
        Prediction("1", "ok", "unsafe", 0.9),
        Prediction("2", "ok", "unsafe", 0.6),
        Prediction("3", "ok", "safe", 0.1),
    ]
    cand = [
        Prediction("1", "ok", "safe", 0.4),
        Prediction("2", "ok", "safe", 0.3),
        Prediction("3", "over_limit"),
    ]
    result = compare_predictions(rows, base, cand)
    assert result["baseline"]["roc_auc"] == 1.0
    assert result["candidate"]["coverage"] == 2 / 3
    assert result["candidate"]["n_truncation_unknown"] == 2
    paired = result["comparison"]
    assert (paired["n_common"], paired["flips"]) == (2, 2)
    assert (paired["fnr"]["delta"], paired["fpr"]["delta"]) == (1, -1)
    assert result["slices"]["technique"]["sample"]["comparison"]["fnr"]["p_holm"] == 1


def test_evaluate_and_replay(tmp_path):
    rows = [QualityRow(str(i), f"private-{i}", "safe") for i in range(3)]
    calls = []

    def predict(batch):
        calls.append([row.id for row in batch])
        return [Prediction(row.id, "ok", "safe", truncated=True) for row in reversed(batch)]

    out = tmp_path / "run"
    result = evaluate_quality(rows, predict, predict, batch_size=2, output_dir=out)
    assert calls == [["0", "1"], ["2"]] * 2
    assert result["candidate"]["n_truncated"] == 3
    predictions = []
    for variant in ("baseline", "candidate"):
        raw = (out / f"{variant}_predictions.jsonl").read_text()
        assert "private-" not in raw
        predictions.append([Prediction(**json.loads(line)) for line in raw.splitlines()])
    dataset_file = (out / "dataset.json").read_text()
    assert "private-" not in dataset_file
    snapshot = json.loads(dataset_file)
    replay_rows = [QualityRow(text="", **row) for row in snapshot["rows"]]
    replay = compare_predictions(replay_rows, *predictions, dataset_metadata=snapshot["metadata"])
    assert result == {**replay, "batch_size": 2}
    assert json.loads((out / "status.json").read_text())["status"] == "complete"


@pytest.mark.parametrize(
    "bad", [[], [Prediction("x", "ok", "safe")] * 2, [Prediction("y", "ok", "safe")]]
)
def test_rejects_contract_violations(tmp_path, bad):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="prediction"):
        evaluate_quality(
            [QualityRow("x", "a", "safe")],
            lambda _: bad,
            lambda _: bad,
            batch_size=1,
            output_dir=out,
        )
    assert json.loads((out / "status.json").read_text())["status"] == "incomplete"


def test_statistical_cancellation():
    rows = [QualityRow(str(i), "sample", "unsafe" if i < 20 else "safe") for i in range(40)]
    base = [Prediction(row.id, "ok", "unsafe") for row in rows]
    cand = [Prediction(row.id, "ok", "safe") for row in rows]
    paired = compare_predictions(rows, base, cand)["comparison"]
    assert paired["fnr"]["worse"] == paired["fpr"]["better"] == 20
    assert paired["fnr"]["p"] < 0.001
    assert paired["fpr"]["p"] < 0.001
