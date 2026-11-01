import pytest

from src.vllm_probe.metrics import summarize


def test_grouping_and_missing_predictions():
    records = [
        dict(dataset='a', split='safe', expected=0, predicted=0),
        dict(dataset='a', split='harm', expected=1, predicted=0),
        dict(dataset='b', split='harm', expected=1, predicted=None),
    ]
    rows = summarize(records)
    overall = next(row for row in rows if row['scope'] == 'overall')
    assert (overall['total'], overall['evaluated'], overall['coverage']) == (3, 2, 2 / 3)
    assert (overall['tn'], overall['fn']) == (1, 1)
    safe = next(row for row in rows if row['dataset'] == 'a' and row['split'] == 'safe')
    harm = next(row for row in rows if row['dataset'] == 'a' and row['split'] == 'harm')
    assert safe['recall'] is None
    assert harm['precision'] is None
    assert {row['dataset'] for row in rows if row['scope'] == 'dataset'} == {'a', 'b'}


def test_empty_and_invalid():
    assert summarize([])[0]['coverage'] is None
    with pytest.raises(ValueError):
        summarize([dict(dataset='a', split='safe', expected=2, predicted=0)])
