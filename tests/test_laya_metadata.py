import pytest

from src.vllm_probe.laya_metadata import parse_laya_metadata


def test_preserves_marker_order_and_question_type():
    assert parse_laya_metadata({'laya': {'marker_positions': [7, 3], 'qtype': 0}}, 9) == ((7, 3), 0)


@pytest.mark.parametrize('markers,qtype', [([], 0), ([3, 3], 0), ([-1, 3], 0), ([3, 9], 0), ([True, 3], 0), ([3, 4], 3)])
def test_rejects_invalid_schema(markers, qtype):
    with pytest.raises(ValueError):
        parse_laya_metadata({'laya': {'marker_positions': markers, 'qtype': qtype}}, 9)


def test_dummy_request_has_no_schema():
    assert parse_laya_metadata(None, 9) is None
