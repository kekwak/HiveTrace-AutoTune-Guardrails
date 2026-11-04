import pytest
from src.vllm_probe.encoder_contracts import sequence_lengths, validate_metadata, parse_pooling


def test_flat_positions_reconstruct_exact_sequences():
    assert sequence_lengths([0, 1, 2, 0, 1]) == [3, 2]
    for positions in [[], [1], [0, 2], [0, 1, 1]]:
        with pytest.raises(ValueError):
            sequence_lengths(positions)


def test_laya_metadata_is_required_and_bounded():
    assert validate_metadata('laya', 5, {'marker_positions': [1, 4], 'qtype': 0})['qtype'] == 0
    for metadata in [None, {'marker_positions': [5], 'qtype': 0},
                     {'marker_positions': [1, 1], 'qtype': 0},
                     {'marker_positions': [1], 'qtype': 7}]:
        with pytest.raises(ValueError):
            validate_metadata('laya', 5, metadata)


def test_pooling_results_fail_on_nonfinite_or_shape_mismatch():
    assert parse_pooling([.2, .8], ['benign', 'malicious'])['label'] == 'malicious'
    with pytest.raises(ValueError):
        parse_pooling([float('nan'), .8], ['benign', 'malicious'])
    with pytest.raises(ValueError):
        parse_pooling([.8], ['benign', 'malicious'])
