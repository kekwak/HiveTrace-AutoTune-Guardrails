import pytest
from src.vllm_probe.reporting import align_outputs, binary_logits, compare_logits


def test_alignment_never_silently_drops_records():
    with pytest.raises(ValueError):
        align_outputs([{}, {}], [[1., 0.]])


@pytest.mark.parametrize('value', [None, [1.], [1., 2., 3.], [float('nan'), 1.], [True, 1.], [[1., 2.]]])
def test_malformed_outputs_not_evaluable(value):
    assert binary_logits(value) is None


def test_common_logit_offset_is_detected_despite_equal_probabilities():
    result = compare_logits([1., 0.], [11., 10.], 2.)
    assert result['max_logit_delta'] == 10.
    assert result['max_probability_delta'] == 0.
    assert result['verdict_flip'] is False
