import pytest
from src.vllm_probe.batching import batch_indices, restore_order


def test_bucketed_batches_preserve_all_rows_with_stable_ties():
    assert batch_indices([5, 1, 5, 2], 2, True) == [[1, 3], [0, 2]]
    assert batch_indices([5, 1, 5, 2], 3, False) == [[0, 1, 2], [3]]


def test_restore_original_request_order():
    assert restore_order([(1, 'b'), (3, 'd'), (0, 'a'), (2, 'c')], 4) == ['a', 'b', 'c', 'd']


def test_reject_missing_or_duplicate_output():
    with pytest.raises(ValueError):
        restore_order([(0, 'a'), (0, 'b')], 2)
    with pytest.raises(ValueError):
        restore_order([(0, 'a')], 2)


def test_reject_nonpositive_batch_size():
    with pytest.raises(ValueError):
        batch_indices([1, 2], 0, True)
