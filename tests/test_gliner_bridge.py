import pytest
from src.vllm_probe.gliner_bridge import hidden_batch


class Encoder:
    def encode(self, inputs):
        return [[[float(token), 1.] for token in row] for row in inputs]


def test_gliner_bridge_strips_padding_and_restores_shape():
    result = hidden_batch(Encoder(), [[1, 2, 0], [3, 0, 0]], [[1, 1, 0], [1, 0, 0]], 2)
    assert result == [[[1., 1.], [2., 1.], [0., 0.]], [[3., 1.], [0., 0.], [0., 0.]]]


def test_gliner_bridge_rejects_holes_empty_rows_and_bad_hidden_counts():
    for mask in [[[1, 0, 1]], [[0, 0, 0]]]:
        with pytest.raises(ValueError): hidden_batch(Encoder(), [[1, 2, 3]], mask, 2)
    class Broken:
        def encode(self, inputs): return []
    with pytest.raises(ValueError): hidden_batch(Broken(), [[1]], [[1]], 2)
