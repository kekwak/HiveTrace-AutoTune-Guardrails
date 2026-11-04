from types import SimpleNamespace as NS
import pytest
from src.vllm_probe.pooling_adapter import PoolingAdapter


class Data:
    def __init__(self, values): self.values = values
    def tolist(self): return self.values


class Engine:
    def __init__(self, values): self.values, self.calls = values, []
    def encode(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return [NS(outputs=NS(data=Data(value))) for value in self.values]


def test_classifier_keeps_native_nine_class_order():
    labels = ['OK', 'S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']
    engine = Engine([[.6] + [.05] * 8])
    adapter = PoolingAdapter('koala', engine, lambda **kw: kw, labels=labels)
    assert adapter.score([[1, 2]])[0]['label'] == 'OK'
    assert engine.calls[0][1]['pooling_task'] == 'token_classify'


def test_laya_cannot_run_without_valid_markers():
    adapter = PoolingAdapter('laya', Engine([]), lambda **kw: kw)
    with pytest.raises(ValueError): adapter.encode([[1, 2]])
    with pytest.raises(ValueError): adapter.encode([[1, 2]], metadata=[{'marker_positions': [2], 'qtype': 0}])


def test_hidden_results_are_not_misrepresented_as_classifications():
    engine = Engine([[[.2, .4], [.3, .5]]])
    adapter = PoolingAdapter('gliner', engine, lambda **kw: kw)
    assert adapter.encode([[1, 2]]) == [[[.2, .4], [.3, .5]]]
    with pytest.raises(ValueError): adapter.score([[1, 2]])


def test_pooling_rejects_length_and_count_errors_before_safe_mapping():
    adapter = PoolingAdapter('llama', Engine([]), lambda **kw: kw, labels=['benign', 'malicious'])
    with pytest.raises(ValueError): adapter.score([[1] * 513])
    with pytest.raises(ValueError): adapter.score([[1]])
