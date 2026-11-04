from types import SimpleNamespace as NS
import pytest
from src.vllm_probe.decoder_adapter import DecoderAdapter


class Engine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, prompts, params, use_tqdm):
        self.calls.append((prompts, params))
        return [NS(outputs=[x]) for x in self.outputs]


def params(**kwargs):
    return kwargs


def test_generation_uses_global_greedy_and_keeps_categories():
    engine = Engine([NS(text='unsafe\nS5', token_ids=[3, 4, 5])])
    adapter = DecoderAdapter('open', engine, params)
    assert adapter.score([[1, 2]]) == [{'label': 'unsafe', 'category': 'S5'}]
    assert engine.calls[0][1]['max_tokens'] == 10
    assert 'allowed_token_ids' not in engine.calls[0][1]


def test_pair_is_conditional_raw_logprob_with_halo_threshold():
    engine = Engine([NS(logprobs=[{3: NS(logprob=-1000.), 4: NS(logprob=-999.)}])])
    adapter = DecoderAdapter('halo', engine, params, pair_ids=(3, 4))
    result = adapter.score([[1]])[0]
    assert result['label'] == 'unsafe'
    assert result['scores']['unsafe'] == pytest.approx(.7310585786)
    assert engine.calls[0][1]['logprob_token_ids'] == [3, 4]


def test_no_silent_truncation_or_missing_batch_rows():
    engine = Engine([])
    adapter = DecoderAdapter('hive', engine, params)
    with pytest.raises(ValueError):
        adapter.score([[1] * 4096])
    assert not engine.calls
    with pytest.raises(ValueError):
        adapter.score([[1]])


def test_missing_pair_score_is_not_safe():
    engine = Engine([NS(logprobs=[{3: NS(logprob=-1.)}])])
    adapter = DecoderAdapter('halo', engine, params, pair_ids=(3, 4))
    with pytest.raises(ValueError):
        adapter.score([[1]])


def test_open_truncated_category_is_not_accepted_as_complete():
    engine = Engine([NS(text='unsafe\nS5', finish_reason='length')])
    adapter = DecoderAdapter('open', engine, params)
    with pytest.raises(ValueError, match='truncated'):
        adapter.score([[1]])
