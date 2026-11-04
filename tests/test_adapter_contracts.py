import pytest

from src.vllm_probe.adapter_contracts import (
    MODELS, classify_scores, parse_verdict, sample_indices, compare_results,
)


def test_all_eight_models_have_pinned_contracts():
    assert set(MODELS) == {'hive', 'open', 'shield', 'halo', 'laya', 'koala', 'llama', 'gliner'}
    assert all(len(m['revision']) == 40 for m in MODELS.values())
    assert MODELS['shield']['dtype'] == 'float32'
    assert MODELS['halo']['threshold'] == .65


def test_native_generation_retains_category_and_rejects_unknown():
    assert parse_verdict('unsafe\nS5, S12', 'open') == {'label': 'unsafe', 'category': 'S5, S12'}
    assert parse_verdict('safe', 'hive') == {'label': 'safe'}
    for value in ['probably safe', 'safe\nS5', '', 'unsafe extra']:
        with pytest.raises(ValueError):
            parse_verdict(value, 'open')
    with pytest.raises(ValueError):
        parse_verdict('unsafe\nS5', 'hive')


def test_scores_preserve_labels_thresholds_and_fail_closed():
    assert classify_scores([.35, .65], ['safe', 'unsafe'], threshold=.65)['label'] == 'unsafe'
    assert classify_scores([.5, .5], ['benign', 'malicious'])['label'] == 'benign'
    for scores in [[float('nan'), 0.], [.2], [.2, .2], [-.1, 1.1]]:
        with pytest.raises(ValueError):
            classify_scores(scores, ['safe', 'unsafe'])


def test_small_sample_is_seeded_unique_and_not_just_first_rows():
    a = sample_indices(345, 8, seed=17)
    assert a == sample_indices(345, 8, seed=17)
    assert len(a) == len(set(a)) == 8
    assert a != list(range(8))
    with pytest.raises(ValueError):
        sample_indices(3, 8, seed=17)


def test_comparison_never_ignores_missing_or_categories():
    a = [{'label': 'unsafe', 'category': 'S5'}, {'label': 'safe'}]
    b = [{'label': 'unsafe', 'category': 'S6'}, None]
    result = compare_results(a, b)
    assert result == {'total': 2, 'paired': 1, 'missing': 1, 'label_flips': 0,
                      'category_mismatches': 1, 'max_score_delta': None}
    with pytest.raises(ValueError):
        compare_results(a, b[:1])


def test_shield_policy_drift_is_compared_even_without_verdict_flip():
    a = [{'label': 'unsafe', 'policy_probabilities': [.9, .1, .2, .3]}]
    b = [{'label': 'unsafe', 'policy_probabilities': [.6, .4, .3, .2]}]
    assert compare_results(a, b)['max_score_delta'] == pytest.approx(.3)
    for probabilities in [[.9], [.9, .1, .2, float('nan')]]:
        with pytest.raises(ValueError):
            compare_results(a, [{'label': 'unsafe', 'policy_probabilities': probabilities}])
