import pytest
from src.vllm_probe.custom_heads import laya_choices, shield_policies, GlinerSafetyAdapter


def test_laya_keeps_calibration_and_native_choice_order():
    result = laya_choices([2., 0.], ['A', 'B'], temperature=2.)
    assert result['label'] == 'A'
    assert result['scores']['A'] == pytest.approx(.7310585786)
    for logits, temperature in [([float('inf'), 0.], 1.), ([1.], 1.), ([1., 0.], 0.)]:
        with pytest.raises(ValueError): laya_choices(logits, ['A', 'B'], temperature)


def test_shield_requires_all_four_policies_and_does_not_hide_nonfinite():
    assert shield_policies([.1, .2, .3, .5]) == {'label': 'unsafe', 'policy_probabilities': [.1, .2, .3, .5]}
    for values in [[.9], [.9, None, .1, .1], [.1, .2, float('nan'), .3]]:
        with pytest.raises(ValueError): shield_policies(values)


def test_gliner_reference_mode_does_not_replace_encoder():
    class Schema:
        def classification(self, **kwargs): return self
    class Model:
        def create_schema(self): return Schema()
    model = Model()
    adapter = GlinerSafetyAdapter(model, encoder_adapter=None)
    assert adapter.model is model
    adapter.close()
