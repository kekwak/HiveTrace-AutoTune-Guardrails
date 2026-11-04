import math
import pytest
from src.vllm_probe.shield_scoring import probability_yes, any_policy_unsafe


def test_conditional_yes_probability_is_stable():
    assert probability_yes(-1000.,-1001.)==pytest.approx(0.73105857863)
    assert probability_yes(-2.,-2.)==0.5


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_scores_rejected(bad):
    with pytest.raises(ValueError):probability_yes(bad,0.)


def test_four_policy_decision_and_missing_coverage():
    assert any_policy_unsafe([0.1,0.2,0.4,0.5])==1
    assert any_policy_unsafe([0.1,0.2,0.3,0.4])==0
    assert any_policy_unsafe([0.9,None,0.1,0.1]) is None
    assert any_policy_unsafe([0.1]) is None
    assert any_policy_unsafe([float('nan'),0.1,0.2,0.3]) is None
