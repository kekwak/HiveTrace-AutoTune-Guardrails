"""Finite Yes/No scoring and conservative four-policy result aggregation."""
import math


def probability_yes(yes_logprob,no_logprob):
    if not math.isfinite(yes_logprob) or not math.isfinite(no_logprob):
        raise ValueError('Yes/No logprobs must be finite')
    maximum=max(yes_logprob,no_logprob)
    yes,no=math.exp(yes_logprob-maximum),math.exp(no_logprob-maximum)
    return yes/(yes+no)


def any_policy_unsafe(probabilities):
    if len(probabilities)!=4 or any(p is None or not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
        return None
    return int(any(p>=0.5 for p in probabilities))
