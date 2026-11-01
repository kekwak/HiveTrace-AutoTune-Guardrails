"""Strict artifact alignment and stable binary logit comparisons."""
import math


def align_outputs(rows, outputs):
    if len(rows) != len(outputs):
        raise ValueError('record/output count mismatch')
    return list(zip(rows, outputs))


def binary_logits(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        return None
    return tuple(float(v) for v in value)


def compare_logits(a, b, temperature):
    a, b = binary_logits(a), binary_logits(b)
    if a is None or b is None:
        return None
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')

    def probabilities(z):
        maximum = max(z)
        exp = [math.exp((v - maximum) / temperature) for v in z]
        return [v / sum(exp) for v in exp]

    pa, pb = probabilities(a), probabilities(b)
    return dict(max_logit_delta=max(abs(x-y) for x,y in zip(a,b)),
                max_probability_delta=max(abs(x-y) for x,y in zip(pa,pb)),
                verdict_flip=(a[0] >= a[1]) != (b[0] >= b[1]))
