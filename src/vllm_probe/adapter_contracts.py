"""Dependency-free, pinned native contracts for the eight-model vLLM study.

Checkpoint provenance: origin/results/haloguard-gliner-u1,
speed_dataset/contracts.json. Prompt-Guard is an explicitly selected public
mirror, not a claim of verified equivalence to Meta's gated original.
"""
import math
import random
import re

MODELS = {
    'hive': dict(repo='hivetrace/HiveTraceGuard-Pro', revision='ed008276c96253e14885e6ade3b7b6d58daa2480', dtype='float16', kind='generate', limit=4096, max_new_tokens=1),
    'open': dict(repo='openguardrails/OpenGuardrails-Text-4B-0124', revision='9bdd1fb549e01b5f6aae4e9c395998492a08830b', dtype='float16', kind='generate', limit=4096, max_new_tokens=10, template_repo='openguardrails/OpenGuardrails-Text-2510', template_revision='cce015e5c8f47be0b8f9022b991ec552bb3cc0b7'),
    'shield': dict(repo='google/shieldgemma-2b', revision='d1dffc9c8c9237a90aab09c61383791e718ef9e8', dtype='float32', kind='policy_pair', limit=8192, threshold=.5),
    'halo': dict(repo='astroware/HaloGuard1-Gen-0.8B', revision='85ff68dd063a7e58487a4251824fd36a20977d5c', dtype='float16', kind='pair', limit=4096, threshold=.65),
    'laya': dict(repo='convaiinnovations/laya', revision='7b928d828b7b0e022f929d9bd2e44165aa270148', dtype='float32', kind='custom', limit=512),
    'koala': dict(repo='KoalaAI/Text-Moderation', revision='838ade0edb66dcffc5532d08ff6ed5c899abfb5c', dtype='float32', kind='classify', limit=512),
    'llama': dict(repo='hipocap/Llama-Prompt-Guard-2-22M', revision='05f1919555db0e711b73cbb7ec2424702b918c20', dtype='float32', kind='classify', limit=512),
    'gliner': dict(repo='hivetrace/gliner-guard-omni', revision='763f8aa9a7714dd2c3264f3751acdd5d854c5ac4', dtype='float32', kind='schema', limit=512),
}


def parse_verdict(text, model):
    text = text.strip()
    if model == 'hive':
        if text not in ('safe', 'unsafe'):
            raise ValueError('Unexpected Hive verdict')
        return {'label': text}
    if model != 'open':
        raise ValueError('Not a generation contract')
    match = re.fullmatch(r'(safe|unsafe)(?:\s+(S\d+(?:\s*,\s*S\d+)*))?', text)
    if not match or (match[1] == 'safe' and match[2]):
        raise ValueError('Unexpected OpenGuardrails verdict/category')
    return {'label': match[1], 'category': match[2]}


def classify_scores(scores, labels, threshold=None):
    scores, labels = list(scores), list(labels)
    if (not scores or len(scores) != len(labels) or len(set(labels)) != len(labels)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores)
            or abs(sum(scores) - 1) > 1e-5):
        raise ValueError('Invalid native probability vector')
    if threshold is not None:
        if labels != ['safe', 'unsafe'] or not 0 <= threshold <= 1:
            raise ValueError('Threshold requires ordered safe/unsafe labels')
        index = int(scores[1] >= threshold)
    else:
        index = max(range(len(scores)), key=scores.__getitem__)
    return {'label': labels[index], 'scores': dict(zip(labels, scores))}


def sample_indices(total, count=8, seed=17):
    if not 0 < count <= total:
        raise ValueError('Sample must fit split, without replacement')
    return sorted(random.Random(seed).sample(range(total), count))


def compare_results(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError('Paired result count mismatch')
    result = dict(total=len(reference), paired=0, missing=0, label_flips=0,
                  category_mismatches=0, max_score_delta=None)
    for a, b in zip(reference, candidate):
        if a is None or b is None:
            result['missing'] += 1
            continue
        result['paired'] += 1
        result['label_flips'] += int(a['label'] != b['label'])
        result['category_mismatches'] += int(a.get('category') != b.get('category'))
        if 'policy_probabilities' in a or 'policy_probabilities' in b:
            aa, bb = a.get('policy_probabilities', []), b.get('policy_probabilities', [])
            if (len(aa) != 4 or len(bb) != 4
                    or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                           or not 0 <= x <= 1 for x in [*aa, *bb])):
                raise ValueError('Four finite paired policy probabilities required')
            delta = max(abs(x-y) for x, y in zip(aa, bb))
            result['max_score_delta'] = max(result['max_score_delta'] or 0., delta)
        if 'scores' in a or 'scores' in b:
            if set(a.get('scores', {})) != set(b.get('scores', {})):
                raise ValueError('Native score labels differ')
            for key in a['scores']:
                x, y = a['scores'][key], b['scores'][key]
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError('Nonfinite paired score')
                delta = abs(x-y)
                result['max_score_delta'] = max(result['max_score_delta'] or 0., delta)
    return result
