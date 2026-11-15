"""Public identities for configurable prompts; executable settings stay private."""
from copy import deepcopy

from .core import digest
from .messages import payload, private_strings

PROMPT_KEYS = {"text", "system_prompt", "prompt", "cue", "guideline", "response_guideline",
               "instructions", "criteria", "api_key", "token", "password", "secret"}
RESPONSE_KEYS = {"label", "scores", "score_type", "category", "generated_tokens",
                 "p_safe", "p_unsafe", "logit_margin", "public_confidence"}


def public_config(config):
    def visit(value, hidden=False):
        if isinstance(value, dict):
            return {k: visit(v, hidden or k in PROMPT_KEYS) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, hidden) for v in value]
        if hidden and isinstance(value, str):
            return "[private-sha256:" + digest(value) + "]"
        return deepcopy(value)
    result = deepcopy(config)
    for name in ("baseline", "candidate"):
        if name in result:
            result[name] = visit(result[name])
    return result


def private_values(rows, config=None):
    values = {s for row in rows for s in private_strings(payload(row)) if s}

    def visit(value, hidden=False):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, hidden or key in PROMPT_KEYS)
        elif isinstance(value, list):
            for item in value:
                visit(item, hidden)
        elif hidden and isinstance(value, str) and value:
            values.add(value)
    visit(config or {})
    return sorted(values, key=len, reverse=True)


def contains_private(value, texts):
    if isinstance(value, dict):
        return any(contains_private(k, texts) or contains_private(v, texts) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_private(v, texts) for v in value)
    return isinstance(value, str) and any(value == s or len(s) >= 4 and s in value for s in texts)


def clean_evidence(value, texts, allowed=()):
    if isinstance(value, str) and value in allowed:
        return value
    if isinstance(value, dict):
        result = {k: (v if k == "score_type" and v in
                      ("probabilities", "logits", "independent_probabilities", "raw")
                      else clean_evidence(v, texts, allowed)) for k, v in value.items()
                  if k in RESPONSE_KEYS or k in allowed or not contains_private(k, texts)}
        if "precision" in value:
            from .precision import validate_precision
            result["precision"] = validate_precision(value["precision"])
        return result
    if isinstance(value, (list, tuple)):
        return [clean_evidence(v, texts, allowed) for v in value]
    if isinstance(value, str):
        for text in texts:
            if value == text:
                value = "[redacted-input]"
            elif len(text) >= 4:
                value = value.replace(text, "[redacted-input]")
    return value
