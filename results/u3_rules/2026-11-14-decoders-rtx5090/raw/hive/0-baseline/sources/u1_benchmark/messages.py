"""Canonical, role-aware inputs shared by indexing and inference."""
from .core import digest


class UnsupportedInput(ValueError):
    """A source format has no approved native preparation for this model."""


def messages(value):
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError("Expected text or nonempty messages")
    roles = [m.get("role") if isinstance(m, dict) else None for m in value]
    if roles not in (["user"], ["assistant"], ["user", "assistant"]):
        raise ValueError("Expected user, assistant or user -> assistant")
    if any(set(m) != {"role", "content"} or not isinstance(m["content"], str) for m in value):
        raise ValueError("Messages require only role and string content")
    return [dict(m) for m in value]


def payload(row):
    return row["messages"] if "messages" in row else row["text"]


def content_hash(value):
    return digest(messages(value))


def select(model, value):
    source = messages(value)
    roles = [m["role"] for m in source]
    mode = {("user",): "user-only", ("assistant",): "assistant-only",
            ("user", "assistant"): "user+assistant"}[tuple(roles)]
    if model in ("gliner", "koala", "llama", "laya"):
        target = len(source) - 1
        context = []
        effective = [source[target]]
    elif model == "halo":
        if "user" not in roles:
            raise UnsupportedInput("assistant-only is not applicable to HaloGuard input guard")
        target, context, effective = 0, [], [source[0]]
    elif model == "hive":
        target = len(source) - 1
        context = list(range(target))
        effective = source
    elif model == "shield":
        if roles == ["assistant"]:
            raise UnsupportedInput("ShieldGemma response mode requires a preceding user")
        target = len(source) - 1
        context, effective = list(range(target)), source
    elif model == "open":
        if roles != ["user"]:
            raise UnsupportedInput("OpenGuardrails 0124 response mode is not confirmed")
        target, context, effective = 0, [], source
    else:
        raise ValueError("Unknown model")
    return {"source_mode": mode, "target_index": target, "target_role": source[target]["role"],
            "context_indices": context, "effective_messages": effective,
            "source_sha256": content_hash(source)}


def private_strings(value):
    return [m["content"] for m in messages(value)]
