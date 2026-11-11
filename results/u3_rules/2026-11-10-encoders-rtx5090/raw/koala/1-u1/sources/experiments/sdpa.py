"""Explicit KV expansion, preserving the pinned Transformers SDPA contract."""


def repeat_attention(module, query, key, value, attention_mask, dropout=0.0,
                     scaling=None, is_causal=None, **kwargs):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if kwargs.get("output_attentions", False):
        raise ValueError("Attention weights are outside the scoring contract")
    groups = query.shape[1] // key.shape[1]
    if query.shape[1] % key.shape[1] or key.shape[1] != value.shape[1]:
        raise ValueError("Invalid GQA dimensions")
    # This allocation is deliberately INSIDE forward and the latency boundary.
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    causal = query.shape[2] > 1 and attention_mask is None and causal
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout,
            scale=scaling, is_causal=causal, enable_gqa=False)
    return output.transpose(1, 2).contiguous(), None


def install_repeat(adapter, registry):
    original = registry["sdpa"]
    registry.register("sdpa", repeat_attention)
    adapter.restore_attention = lambda: registry.register("sdpa", original)
    adapter.metadata["sdpa_policy"] = "explicit_repeat_kv_forced_efficient"
