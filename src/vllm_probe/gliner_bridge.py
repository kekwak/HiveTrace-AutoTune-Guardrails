"""Keep GLiNER's native schema processing/heads; route encoder through vLLM.

This experimental bridge copies hidden states through host memory. It is not
zero-copy and is not claimed to improve speed. No private texts enter vLLM
metadata or exported diagnostics.
"""
import math


def hidden_batch(adapter, input_ids, attention_mask, hidden_size):
    if not input_ids or len(input_ids) != len(attention_mask):
        raise ValueError('Encoder batch/mask mismatch')
    width = len(input_ids[0])
    unpadded = []
    for ids, mask in zip(input_ids, attention_mask):
        n = sum(mask)
        if (len(ids) != width or len(mask) != width or n == 0
                or mask != [1] * n + [0] * (width-n)):
            raise ValueError('Expected nonempty right-padded binary attention mask')
        unpadded.append(ids[:n])
    outputs = adapter.encode(unpadded)
    if len(outputs) != len(unpadded):
        raise ValueError('Hidden result count mismatch')
    result = []
    for ids, hidden in zip(unpadded, outputs):
        if (len(hidden) != len(ids) or any(len(row) != hidden_size for row in hidden)
                or any(not math.isfinite(x) for row in hidden for x in row)):
            raise ValueError('Malformed or nonfinite encoder hidden states')
        result.append(hidden + [[0.] * hidden_size for _ in range(width-len(ids))])
    return result


def install_encoder_bridge(model, adapter):
    """Return restore callback; model is the pinned FP32 GLiNER2 instance."""
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    encoder = model.encoder
    original = encoder.forward
    hidden_size = encoder.config.hidden_size

    def forward(input_ids=None, attention_mask=None, **kwargs):
        if any(value is not None for value in kwargs.values()):
            raise ValueError('Unexpected native encoder arguments')
        hidden = hidden_batch(adapter, input_ids.detach().cpu().tolist(),
                              attention_mask.detach().cpu().tolist(), hidden_size)
        device = input_ids.device
        return BaseModelOutput(last_hidden_state=torch.tensor(hidden, device=device, dtype=torch.float32))

    encoder.forward = forward
    return lambda: setattr(encoder, 'forward', original)
