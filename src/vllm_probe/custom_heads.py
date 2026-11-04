"""Native custom-head semantics kept separate from engine token transport."""
import math
from .adapter_contracts import classify_scores
from .shield_scoring import any_policy_unsafe


def laya_choices(logits, labels, temperature):
    if (len(logits) != len(labels) or not logits or not math.isfinite(temperature)
            or temperature <= 0 or any(not math.isfinite(x) for x in logits)):
        raise ValueError('Invalid calibrated Laya decision logits')
    maximum = max(logits)
    values = [math.exp((x-maximum)/temperature) for x in logits]
    total = sum(values)
    return classify_scores([x/total for x in values], labels)


def shield_policies(probabilities):
    decision = any_policy_unsafe(probabilities)
    if decision is None:
        raise ValueError('All four finite official policy scores required')
    return {'label': 'unsafe' if decision else 'safe',
            'policy_probabilities': list(probabilities)}


class LayaChoicesAdapter:
    """Decision-only choice API, not action-head or complete SDK replacement."""
    def __init__(self, pooling_adapter, labels, temperature):
        self.pooling = pooling_adapter
        self.labels, self.temperature = labels, temperature

    def score(self, input_ids, metadata):
        logits = self.pooling.encode(input_ids, metadata=metadata)
        return [laya_choices(row, self.labels, self.temperature) for row in logits]


class ShieldPoliciesAdapter:
    def __init__(self, decoder_adapter):
        self.decoder = decoder_adapter

    def score(self, policy_input_ids):
        if any(len(row) != 4 for row in policy_input_ids):
            raise ValueError('Four independently prepared policy inputs per record required')
        outputs = []
        for policies in policy_input_ids:
            values = self.decoder.score(policies)
            outputs.append(shield_policies([row['scores']['unsafe'] for row in values]))
        return outputs


class GlinerSafetyAdapter:
    """Native GLiNER schema/head with the vLLM encoder bridge installed."""
    def __init__(self, native_model, encoder_adapter):
        from .gliner_bridge import install_encoder_bridge
        self.model = native_model
        self.schema = native_model.create_schema().classification(
            task='safety', labels=['safe', 'unsafe'], multi_label=False)
        self.restore = (install_encoder_bridge(native_model, encoder_adapter)
                        if encoder_adapter is not None else lambda: None)

    def score(self, texts):
        import torch
        captured = []
        batch = self.model.processor.collate_fn_inference(
            [(text, self.schema.build()) for text in texts], max_len=None,
            error_policy='raise', architecture='span')
        if batch.original_texts != texts or (batch.attention_mask.sum(-1) > 512).any():
            raise ValueError('Processor changed text or input exceeds encoder limit')
        handle = self.model.classifier.register_forward_hook(
            lambda module, args, output: captured.append(output.detach().float().reshape(-1)))
        try:
            with torch.inference_mode():
                public = self.model.batch_extract(texts, schemas=self.schema,
                    batch_size=len(texts), include_confidence=True, max_len=None)
        finally:
            handle.remove()
        if len(captured) != len(texts) or len(public) != len(texts):
            raise ValueError('Native GLiNER result count mismatch')
        results = []
        for logits, raw in zip(captured, public):
            # Native argmax: first class (safe) wins ties, not >= .5 unsafe.
            result = classify_scores(logits.softmax(-1).cpu().tolist(), ['safe', 'unsafe'])
            confidence = raw['safety']['confidence']
            if result['label'] != raw['safety']['label'] or not math.isfinite(confidence):
                raise ValueError('GLiNER public output disagrees with native logits')
            results.append(result)
        return results

    def close(self):
        self.restore()
