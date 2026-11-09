"""Классификатор последовательностей (KoalaAI и подобные): P(unsafe) = 1 - P(безопасных классов)."""

import torch

from .base import Adapter


class SeqClsAdapter(Adapter):
    def load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        rev = self.spec.get("revision")
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["model_id"], revision=rev)
        model = AutoModelForSequenceClassification.from_pretrained(self.spec["model_id"], revision=rev, dtype=self.dtype)
        labels = model.config.id2label
        safe = set(self.spec["safe_labels"])
        unknown = safe - set(labels.values())
        if unknown:
            raise ValueError(f"Классов {sorted(unknown)} нет в модели; есть {sorted(labels.values())}")
        self.safe_ids = [i for i, name in labels.items() if name in safe]
        self._finish_load(model)

    def predict(self, texts):
        full = [len(ids) for ids in self.tokenizer(texts, add_special_tokens=True)["input_ids"]]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
        with torch.inference_mode():
            probs = self.model(**batch).logits.float().softmax(-1)
        p_safe = probs[:, self.safe_ids].sum(-1).clamp(0, 1)
        return [{"p_unsafe": float(1 - p), "n_tokens": min(n, self.max_length), "truncated": n > self.max_length}
                for p, n in zip(p_safe.cpu(), full)]
