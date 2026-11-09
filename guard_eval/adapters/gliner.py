"""GLiNER2 (gliner-guard-omni): классификация safety safe/unsafe, как в адаптере У1.

P(unsafe) — softmax двух логитов классификатора, перехваченных хуком; вердикт safe при равенстве.
"""

import torch

from .base import Adapter


class GlinerAdapter(Adapter):
    def load(self):
        from gliner2 import GLiNER2
        from huggingface_hub import snapshot_download
        path = snapshot_download(self.spec["model_id"], revision=self.spec.get("revision"),
                                 allow_patterns=["README.md", "config.json", "encoder_config/config.json",
                                                 "tokenizer.json", "tokenizer_config.json", "model.safetensors"])
        model = GLiNER2.from_pretrained(path, attention_backend="standard", quantize=False, compile=False,
                                        map_location="cpu").to(dtype=self.dtype)
        model.strict_extraction = True
        self.schema = model.create_schema().classification(task=self.spec["task"], labels=self.spec["labels"],
                                                           multi_label=False)
        self.tokenizer = model.processor.tokenizer
        self._finish_load(model)

    def _compile(self):
        # компилируем только encoder: обвязка GLiNER2 на Python плохо переносит torch.compile целиком
        self.model.encoder.forward = torch.compile(self.model.encoder.forward, dynamic=True)
        self.quant_info["compiled"] = "encoder.forward"

    def predict(self, texts):
        prepared = [self._truncate_text(t, self.max_length - 32) for t in texts]
        captured = []
        hook = self.model.classifier.register_forward_hook(lambda m, a, out: captured.append(out.detach()))
        try:
            with torch.inference_mode():
                self.model.batch_extract([t for t, _ in prepared], schemas=self.schema, batch_size=len(texts),
                                         include_confidence=True, max_len=None)
        finally:
            hook.remove()
        if len(captured) != len(texts):
            raise ValueError(f"Ожидалось {len(texts)} вызовов классификатора, получено {len(captured)}")
        rows = []
        for logits, (text, cut) in zip(captured, prepared):
            p = logits.float().reshape(-1).softmax(-1)
            rows.append({"p_unsafe": float(p[1]), "n_tokens": self.user_length(text), "truncated": cut})
        return rows
