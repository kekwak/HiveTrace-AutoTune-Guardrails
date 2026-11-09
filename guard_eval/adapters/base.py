"""Общий интерфейс адаптера: текст -> P(unsafe)."""

import torch

from .. import quant
from ..spec import resolve_dtype

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class Adapter:
    def __init__(self, spec, device):
        spec = resolve_dtype(spec, device)   # dtype: auto -> bf16 / fp16 / fp32 по устройству
        self.spec, self.device = spec, torch.device(device)
        self.dtype = DTYPES[spec["dtype"]]
        self.max_length = spec["max_length"]
        self.model = self.tokenizer = None
        self.quant_info = None

    def load(self):
        """Загрузить модель на CPU, затем quant.quantize переносит её на устройство."""
        raise NotImplementedError

    def predict(self, texts):
        """Список словарей {p_unsafe, n_tokens, truncated} в порядке texts."""
        raise NotImplementedError

    def user_length(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _finish_load(self, model):
        model.eval()
        # transformers оставляет часть параметров в fp32 (у DeBERTa — q_bias/v_bias), и смешанные dtype ломают matmul
        stray = sum(1 for p in model.parameters() if p.is_floating_point() and p.dtype != self.dtype)
        model.to(dtype=self.dtype)
        self.quant_info = quant.quantize(model, self.spec.get("quant", "none"), self.device)
        self.quant_info["params_cast_to_dtype"] = stray
        self.model = model
        if self.spec.get("compile"):
            self._compile()

    def _compile(self):
        """torch.compile основного модуля: torchao без компиляции квантует активации отдельными ядрами и почти не ускоряет."""
        self.model.forward = torch.compile(self.model.forward, dynamic=True)
        self.quant_info["compiled"] = "forward"

    def _truncate_text(self, text, max_tokens):
        """Обрезает только текст пользователя по токенам; шаблон и инструкция не трогаются."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text, False
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True), True

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
