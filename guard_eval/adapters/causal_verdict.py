"""Decoder-guardrail с вердиктом следующим токеном (HiveTraceGuard-Pro, WildGuard, OpenGuardrails).

Один forward с logits_to_keep=1: P(unsafe) = softmax(logit_unsafe, logit_safe)[unsafe].
Без generate: бинарный вердикт определяется первым токеном ответа.
"""

import torch

from .base import Adapter


class CausalVerdictAdapter(Adapter):
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rev = self.spec.get("revision")
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["model_id"], revision=rev)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.chat_template = None
        source = self.spec["prompt"].get("chat_template_from")
        if source:   # шаблон чата из другого репозитория (у OpenGuardrails-0124 своего рабочего нет)
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(source["repo"], source.get("file", "chat_template.jinja"))) as f:
                self.chat_template = f.read()
        model = AutoModelForCausalLM.from_pretrained(self.spec["model_id"], revision=rev, dtype=self.dtype)
        self.safe_id, self.unsafe_id = self._verdict_ids()
        self.prompt_overhead = len(self._encode(self._prompt("")))
        self._finish_load(model)

    def _prompt(self, text):
        p, prefix = self.spec["prompt"], self.spec["verdict"].get("prefix", "")
        if p["kind"] == "chat_template":
            body = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                      chat_template=self.chat_template,
                                                      add_generation_prompt=p.get("add_generation_prompt", False))
        elif p["kind"] == "format":
            body = p["template"].replace("{text}", text)
        else:
            raise ValueError(f"Неизвестный prompt.kind {p['kind']}")
        return body + prefix

    def _encode(self, prompt):
        add = self.spec["prompt"].get("add_special_tokens", self.spec["prompt"]["kind"] != "chat_template")
        return self.tokenizer.encode(prompt, add_special_tokens=add)

    def _verdict_ids(self):
        """Id первого токена ответа для safe/unsafe — в том контексте, где он реально стоит после промпта."""
        base = self._encode(self._prompt("test"))
        ids = []
        for word in (self.spec["verdict"]["safe"], self.spec["verdict"]["unsafe"]):
            full = self._encode(self._prompt("test") + word)
            if full[:len(base)] != base or len(full) <= len(base):
                raise ValueError(f"Токенизация промпта меняется при добавлении {word!r}; задайте другой verdict.prefix")
            ids.append(full[len(base)])
        if ids[0] == ids[1]:
            raise ValueError("Первые токены safe и unsafe совпадают")
        return ids

    def predict(self, texts):
        limit = self.max_length - self.prompt_overhead
        prompts, meta = [], []
        for text in texts:
            text, cut = self._truncate_text(text, limit)
            ids = self._encode(self._prompt(text))
            prompts.append(ids)
            meta.append((len(ids), cut))
        width = max(len(ids) for ids in prompts)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.tensor([[pad] * (width - len(ids)) + ids for ids in prompts], device=self.device)
        mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts], device=self.device)
        # при левом паддинге позиции считаем по маске, иначе батч даёт другие логиты, чем одиночный запрос
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                                logits_to_keep=1).logits[:, -1].float()
        pair = logits[:, [self.safe_id, self.unsafe_id]].softmax(-1)[:, 1].cpu()
        return [{"p_unsafe": float(p), "n_tokens": n, "truncated": cut} for p, (n, cut) in zip(pair, meta)]
