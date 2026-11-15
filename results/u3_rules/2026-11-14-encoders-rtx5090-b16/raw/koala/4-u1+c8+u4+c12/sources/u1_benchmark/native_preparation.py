"""Version-three native preparation; historical speed_dataset preparation is frozen."""
import inspect
import json
from importlib.metadata import version
from pathlib import Path

from speed_dataset.preparation import Preparer as LegacyPreparer, GUIDELINE, normalized
from .contracts import MODEL_SPECS
from .core import digest, sha
from .messages import UnsupportedInput, messages, select

RESPONSE_GUIDELINE = GUIDELINE.replace("The prompt shall not contain or seek generation of content",
                                     "The chatbot shall not generate content")


class Preparer:
    def __init__(self, model):
        self.name, self.spec = model, MODEL_SPECS[model]
        if model != "open":
            self.legacy = LegacyPreparer(model)
            self.tokenizer = self.legacy.tokenizer
            self.assets = dict(self.legacy.assets)
            self.versions = dict(self.legacy.versions)
        else:
            # Do not initialize the historical Open preparer: it downloads a 2510 template.
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
            from speed_dataset.core import CONTRACTS
            required = CONTRACTS["dependencies"]
            self.versions = {k: version(k) for k in required}
            if self.versions != required:
                raise ValueError("Tokenizer environment differs from pinned dependencies")
            names = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "config.json", "README.md"]
            path = Path(snapshot_download(self.spec["repo"], revision=self.spec["revision"], allow_patterns=names))
            self.tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
            self.tokenizer.model_max_length = 10**9
            self.assets = {name: sha(path / name) for name in names if (path / name).is_file()}
            self.assets["effective_chat_template"] = digest(self.tokenizer.chat_template)
            self.legacy = None
        self.assets["native_role_preparation"] = digest(inspect.getsource(type(self)))

    def render(self, value):
        choice = select(self.name, value)
        effective = choice["effective_messages"]
        if self.name == "hive":
            return self.tokenizer.apply_chat_template(effective, tokenize=False)
        if self.name == "shield":
            policy = RESPONSE_GUIDELINE if choice["target_role"] == "assistant" else GUIDELINE
            return self.tokenizer.apply_chat_template(effective, tokenize=False, guideline=policy)
        if self.name == "open":
            return self.tokenizer.apply_chat_template(effective, tokenize=False, add_generation_prompt=True)
        return self.legacy.render(effective[0]["content"])

    def prepare(self, value):
        choice = select(self.name, value)
        if self.name not in ("hive", "shield", "open"):
            return self.legacy.prepare(choice["effective_messages"][0]["content"])
        kwargs = {"truncation": False, "padding": False, "return_attention_mask": True}
        if self.name == "shield":
            kwargs["add_special_tokens"] = False
        return normalized(self.tokenizer(self.render(value), **kwargs))

    def index(self, row):
        value = row.get("messages", row.get("text"))
        source = messages(value)
        common = {"id": row["id"], "source_sha256": digest(source)}
        try:
            choice = select(self.name, source)
        except UnsupportedInput as error:
            return {**common, "supported": False, "reason": str(error), "total_tokens": None,
                    "bucket": None, "input_sha256": None}
        data = self.prepare(source)
        count = len(data["input_ids"])
        changed = self.name == "gliner" and self.legacy.input_changed
        supported = 0 < count <= self.spec["limit"] and not changed
        if self.name == "laya" and supported:
            from speed_dataset.preparation import QUESTIONS
            text = choice["effective_messages"][0]["content"]
            native = self.legacy.agent._encode_state(text, list(QUESTIONS), self.legacy.internal)[0]["ids"]
            if native != data["input_ids"]:
                raise ValueError("Laya SDK changes/truncates a supposedly supported input")
        from speed_dataset.core import bucket
        effective = {k: v for k, v in choice.items() if k != "effective_messages"}
        return {**common, **effective, "user_tokens": sum(len(self.tokenizer.encode(m["content"], add_special_tokens=False))
                for m in source if m["role"] == "user"),
                "content_tokens": sum(len(self.tokenizer.encode(m["content"], add_special_tokens=False)) for m in source),
                "total_tokens": count, "bucket": bucket(count), "supported": supported,
                "reason": "processor_changed_text" if changed else None if supported else "outside_input_limit",
                "input_changed": changed, "ids_sha256": digest(data["input_ids"]),
                "mask_sha256": digest(data["attention_mask"]), "input_sha256": digest(data)}

    def parity(self):
        checks = []
        cases = [[{"role": "user", "content": "USER: library hours?"}],
                 [{"role": "assistant", "content": "ASSISTANT: nine oclock."}],
                 [{"role": "user", "content": "USER: library hours?"},
                  {"role": "assistant", "content": "ASSISTANT: nine oclock."}]]
        for source in cases:
            try:
                selected = select(self.name, source)
                actual = self.prepare(source)
                if self.name in ("hive", "shield", "open"):
                    opts = {"tokenize": True, "return_dict": True, "return_attention_mask": True}
                    if self.name == "shield":
                        opts["guideline"] = RESPONSE_GUIDELINE if selected["target_role"] == "assistant" else GUIDELINE
                    if self.name == "open":
                        opts["add_generation_prompt"] = True
                    expected = normalized(self.tokenizer.apply_chat_template(selected["effective_messages"], **opts))
                else:
                    expected = self.legacy.prepare(selected["effective_messages"][0]["content"])
                if actual != expected:
                    raise ValueError("Native role preparation parity failed: " + self.name)
                checks.append({"source_sha256": digest(source), "status": "passed", "input_sha256": digest(actual)})
            except UnsupportedInput:
                checks.append({"source_sha256": digest(source), "status": "not_applicable"})
        return {"status": "passed", "scope": "CPU native preparation; not quality or GPU acceptance", "checks": checks}
