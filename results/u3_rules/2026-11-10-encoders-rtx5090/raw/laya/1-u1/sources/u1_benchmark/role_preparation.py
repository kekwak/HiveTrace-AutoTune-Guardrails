"""Current role contracts; preserve the historical native preparation unchanged."""
import inspect

from speed_dataset.preparation import CHECK_TEXTS, Preparer as LegacyPreparer
from .core import digest
from .messages import UnsupportedInput, select
from .native_preparation import Preparer as NativePreparer


class OpenTaskPreparer(NativePreparer):
    """0124 tokenizer/weights with the explicitly pinned 2510 task template."""
    def __init__(self):
        self.name = "open"
        self.legacy = LegacyPreparer(self.name)
        self.spec, self.tokenizer = self.legacy.spec, self.legacy.tokenizer
        self.versions, self.assets = dict(self.legacy.versions), dict(self.legacy.assets)
        self.assets["tokenizer_default_chat_template"] = self.assets["effective_chat_template"]
        self.assets["effective_chat_template"] = digest(self.legacy.template)
        self.assets["task_role_preparation"] = digest(inspect.getsource(type(self)))

    def render(self, value):
        selected = select(self.name, value)
        return self.legacy.render(selected["effective_messages"][0]["content"])

    def parity(self):
        checks = []
        for text in CHECK_TEXTS:
            value = [{"role": "user", "content": text}]
            actual = self.prepare(value)
            if actual != self.legacy.prepare(text):
                raise ValueError("Open task template/tokenization parity failed")
            checks.append({"source_sha256": digest(value), "status": "passed", "input_sha256": digest(actual)})
        for value in ([{"role": "assistant", "content": "answer"}],
                      [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]):
            try:
                self.prepare(value)
            except UnsupportedInput:
                checks.append({"source_sha256": digest(value), "status": "not_applicable"})
            else:
                raise ValueError("Open task contract must remain user-only")
        return {"status": "passed", "scope": "CPU parity with pinned 2510 task template; 0124 checkpoint; not quality", "checks": checks}


def Preparer(model):
    if model == "open":
        return OpenTaskPreparer()
    # Late import keeps the other seven native contracts and their test seams intact.
    from .native_preparation import Preparer as Native
    return Native(model)
