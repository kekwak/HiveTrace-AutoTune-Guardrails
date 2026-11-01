"""CPU-only contract fixture, NOT a quantized model or a speed implementation.

Use create(...) and this method surface for a real adapter. Replace every toy
operation with your runtime. Native tokenizer/task/rules must remain unchanged.
The fixture intentionally changes one verdict and is rejected for GPU acceptance.
"""
from u1_benchmark.core import digest


class Fixture:
    def __init__(self, contract, spec, model="gliner"):
        self.contract, self.spec = contract, spec
        self.name = model
        self.limit = contract["limit"]
        self.compile_count = 0
        self.metadata = {"test_fixture": True, "backend_actual": "cpu_fixture_not_quantization"}

    def index(self, text):
        if self.contract.get("input_version") == 3:
            from u1_benchmark.messages import messages, select
            choice = select(self.name, text)
            value = self.index_text(choice["effective_messages"][-1]["content"])
            return {**value, **{k: v for k, v in choice.items() if k != "effective_messages"}}
        return self.index_text(text)

    def index_text(self, text):
        ids = [ord(ch) for ch in text] or [0]
        mask = [1] * len(ids)
        return {"user_tokens": len(text), "total_tokens": len(ids), "ids_sha256": digest(ids),
                "mask_sha256": digest(mask), "input_sha256": digest({"input_ids": ids, "attention_mask": mask})}

    def request(self, text):
        if self.index(text)["total_tokens"] > self.limit:
            raise ValueError("Input exceeds limit")
        if self.contract.get("input_version") == 3:
            from u1_benchmark.messages import select
            text = select(self.name, text)["effective_messages"][-1]["content"]
        changed = self.spec.get("options", {}).get("change_verdict", False) and text.startswith("change")
        return {"label": "unsafe" if changed else "safe"}

    def verify(self, text, expected):
        from u1_benchmark.adapters import check_index
        check_index(self.index(text), expected)
        return self.request(text)

    def prepare_backend(self, texts):
        for text in texts:
            self.request(text)

    def boundary(self):
        return "x" * self.limit, "x" * (self.limit + 1)

    def reject_over_limit(self, text):
        try:
            self.request(text)
        except ValueError:
            return
        raise AssertionError("Limit not enforced")

    def size(self):
        return {"test_fixture": True, "unique_parameter_buffer_storage_bytes": None,
                "checkpoint_files_bytes": {}, "export_files_bytes": {}, "dtypes": []}

    def close(self):
        pass


def create(*, model, spec, directory, contract):
    return Fixture(contract, spec, model)
