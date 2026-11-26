"""Apply team rules at the existing native adapter's computation boundary."""

import sys
from copy import deepcopy

from .core import ROOT, digest

RULES = ("c7", "c12", "c8", "u4", "u1")


def split_rules(spec):
    native = deepcopy(spec)
    rules = native.get("execution", {}).pop("rules", [])
    if (
        not isinstance(rules, list)
        or any(not isinstance(r, str) or r not in RULES for r in rules)
        or len(set(rules)) != len(rules)
    ):
        raise ValueError("Invalid team rules")
    if rules:
        family = spec.get("adapter")
        allowed = {"c7", "c12"} if family == "d3_decision" else {"c12", "c8", "u4", "u1"}
        if (
            family not in ("hf_sequence_classification", "d3_decision")
            or set(rules) - allowed
            or native["execution"]["runtime"] != "eager"
        ):
            raise ValueError("Rules require a supported native eager adapter")
    return native, rules


def rules_module():
    import importlib

    path = ROOT / "src"
    if not (path / "guardrail_tuner/rules.py").is_file():
        raise ValueError("Team rules require the integrated repository checkout")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module("guardrail_tuner.rules")


class RuleAdapter:
    def __init__(self, native, spec, rules):
        self.native, self.spec, self.requested = native, spec, list(rules)
        self.ruled, self.original = None, None
        self.metadata = native.metadata
        self.metadata["variant_sha256"] = digest(spec)
        self.metadata["team_rules"] = {
            "requested": self.requested,
            "applied": [],
            "skipped": {},
            "already_active": [],
        }

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def compile_count(self):
        return self.native.compile_count + (self.ruled.compile_count if self.ruled else 0)

    def _install(self):
        module = rules_module()
        rules = list(self.requested)
        if "c7" in rules:
            if self.native.spec["execution"]["linear_attention"] not in ("native", "fla"):
                raise ValueError("c7 requires an existing native linear fast path")
            rules.remove("c7")
            self.metadata["team_rules"]["already_active"] = ["c7"]
        model = self.native.model
        self.original = model.forward
        self.ruled = module.RuledModel(
            model,
            rules,
            self.native.tokenizer.pad_token_id,
            forward=lambda values: self.original(**values).logits,
            strict=True,
        )
        if self.spec["adapter"] == "hf_sequence_classification":
            model.forward = self.ruled
        self.metadata["team_rules"]["applied"] = list(self.ruled.applied)

    def prepare_batch_backend(self, batches):
        if not batches:
            raise ValueError("No rule preparation inputs")
        if self.ruled is None:
            self._install()
        self.native.prepare_batch_backend(batches)
        shapes = {}
        for batch in batches:
            key = (len(batch), max(self.index(row)["total_tokens"] for row in batch))
            shapes.setdefault(key, batch)
        for batch in shapes.values():
            self.request_batch(batch)
        if "u4" in self.requested and not self.compile_count:
            raise ValueError("Requested compiled forward did not execute")
        self.metadata["prepared_compile_count"] = self.compile_count
        self.metadata["team_rules"]["prepared_shapes"] = [list(s) for s in shapes]
        return {
            "runtime": self.backend,
            "compile_count": self.compile_count,
            "team_rules": deepcopy(self.metadata["team_rules"]),
        }

    def request_batch(self, payloads):
        if self.ruled is None:
            raise ValueError("Team rules have not been prepared")
        try:
            return self.native.request_batch(payloads)
        finally:
            # Fusion may reuse an intermediate within one forward, never across requests.
            for child in self.native.model.modules():
                if hasattr(child, "sizes") and hasattr(child, "x") and hasattr(child, "out"):
                    child.x = child.out = None

    def request(self, payload):
        return self.request_batch([payload])[0]

    def prove_batch_backend(self, batch):
        proof = self.native.prove_batch_backend(batch)
        evidence = deepcopy(self.metadata["team_rules"])
        evidence["applied"] = list(self.ruled.applied)
        evidence["skipped"] = {}
        if "u1" in self.requested:
            if len(batch) != 1:
                evidence["applied"].remove("u1")
                evidence["skipped"]["u1"] = "batch_size != 1"
            elif not self.ruled.graphs:
                raise ValueError("Requested CUDA graph did not execute")
        if "c7" in self.requested and not proof.get("dispatch"):
            raise ValueError("Native linear fast path lacks dispatch evidence")
        evidence["compile_count"] = self.compile_count
        proof["team_rules"] = evidence
        self.metadata["team_rules"] = evidence
        self.metadata["backend_proof"] = proof
        self.metadata["batch_backend_proof"] = proof
        return proof

    def close(self):
        if self.original is not None and self.native.model is not None:
            self.native.model.forward = self.original
        compiled = self.compile_count
        self.ruled = self.original = None
        self.native.close()
        if compiled:
            import torch

            torch._dynamo.reset()
