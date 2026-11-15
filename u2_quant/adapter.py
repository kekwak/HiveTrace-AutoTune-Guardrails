"""External U1 v4 adapter: native preparation/scoring, then optional FP8 and torch.compile.

The U1 native adapter loads the pinned checkpoint in eager mode; this module only quantizes the model body
(heads stay bf16), compiles it with an inductor backend that counts compilations, and for HiveTraceGuard
adds safe/unsafe probabilities from the same greedy step.
"""
from copy import deepcopy

import torch

from u1_benchmark.adapters import parse_generated
from u1_benchmark.batch_adapters import generated_ids
from u1_benchmark.core import digest
from u1_benchmark.v4_adapters import NativeAdapter
from u1_benchmark.v4_preparation import create_preparer as native_preparer
from u2_quant import quantize

REPOS = {"hivetrace/gliner-guard-omni": "gliner", "hivetrace/HiveTraceGuard-Pro": "hive"}


def native_spec(spec):
    repo = spec["weights"]["repo"]
    if repo not in REPOS:
        raise ValueError(f"u2_quant adapter supports only {sorted(REPOS)}, not {repo}")
    result = deepcopy(spec)
    result.pop("quantization", None)
    result["adapter"] = REPOS[repo]
    result["execution"] = {**result["execution"], "runtime": "eager"}
    return result


def _inductor():
    from torch._dynamo.backends.registry import lookup_backend
    return lookup_backend("inductor")


class QuantAdapter(NativeAdapter):
    def __init__(self, spec, directory):
        super().__init__(native_spec(spec), directory)
        settings = spec.get("quantization", {})
        method = settings.get("method", "none")
        self.metadata["quantization"] = quantize.apply(self._quant_body(), method, settings.get("skip_modules", []))
        self.metadata["variant_sha256"] = digest(spec)
        try:
            self.check_device()
        except ValueError as error:
            raise ValueError(f"Precision check failed after quantization method={method}: {error}") from error
        self.backend, self._backend_installed = "compile", False

    def _quant_body(self):
        return self.model.encoder if self.name == "gliner" else self.model.model

    def _compile_body(self):
        return self.model.encoder if self.name == "gliner" else self.model

    def _install_compile(self):
        body, inductor = self._compile_body(), _inductor()

        def counted(graph, inputs):
            self._compile_count += 1
            return inductor(graph, inputs)

        original = body.forward
        body.forward = torch.compile(original, backend=counted, dynamic=True, fullgraph=True)
        self._restore_encoder = lambda: setattr(body, "forward", original)
        self._backend_installed = True
        self.metadata["compile"] = {"backend": "inductor", "mode": "default", "dynamic": True, "fullgraph": True,
                                    "target": "encoder" if self.name == "gliner" else "causal_lm"}

    def _shape(self, batch):
        if self.name == "gliner":
            return tuple(self.gliner.prepare([self.legacy_text(t) for t in batch])["input_ids"].shape)
        return tuple(self._hf_inputs(batch)["input_ids"].shape)

    def prepare_batch_backend(self, batches):
        if not batches:
            raise ValueError("No batches supplied for backend preparation")
        if not self._backend_installed:
            self._install_compile()
        shapes = {}
        for batch in batches:
            shapes.setdefault(self._shape(batch), batch)
        for batch in shapes.values():
            self.request_batch(batch)
        if not self.compile_count:
            raise ValueError("Compiled model never executed: torch.compile produced no graph")
        self._prepared = True
        self.metadata.update(prepared_compile_count=self.compile_count, backend_actual="compile")
        self.metadata["backend"] = {"runtime": "compile", "attention": self.metadata.get("attention"),
                                    "compile_count": self.compile_count,
                                    "quantization": self.metadata["quantization"]["method"]}
        return {"compile_count": self.compile_count, "runtime": "compile"}

    def _label_ids(self):
        ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in ("safe", "unsafe")]
        if any(len(i) != 1 for i in ids):
            raise ValueError("HiveTraceGuard verdict words must be single tokens")
        return [i[0] for i in ids]

    def _request(self, payloads):
        if self.name != "hive":
            return super()._request(payloads)
        inputs = self._hf_inputs(payloads)
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.spec["scoring"]["max_new_tokens"],
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     output_scores=True, return_dict_in_generate=True)
        eos = self.model.generation_config.eos_token_id
        pairs = output.scores[0].float()[:, self._label_ids()].softmax(-1).cpu().tolist()
        answers = []
        for row, (safe, unsafe) in zip(output.sequences[:, inputs["input_ids"].shape[1]:].cpu().tolist(), pairs):
            ids = generated_ids(row, eos)
            answer = parse_generated(self.tokenizer.decode(ids, skip_special_tokens=True), "hive", len(ids))
            answers.append({**answer, "scores": {"safe": safe, "unsafe": unsafe}, "score_type": "probabilities"})
        return answers


def create_preparer(spec):
    return native_preparer(native_spec(spec))


def create(spec, directory):
    return QuantAdapter(spec, directory)
