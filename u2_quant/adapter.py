"""External U1 v4 adapter: native preparation/scoring, then optional FP8 and torch.compile.

The U1 native adapter loads the pinned checkpoint in eager mode; this module only quantizes the model body
(heads stay bf16), compiles it with an inductor backend that counts compilations, and for HiveTraceGuard
adds safe/unsafe probabilities from the same greedy step.
"""
from copy import deepcopy
from pathlib import Path

import torch

from u1_benchmark.adapters import parse_generated
from u1_benchmark.batch_adapters import generated_ids
from u1_benchmark.core import digest, sha
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
        expected = settings.get("quantize_sha256")
        if expected is not None and expected != sha(Path(quantize.__file__)):
            self.close()
            raise ValueError("u2_quant/quantize.py differs from the configured quantize_sha256")
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
        if torch._dynamo.config.suppress_errors:
            raise ValueError("torch._dynamo suppress_errors would allow a silent eager fallback")
        # Превышение лимита перекомпиляций не должно тихо уходить в eager во время замера.
        torch._dynamo.config.fail_on_recompile_limit_hit = True
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
        q = self.metadata["quantization"]
        # metadata["backend"] попадает в result.json (safe_metadata У1), metadata["quantization"] — нет.
        self.metadata["backend"] = {"runtime": "compile", "attention": self.metadata.get("attention"),
                                    "compile_count": self.compile_count,
                                    "quantization": {"method": q["method"], "granularity": q.get("granularity"),
                                                     "quantized_count": len(q.get("quantized", [])),
                                                     "quantized": list(q.get("quantized", [])),
                                                     "skipped": list(q.get("skipped", [])), "torchao": q.get("torchao")}}
        return {"compile_count": self.compile_count, "runtime": "compile"}

    def size(self):
        """As NativeAdapter.size, but counts inner storages of tensor subclasses (torchao Float8Tensor)."""
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        def leaves(tensor):
            if is_traceable_wrapper_subclass(tensor):
                for name in tensor.__tensor_flatten__()[0]:
                    yield from leaves(getattr(tensor, name))
            else:
                yield tensor

        tensors = [leaf for t in list(self.model.parameters()) + list(self.model.buffers()) for leaf in leaves(t)]
        storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
        checkpoint = {str(p.relative_to(self.weight_path)): p.stat().st_size for p in self.weight_path.rglob("*")
                      if p.is_file() and p.suffix in (".safetensors", ".bin", ".pt", ".pth")}
        if not checkpoint:
            raise ValueError("Loaded checkpoint weight files missing")
        return {"unique_parameter_buffer_storage_bytes": sum(storages.values()),
                "dtypes": sorted({str(t.dtype) for t in tensors}), "checkpoint_files_bytes": checkpoint,
                "export_files_bytes": {}, "original_encoder_retained": False}

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
