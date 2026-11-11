"""New wrappers around historical implementations; imports are worker-local."""
import inspect
import time
from contextlib import contextmanager
from pathlib import Path

from .core import digest, score_check, write


@contextmanager
def quiet_dispatch(module, implementations):
    originals = {k: getattr(module, k) for k in implementations}
    try:
        for key, fn in implementations.items():
            setattr(module, key, fn)
        yield
    finally:
        for key, fn in originals.items():
            setattr(module, key, fn)


def filtered(fn):
    keys = tuple(inspect.signature(fn).parameters)
    def call(*args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items() if k in keys})
    return call


def normalized(inputs):
    ids = inputs["input_ids"].detach().cpu().tolist()[0]
    mask = inputs["attention_mask"].detach().cpu().tolist()[0]
    if len(ids) != len(mask) or any(m != 1 for m in mask):
        raise ValueError("Unexpected padding/mask")
    return {"input_ids": ids, "attention_mask": mask}


def check_input(adapter, row):
    value = normalized(adapter.prepare([row["text"]]))
    expected = row["index"]
    if (digest(value) != expected["input_sha256"] or
        digest(value["input_ids"]) != expected["ids_sha256"] or
        digest(value["attention_mask"]) != expected["mask_sha256"] or
        len(value["input_ids"]) != expected["total_tokens"] or
        adapter.user_length(row["text"]) != expected["user_tokens"]):
        raise ValueError("Actual preparation differs from published index: " + row["id"])


def boundary_text(adapter, target):
    # Repeat a simple token only for execution-boundary diagnostics, never speed data.
    old = adapter.limit
    adapter.limit = target + 64
    try:
        # Some tokenizers encode "a" with two tokens, leaving gaps in reachable lengths.
        for unit in ("a ", "hello ", "text ", "0 "):
            low, high = 0, target * 2
            while low <= high:
                n = (low + high) // 2
                text = unit * n + "."
                try:
                    length = len(normalized(adapter.prepare([text]))["input_ids"])
                except ValueError as error:
                    if "exceeds" not in str(error):
                        raise
                    # A long system prompt can make the initial midpoint over-limit.
                    high = n - 1
                    continue
                if length == target:
                    return text
                if length < target:
                    low = n + 1
                else:
                    high = n - 1
        raise ValueError(f"Cannot construct exact {target}-token boundary; do not weaken check")
    finally:
        adapter.limit = old


def load(model, variant, candidate, directory):
    import torch
    from experiments.adapters import GlinerAdapter, HaloAdapter
    from followup.team_halo import TeamHaloAdapter, check_dispatch
    if model == "halo":
        class Halo(TeamHaloAdapter):
            limit = 4096

            def _tensor(self, inputs):
                with quiet_dispatch(self.qwen, self.quiet_implementations):
                    return HaloAdapter._tensor(self, inputs)

            def prove_backend(self, text):
                self.calls.clear()
                originals = {k: getattr(self.qwen, k) for k in self.implementations}
                inputs = self.prepare([text])
                expected = TeamHaloAdapter._tensor(self, inputs).detach().clone()
                result = check_dispatch(self.calls, self.spec["fast"])
                actual = self._tensor(inputs)
                if not torch.equal(expected, actual):
                    raise ValueError("Diagnostic/quiet Halo implementations differ")
                if any(getattr(self.qwen, k) is not v for k, v in originals.items()):
                    raise ValueError("Halo global dispatch not restored")
                self.calls.clear()
                return result
        a = Halo({"dtype": "float16", "fast": "linear" if variant == "candidate" else "none",
                  "attention": "eager"}, directory)
        a.quiet_implementations = {k: filtered(fn) for k, fn in a.implementations.items()}
    else:
        a = GlinerAdapter({"dtype": "float32"}, directory)
        original = a.model.processor.collate_fn_inference
        def checked(batch, *args, **kwargs):
            prepared = original(batch, *args, **kwargs)
            if prepared.original_texts != [item[0] for item in batch]:
                raise ValueError("Processor changed text")
            if any(not 0 < n <= a.limit for n in prepared.attention_mask.sum(-1).tolist()):
                raise ValueError("Input exceeds limit; rejected before encoder")
            return prepared
        a.model.processor.collate_fn_inference = checked
    if any(p.device != a.device for p in list(a.model.parameters()) + list(a.model.buffers())):
        raise ValueError("Unexpected CPU/offloaded parameter or buffer")
    a.compile_count = 0
    a.metadata.update(measurement_adapter="u1-v1", limit=a.limit)
    return a


def activate(a, candidate, variant, texts, directory):
    import torch
    directory = Path(directory)
    durations = {}
    if variant == "candidate" and candidate == "compile":
        from torch._dynamo.backends.registry import lookup_backend
        from transformers.modeling_outputs import BaseModelOutput
        if torch._dynamo.config.suppress_errors:
            raise RuntimeError("Dynamo suppress_errors would permit fallback")
        original = a.model.encoder.forward
        backend = lookup_backend("inductor")
        def counted(graph, inputs):
            a.compile_count += 1
            return backend(graph, inputs)
        fn = torch.compile(lambda ids, mask: original(input_ids=ids, attention_mask=mask).last_hidden_state,
                           backend=counted, dynamic=True, fullgraph=True)
        def forward(input_ids=None, attention_mask=None, **kwargs):
            if any(v is not None for v in kwargs.values()):
                raise ValueError("Unexpected encoder arguments")
            return BaseModelOutput(last_hidden_state=fn(input_ids, attention_mask))
        a.model.encoder.forward = forward
        a.restore_encoder = lambda: setattr(a.model.encoder, "forward", original)
        a.spec.update(compile="default", dynamic=True)
        a.metadata["compile"] = {"dynamic": True, "fullgraph": True, "backend": "inductor", "mode": "default"}
    if variant == "candidate" and candidate == "ort":
        from followup import onnx_service
        from followup.team_smoke import preload_ort
        preload_ort(directory)
        old = onnx_service.VALIDATION_TEXTS
        try:
            onnx_service.VALIDATION_TEXTS = texts
            start = time.monotonic()
            onnx_service.install(a, directory)
            torch.cuda.synchronize()
            durations["export_and_session_setup_seconds"] = time.monotonic() - start
            start = time.monotonic()
            onnx_service.finish(a, directory, texts[0])
            torch.cuda.synchronize()
            durations["provider_validation_seconds"] = time.monotonic() - start
        finally:
            onnx_service.VALIDATION_TEXTS = old
        a.metadata["ort_profile_active_during_latency"] = False
    if hasattr(a, "prove_backend"):
        start = time.monotonic()
        a.metadata["backend_evidence"] = a.prove_backend(texts[0])
        torch.cuda.synchronize()
        durations["dispatch_proof_seconds"] = time.monotonic() - start
    # All preparations and backend warmups are outside the measured request series.
    start = time.monotonic()
    for text in texts:
        score_check(a.scores([text])[0])
    if variant == "candidate" and candidate == "compile" and a.compile_count == 0:
        raise RuntimeError("Compiled encoder never executed")
    torch.cuda.synchronize()
    durations["shape_warmup_including_compile_seconds"] = time.monotonic() - start
    a.metadata["preparation_times"] = durations
    a.metadata["prepared_compile_count"] = a.compile_count
    write(directory / "backend.json", a.metadata)


def stable_call(a, text):
    before = a.compile_count
    result = a.scores([text])
    if a.compile_count != before:
        raise RuntimeError("New compilation during measured/validated series")
    if len(result) != 1:
        raise ValueError("Expected one result")
    score_check(result[0])
    return result[0]


def model_size(a, directory):
    tensors = list(a.model.parameters()) + list(a.model.buffers())
    storages = {}
    for t in tensors:
        storage = t.untyped_storage()
        storages[(str(t.device), storage.data_ptr())] = storage.nbytes()
    from huggingface_hub import hf_hub_download
    # Find the loaded snapshot through its config; unused model-card images need not be cached.
    snapshot = Path(hf_hub_download(a.contract["model"], "config.json", revision=a.contract["revision"],
                                   local_files_only=True)).parent
    checkpoint = {str(p.relative_to(snapshot)): p.stat().st_size for p in snapshot.rglob("*")
                  if p.is_file() and p.suffix in (".safetensors", ".bin")}
    if not checkpoint:
        raise ValueError("Loaded checkpoint weight files missing from snapshot")
    exports = {p.name: p.stat().st_size for p in Path(directory).iterdir()
               if p.is_file() and (p.suffix in (".onnx", ".data") or ".onnx." in p.name)}
    return {"unique_parameter_buffer_storage_bytes": sum(storages.values()),
            "dtypes": sorted({str(t.dtype) for t in tensors}), "checkpoint_files_bytes": checkpoint,
            "export_files_bytes": exports, "original_encoder_retained": hasattr(a, "ort_session"),
            "note": "Tensor storage bytes, on-disk artifact bytes and VRAM are distinct"}
