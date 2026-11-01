"""Full-request adapter boundary for built-in and explicitly trusted local code."""
import importlib.util
import importlib.metadata
import re
from contextlib import nullcontext
from pathlib import Path

from .contracts import DTYPES, MODEL_SPECS, model_contract, validate_response
from .core import digest, sha


def normalize(inputs):
    from speed_dataset.preparation import normalized
    return normalized(inputs)


def input_index(inputs, user_tokens):
    value = normalize(inputs)
    n = len(value["input_ids"])
    return {"total_tokens": n, "user_tokens": user_tokens, "input_sha256": digest(value),
            "ids_sha256": digest(value["input_ids"]), "mask_sha256": digest(value["attention_mask"])}


def check_index(actual, expected):
    if any(actual[k] != expected[k] for k in ("total_tokens", "user_tokens", "input_sha256", "ids_sha256", "mask_sha256")):
        raise ValueError("Actual token IDs/mask differ from dataset index")
    for key in ("target_role", "target_index", "context_indices", "source_mode", "source_sha256"):
        if key in expected and actual.get(key) != expected[key]:
            raise ValueError("Actual evaluated message/context differs from dataset index")


def parse_generated(text, model, count):
    if model == "hive":
        if text.strip() not in ("safe", "unsafe"):
            raise ValueError("Unexpected Hive verdict (not mapped to safe)")
        return {"label": text.strip(), "generated_tokens": count}
    match = re.fullmatch(r"(safe|unsafe)(?:\s+(S\d+(?:\s*,\s*S\d+)*))?", text.strip())
    if not match or (match[1] == "safe" and match[2]):
        raise ValueError("Unexpected OpenGuardrails verdict/category")
    return {"label": match[1], "category": match[2], "generated_tokens": count}


class Builtin:
    def __init__(self, model, spec, directory, input_version=2):
        import torch
        self.name, self.spec, self.directory = model, spec, Path(directory)
        self.input_version = input_version
        self.contract = model_contract(model, input_version)
        self.limit, self.backend = self.contract["limit"], spec.get("backend", "eager")
        self.legacy = None
        self._restore_laya_encoding = None
        self.device = torch.device("cuda:0")
        self.forward_calls = 0
        self.metadata = {"model": model, "backend_requested": self.backend, "dtype": DTYPES[model], "input_version": input_version}
        if model in ("halo", "gliner"):
            from .models import load
            self.legacy = load(model, "baseline" if self.backend == "eager" else "candidate", self.backend, directory)
            self.model, self.tokenizer = self.legacy.model, self.legacy.tokenizer
            if input_version == 3:
                from .role_preparation import Preparer
                self.role_preparer = Preparer(model)
        else:
            if input_version == 3:
                from .role_preparation import Preparer
            else:
                from speed_dataset.preparation import Preparer
            self.preparer = Preparer(model)
            self.metadata["preparation_parity"] = self.preparer.parity()
            self.tokenizer = self.preparer.tokenizer
            spec0 = MODEL_SPECS[model]
            if model == "laya":
                import laya
                from huggingface_hub import snapshot_download
                local = snapshot_download(spec0["repo"], revision=spec0["revision"],
                    allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
                self.agent = laya.load(local, device="cuda", fast=False)
                self.model = self.agent.model
                if self.agent.dtype != torch.float16 or not self.agent.amp_enabled:
                    raise ValueError("Laya native FP16 autocast changed")
            else:
                from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
                loader = AutoModelForSequenceClassification if model in ("koala", "llama") else AutoModelForCausalLM
                options = {"attn_implementation": "sdpa" if model in ("hive", "open") else "eager"}
                self.model = loader.from_pretrained(spec0["repo"], revision=spec0["revision"],
                    torch_dtype=getattr(torch, DTYPES[model]), trust_remote_code=False, **options).to(self.device).eval()
                if model in ("koala", "llama"):
                    self.labels = [str(self.model.config.id2label[i]) for i in range(self.model.config.num_labels)]
                    if len(set(self.labels)) != (9 if model == "koala" else 2):
                        raise ValueError("Native label cardinality changed")
                    self.metadata["labels"] = self.labels
                if model == "shield":
                    token_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")]
                    if any(len(ids) != 1 for ids in token_ids):
                        raise ValueError("Shield Yes/No are not single tokens")
                    self.label_ids = [ids[0] for ids in token_ids]
        self.target = self.model.encoder if model == "gliner" else self.model
        self._observing = False
        self._observed = []
        # Hook enabled only for diagnostics; removed before timing.
        self._hook = None
        self.check_device()
        if self.name == "laya":
            self._install_laya_input_guard()

    def _install_laya_input_guard(self):
        if self._restore_laya_encoding is not None:
            raise RuntimeError("Laya input guard already installed")
        if self.agent.cfg.get("max_len") != self.limit:
            raise ValueError("Laya checkpoint input budget changed")
        original = self.agent._encode_state
        agent = self.agent
        had_override = "_encode_state" in vars(agent)

        def encode(state, ids, internal, max_len=None, head_max_len=None):
            if not isinstance(state, str) or max_len not in (None, self.limit):
                raise ValueError("Unexpected Laya state or input budget override")
            # Observe the full SDK sequence once; never truncate and then accept it.
            items = original(state, ids, internal,
                             max_len=max(self.limit, len(state) * 8 + 4096),
                             head_max_len=head_max_len)
            if not items or any(not 0 < len(item["ids"]) <= self.limit for item in items):
                raise ValueError("Input exceeds limit before forward")
            return items

        def restore():
            if had_override:
                agent._encode_state = original
            else:
                del agent._encode_state

        agent._encode_state = encode
        self._restore_laya_encoding = restore
        self.metadata["input_guard"] = "single SDK encoding; full length checked before forward"

    def check_device(self):
        import torch
        if any(t.device != self.device for t in list(self.model.parameters()) + list(self.model.buffers())):
            raise ValueError("Unexpected CPU/offloaded tensor")
        if self.legacy is None and any(t.is_floating_point() and t.dtype != getattr(torch, DTYPES[self.name]) for t in self.model.parameters()):
            raise ValueError("Native baseline parameter dtype changed")
        if self.name == "laya" and self.agent.device.type != "cuda":
            raise ValueError("Laya SDK CPU fallback")

    @property
    def compile_count(self):
        return getattr(self.legacy, "compile_count", 0)

    def prepare(self, text, enforce=True):
        import torch
        if self.legacy is not None:
            text = self.legacy_text(text)
            old = self.legacy.limit
            if not enforce:
                self.legacy.limit = max(1000000, len(text) * 8 + 4096)
            try:
                return self.legacy.prepare([text])
            finally:
                self.legacy.limit = old
        raw = self.preparer.prepare(text)
        if self.name == "gliner" and self.preparer.input_changed:
            raise ValueError("Preparation changed text")
        if enforce and not 0 < len(raw["input_ids"]) <= self.limit:
            raise ValueError("Input exceeds limit before forward")
        return {key: torch.tensor([value], device=self.device) for key, value in raw.items()}

    def index(self, text):
        if self.input_version == 3:
            from .messages import UnsupportedInput, messages, select
            p = self.role_preparer if self.legacy is not None else self.preparer
            value = p.index({"id": "index", "messages": messages(text)})
            if value.get("total_tokens") is None:
                raise UnsupportedInput(value["reason"])
            # Also execute actual adapter preparation; never trust a parallel tokenizer path alone.
            actual = input_index(self.prepare(text, enforce=False), value["user_tokens"])
            actual.update({k: v for k, v in select(self.name, text).items() if k != "effective_messages"})
            check_index(actual, value)
            return {k: v for k, v in value.items() if k != "id"}
        n = len(self.tokenizer.encode(text, add_special_tokens=False))
        return input_index(self.prepare(text, enforce=False), n)

    def request(self, text):
        import torch
        if self.legacy is not None:
            value = self.legacy.scores([self.legacy_text(text)])[0]
            return {**value, "scores": {"safe": value["p_safe"], "unsafe": value["p_unsafe"]},
                    "score_type": "probabilities"}
        if self.name == "laya":
            from speed_dataset.preparation import QUESTIONS
            answer = self.agent.predict(self.legacy_text(text), QUESTIONS)["answers"]["danger"]
            self.check_device()
            if answer["choice"] not in ("A", "B"):
                raise ValueError("Invalid Laya native choice")
            return {"label": answer["choice"], "scores": answer["probabilities"], "score_type": "probabilities"}
        inputs = self.prepare(text)
        context = nullcontext()
        if self.name in ("hive", "open"):
            from torch.nn.attention import sdpa_kernel, SDPBackend
            context = sdpa_kernel(SDPBackend.MATH)
        with context:
            if self.name in ("koala", "llama"):
                logits = self.model(**inputs).logits[0].float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite classifier logits")
                probs = logits.softmax(-1).cpu().tolist()
                return {"label": self.labels[probs.index(max(probs))], "scores": dict(zip(self.labels, probs)), "score_type": "probabilities"}
            if self.name == "shield":
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1, self.label_ids].float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite Shield logits")
                probs = logits.softmax(-1).cpu().tolist()
                return {"label": "Yes" if probs[0] > probs[1] else "No",
                        "scores": dict(zip(("Yes", "No"), probs)), "score_type": "probabilities"}
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=1 if self.name == "hive" else 10)
            ids = output[0, inputs["input_ids"].shape[-1]:].cpu().tolist()
            return parse_generated(self.tokenizer.decode(ids, skip_special_tokens=True), self.name, len(ids))

    def verify(self, text, expected):
        check_index(self.index(text), expected)
        observed = []
        def hook(module, args, kwargs):
            self.forward_calls += 1
            if not observed:
                ids = kwargs.get("input_ids", args[0] if args else None)
                mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
                if mask is None and ids is not None and self.name in ("hive", "open"):
                    # generate may elide an all-ones prefill mask. Batch=1, no padding;
                    # compare its effective mask, while still verifying every token ID.
                    import torch
                    mask = torch.ones_like(ids)
                    self.metadata["generation_implicit_all_ones_mask"] = True
                observed.append(digest(normalize({"input_ids": ids, "attention_mask": mask})))
        handle = self.target.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            result = self.request(text)
        finally:
            handle.remove()
        if observed != [expected["input_sha256"]]:
            raise ValueError("Full request does not use indexed input")
        self.check_device()
        return validate_response(result)

    def prepare_backend(self, texts):
        if self.legacy is not None:
            from .models import activate
            activate(self.legacy, self.backend, "baseline" if self.backend == "eager" else "candidate",
                     [self.legacy_text(t) for t in texts], self.directory)
            self.metadata.update(self.legacy.metadata)
        else:
            for text in texts:
                validate_response(self.request(text))
            self.metadata.update(backend_actual="native_eager", attention=getattr(self.model.config, "_attn_implementation", None) if hasattr(self.model, "config") else "sdk")

    def prove_backend(self, text):
        import gzip
        import json
        import torch
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
            validate_response(self.request(text))
            torch.cuda.synchronize()
        trace = self.directory / "backend-proof.trace.json"
        p.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = sorted({e["name"] for e in events if e.get("cat") == "kernel"})
        operations = sorted({e.key for e in p.key_averages()})
        if not kernels:
            raise ValueError("No CUDA kernels in backend proof")
        if self.name in ("hive", "open") and "aten::_scaled_dot_product_attention_math" not in operations:
            raise ValueError("Requested native SDPA math was not observed")
        if self.name == "laya" and "aten::_scaled_dot_product_efficient_attention" not in operations:
            raise ValueError("Native Laya efficient SDPA was not observed")
        with gzip.open(str(trace) + ".gz", "wb") as output:
            output.write(trace.read_bytes())
        trace.unlink()
        self.metadata["backend_proof"] = {"kernels": kernels, "operators": operations,
                                          "note": "Separate from latency; fused kernels not attributed by substring"}

    def boundary(self):
        for target in (self.limit, self.limit + 1):
            found = None
            for unit in ("a ", "hello ", "text ", "0 "):
                lo, hi = 0, target * 2
                while lo <= hi:
                    mid = (lo + hi) // 2
                    text = unit * mid + "."
                    n = self.index(text)["total_tokens"]
                    if n == target:
                        found = text
                        break
                    if n < target:
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if found is not None:
                    break
            if found is None:
                raise ValueError("Could not construct exact boundary")
            yield found

    def reject_over_limit(self, text):
        called = []
        hook = self.target.register_forward_pre_hook(lambda *args: called.append(1))
        try:
            try:
                self.request(text)
            except ValueError as e:
                if "limit" not in str(e) and "exceeds" not in str(e):
                    raise
            else:
                raise ValueError("Over-limit request accepted")
        finally:
            hook.remove()
        if called:
            raise ValueError("Over-limit request reached model")

    def size(self):
        if self.legacy is not None:
            from .models import model_size
            return model_size(self.legacy, self.directory)
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
        from huggingface_hub import try_to_load_from_cache
        spec = MODEL_SPECS[self.name]
        # Loading weights need not fetch unrelated exports from the same repository.
        anchor = try_to_load_from_cache(spec["repo"],
            "rl_agent_config.json" if self.name == "laya" else "config.json", revision=spec["revision"])
        if not isinstance(anchor, str) or not Path(anchor).is_file():
            raise ValueError("Loaded checkpoint cache anchor is unavailable")
        snapshot = Path(anchor).parent
        files = {str(p.relative_to(snapshot)): p.stat().st_size for p in snapshot.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".bin")}
        return {"unique_parameter_buffer_storage_bytes": sum(storages.values()), "checkpoint_files_bytes": files,
                "dtypes": sorted({str(t.dtype) for t in tensors}), "export_files_bytes": {}}

    def close(self):
        restore = getattr(self, "_restore_laya_encoding", None)
        if restore is not None:
            restore()
            self._restore_laya_encoding = None
        if self.legacy is not None:
            self.legacy.close()

    def legacy_text(self, value):
        if self.input_version == 3:
            from .messages import select
            return select(self.name, value)["effective_messages"][0]["content"]
        return value


def load(model, spec, directory, input_version=2):
    if "module" not in spec:
        return Builtin(model, spec, directory, input_version)
    if sha(spec["module"]) != spec["source_sha256"]:
        raise ValueError("External module changed")
    for package, expected in spec["dependencies"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("External adapter dependency mismatch: " + package)
    definition = importlib.util.spec_from_file_location("u1_external_" + spec["source_sha256"], spec["module"])
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    adapter = module.create(model=model, spec=spec, directory=Path(directory), contract=model_contract(model, input_version))
    for method in ("index", "request", "verify", "prepare_backend", "boundary", "reject_over_limit", "size", "close"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError("External adapter missing " + method)
    if adapter.contract != model_contract(model, input_version):
        raise ValueError("External adapter changed checkpoint/tokenizer/task contract")
    return adapter
