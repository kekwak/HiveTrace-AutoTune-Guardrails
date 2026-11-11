"""Inference contracts pinned to the historical notebooks."""

import ast
import gc
import math
import re
from pathlib import Path

from .common import check_score, digest

HALO_ID = "astroware/HaloGuard1-Gen-0.8B"
HALO_REVISION = "85ff68dd063a7e58487a4251824fd36a20977d5c"
TRANSFORMERS_COMMIT = "394b1a0eaa8e6199e372334da0aff3753a117fdb"
GLINER_ID = "hivetrace/gliner-guard-omni"
GLINER_REVISION = "763f8aa9a7714dd2c3264f3751acdd5d854c5ac4"


class UnsupportedExperiment(RuntimeError):
    pass


def guard_prompt(readme):
    candidates = []
    for block in re.findall(r"```python\s*\n(.*?)```", readme, re.S):
        try:
            nodes = ast.parse(block).body
        except SyntaxError:
            continue
        for node in nodes:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                                                     t.id == "GUARD_SYSTEM_PROMPT" for t in node.targets):
                candidates.append(ast.literal_eval(node.value))
    if len(candidates) != 1 or not isinstance(candidates[0], str) or not candidates[0].strip():
        raise ValueError("Expected one literal author system prompt")
    return candidates[0]


def score_from_logits(logits, threshold, tie_safe=False):
    import torch
    logits = logits.detach().float().reshape(-1)
    if logits.numel() != 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected two finite logits")
    p = torch.softmax(logits, -1).cpu().tolist()
    unsafe = p[1] > threshold if tie_safe else p[1] >= threshold
    result = {"p_safe": p[0], "p_unsafe": p[1],
              "logit_margin": float((logits[1] - logits[0]).cpu()),
              "label": "unsafe" if unsafe else "safe"}
    check_score(result)
    return result


class Adapter:
    def user_length(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def validate_inputs(self, inputs):
        lengths = inputs["attention_mask"].sum(-1).tolist()
        if not lengths or any(not 0 < n <= self.limit for n in lengths):
            raise ValueError(f"Input exceeds {self.limit}; truncation forbidden")
        return inputs

    def input_hash(self, texts):
        prepared = self.prepare(texts)
        return digest({"text": texts[0], "contract": self.contract,
                       "ids": prepared["input_ids"].cpu().tolist(),
                       "mask": prepared["attention_mask"].cpu().tolist()})

    def close(self):
        import torch
        if hasattr(self, "restore_attention"):
            self.restore_attention()
            del self.restore_attention
        if hasattr(self, "restore_encoder"):
            self.restore_encoder()
            del self.restore_encoder
        if getattr(self, "projection_memo", None) is not None:
            self.projection_memo.remove()
            self.projection_memo = None
        if hasattr(self, "ort_session"):
            self.ort_session = None
        if getattr(self, "spec", {}).get("compile"):
            torch._dynamo.reset()
        self.tensor_call = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HaloAdapter(Adapter):
    limit, threshold, boundary = 1200, .65, "forward_gpu_inputs"

    def __init__(self, spec, directory, device="cuda:0"):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

        self.device, self.spec = torch.device(device), spec
        symbols = ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                   "fused_recurrent_gated_delta_rule", "FusedRMSNormGated")
        # Restore upstream bindings before EVERY load; baseline never inherits a prior patch.
        if not hasattr(qwen, "_experiment_originals"):
            qwen._experiment_originals = {name: getattr(qwen, name, None) for name in symbols}
            qwen._experiment_originals["is_fast_path_available"] = qwen.is_fast_path_available
        for name, value in qwen._experiment_originals.items():
            setattr(qwen, name, value)
        fast = spec.get("fast", "none")
        requested = {"linear": {"chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"},
                     "conv": {"causal_conv1d_fn", "causal_conv1d_update"},
                     "norm": {"FusedRMSNormGated"}, "all": set(symbols), "none": set()}[fast]
        for name in requested:
            if getattr(qwen, name) is None:
                raise UnsupportedExperiment(f"Optional kernel unavailable: {name}; install in a separate environment")
        for name in set(symbols) - requested:
            setattr(qwen, name, None)
        qwen.is_fast_path_available = fast == "all"
        card = Path(hf_hub_download(HALO_ID, "README.md", revision=HALO_REVISION)).read_text()
        self.prompt = guard_prompt(card)
        self.tokenizer = AutoTokenizer.from_pretrained(HALO_ID, revision=HALO_REVISION, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("safe", "unsafe")]
        if any(len(x) != 1 for x in self.label_ids) or self.label_ids[0] == self.label_ids[1]:
            raise ValueError("Labels must be distinct single tokens")
        self.label_ids = [x[0] for x in self.label_ids]
        prefix = self.render("Library hours?")
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        for label, token in zip(("safe", "unsafe"), self.label_ids):
            if self.tokenizer.decode(ids + [token], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False) != prefix + label:
                raise ValueError("Label does not continue the official cue")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            HALO_ID, revision=HALO_REVISION, dtype=getattr(torch, spec["dtype"]),
            attn_implementation=spec.get("attention", "eager"), trust_remote_code=False).to(self.device).eval()
        layers = [m for m in self.model.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]
        if not layers:
            raise ValueError("No GatedDeltaNet layers found")
        if fast == "none" and any(m.chunk_gated_delta_rule is not qwen.torch_chunk_gated_delta_rule for m in layers):
            raise ValueError("Reference linear attention changed")
        if fast in ("linear", "all") and any(m.chunk_gated_delta_rule is qwen.torch_chunk_gated_delta_rule for m in layers):
            raise UnsupportedExperiment("Requested fast linear path was not activated")
        self.contract = {"model": HALO_ID, "revision": HALO_REVISION, "prompt": self.prompt,
                         "template": self.tokenizer.chat_template, "cue": "Safety: ",
                         "thinking": False, "threshold": self.threshold, "label_ids": self.label_ids}
        self.metadata = {"contract_hash": digest(self.contract), "dtype": spec["dtype"],
                         "attention": self.model.config.text_config._attn_implementation,
                         "linear": [m.chunk_gated_delta_rule.__module__ + "." + m.chunk_gated_delta_rule.__name__ for m in layers],
                         "norm": sorted({type(m.norm).__name__ for m in layers}), "fast_requested": fast}
        self.tensor_call = self._tensor
        if spec.get("repeat_kv"):
            from .sdpa import install_repeat
            install_repeat(self, qwen.ALL_ATTENTION_FUNCTIONS)

    def render(self, text):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.prompt}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Safety: "

    def prepare(self, texts):
        inputs = self.tokenizer([self.render(t) for t in texts], padding=True, truncation=False,
                                return_tensors="pt", add_special_tokens=False)
        return self.validate_inputs({k: v.to(self.device) for k, v in inputs.items()})

    def _tensor(self, inputs):
        return self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1, :]

    def scores(self, texts):
        logits = self.tensor_call(self.prepare(texts))
        return [score_from_logits(row[self.label_ids], self.threshold) for row in logits]


class GlinerAdapter(Adapter):
    limit, threshold, boundary = 512, .5, "encoder_forward_gpu_inputs"

    def __init__(self, spec, directory, device="cuda:0"):
        import torch
        from gliner2 import GLiNER2
        from huggingface_hub import snapshot_download
        self.device, self.spec = torch.device(device), spec
        snapshot = snapshot_download(GLINER_ID, revision=GLINER_REVISION,
                                     allow_patterns=["README.md", "config.json", "encoder_config/config.json",
                                                     "tokenizer.json", "tokenizer_config.json", "model.safetensors"])
        self.model = GLiNER2.from_pretrained(snapshot, attention_backend="standard", quantize=False,
                                            compile=False, map_location="cpu").to(
                                                device=self.device, dtype=getattr(torch, spec["dtype"])).eval()
        self.model.strict_extraction = True
        self.schema = self.model.create_schema().classification(task="safety", labels=["safe", "unsafe"], multi_label=False)
        self.tokenizer = self.model.processor.tokenizer
        self.contract = {"model": GLINER_ID, "revision": GLINER_REVISION, "schema": self.schema.to_dict(),
                         "threshold": self.threshold, "tie": "safe", "word_cap": None}
        self.metadata = {"contract_hash": digest(self.contract), "dtype": spec["dtype"],
                         "encoder": type(self.model.encoder).__name__,
                         "attention": self.model.encoder.config._attn_implementation}
        self.tensor_call = lambda inputs: self.model.encoder(**inputs).last_hidden_state

    def prepare(self, texts):
        batch = self.model.processor.collate_fn_inference(
            [(text, self.schema.build()) for text in texts], max_len=None, error_policy="raise", architecture="span")
        if batch.original_texts != texts:
            raise ValueError("Processor changed or dropped text")
        return self.validate_inputs({"input_ids": batch.input_ids.to(self.device),
                                     "attention_mask": batch.attention_mask.to(self.device)})

    def scores(self, texts):
        captured = []
        hook = self.model.classifier.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
        try:
            results = self.model.batch_extract(texts, schemas=self.schema, batch_size=len(texts),
                                               include_confidence=True, max_len=None)
        finally:
            hook.remove()
        if len(captured) != len(texts) or len(results) != len(texts):
            raise ValueError("Unexpected number of classifier calls")
        scores = [score_from_logits(row, self.threshold, tie_safe=True) for row in captured]
        for score, raw in zip(scores, results):
            public = raw["safety"]
            if public["label"] != score["label"] or not math.isfinite(public["confidence"]):
                raise ValueError("Public output disagrees with captured classification logits")
            # FP16 public softmax can round scores; record rather than conceal this difference.
            score["public_confidence"] = public["confidence"]
        return scores


def load_adapter(model, spec, directory, device="cuda:0"):
    if model not in ("halo", "gliner"):
        raise ValueError("Unknown model")
    adapter = (HaloAdapter if model == "halo" else GlinerAdapter)(spec, directory, device)
    if any(p.device != adapter.device for p in adapter.model.parameters()):
        raise ValueError("Model contains parameters outside requested device")
    return adapter
