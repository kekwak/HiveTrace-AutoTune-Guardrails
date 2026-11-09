"""Native batch boundary; diagnostics never instrument the timed request path."""
from contextlib import nullcontext
from copy import deepcopy
import gzip
import json

from .adapters import Builtin, check_index, input_index, parse_generated
from .contracts import validate_response


class UnsupportedBatch(RuntimeError):
    """The requested batch implementation cannot satisfy the measurement contract."""


def generated_ids(ids, eos_token_id):
    """Include the first generated EOS, never the padding that follows it."""
    stops = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    for i, token in enumerate(ids):
        if token in stops:
            return ids[:i + 1]
    return ids


def _payloads(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Expected a nonempty list of batch payloads")
    return list(values)


def _responses(values, count):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Batch response cardinality changed")
    return [validate_response(value) for value in values]


def validate_forward_rows(ids, mask, expected, pad_id, side):
    """Check the actual prefill/encoder tensors, preserving order and masked IDs."""
    import torch
    if ids is None or mask is None or ids.ndim != 2 or ids.shape != mask.shape:
        raise ValueError("Expected explicit two-dimensional input IDs and attention mask")
    if ids.shape[0] != len(expected) or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Unexpected batch size or nonbinary input mask")
    lengths = []
    for row_ids, row_mask, index in zip(ids, mask, expected):
        keep = row_mask.bool()
        n = int(keep.sum())
        if not n:
            raise ValueError("Empty effective batch input")
        canonical = torch.arange(len(row_ids), device=row_ids.device)
        contiguous = canonical >= len(row_ids) - n if side == "left" else canonical < n
        if not torch.equal(keep, contiguous):
            raise ValueError("Padding side or contiguity changed")
        if bool((~keep).any()) and (pad_id is None or not bool((row_ids[~keep] == pad_id).all())):
            raise ValueError("Unexpected padded token IDs")
        actual = input_index({"input_ids": row_ids[keep][None],
                              "attention_mask": row_mask[keep][None]}, index["user_tokens"])
        check_index(actual, {k: index[k] for k in actual})
        lengths.append(n)
    if ids.shape[1] != max(lengths):
        raise ValueError("Padding is not limited to the longest input")
    return {"shape": list(ids.shape), "lengths": lengths,
            "useful_tokens": sum(lengths), "padded_tokens": ids.numel(),
            "padding_tokens": ids.numel() - sum(lengths)}


class BatchAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self._builtin = isinstance(adapter, Builtin)
        self._prepared = False
        self._backend_installed = not (self._builtin and adapter.name == "gliner" and
                                       adapter.backend in ("compile", "ort"))
        self._restore_padding = None
        if self._builtin:
            side = "left" if adapter.name in ("halo", "hive", "open", "shield") else "right"
            self.batch_policy = {"padding": "longest", "padding_side": side,
                                 "truncation": False, "sort_by_length": False,
                                 "implementation": "native", "generation_eos_in_count": True}
            if adapter.name in ("hive", "open", "shield"):
                tok = adapter.tokenizer
                original = (tok.padding_side, tok.pad_token)
                self._restore_padding = lambda: self._reset_padding(tok, original)
                tok.padding_side = "left"
                if tok.pad_token_id is None:
                    if tok.eos_token_id is None:
                        raise UnsupportedBatch("Decoder tokenizer has neither PAD nor EOS")
                    tok.pad_token = tok.eos_token
            tokenizer = adapter.agent.tok if adapter.name == "laya" else adapter.tokenizer
            pad_id = tokenizer.pad_token_id
            if type(pad_id) is not int or pad_id < 0:
                raise UnsupportedBatch("Native batch tokenizer must provide an integer pad_token_id")
            self.batch_policy["pad_token_id"] = pad_id
        else:
            policy = getattr(adapter, "batch_policy", adapter.metadata.get("batch_policy"))
            self._external_policy = (isinstance(policy, dict) and policy.get("padding") == "longest"
                                     and policy.get("padding_side") in ("left", "right")
                                     and type(policy.get("pad_token_id")) is int
                                     and policy["pad_token_id"] >= 0
                                     and policy.get("truncation") is False
                                     and policy.get("sort_by_length") is False)
            self.batch_policy = dict(policy) if isinstance(policy, dict) else {
                "padding": "none", "implementation": "single_only", "truncation": False}
        self.metadata["batch_policy"] = self.batch_policy

    @staticmethod
    def _reset_padding(tokenizer, original):
        tokenizer.padding_side, tokenizer.pad_token = original

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def _external(self, method, payloads, *args):
        fn = getattr(self.adapter, method, None)
        if callable(fn):
            if not self._external_policy:
                raise UnsupportedBatch("External batch adapter must declare longest padding, side, pad_token_id, no truncation or sorting")
            return fn(payloads, *args)
        if len(payloads) != 1:
            raise UnsupportedBatch("External adapter missing " + method)
        if method == "request_batch":
            return [self.adapter.request(payloads[0])]
        if method == "verify_batch":
            return [self.adapter.verify(payloads[0], args[0][0])]
        if method == "reject_batch_over_limit":
            return self.adapter.reject_over_limit(payloads[0])
        if method == "prove_batch_backend":
            proof = getattr(self.adapter, "prove_backend", None)
            if not callable(proof):
                raise UnsupportedBatch("External adapter must provide backend proof")
            return proof(payloads[0])
        raise UnsupportedBatch("External adapter missing " + method)

    def _hf_inputs(self, payloads):
        import torch
        rows = [self.adapter.preparer.prepare(payload) for payload in payloads]
        if any(not 0 < len(row["input_ids"]) <= self.limit for row in rows):
            raise ValueError("Input exceeds limit before forward")
        if self.tokenizer.padding_side != self.batch_policy["padding_side"]:
            raise ValueError("Tokenizer padding policy changed")
        inputs = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    def request_batch(self, payloads):
        payloads = _payloads(payloads)
        if not self._builtin:
            return _responses(self._external("request_batch", payloads), len(payloads))
        import torch
        with torch.inference_mode():
            try:
                return _responses(self._request(payloads), len(payloads))
            finally:
                self.check_device()

    def _request(self, payloads):
        import torch
        a = self.adapter
        if not self._backend_installed:
            raise UnsupportedBatch("Batch backend has not been prepared")
        tokenizer = a.agent.tok if a.name == "laya" else a.tokenizer
        if tokenizer.pad_token_id != self.batch_policy["pad_token_id"]:
            raise ValueError("Native batch pad_token_id changed")
        if a.legacy is not None:
            values = a.legacy.scores([a.legacy_text(t) for t in payloads])
            return [{**v, "scores": {"safe": v["p_safe"], "unsafe": v["p_unsafe"]},
                     "score_type": "probabilities"} for v in values]
        if a.name == "laya":
            from speed_dataset.preparation import QUESTIONS
            values = a.agent.predict_batch([a.legacy_text(t) for t in payloads], QUESTIONS,
                                           batch_size=len(payloads), sort_by_length=False)
            answers = [v["answers"]["danger"] for v in values]
            if any(v["choice"] not in ("A", "B") for v in answers):
                raise ValueError("Invalid Laya native choice")
            return [{"label": v["choice"], "scores": v["probabilities"],
                     "score_type": "probabilities"} for v in answers]
        inputs = self._hf_inputs(payloads)
        context = nullcontext()
        if a.name in ("hive", "open"):
            from torch.nn.attention import sdpa_kernel, SDPBackend
            context = sdpa_kernel(SDPBackend.MATH)
        with context:
            if a.name in ("koala", "llama", "shield"):
                if a.name == "shield":
                    logits = a.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, a.label_ids].float()
                    labels = ["Yes", "No"]
                else:
                    logits = a.model(**inputs).logits.float()
                    labels = a.labels
                if logits.shape != (len(payloads), len(labels)) or not bool(torch.isfinite(logits).all()):
                    raise ValueError("Unexpected or nonfinite batch logits")
                probs = logits.softmax(-1).cpu().tolist()
                return [{"label": ("Yes" if p[0] > p[1] else "No") if a.name == "shield"
                         else labels[p.index(max(p))], "scores": dict(zip(labels, p)),
                         "score_type": "probabilities"} for p in probs]
            output = a.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=1 if a.name == "hive" else 10)
            eos = a.model.generation_config.eos_token_id
            answers = []
            for row in output[:, inputs["input_ids"].shape[1]:].cpu().tolist():
                ids = generated_ids(row, eos)
                answers.append(parse_generated(a.tokenizer.decode(ids, skip_special_tokens=True), a.name, len(ids)))
            return answers

    def verify_batch(self, payloads, expected_indices):
        payloads = _payloads(payloads)
        if len(payloads) != len(expected_indices):
            raise ValueError("Batch index cardinality changed")
        if not self._builtin:
            return _responses(self._external("verify_batch", payloads, expected_indices), len(payloads))
        for payload, expected in zip(payloads, expected_indices):
            check_index(self.index(payload), expected)
        observed = []

        def hook(module, args, kwargs):
            if observed and self.name in ("hive", "open"):
                return  # Subsequent autoregressive decode steps are not input prefill.
            ids = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            if mask is None and self.name in ("hive", "open"):
                import torch
                if ids is not None and all(e["total_tokens"] == ids.shape[-1] for e in expected_indices):
                    mask = torch.ones_like(ids)
            pad_id = self.agent.tok.pad_token_id if self.name == "laya" else self.tokenizer.pad_token_id
            observed.append(validate_forward_rows(ids, mask, expected_indices, pad_id,
                                                   self.batch_policy["padding_side"]))

        handle = self.target.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            result = self.request_batch(payloads)
        finally:
            handle.remove()
        if len(observed) != 1:
            raise ValueError("Expected exactly one native batch prefill/encoder call")
        self.metadata["batch_verification"] = observed[0]
        return result

    def prepare_batch_backend(self, batches):
        batches = [_payloads(b) for b in batches]
        if not batches:
            raise ValueError("No batches supplied for backend preparation")
        if not self._builtin:
            fn = getattr(self.adapter, "prepare_batch_backend", None)
            if callable(fn):
                if not self._external_policy:
                    raise UnsupportedBatch("External batch adapter must declare batch_policy")
                return fn(batches)
            if any(len(b) != 1 for b in batches):
                raise UnsupportedBatch("External adapter missing prepare_batch_backend")
            return self.adapter.prepare_backend([b[0] for b in batches])
        if self.name == "gliner" and self.backend in ("compile", "ort"):
            unique = {}
            for batch in batches:
                inputs = self.legacy.prepare([self.adapter.legacy_text(t) for t in batch])
                unique.setdefault(tuple(inputs["input_ids"].shape), batch)
            del inputs
            batches = list(unique.values())
        if self.name == "gliner" and self.backend == "ort":
            from .batch_ort import prepare
            prepare(self, batches)
        elif self.name == "gliner" and self.backend == "compile":
            if not self._prepared:
                self._install_compile()
            for batch in batches:
                self.request_batch(batch)
            if not self.compile_count:
                raise UnsupportedBatch("Compiled batch encoder never executed")
        self._prepared = True
        self.metadata["prepared_compile_count"] = self.compile_count
        self.metadata["batch_policy"] = self.batch_policy
        self.metadata["backend_actual"] = self.backend

    def _install_compile(self):
        import torch
        from torch._dynamo.backends.registry import lookup_backend
        from transformers.modeling_outputs import BaseModelOutput
        a = self.legacy
        if torch._dynamo.config.suppress_errors or hasattr(a, "restore_encoder"):
            raise UnsupportedBatch("Compile requires an unpatched encoder and no fallback")
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
        self._backend_installed = True
        a.metadata["compile"] = {"dynamic": True, "fullgraph": True, "backend": "inductor", "mode": "default"}
        self.metadata["compile"] = a.metadata["compile"]

    def prove_batch_backend(self, payloads):
        payloads = _payloads(payloads)
        if not self._builtin:
            result = self._external("prove_batch_backend", payloads)
            if isinstance(result, dict) and "kernels" in result:
                proof = result
            else:
                source = result if isinstance(result, dict) and any(
                    key in result for key in ("batch_backend_proof", "backend_proof")) else self.metadata
                proof = source.get("batch_backend_proof", source.get("backend_proof"))
                if proof is None and isinstance(result, dict):
                    proof = result
            if not isinstance(proof, dict):
                raise UnsupportedBatch("External adapter did not provide a backend proof dictionary")
            kernels = proof.get("kernels")
            if not self.metadata.get("test_fixture") and not (
                    isinstance(kernels, list) and kernels and all(isinstance(k, str) and k for k in kernels)):
                raise UnsupportedBatch("External batch backend proof has no CUDA kernels")
            proof = deepcopy(proof)
            self.metadata["batch_backend_proof"] = proof
            return proof
        import torch
        self.verify_batch(payloads, [self.index(t) for t in payloads])
        if self.name == "halo":
            from followup.team_halo import TeamHaloAdapter, check_dispatch
            a = self.legacy
            inputs = a.prepare([self.adapter.legacy_text(t) for t in payloads])
            originals = {k: getattr(a.qwen, k) for k in a.implementations}
            a.calls.clear()
            with torch.inference_mode():
                expected = TeamHaloAdapter._tensor(a, inputs).detach().clone()
                evidence = check_dispatch(a.calls, a.spec["fast"])
                actual = a._tensor(inputs)
            a.calls.clear()
            if not torch.equal(expected, actual) or any(getattr(a.qwen, k) is not v for k, v in originals.items()):
                raise ValueError("Halo batch diagnostic/quiet dispatch differs or was not restored")
            self.metadata["batch_dispatch"] = evidence
        if self.name == "gliner" and self.backend == "ort":
            if not getattr(self.legacy, "ort_profile_checked", False):
                raise UnsupportedBatch("ORT shapes have not been validated")
        if self.name == "gliner" and self.backend == "compile" and not self.compile_count:
            raise UnsupportedBatch("Compiled batch encoder never executed")
        before = self.compile_count
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as p:
            self.request_batch(payloads)
            torch.cuda.synchronize()
        if self.compile_count != before:
            raise UnsupportedBatch("New compilation during backend proof")
        self.directory.mkdir(parents=True, exist_ok=True)
        trace = self.directory / "batch-backend-proof.trace.json"
        p.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = sorted({e["name"] for e in events if e.get("cat") == "kernel"})
        operators = sorted({e.key for e in p.key_averages()})
        if not kernels:
            raise UnsupportedBatch("No CUDA kernels in batch backend proof")
        required = "aten::_scaled_dot_product_attention_math" if self.name in ("hive", "open") else (
            "aten::_scaled_dot_product_efficient_attention" if self.name == "laya" else None)
        if required and required not in operators:
            raise UnsupportedBatch("Requested native attention was not observed")
        with gzip.open(str(trace) + ".gz", "wb") as output:
            output.write(trace.read_bytes())
        trace.unlink()
        proof = {"kernels": kernels, "operators": operators, "trace": trace.name + ".gz",
                 "input": self.metadata["batch_verification"]}
        self.metadata["batch_backend_proof"] = proof
        return proof

    def reject_batch_over_limit(self, payloads):
        payloads = _payloads(payloads)
        if not self._builtin:
            return self._external("reject_batch_over_limit", payloads)
        called = []
        hook = self.target.register_forward_pre_hook(lambda *args: called.append(True))
        try:
            try:
                self.request_batch(payloads)
            except ValueError as error:
                if "limit" not in str(error) and "exceeds" not in str(error):
                    raise
            else:
                raise ValueError("Over-limit batch accepted")
        finally:
            hook.remove()
        if called:
            raise ValueError("Over-limit batch reached model")

    def close(self):
        try:
            self.adapter.close()
        finally:
            if self._restore_padding is not None:
                self._restore_padding()
                self._restore_padding = None


def wrap(adapter):
    return adapter if isinstance(adapter, BatchAdapter) else BatchAdapter(adapter)
