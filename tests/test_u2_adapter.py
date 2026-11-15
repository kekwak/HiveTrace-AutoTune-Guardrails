"""External U1 v4 adapter: spec mapping, quantize -> check -> compile order, Hive probabilities.

NativeAdapter.__init__ (CUDA, real checkpoints) is replaced by a tiny CPU fake; kernels are checked on the pod.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from u1_benchmark.contracts import validate_response
from u1_benchmark.core import digest
from u1_benchmark.v4_adapters import NativeAdapter
from u2_quant import adapter as qa

SOURCE = {"repo": "hivetrace/gliner-guard-omni", "revision": "763f8aa9a7714dd2c3264f3751acdd5d854c5ac4"}


def spec(repo="hivetrace/gliner-guard-omni", method=None):
    value = {"adapter": {"module": "u2_quant/adapter.py", "source_sha256": "0" * 64, "dependencies": {}},
             "weights": dict(SOURCE, repo=repo), "tokenizer": dict(SOURCE, repo=repo), "template": None,
             "preparation": {"limit": 512}, "scoring": {"max_new_tokens": 1} if "Hive" in repo else {},
             "execution": {"runtime": "compile", "dtype": "bfloat16", "attention": "native",
                           "sdpa_backend": None, "linear_attention": "reference"}}
    if method:
        value["quantization"] = {"method": method, "granularity": "per_row", "skip_modules": [],
                                 "baseline_identity": "b" * 64}
    return value


class Tokenizer:
    pad_token_id = 0
    vocab = {"safe": 5, "unsafe": 6}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]]

    def decode(self, ids, skip_special_tokens=True):
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse.get(i, "") for i in ids)


def fake_init(calls):
    def init(self, native, directory):
        calls.append(("native_init", deepcopy(native)))
        self.spec, self.name = native, native["adapter"]
        self.backend, self._backend_installed = native["execution"]["runtime"], True
        self._compile_count, self._prepared = 0, False
        self.metadata = {"variant_sha256": digest(native)}
        self.tokenizer = Tokenizer()
        body = nn.Sequential(nn.Linear(4, 4))
        self.model = SimpleNamespace(encoder=body, model=body)
        self.target = body
    return init


class Mapping(unittest.TestCase):
    def test_native_spec_maps_repo_and_strips_quantization(self):
        for repo, name in (("hivetrace/gliner-guard-omni", "gliner"), ("hivetrace/HiveTraceGuard-Pro", "hive")):
            original = spec(repo, "fp8")
            native = qa.native_spec(original)
            self.assertEqual(native["adapter"], name)
            self.assertEqual(native["execution"]["runtime"], "eager")
            self.assertNotIn("quantization", native)
            for key in ("weights", "tokenizer", "template", "preparation", "scoring"):
                self.assertEqual(native[key], original[key])
            self.assertEqual(native["execution"]["dtype"], "bfloat16")

    def test_unknown_repo_raises(self):
        with self.assertRaises(ValueError):
            qa.native_spec(spec("KoalaAI/Text-Moderation"))


class Construction(unittest.TestCase):
    def build(self, value, check=None):
        calls = []
        apply = lambda module, method, skip: calls.append(("apply", method)) or {"method": method}
        check = check or (lambda self: calls.append(("check",)))
        with patch.object(NativeAdapter, "__init__", fake_init(calls)), \
                patch.object(NativeAdapter, "check_device", check), patch.object(qa.quantize, "apply", apply):
            made = qa.create(value, "/tmp/unused")
        return made, calls

    def test_metadata_records_quantization_and_variant(self):
        value = spec(method="fp8")
        made, _ = self.build(value)
        self.assertEqual(made.metadata["quantization"], {"method": "fp8"})
        self.assertEqual(made.metadata["variant_sha256"], digest(value))

    def test_check_device_called_after_quantization(self):
        _, calls = self.build(spec(method="fp8"))
        self.assertEqual([c[0] for c in calls], ["native_init", "apply", "check"])

        def failing(self):
            raise ValueError("Physical parameter precision differs from requested policy")
        with self.assertRaisesRegex(ValueError, "fp8"):
            self.build(spec(method="fp8"), failing)

    def test_baseline_without_quantization_block_uses_none(self):
        made, calls = self.build(spec())
        self.assertIn(("apply", "none"), calls)


class Compile(unittest.TestCase):
    def make(self, compiles):
        calls = []
        with patch.object(NativeAdapter, "__init__", fake_init(calls)), \
                patch.object(NativeAdapter, "check_device", lambda self: None):
            made = qa.create(spec(), "/tmp/unused")
        made.request_batch = lambda batch: made.target(torch.zeros(len(batch), 4))
        made._shape = lambda batch: (len(batch), 1)

        def fake_compile(fn, backend, **kwargs):
            def run(*args, **kw):
                if compiles:
                    backend(None, None)
                return fn(*args, **kw)
            return run
        return made, fake_compile

    def test_prepare_raises_without_compile(self):
        made, fake = self.make(compiles=False)
        with patch.object(qa.torch, "compile", fake), patch.object(qa, "_inductor", lambda: lambda gm, inputs: gm):
            with self.assertRaisesRegex(ValueError, "compile"):
                made.prepare_batch_backend([["a"], ["b", "c"]])

    def test_prepare_records_compile(self):
        made, fake = self.make(compiles=True)
        with patch.object(qa.torch, "compile", fake), patch.object(qa, "_inductor", lambda: lambda gm, inputs: gm):
            result = made.prepare_batch_backend([["a"], ["b", "c"]])
        self.assertGreater(made.compile_count, 0)
        self.assertEqual(made.metadata["backend"]["runtime"], "compile")
        self.assertEqual(made.metadata["prepared_compile_count"], made.compile_count)
        self.assertEqual(result["runtime"], "compile")


class HiveScores(unittest.TestCase):
    def test_hive_response_has_binary_probabilities(self):
        calls = []
        with patch.object(NativeAdapter, "__init__", fake_init(calls)), \
                patch.object(NativeAdapter, "check_device", lambda self: None):
            made = qa.create(spec("hivetrace/HiveTraceGuard-Pro"), "/tmp/unused")
        logits = torch.full((2, 8), -5.0)
        logits[0, 6], logits[0, 5] = 3.0, 1.0      # unsafe
        logits[1, 5], logits[1, 6] = 2.0, 0.5      # safe
        made._hf_inputs = lambda payloads: {"input_ids": torch.zeros(2, 3, dtype=torch.long)}
        made.model.generation_config = SimpleNamespace(eos_token_id=7)
        made.model.generate = lambda **kw: SimpleNamespace(
            sequences=torch.tensor([[0, 0, 0, 6], [0, 0, 0, 5]]), scores=(logits,))
        answers = made._request(["a", "b"])
        self.assertEqual([a["label"] for a in answers], ["unsafe", "safe"])
        for answer in answers:
            validate_response(answer)
            self.assertEqual(answer["score_type"], "probabilities")
            self.assertAlmostEqual(sum(answer["scores"].values()), 1.0, places=6)
        self.assertGreater(answers[0]["scores"]["unsafe"], 0.5)

    def test_gliner_response_unchanged(self):
        calls = []
        with patch.object(NativeAdapter, "__init__", fake_init(calls)), \
                patch.object(NativeAdapter, "check_device", lambda self: None), \
                patch.object(NativeAdapter, "_request", lambda self, payloads: [{"label": "safe"}]):
            made = qa.create(spec(), "/tmp/unused")
            self.assertEqual(made._request(["a"]), [{"label": "safe"}])


class Review(unittest.TestCase):
    def make(self, value=None):
        calls = []
        with patch.object(NativeAdapter, "__init__", fake_init(calls)), \
                patch.object(NativeAdapter, "check_device", lambda self: None):
            return qa.create(value or spec(), "/tmp/unused")

    def test_size_counts_fp8_inner_storages(self):
        import tempfile
        from pathlib import Path
        from torchao.quantization import Float8Tensor
        made = self.make()
        layer = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
        layer.weight = nn.Parameter(Float8Tensor.from_hp(layer.weight.detach()), requires_grad=False)
        made.model = nn.Sequential(layer)
        with tempfile.TemporaryDirectory() as root:
            made.weight_path, made.directory = Path(root), Path(root)
            (Path(root) / "model.safetensors").write_bytes(b"0" * 8)
            size = made.size()
        self.assertIn("torch.float8_e4m3fn", size["dtypes"])
        self.assertLess(size["unique_parameter_buffer_storage_bytes"], 64 * 64 * 2)
        self.assertGreaterEqual(size["unique_parameter_buffer_storage_bytes"], 64 * 64)

    def test_backend_metadata_carries_quantization_details(self):
        made = self.make()
        made.metadata["quantization"] = {"method": "fp8", "granularity": "per_row", "quantized": ["0", "1"],
                                         "skipped": [], "torchao": "0.17.0"}
        made.request_batch = lambda batch: None
        made._shape = lambda batch: (len(batch), 1)
        made._install_compile = lambda: setattr(made, "_compile_count", 1) or setattr(made, "_backend_installed", True)
        made.prepare_batch_backend([["a"]])
        details = made.metadata["backend"]["quantization"]
        self.assertEqual((details["method"], details["quantized_count"], details["torchao"]), ("fp8", 2, "0.17.0"))
        self.assertEqual(details["quantized"], ["0", "1"])

    def test_install_compile_refuses_silent_fallback(self):
        made = self.make()
        with patch.object(torch._dynamo.config, "suppress_errors", True):
            with self.assertRaisesRegex(ValueError, "fallback"):
                made._install_compile()

    def test_install_compile_fails_on_recompile_limit(self):
        made = self.make()
        with patch.object(qa, "_inductor", lambda: lambda gm, inputs: gm), \
                patch.object(torch._dynamo.config, "fail_on_recompile_limit_hit", False):
            made._install_compile()
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)

    def test_quantize_source_hash_is_enforced(self):
        value = spec(method="none")
        value["quantization"]["quantize_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "quantize.py"):
            self.make(value)


if __name__ == "__main__":
    unittest.main()
