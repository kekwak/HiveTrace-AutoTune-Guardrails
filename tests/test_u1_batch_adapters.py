"""Native SDK/tiny CPU tests, not a claim of CUDA batch compatibility."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from u1_benchmark.adapters import Builtin, input_index
from u1_benchmark.batch_adapters import UnsupportedBatch, generated_ids, validate_forward_rows, wrap
from u1_fixtures import tiny_gliner_adapter, tiny_laya_agent, tiny_tokenizer


def builtin(name, model, tokenizer):
    a = Builtin.__new__(Builtin)
    a.name, a.model, a.tokenizer = name, model, tokenizer
    a.target, a.device = model, torch.device("cpu")
    a.limit = 512 if name in ("laya", "gliner", "koala", "llama") else 4096
    a.backend, a.legacy, a.input_version = "eager", None, 2
    a.metadata, a._restore_laya_encoding = {}, None
    a.check_device = Mock()  # Random CPU fixture, never production acceptance.
    a.preparer = SimpleNamespace(prepare=lambda text: tokenizer(text, add_special_tokens=False,
                                                               return_token_type_ids=False))
    a.index = lambda text: input_index(a.prepare(text, enforce=False), len(tokenizer.encode(text)))
    return a


class ToyModel(torch.nn.Module):
    def __init__(self, classes=2, causal=False):
        super().__init__()
        self.classes, self.causal = classes, causal
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        if self.causal:
            logits = torch.arange(self.classes).float().expand(input_ids.shape[0], 1, -1)
        else:
            logits = torch.arange(self.classes).float().expand(input_ids.shape[0], -1)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self(input_ids=input_ids, attention_mask=attention_mask)
        tails = torch.tensor([[5, 3, 0, 0], [6, 3, 0, 0]])[:len(input_ids)]
        return torch.cat([input_ids, tails], 1)


class HFBatches(unittest.TestCase):
    def test_classifier_native_labels_order_and_forward(self):
        for name, count in (("koala", 9), ("llama", 2)):
            with self.subTest(name=name):
                a = builtin(name, ToyModel(count), tiny_tokenizer())
                a.labels = [str(i) for i in range(count)]
                b = wrap(a)
                texts = ["library", "The public library"]
                results = b.verify_batch(texts, [a.index(t) for t in texts])
                self.assertEqual(len(a.model.calls), 1)
                self.assertEqual(results[0]["label"], str(count - 1))
                self.assertEqual(len(results[0]["scores"]), count)
                self.assertEqual(b.metadata["batch_verification"]["shape"], [2, 3])
                self.assertEqual(b.batch_policy["padding_side"], "right")
                self.assertEqual(b.batch_policy["pad_token_id"], a.tokenizer.pad_token_id)

    def test_shield_left_padding_and_tie_no(self):
        a = builtin("shield", ToyModel(15, causal=True), tiny_tokenizer())
        a.label_ids = [5, 6]
        b = wrap(a)
        out = b.verify_batch(["library", "public library"], [a.index(t) for t in ("library", "public library")])
        self.assertEqual([v["label"] for v in out], ["No", "No"])
        self.assertEqual(a.model.calls[0][1].tolist(), [[0, 1], [1, 1]])
        a.label_ids = [5, 5]
        self.assertEqual(b.request_batch(["library"])[0]["label"], "No")
        b.close()
        self.assertEqual(a.tokenizer.padding_side, "right")

    def test_generated_rows_exclude_padding_and_preserve_order(self):
        for name in ("hive", "open"):
            a = builtin(name, ToyModel(), tiny_tokenizer())
            b = wrap(a)
            texts = ["library", "public library"]
            out = b.verify_batch(texts, [a.index(t) for t in texts])
            self.assertEqual([v["label"] for v in out], ["safe", "unsafe"])
            self.assertEqual([v["generated_tokens"] for v in out], [2, 2])
            self.assertEqual(b.metadata["batch_verification"]["shape"], [2, 2])
            b.close()

    def test_limits_before_any_forward_and_oom_propagation(self):
        a = builtin("llama", ToyModel(), tiny_tokenizer())
        a.labels = ["a", "b"]
        b = wrap(a)
        b.request_batch(["library " * 512, "library"])
        a.model.calls.clear()
        b.reject_batch_over_limit(["library", "library " * 513])
        self.assertEqual(a.model.calls, [])
        with patch.object(a.model, "forward", side_effect=torch.OutOfMemoryError("fixture")):
            with self.assertRaises(torch.OutOfMemoryError):
                b.request_batch(["library", "library"])
        a.check_device.assert_called()

    def test_bad_index_and_no_forward_hook_during_request(self):
        a = builtin("llama", ToyModel(), tiny_tokenizer())
        a.labels = ["a", "b"]
        b = wrap(a)
        expected = [a.index("library"), a.index("public library")]
        expected[1] = {**expected[1], "ids_sha256": "wrong"}
        with self.assertRaises(ValueError):
            b.verify_batch(["library", "public library"], expected)
        self.assertEqual(len(a.model._forward_pre_hooks), 0)
        b.request_batch(["library", "public library"])
        self.assertNotIn("batch_verification", b.metadata)

    def test_eos_and_no_eos_counts(self):
        self.assertEqual(generated_ids([5, 3, 3, 3], 3), [5, 3])
        self.assertEqual(generated_ids([6, 8, 9, 3], [2, 3]), [6, 8, 9, 3])
        self.assertEqual(generated_ids([5, 8, 9], 3), [5, 8, 9])
        self.assertEqual(generated_ids([5, 8], None), [5, 8])

    def test_pad_identity_is_fixed_and_decoder_eos_fallback_is_recorded(self):
        a = builtin("shield", ToyModel(15, causal=True), tiny_tokenizer())
        a.label_ids = [5, 6]
        a.tokenizer.eos_token = "[SEP]"
        a.tokenizer.pad_token = None
        b = wrap(a)
        self.assertEqual(b.batch_policy["pad_token_id"], a.tokenizer.eos_token_id)
        a.tokenizer.pad_token = "[PAD]"
        with self.assertRaisesRegex(ValueError, "pad_token_id changed"):
            b.request_batch(["library", "public library"])
        self.assertEqual(a.model.calls, [])
        b.close()
        self.assertIsNone(a.tokenizer.pad_token)

    def test_actual_masks_padding_and_cardinality(self):
        ids = torch.tensor([[0, 5], [6, 5]])
        mask = torch.tensor([[0, 1], [1, 1]])
        expected = [input_index({"input_ids": torch.tensor([[5]]), "attention_mask": torch.ones(1, 1, dtype=torch.long)}, 1),
                    input_index({"input_ids": ids[1:], "attention_mask": mask[1:]}, 2)]
        validate_forward_rows(ids, mask, expected, 0, "left")
        for bad_ids, bad_mask in ((torch.tensor([[9, 5], [6, 5]]), mask), (ids, torch.ones_like(mask))):
            with self.assertRaises(ValueError):
                validate_forward_rows(bad_ids, bad_mask, expected, 0, "left")
        with self.assertRaises(ValueError):
            validate_forward_rows(ids, mask, expected[::-1], 0, "left")


class NativeSDKBatch(unittest.TestCase):
    def test_laya_one_encoding_and_native_batch(self):
        from speed_dataset.preparation import QUESTIONS
        agent = tiny_laya_agent()
        a = builtin("laya", agent.model, agent.tok)
        a.agent = agent
        a._install_laya_input_guard()
        b = wrap(a)
        self.addCleanup(b.close)
        texts = ["library", "The public library"]
        native = agent.predict_batch(texts, QUESTIONS, batch_size=2, sort_by_length=False)
        internal = {k: agent._to_internal(v) for k, v in QUESTIONS.items()}

        def index(text):
            ids = agent._encode_state(text, list(QUESTIONS), internal)[0]["ids"]
            return input_index({"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)},
                               len(agent.tok.encode(text)))

        a.index = index
        expected = [index(t) for t in texts]
        with patch.object(agent, "_encode_state", wraps=agent._encode_state) as encode, \
                patch.object(agent.model, "forward", wraps=agent.model.forward) as forward:
            output = b.request_batch(texts)
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(output, [{"label": v["answers"]["danger"]["choice"],
                                  "scores": v["answers"]["danger"]["probabilities"],
                                  "score_type": "probabilities"} for v in native])
        b.verify_batch(texts, expected)
        overhead = len(agent._encode_state("", list(QUESTIONS), internal)[0]["ids"])
        b.request_batch(["library " * (512 - overhead), "library"])
        b.reject_batch_over_limit(["library", "library " * (513 - overhead)])
        b.close()
        self.assertNotIn("_encode_state", vars(agent))

    def gliner(self):
        legacy = tiny_gliner_adapter()
        legacy.model.strict_extraction = True
        legacy.compile_count = 0
        a = builtin("gliner", legacy.model, legacy.tokenizer)
        a.legacy = legacy
        a.target = legacy.model.encoder
        return a

    def test_gliner_native_batch_encoder_and_head(self):
        a = self.gliner()
        b = wrap(a)
        texts = ["library.", "The public library."]
        expected = [a.index(t) for t in texts]
        with patch.object(a.model.encoder, "forward", wraps=a.model.encoder.forward) as forward:
            values = b.verify_batch(texts, expected)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(len(values), 2)
        self.assertTrue(all(set(v["scores"]) == {"safe", "unsafe"} for v in values))
        self.assertEqual(b.metadata["batch_verification"]["shape"][0], 2)

    def test_gliner_compile_real_score_route_and_restore(self):
        a = self.gliner()
        original = a.model.encoder.forward
        batches = [["library.", "The public library."], ["public library."]]
        reference = [wrap(a).request_batch(batch) for batch in batches]
        a.backend = "compile"
        b = wrap(a)
        with self.assertRaisesRegex(UnsupportedBatch, "not been prepared"):
            b.request_batch(batches[0])
        # Real Dynamo graph routing; CPU eager backend replaces Inductor only in this fixture.
        with patch("torch._dynamo.backends.registry.lookup_backend",
                   side_effect=lambda name: name if callable(name) else lambda graph, inputs: graph.forward):
            b.prepare_batch_backend(batches)
        self.assertGreater(b.compile_count, 0)
        before = b.compile_count
        for batch, expected in zip(batches, reference):
            actual = b.verify_batch(batch, [a.index(t) for t in batch])
            self.assertEqual(actual, expected)
        self.assertEqual(b.compile_count, before)
        a.legacy.restore_encoder()
        self.assertEqual(a.model.encoder.forward, original)
        torch._dynamo.reset()

    def test_halo_reuses_native_batched_scores(self):
        a = builtin("halo", ToyModel(), tiny_tokenizer())
        value = {"label": "safe", "p_safe": .8, "p_unsafe": .2, "logit_margin": -1.0}
        a.legacy = SimpleNamespace(scores=Mock(return_value=[value, value]))
        result = wrap(a).request_batch(["library", "public library"])
        a.legacy.scores.assert_called_once_with(["library", "public library"])
        self.assertEqual(result[0]["scores"], {"safe": .8, "unsafe": .2})


class ExternalBatch(unittest.TestCase):
    def test_single_fallback_only_and_delegation(self):
        a = SimpleNamespace(metadata={}, name="test", compile_count=0, request=Mock(return_value={"label": "a"}),
                            verify=Mock(return_value={"label": "a"}), close=Mock(), size=Mock(return_value={"x": 1}))
        b = wrap(a)
        self.assertIs(b.metadata, a.metadata)
        self.assertEqual(b.request_batch(["a"]), [{"label": "a"}])
        self.assertEqual(b.verify_batch(["a"], [{}]), [{"label": "a"}])
        for method, args in ((b.request_batch, (["a", "b"],)), (b.verify_batch, (["a", "b"], [{}, {}])),
                             (b.prepare_batch_backend, ([["a", "b"]],)), (b.prove_batch_backend, (["a", "b"],))):
            with self.assertRaises(UnsupportedBatch):
                method(*args)
        self.assertEqual(b.size(), {"x": 1})
        b.close()
        a.close.assert_called_once()

    def test_explicit_external_contract_and_cardinality(self):
        a = SimpleNamespace(metadata={}, batch_policy={"padding": "longest", "padding_side": "right", "pad_token_id": 0,
                                                      "truncation": False, "sort_by_length": False},
                            request_batch=Mock(return_value=[{"label": "a"}, {"label": "b"}]))
        b = wrap(a)
        self.assertEqual(len(b.request_batch(["a", "b"])), 2)
        a.request_batch.return_value = [{"label": "a"}]
        with self.assertRaises(ValueError):
            b.request_batch(["a", "b"])

    def test_synthetic_single_policy_does_not_authorize_external_batch(self):
        a = SimpleNamespace(metadata={}, request_batch=Mock(return_value=[{"label": "a"}, {"label": "a"}]))
        b = wrap(a)
        with self.assertRaises(UnsupportedBatch):
            b.request_batch(["a", "b"])
        a.request_batch.assert_not_called()

    def test_external_pad_id_must_be_explicit_integer(self):
        base = {"padding": "longest", "padding_side": "right", "truncation": False, "sort_by_length": False}
        for value in (None, True, 0.0, "0", -1):
            with self.subTest(value=value):
                a = SimpleNamespace(metadata={}, batch_policy={**base, "pad_token_id": value},
                                    request_batch=Mock(return_value=[{"label": "safe"}] * 2))
                with self.assertRaises(UnsupportedBatch):
                    wrap(a).request_batch(["a", "b"])
                a.request_batch.assert_not_called()

    def test_external_proof_normalized_from_return_or_metadata(self):
        policy = {"padding": "longest", "padding_side": "right", "pad_token_id": 0,
                  "truncation": False, "sort_by_length": False}
        for location in ("return", "backend_proof", "batch_backend_proof", "returned_metadata"):
            with self.subTest(location=location):
                proof = {"kernels": ["fixture_cuda_kernel"], "operators": ["fixture_op"]}
                a = SimpleNamespace(metadata={}, batch_policy=policy)

                def prove(payloads):
                    if location == "return":
                        return proof
                    if location == "returned_metadata":
                        return {"backend_proof": proof}
                    a.metadata[location] = proof

                a.prove_batch_backend = prove
                b = wrap(a)
                result = b.prove_batch_backend(["a", "b"])
                self.assertEqual(result, proof)
                self.assertIs(result, a.metadata["batch_backend_proof"])
                self.assertIsNot(result, proof)

    def test_single_legacy_proof_and_kernel_requirement(self):
        for fixture in (False, True):
            a = SimpleNamespace(metadata={"test_fixture": fixture})

            def prove(text):
                a.metadata["backend_proof"] = {"kernels": []}

            a.prove_backend = prove
            b = wrap(a)
            if fixture:
                self.assertEqual(b.prove_batch_backend(["a"]), {"kernels": []})
            else:
                with self.assertRaisesRegex(UnsupportedBatch, "no CUDA kernels"):
                    b.prove_batch_backend(["a"])
            a.prove_backend = lambda text: {"kernels": ["legacy_kernel"]}
            self.assertEqual(b.prove_batch_backend(["a"]), {"kernels": ["legacy_kernel"]})
            self.assertEqual(b.batch_policy["implementation"], "single_only")


class ORTBatch(unittest.TestCase):
    def run_stubbed_prepare(self, fail=False):
        from followup import onnx_service
        from u1_benchmark import batch_ort
        legacy = tiny_gliner_adapter()
        legacy.compile_count = 0
        legacy.model.strict_extraction = True
        a = builtin("gliner", legacy.model, legacy.tokenizer)
        a.legacy, a.target, a.backend = legacy, legacy.model.encoder, "ort"
        b = wrap(a)
        original = legacy.model.encoder.forward
        sessions, visits = [], []
        bad = {"cat": "Node", "name": "activation_kernel_time", "args": {
            "provider": "CPUExecutionProvider", "op_name": "MatMul",
            "input_type_shape": [{"float": [2, 3]}], "output_type_shape": [{"float": [2, 3]}]}}

        class Session:
            def __init__(self, path, sess_options, providers):
                self.profile = sess_options.enable_profiling
                self.ended = False
                self.ordinal = len(sessions)
                self.fallback_disabled = False
                sessions.append(self)

            def get_providers(self):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            def disable_fallback(self):
                self.fallback_disabled = True

            def end_profiling(self):
                self.ended = True
                path = a.directory / (str(self.ordinal) + ".json")
                events = [{"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}}]
                path.write_text(json.dumps(events + ([bad] if fail else [])))
                return str(path)

        def install(adapter, directory):
            session = Session("fixture", SimpleNamespace(enable_profiling=True), [])
            adapter.ort_session = session
            adapter.restore_encoder = lambda: setattr(adapter.model.encoder, "forward", original)
            adapter.service_proven_names = set()
            adapter.service_checked_shapes = set()
            adapter.metadata["ort"] = {"cpu_policy": "metadata-cpu-v1"}

        def binding(session, encoder, shapes, checked):
            def forward(input_ids=None, attention_mask=None, **kwargs):
                shape = tuple(input_ids.shape)
                if checked() and shape not in shapes:
                    raise UnsupportedBatch("Unvalidated ORT batch shape")
                visits.append((session.ordinal, shape, session.profile))
                return original(input_ids=input_ids, attention_mask=attention_mask)
            return forward

        fake_ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(enable_profiling=False),
                                   GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
                                   InferenceSession=Session)
        fake_onnx = SimpleNamespace(load=lambda *args, **kwargs: SimpleNamespace(graph=None))
        before = onnx_service.VALIDATION_TEXTS
        with tempfile.TemporaryDirectory() as tmp:
            a.directory = Path(tmp)
            with patch.dict(sys.modules, {"onnxruntime": fake_ort, "onnx": fake_onnx}), \
                    patch("followup.team_smoke.preload_ort"), patch("torch.cuda.synchronize"), \
                    patch.object(onnx_service, "install", side_effect=install), \
                    patch.object(onnx_service, "prove_metadata_nodes", return_value={"proven": []}), \
                    patch.object(batch_ort, "binding_forward", side_effect=binding):
                batches = [["library.", "The public library."]]
                if fail:
                    with self.assertRaisesRegex(UnsupportedBatch, "metadata-cpu-v1"):
                        b.prepare_batch_backend(batches)
                    self.assertEqual(legacy.model.encoder.forward, original)
                    self.assertFalse(b._backend_installed)
                else:
                    b.prepare_batch_backend(batches * 3)
                    self.assertTrue(sessions[0].ended)
                    self.assertFalse(sessions[-1].profile)
                    self.assertTrue(all(s.fallback_disabled for s in sessions))
                    b.verify_batch(batches[0], [a.index(t) for t in batches[0]])
                    self.assertEqual(visits[-1][0], sessions[-1].ordinal)
                    prior = len(sessions)
                    b.prepare_batch_backend(batches)
                    self.assertEqual(len(sessions), prior)
                    # The separate limit probe introduces a new validated shape, not a fallback.
                    new = [["The public library opens every morning.", "library."]]
                    b.prepare_batch_backend(new)
                    self.assertEqual(len(sessions), prior + 2)
                    self.assertEqual(len(legacy.service_checked_shapes), 2)
                    with self.assertRaisesRegex(UnsupportedBatch, "Unvalidated"):
                        b.request_batch(["library."])
                    legacy.restore_encoder()
                    self.assertEqual(legacy.model.encoder.forward, original)
                self.assertIs(onnx_service.VALIDATION_TEXTS, before)
        return visits

    def test_ort_score_routing_profile_separation_shape_extension_restore(self):
        self.assertTrue(self.run_stubbed_prepare())

    def test_ort_cpu_policy_failure_restores_and_does_not_fallback(self):
        self.run_stubbed_prepare(fail=True)

    def test_binding_rejects_cpu_and_unexpected_arguments(self):
        from u1_benchmark.batch_ort import binding_forward
        fn = binding_forward(Mock(), Mock(), {(2, 3)}, lambda: True)
        with self.assertRaisesRegex(UnsupportedBatch, "CUDA 0"):
            fn(torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(UnsupportedBatch, "Unexpected"):
            fn(token_type_ids=torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
