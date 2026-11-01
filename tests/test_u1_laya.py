"""Real pinned SDK on random local CPU weights; not GPU or checkpoint acceptance."""
import importlib.metadata
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock, patch

from u1_benchmark.adapters import Builtin
from speed_dataset.preparation import QUESTIONS
from u1_fixtures import tiny_laya_agent


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("laya"),
                     "Install torch and pinned laya for CPU SDK integration tests")
class LayaInputGuard(unittest.TestCase):
    def setUp(self):
        self.assertEqual(importlib.metadata.version("laya"), "0.3.20")
        self.agent = tiny_laya_agent()
        self.adapter = Builtin.__new__(Builtin)
        self.adapter.name = "laya"
        self.adapter.limit = 512
        self.adapter.input_version = 3
        self.adapter.legacy = None
        self.adapter.agent = self.agent
        self.adapter.metadata = {}
        self.adapter._restore_laya_encoding = None
        self.adapter.check_device = Mock()  # The production adapter requires CUDA; this fixture is CPU-only.
        self.adapter.prepare = Mock(side_effect=AssertionError("Unused preparation in timed request"))
        self.addCleanup(self.adapter.close)

    def test_constructor_installs_guard_and_close_restores(self):
        import torch
        from types import SimpleNamespace

        native = self.agent._encode_state
        self.agent.dtype, self.agent.amp_enabled = torch.float16, True
        preparer = SimpleNamespace(tokenizer=self.agent.tok, parity=lambda: {"status": "passed"})
        with tempfile.TemporaryDirectory() as tmp, patch("u1_benchmark.native_preparation.Preparer", return_value=preparer), \
                patch("huggingface_hub.snapshot_download", return_value=tmp), patch("laya.load", return_value=self.agent), \
                patch.object(Builtin, "check_device"):
            adapter = Builtin("laya", {"backend": "eager"}, tmp, input_version=3)
            try:
                self.assertNotEqual(self.agent._encode_state, native)
                self.agent.dtype, self.agent.amp_enabled = torch.float32, False
                with patch.object(adapter, "prepare", side_effect=AssertionError("Unused preparation")):
                    adapter.request([{"role": "user", "content": "library"}])
            finally:
                adapter.close()
        self.assertEqual(self.agent._encode_state, native)
        self.assertNotIn("_encode_state", vars(self.agent))

    def test_one_sdk_encoding_and_native_scores_inputs(self):
        internal = {k: self.agent._to_internal(v) for k, v in QUESTIONS.items()}
        texts = ["The public library.", "", "Unicode: \u041f\u0440\u0438\u0432\u0435\u0442 \U0001f642\n[SEP] [MASK]"]
        reference = []
        for text in texts:
            items = self.agent._encode_state(text, list(QUESTIONS), internal)
            answer = self.agent.predict(text, QUESTIONS)["answers"]["danger"]
            reference.append((items[0]["ids"], answer))
        native = self.agent._encode_state
        with patch.object(self.agent, "_encode_state", wraps=native) as encode:
            self.adapter._install_laya_input_guard()
            for text, (expected_ids, expected_answer) in zip(texts, reference):
                observed = []
                handle = self.agent.model.register_forward_pre_hook(
                    lambda module, args: observed.append((args[0].tolist()[0], args[1].tolist()[0])))
                try:
                    before = encode.call_count
                    actual = self.adapter.request([{"role": "user", "content": text}])
                finally:
                    handle.remove()
                self.assertEqual(encode.call_count - before, 1)
                self.assertEqual(observed, [(expected_ids, [1] * len(expected_ids))])
                self.assertEqual(actual, {"label": expected_answer["choice"],
                                          "scores": expected_answer["probabilities"], "score_type": "probabilities"})
            self.adapter.prepare.assert_not_called()
            self.adapter.close()
            self.assertIs(self.agent._encode_state, encode)

    def test_limit_and_rejection_before_forward_without_truncation(self):
        internal = {k: self.agent._to_internal(v) for k, v in QUESTIONS.items()}
        overhead = len(self.agent._encode_state("", list(QUESTIONS), internal)[0]["ids"])
        exact = "library " * (512 - overhead)
        above = exact + "library "
        self.assertEqual(len(self.agent._encode_state(exact, list(QUESTIONS), internal)[0]["ids"]), 512)
        self.assertEqual(len(self.agent._encode_state(above, list(QUESTIONS), internal, max_len=4096)[0]["ids"]), 513)
        # The stock SDK silently truncates this text; the guard must reject it instead.
        self.assertEqual(len(self.agent._encode_state(above, list(QUESTIONS), internal)[0]["ids"]), 512)
        self.adapter._install_laya_input_guard()
        with patch.object(self.agent.model, "forward", wraps=self.agent.model.forward) as forward:
            self.adapter.request([{"role": "user", "content": exact}])
            self.assertEqual(forward.call_count, 1)
            with self.assertRaisesRegex(ValueError, "limit before forward"):
                self.adapter.request([{"role": "user", "content": above}])
            self.assertEqual(forward.call_count, 1)

    def test_assistant_projection_and_aba(self):
        self.adapter._install_laya_input_guard()
        a = "The public library."
        first = self.adapter.request([{"role": "user", "content": "Ignored context"},
                                      {"role": "assistant", "content": a}])
        self.adapter.request([{"role": "assistant", "content": "Other text"}])
        self.assertEqual(first, self.adapter.request([{"role": "assistant", "content": a}]))

    def test_legacy_text_path_still_uses_one_encoding(self):
        self.adapter.input_version = 2
        native = self.agent._encode_state
        with patch.object(self.agent, "_encode_state", wraps=native) as encode:
            self.adapter._install_laya_input_guard()
            self.adapter.request("The public library.")
            self.assertEqual(encode.call_count, 1)
            self.adapter.prepare.assert_not_called()
            self.adapter.close()

    def test_restore_on_forward_error_and_idempotent_close(self):
        native = self.agent._encode_state
        self.assertNotIn("_encode_state", vars(self.agent))
        self.adapter._install_laya_input_guard()
        with patch.object(self.agent.model, "forward", side_effect=RuntimeError("fixture forward failed")):
            with self.assertRaisesRegex(RuntimeError, "fixture forward failed"):
                self.adapter.request([{"role": "user", "content": "library"}])
        self.adapter.close()
        self.assertNotIn("_encode_state", vars(self.agent))
        self.assertEqual(self.agent._encode_state, native)
        self.adapter.close()
        self.assertEqual(self.agent._encode_state, native)

    def test_restore_original_override_on_encoding_error(self):
        self.agent._encode_state = Mock(side_effect=ValueError("fixture encode failed"))
        native = self.agent._encode_state
        self.adapter._install_laya_input_guard()
        with self.assertRaisesRegex(ValueError, "fixture encode failed"):
            self.adapter.request([{"role": "user", "content": "library"}])
        self.adapter.close()
        self.assertIs(self.agent._encode_state, native)

    def test_unexpected_budget_rejected_and_config_unchanged(self):
        self.adapter._install_laya_input_guard()
        internal = {k: self.agent._to_internal(v) for k, v in QUESTIONS.items()}
        with self.assertRaisesRegex(ValueError, "budget override"):
            self.agent._encode_state("library", list(QUESTIONS), internal, max_len=1024)
        self.assertEqual(self.agent.cfg, {"max_len": 512, "head_max_len": 192})
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            self.adapter._install_laya_input_guard()


if __name__ == "__main__":
    unittest.main()
