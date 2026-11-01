import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from u1_benchmark import contracts, datasets, role_preparation
from u1_benchmark.core import ROOT, digest, sha
from u1_benchmark.messages import UnsupportedInput
from test_u1_roles import ToyRolePreparer


class OpenTask(unittest.TestCase):
    def test_other_seven_preparation_contracts_are_unchanged(self):
        expected = digest({name: sha(ROOT / "u1_benchmark" / name)
                           for name in ("messages.py", "native_preparation.py")})
        for model in contracts.MODEL_SPECS:
            current = contracts.model_contract(model, 3)
            if model == "open":
                self.assertNotEqual(current["preparation_sha256"], expected)
                self.assertIn("template_revision", current["preparation_spec"])
            else:
                self.assertEqual(current["preparation_sha256"], expected)

    def test_factory_only_overrides_open(self):
        with patch("u1_benchmark.native_preparation.Preparer", return_value="native") as native, \
             patch.object(role_preparation, "OpenTaskPreparer", return_value="task") as task:
            self.assertEqual(role_preparation.Preparer("open"), "task")
            for model in set(contracts.MODEL_SPECS) - {"open"}:
                self.assertEqual(role_preparation.Preparer(model), "native")
            self.assertEqual(native.call_count, 7)
            task.assert_called_once_with()

    def test_task_uses_selected_user_with_no_context_or_parser_change(self):
        class Legacy:
            def render(self, text):
                return "TASK:" + text
        p = role_preparation.OpenTaskPreparer.__new__(role_preparation.OpenTaskPreparer)
        p.name, p.legacy = "open", Legacy()
        self.assertEqual(p.render([{"role": "user", "content": "hello 🙂"}]), "TASK:hello 🙂")
        for value in ([{"role": "assistant", "content": "answer"}],
                      [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]):
            with self.assertRaises(UnsupportedInput):
                p.render(value)
        from u1_benchmark.adapters import parse_generated
        with self.assertRaises(ValueError):
            parse_generated("unsafe\nunsafe\n", "open", 4)

    def test_reindex_keeps_corpus_and_other_indices_and_repeats(self):
        from tools.reindex_role_model import rebuild
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            datasets.jsonl(source, [{"id": str(i), "messages": [{"role": "user", "content": str(i)}]}
                                   for i in range(400)])
            spec = {"schema_version": 3, "kind": "speed", "source": str(source),
                    "fields": {"id": "id", "messages": "messages"}, "models": ["gliner", "koala"],
                    "version": "fixture", "provenance": "test"}
            datasets.prepare(spec, root / "base", ToyRolePreparer)
            first = rebuild(root / "base", root / "new", ["gliner"], "fixture-new", ToyRolePreparer)
            second = rebuild(root / "base", root / "repeat", ["gliner"], "fixture-new", ToyRolePreparer)
            self.assertEqual(first["status"], "ready")
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            for name in ("corpus.jsonl", "indices/koala.jsonl", "buckets/koala/B1.json"):
                self.assertEqual((root / "base" / name).read_bytes(), (root / "new" / name).read_bytes())
