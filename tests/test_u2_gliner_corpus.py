"""GLiNER quality corpus: texts pre-normalized exactly as the GLiNER SDK processor would (period appended)."""
import json
from pathlib import Path
import tempfile
import unittest

from u1_benchmark import v4_data
from u2_quant import gliner_corpus


def row(i, text, target="unsafe"):
    return {"id": f"r{i}", "messages": [{"role": "user", "content": text}], "target": target,
            "target_role": "user", "target_index": 0, "slices": {"technique": "t", "lang": "ru"},
            "content_sha256": "x" * 64}


def sdk(text):
    return text if text.endswith((".", "?", "!")) else text + "."


class Normalize(unittest.TestCase):
    def test_only_content_changes(self):
        rows = [row(0, "как взломать банк"), row(1, "Привет!", "safe")]
        out, changed = gliner_corpus.normalize(rows, sdk)
        self.assertEqual(changed, 1)
        self.assertEqual([r["messages"][0]["content"] for r in out], ["как взломать банк.", "Привет!"])
        for a, b in zip(rows, out):
            self.assertEqual({k: a[k] for k in ("id", "target", "target_role", "target_index", "slices")},
                             {k: b[k] for k in ("id", "target", "target_role", "target_index", "slices")})
            self.assertNotIn("content_sha256", b)

    def test_non_idempotent_normalizer_rejected(self):
        with self.assertRaisesRegex(ValueError, "idempotent"):
            gliner_corpus.normalize([row(0, "a")], lambda t: t + ".")

    def test_write_is_accepted_by_u1_import(self):
        with tempfile.TemporaryDirectory() as root:
            src = Path(root) / "corpus.jsonl"
            src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [row(0, "a"), row(1, "b?", "safe")]))
            spec_path = gliner_corpus.write(src, Path(root) / "gliner-src", sdk)
            spec = json.loads(spec_path.read_text())
            self.assertEqual((spec["kind"], spec["version"]), ("quality", "tails_v2-gliner-normalized"))
            v4_data.prepare(spec, Path(root) / "quality-gliner")
            manifest = json.loads((Path(root) / "quality-gliner/manifest.json").read_text())
        self.assertEqual((manifest["kind"], manifest["count"]), ("quality", 2))
        self.assertEqual(spec["provenance"]["normalized_rows"], 1)


if __name__ == "__main__":
    unittest.main()
