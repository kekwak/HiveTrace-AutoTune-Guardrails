"""U1 v4 quality worker: checkpoint at most every SAVE_INTERVAL seconds plus a final save (was O(N^2))."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from u1_benchmark import v4_worker


class Guard:
    def call(self, function, *args, **kwargs):
        return function(*args, **kwargs)


class Adapter:
    def prepare_batch_backend(self, groups):
        pass

    def prove_batch_backend(self, group):
        return {}

    def verify_batch(self, group, indices):
        return [{"label": "safe"} for _ in group]


def plan(n, size):
    ids = [f"r{i}" for i in range(n)]
    return {"ordered_ids": ids, "excluded": {}, "indices": {i: {} for i in ids},
            "batches": [{"ids": ids[k:k + size]} for k in range(0, n, size)]}


class Throttle(unittest.TestCase):
    def run_quality(self, n, size, clock):
        rows = [{"id": f"r{i}", "text": f"t{i}", "target": "safe"} for i in range(n)]
        saves = []
        with tempfile.TemporaryDirectory() as root, \
                patch("u1_benchmark.quality_batches.prepare_plan", lambda *a: plan(n, size)), \
                patch.object(v4_worker, "private_values", lambda rows, spec: []), \
                patch.object(v4_worker, "public_labels", lambda adapter, spec: set()), \
                patch.object(v4_worker, "clean_evidence", lambda score, texts, allowed: score), \
                patch("u1_benchmark.quality.metrics", lambda rows, values, mapping: {"overall": {"evaluated": len(values)}}), \
                patch.object(v4_worker.time, "monotonic", clock):
            result = {}
            v4_worker.quality(Adapter(), rows, {"quality_batch_size": size, "seed": 17, "label_mapping": {}},
                              Guard(), result, Path(root), lambda: saves.append(1))
            lines = [json.loads(x) for x in (Path(root) / "predictions.jsonl").read_text().splitlines()]
        return saves, lines

    def test_few_checkpoints_and_complete_final_predictions(self):
        saves, lines = self.run_quality(500, 1, lambda: 0.0)        # время стоит — только финальное сохранение
        self.assertLessEqual(len(saves), 2)
        self.assertEqual([x["status"] for x in lines], ["passed"] * 500)

    def test_checkpoint_after_interval(self):
        ticks = iter(range(0, 10_000, 10))                           # каждый вызов +10 c
        saves, lines = self.run_quality(30, 1, lambda: next(ticks))
        self.assertGreater(len(saves), 2)
        self.assertLess(len(saves), 31)
        self.assertEqual(len(lines), 30)


if __name__ == "__main__":
    unittest.main()
