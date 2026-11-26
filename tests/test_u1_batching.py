"""CPU-only batch planning contract tests, no model dependencies."""
import copy
import json
import math
import random
import unittest
from collections import Counter

from u1_benchmark import batching as b


def rows(n=400, bucket="B1"):
    return [{"id": f"{bucket}-{i:04d}", "bucket": bucket,
             "text": "PRIVATE INPUT", "messages": [{"role": "user", "content": "PRIVATE"}],
             "index": {"total_tokens": 1 + i, "input_sha256": b.plan_hash([bucket, i])}}
            for i in range(n)]


def config(model="gliner", sizes=None, mode="screening", candidate=True):
    value = {"model": model, "seed": 17, "mode": mode,
             "batch_sizes_by_bucket": sizes or b.default_grid(model)}
    if candidate:
        value["candidate"] = {"backend": "compile"}
    return value


class GridTests(unittest.TestCase):
    def test_all_model_defaults_and_freshness(self):
        for model in ("gliner", "koala", "llama", "laya"):
            self.assertEqual(b.default_grid(model), {"B1": [1, 8, 16, 32, 64]})
        for model in ("halo", "hive", "open", "shield"):
            expected = {"B1": [1, 4, 8, 16, 32], "B2": [1, 4, 8], "B3": [1, 4], "B4": [1]}
            self.assertEqual(b.default_grid(model), expected)
        grid = b.default_grid("laya")
        grid["B1"].clear()
        self.assertEqual(b.default_grid("laya")["B1"], [1, 8, 16, 32, 64])

    def test_custom_subset_canonicalization_and_legacy(self):
        grid = {"B4": [1], "B3": [8, 1], "B2": [4], "B1": [64, 8]}
        original = copy.deepcopy(grid)
        self.assertEqual(b.cells(config("halo", grid)), [("B1", 8), ("B1", 64), ("B2", 4), ("B3", 1), ("B3", 8), ("B4", 1)])
        self.assertEqual(grid, original)
        self.assertEqual(b.cells({"model": "halo"}), [(key, 1) for key in ("B1", "B2", "B3", "B4")])

    def test_reject_malformed_grids(self):
        for grid in (None, [], {}, {"B2": [1]}, {"B1": []}, {"B1": [1, 1]},
                     {"B1": [True]}, {"B1": [1.0]}, {"B1": ["1"]}, {"B1": [0]},
                     {"B1": [-1]}, {"B1": [2]}, {"B1": [256]}, {"B1": [401]},
                     {"B1": {1}}, {"B1": [1], "B2": [1]}):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                b.validate_grid("laya", grid)
        with self.assertRaises(ValueError):
            b.default_grid("unknown")
        with self.assertRaises(ValueError):
            b.validate_grid("halo", {"B1": [1]})


class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {"B1": rows()}
        cls.cfg = config(sizes={"B1": list(b.SIZES)})
        cls.jobs = b.schedule(cls.cfg, cls.data)

    def test_exact_counts_ratios_and_no_tail(self):
        for size in b.SIZES:
            jobs = [job for job in self.jobs if job["batch_size"] == size]
            self.assertEqual(len(jobs), 10)
            for job in jobs:
                expected = 30 * size if job["phase"] == "aa" else max(400, 100 * size)
                self.assertEqual(len(job["ids"]), expected)
                self.assertEqual(len(job["batches"]), expected // size)
                occurrences = Counter()
                for batch in job["batches"]:
                    self.assertEqual(len(set(batch["ids"])), size)
                    self.assertEqual(batch["occurrences"], [occurrences[key] for key in batch["ids"]])
                    occurrences.update(batch["ids"])
                if job["phase"] == "ab":
                    self.assertEqual(len(occurrences), 400)
                    self.assertEqual(set(occurrences.values()), {expected // 400})
                    for start in range(400, expected, 400):
                        self.assertEqual(job["ids"][:400], job["ids"][start:start + 400])

    def test_cell_first_pair_order_and_copy_isolation(self):
        self.assertEqual([job["batch_size"] for job in self.jobs], sorted(job["batch_size"] for job in self.jobs))
        for a, c in zip(self.jobs[::2], self.jobs[1::2]):
            self.assertEqual(a["batches"], c["batches"])
            self.assertIsNot(a["batches"], c["batches"])
            self.assertIsNot(a["batches"][0], c["batches"][0])
            self.assertEqual(a["plan_sha256"], b.plan_hash(c["batches"]))
            self.assertEqual((a["side"], c["side"]), ("a", "b") if a["block"] % 2 == 0 else ("b", "a"))
            if a["phase"] == "aa":
                self.assertEqual(a["variant"], "baseline")
                self.assertEqual(c["variant"], "baseline")
            else:
                self.assertEqual({a["side"]: a["variant"], c["side"]: c["variant"]}, {"a": "baseline", "b": "candidate"})

    def test_reproducible_json_safe_private_input_independent(self):
        before = copy.deepcopy(self.data)
        state = random.getstate()
        self.assertEqual(self.jobs, b.schedule(self.cfg, {"B1": list(reversed(self.data["B1"]))}))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(self.data, before)
        encoded = json.dumps(self.jobs)
        self.assertEqual(json.loads(encoded), self.jobs)
        self.assertNotIn("PRIVATE", encoded)
        self.assertNotIn('"text"', encoded)
        self.assertNotIn('"messages"', encoded)
        changed = b.schedule({**self.cfg, "seed": 42}, self.data)
        self.assertNotEqual(changed[0]["plan_sha256"], self.jobs[0]["plan_sha256"])

    def test_pilot_without_aa_and_small_dataset(self):
        for candidate in (True, False):
            jobs = b.schedule(config(sizes={"B1": [64]}, mode="pilot", candidate=candidate), {"B1": rows(64)})
            self.assertEqual(len(jobs), 2 if candidate else 1)
            for job in jobs:
                self.assertEqual(len(job["batches"]), 12)
                self.assertEqual(len(job["ids"]), 768)
                self.assertEqual(job["phase"], "ab" if candidate else "baseline")

    def test_baseline_and_non_latency_modes(self):
        jobs = b.schedule(config(sizes={"B1": [1]}, candidate=False), self.data)
        self.assertEqual(Counter(job["phase"] for job in jobs), {"aa": 4, "baseline": 3})
        self.assertEqual({job["variant"] for job in jobs}, {"baseline"})
        for mode in ("quality", "smoke"):
            self.assertEqual(b.schedule(config(mode=mode), {}), [])

    def test_all_decoder_cells_and_dataset_errors(self):
        cfg = config("halo", mode="pilot")
        data = {bucket: rows(64, bucket) for bucket in b.default_grid("halo")}
        jobs = b.schedule(cfg, data)
        self.assertEqual([(job["bucket"], job["batch_size"]) for job in jobs[::2]], b.cells(cfg))
        for data in ({}, {"B2": rows()}, {"B1": rows(399)}, {"B1": rows(401)}):
            with self.assertRaises(ValueError):
                b.schedule(self.cfg, data)
        with self.assertRaises(ValueError):
            b.schedule(config(mode="pilot", sizes={"B1": [64]}), {"B1": rows(12)})
        with self.assertRaises(ValueError):
            b.schedule({**self.cfg, "mode": "typo"}, self.data)


class AuxiliaryPlanTests(unittest.TestCase):
    def test_smoke_profile_full_unique_short_long_mixed(self):
        source = rows()
        for size in b.SIZES:
            short, long, mixed = b.smoke_plans(source, size)
            self.assertEqual(short["total_tokens"], list(range(1, size + 1)))
            self.assertEqual(long["total_tokens"], list(range(401 - size, 401)))
            for plan in (short, long, mixed):
                self.assertEqual(len(set(plan["ids"])), size)
            self.assertEqual(b.profile_plans(source, size), [short, long])
            self.assertEqual([short, long, mixed], b.smoke_plans(list(reversed(source)), size))
        with self.assertRaises(ValueError):
            b.smoke_plans(rows(12), 64)

    def test_memory_minimal_replays_and_occurrences(self):
        for count in (65, 400):
            for size in b.SIZES:
                if size > count:
                    with self.assertRaises(ValueError):
                        b.memory_plans(rows(count), size)
                    continue
                plans = b.memory_plans(rows(count), size)
                self.assertEqual(len(plans) * size, math.lcm(count, size))
                seen = Counter()
                for plan in plans:
                    self.assertEqual(len(set(plan["ids"])), size)
                    self.assertEqual(plan["occurrences"], [seen[key] for key in plan["ids"]])
                    seen.update(plan["ids"])
                self.assertEqual(len(seen), count)
                self.assertEqual(set(seen.values()), {math.lcm(count, size) // count})
        with self.assertRaises(ValueError):
            b.memory_plans(rows(12), 64)

    def test_metadata_accounting_and_materialize(self):
        source = rows()
        plan = b.smoke_plans(source, 4)[0]
        self.assertEqual(plan["useful_tokens"], 10)
        self.assertEqual(plan["padded_length"], 4)
        self.assertEqual(plan["padded_tokens"], 16)
        self.assertEqual(plan["padding_tokens"], 6)
        self.assertEqual(plan["id"], b.plan_hash({k: v for k, v in plan.items() if k != "id"}))
        materialized = b.materialize(plan, source)
        self.assertEqual([row["id"] for row in materialized], plan["ids"])
        self.assertIs(materialized[0], source[0])
        for field, value in (("id", "tampered"), ("padded_length", 900), ("occurrences", [-1] * 4),
                             ("ids", ["absent"] * 4), ("unexpected", "PRIVATE")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                b.materialize({**plan, field: value}, source)
        altered = copy.deepcopy(source)
        altered[0]["index"]["input_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            b.materialize(plan, altered)

    def test_structural_failures(self):
        bad = [[], rows(1) * 2, [{"id": "x"}], [None]]
        for field, value in (("total_tokens", 0), ("total_tokens", True),
                             ("input_sha256", "private text"), ("input_sha256", "g" * 64)):
            altered = rows(1)
            altered[0]["index"][field] = value
            bad.append(altered)
        for source in bad:
            with self.subTest(source=source), self.assertRaises(ValueError):
                b.smoke_plans(source, 1)


if __name__ == "__main__":
    unittest.main()
