"""CPU-only v4 plan tests; models and messages are never rendered."""
import copy
import json
import random
import unittest
from collections import Counter

from u1_benchmark import batching as legacy
from u1_benchmark import v4_planning as p


def rows(n=257, bucket="custom"):
    return [{"id": f"{bucket}-{i:05d}",
             "index": {"total_tokens": i + 1, "input_sha256": p.plan_hash([bucket, i])},
             "messages": [{"role": "user", "content": "PRIVATE INPUT"}]}
            for i in range(n)]


def config(sizes=(3,), mode="baseline", pilot=False):
    cfg = {"schema_version": 4, "mode": mode, "pilot": pilot, "seed": 17,
           "buckets": {"custom": {"min": 1, "max": 4096}},
           "batch_sizes_by_bucket": {"custom": list(sizes)},
           "baseline": {"adapter": {"module": "custom.module:Adapter"}}}
    if mode in ("search", "confirm"):
        cfg["candidate"] = copy.deepcopy(cfg["baseline"])
    return cfg


class CellTests(unittest.TestCase):
    def test_custom_bucket_arbitrary_size_canonical_and_fresh(self):
        cfg = config((257, 3, 256, 24))
        before = copy.deepcopy(cfg)
        self.assertEqual(p.cells(cfg), [("custom", n) for n in (3, 24, 256, 257)])
        self.assertEqual(cfg, before)

    def test_bad_grids_and_no_coercion(self):
        for grid in (None, [], {}, {"unknown": [3]}, {"custom": []},
                     {"custom": [3, 3]}, {"custom": [True]}, {"custom": [3.0]},
                     {"custom": ["3"]}, {"custom": [0]}, {"custom": [-1]},
                     {"custom": {3}}, {"custom": [3], "unknown": [1]}):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                p.cells({**config(), "batch_sizes_by_bucket": grid})
        for bounds in ({"min": True, "max": 3}, {"min": 1, "max": "3"},
                       {"min": 3, "max": 2}, {"min": 0, "max": 3}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                p.cells({**config(), "buckets": {"custom": bounds}})


class ScheduleTests(unittest.TestCase):
    def test_arbitrary_sizes_full_unique_batches_and_occurrences(self):
        data = {"custom": rows()}
        jobs = p.schedule(config((3, 24, 256, 257)), data)
        self.assertEqual(len(jobs), 28)
        for size in (3, 24, 256, 257):
            selected = [j for j in jobs if j["batch_size"] == size]
            self.assertEqual(Counter(j["phase"] for j in selected), {"aa": 4, "baseline": 3})
            for job in selected:
                count = 30 if job["phase"] == "aa" else 100
                self.assertEqual(len(job["batches"]), count)
                self.assertEqual(len(job["ids"]), count * size)
                seen = Counter()
                for batch in job["batches"]:
                    self.assertEqual(len(batch["ids"]), size)
                    self.assertEqual(len(set(batch["ids"])), size)
                    self.assertEqual(batch["occurrences"], [seen[key] for key in batch["ids"]])
                    seen.update(batch["ids"])
                if job["phase"] == "baseline":
                    self.assertEqual(len(seen), 257)
                    self.assertLessEqual(max(seen.values()) - min(seen.values()), 1)
                self.assertEqual(p.plan_hash(job["batches"]), job["plan_sha256"])

    def test_ceiling_and_original_seed_string(self):
        source = rows(401)
        jobs = p.schedule(config((3,)), {"custom": source})
        main = [j for j in jobs if j["phase"] == "baseline"]
        self.assertEqual([len(j["batches"]) for j in main], [134] * 3)
        for job in jobs:
            expected = [row["id"] for row in source]
            random.Random(f"17:{job['phase']}:{job['block']}:custom").shuffle(expected)
            self.assertEqual(job["ids"], [expected[i % len(source)] for i in range(len(job["ids"]))])

    def test_search_only_candidate_and_exact_baseline_main_replay(self):
        data = {"custom": rows(31)}
        cfg = config((3, 24))
        baseline = p.schedule(cfg, data)
        before = copy.deepcopy(baseline)
        search = p.schedule(config((3, 24), "search"), data, baseline)
        self.assertEqual(len(search), 6)
        self.assertEqual({j["variant"] for j in search}, {"candidate"})
        self.assertEqual({j["phase"] for j in search}, {"search"})
        self.assertEqual({j["side"] for j in search}, {"b"})
        self.assertEqual(search, p.schedule(config((3, 24), "search"), data))
        for job, ref in zip(search, [j for j in baseline if j["phase"] == "baseline"]):
            self.assertEqual(job["batches"], ref["batches"])
            self.assertEqual(job["plan_sha256"], ref["plan_sha256"])
            self.assertIsNot(job["batches"][0], ref["batches"][0])
        self.assertEqual(baseline, before)

    def test_reference_failures(self):
        data = {"custom": rows(9)}
        baseline = p.schedule(config(), data)
        main = [j for j in baseline if j["phase"] == "baseline"]
        for reference in (None, [], main[:-1], main + [main[0]], {"jobs": main}):
            if reference is None:
                continue
            with self.subTest(reference=type(reference)), self.assertRaises(ValueError):
                p.schedule(config(mode="search"), data, reference)
        for field, value in (("plan_sha256", "0" * 64), ("side", "b"),
                             ("variant", "candidate"), ("batch_size", 3.0),
                             ("block", False), ("ids", []), ("batches", [])):
            changed = copy.deepcopy(main)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.schedule(config(mode="search"), data, changed)
        with self.assertRaises(ValueError):
            p.schedule({**config(mode="search"), "seed": 123}, data, main)
        altered = copy.deepcopy(data)
        altered["custom"][0]["index"]["input_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            p.schedule(config(mode="search"), altered, main)

    def test_pilot_search_replays_full_reference_block_zero_prefix_per_cell(self):
        cfg = config((3, 24, 256, 257))
        cfg["buckets"]["other"] = {"min": 1, "max": 4096}
        cfg["batch_sizes_by_bucket"]["other"] = [3, 24]
        data = {"custom": rows(257), "other": rows(31, "other")}
        baseline = p.schedule(cfg, data)
        before = copy.deepcopy(baseline)
        pilot = {**cfg, "mode": "search", "pilot": True, "candidate": cfg["baseline"]}
        jobs = p.schedule(pilot, data, baseline)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(jobs, p.schedule(pilot, data))
        first = {(j["bucket"], j["batch_size"]): j for j in baseline
                 if j["phase"] == "baseline" and j["block"] == 0}
        for job in jobs:
            original = first[job["bucket"], job["batch_size"]]
            self.assertEqual((job["phase"], job["variant"], job["side"], job["block"]),
                             ("search", "candidate", "b", 0))
            self.assertEqual(job["batches"], original["batches"][:12])
            self.assertEqual(job["ids"], original["ids"][:12 * job["batch_size"]])
            self.assertEqual(job["plan_sha256"], p.plan_hash(original["batches"][:12]))
            self.assertNotEqual(job["plan_sha256"], original["plan_sha256"])
            self.assertIsNot(job["batches"][0], original["batches"][0])
        self.assertEqual(baseline, before)
        counts = p.workload(pilot, data, baseline)
        self.assertEqual((counts["jobs"], counts["batches"]), (6, 72))
        self.assertEqual(counts["processed_rows"], 12 * sum(size for _, size in p.cells(cfg)))

    def test_pilot_search_still_requires_full_reference_cells_and_blocks(self):
        data = {"custom": rows(31)}
        baseline = p.schedule(config(), data)
        main = [j for j in baseline if j["phase"] == "baseline"]
        pilot = config(mode="search", pilot=True)
        references = [p.schedule(config(pilot=True), data), main[:1], main[:2],
                      main + [main[0]], main + [{**main[1], "block": 3}],
                      main + [{**main[1], "bucket": "unrelated"}],
                      main + [{**main[1], "batch_size": 24}]]
        truncated = copy.deepcopy(main)
        for job in truncated:
            job["batches"] = job["batches"][:12]
            job["ids"] = [key for batch in job["batches"] for key in batch["ids"]]
            job["plan_sha256"] = p.plan_hash(job["batches"])
        references.append(truncated)
        for number, reference in enumerate(references):
            with self.subTest(reference=number), self.assertRaises(ValueError):
                p.schedule(pilot, data, reference)

    def test_pilot_search_validates_prefix_and_full_reference_hashes(self):
        data = {"custom": rows(31)}
        main = [j for j in p.schedule(config(), data) if j["phase"] == "baseline"]
        pilot = config(mode="search", pilot=True)
        for block in range(3):
            for field, value in (("plan_sha256", "0" * 64), ("ids", [])):
                changed = copy.deepcopy(main)
                changed[block][field] = value
                with self.subTest(block=block, field=field), self.assertRaises(ValueError):
                    p.schedule(pilot, data, changed)
        changed = copy.deepcopy(main)
        changed[0]["batches"][0] = copy.deepcopy(changed[0]["batches"][1])
        changed[0]["ids"] = [key for batch in changed[0]["batches"] for key in batch["ids"]]
        changed[0]["plan_sha256"] = p.plan_hash(changed[0]["batches"])
        with self.assertRaises(ValueError):
            p.schedule(pilot, data, changed)
        with self.assertRaises(ValueError):
            p.schedule({**pilot, "seed": 123}, data, main)

    def test_confirm_pairs_and_copy_isolation(self):
        jobs = p.schedule(config(mode="confirm"), {"custom": rows(9)})
        self.assertEqual(Counter(j["phase"] for j in jobs), {"aa": 4, "ab": 6})
        for a, b in zip(jobs[::2], jobs[1::2]):
            self.assertEqual((a["side"], b["side"]), ("a", "b") if a["block"] % 2 == 0 else ("b", "a"))
            self.assertEqual(a["batches"], b["batches"])
            self.assertIsNot(a["batches"][0], b["batches"][0])
            for job in (a, b):
                expected = "candidate" if job["phase"] == "ab" and job["side"] == "b" else "baseline"
                self.assertEqual(job["variant"], expected)

    def test_pilot_and_non_latency(self):
        for mode in ("baseline", "search", "confirm"):
            jobs = p.schedule(config((24,), mode, True), {"custom": rows(24)})
            self.assertEqual(len(jobs), 2 if mode == "confirm" else 1)
            self.assertTrue(all(j["phase"] != "aa" and len(j["batches"]) == 12 for j in jobs))
        for mode in ("smoke", "quality"):
            self.assertEqual(p.schedule({"mode": mode}, {}), [])

    def test_reproducible_private_free_and_no_mutation(self):
        cfg, data = config(), {"custom": rows(29)}
        before, state = copy.deepcopy(data), random.getstate()
        result = p.schedule(cfg, data)
        self.assertEqual(result, p.schedule(cfg, {"custom": list(reversed(data["custom"]))}))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(data, before)
        encoded = json.dumps(result)
        self.assertEqual(json.loads(encoded), result)
        self.assertNotIn("PRIVATE", encoded)
        self.assertNotIn("messages", encoded)

    def test_bad_inputs(self):
        data = {"custom": rows(9)}
        for field, value in (("seed", "17"), ("seed", True), ("seed", 17.0),
                             ("pilot", 1), ("mode", "screening")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                p.schedule({**config(), field: value}, data)
        for value in ({}, {"unknown": rows(9)}, {"custom": rows(2)},
                      {"custom": rows(3) * 2}, {"custom": rows(4097)}):
            with self.subTest(keys=list(value)), self.assertRaises(ValueError):
                p.schedule(config(), value)
        cfg = config()
        cfg["buckets"]["other"] = {"min": 1, "max": 10}
        cfg["batch_sizes_by_bucket"]["other"] = [1]
        with self.assertRaises(ValueError):
            p.schedule(cfg, {**data, "other": data["custom"]})

    def test_workload_counts_rows_not_batches(self):
        result = p.workload(config((24,), pilot=True), {"custom": rows(25)})
        self.assertEqual(result, {"scope": "timed_jobs_only", "rows": 25,
                                  "rows_by_bucket": {"custom": 25}, "jobs": 1,
                                  "batches": 12, "processed_rows": 288})
        self.assertNotIn("PRIVATE", json.dumps(result))


class BatchTests(unittest.TestCase):
    def test_smoke_profile_and_legacy_shape_hash(self):
        source = rows()
        for size in (1, 3, 24, 256, 257):
            short, long, mixed = p.smoke_plans(source, size)
            self.assertEqual(short["total_tokens"], list(range(1, size + 1)))
            self.assertEqual(long["total_tokens"], list(range(258 - size, 258)))
            for plan in (short, long, mixed):
                self.assertEqual(len(set(plan["ids"])), size)
            self.assertEqual(p.profile_plans(source, size), [short, long])
        self.assertEqual(p.smoke_plans(source, 4), legacy.smoke_plans(source, 4))

    def test_materialize_original_objects_and_authenticated_fields(self):
        source = rows(9)
        plan = p.smoke_plans(source, 3)[0]
        self.assertIs(p.materialize(plan, source)[0], source[0])
        for field, value in (("id", "changed"), ("padded_length", 100),
                             ("occurrences", [False, 0, 0]), ("ids", ["missing"] * 3),
                             ("total_tokens", [True, 2, 3]), ("unexpected", "PRIVATE")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.materialize({**plan, field: value}, source)
        source[0]["index"]["input_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            p.materialize(plan, source)

    def test_row_and_size_failures(self):
        for size in (True, 3.0, "3", 0, -1, 10):
            with self.subTest(size=size), self.assertRaises(ValueError):
                p.smoke_plans(rows(9), size)
        for source in ([], [None], [{"id": "x"}], rows(1) * 2):
            with self.assertRaises(ValueError):
                p.smoke_plans(source, 1)
        for field, value in (("total_tokens", True), ("total_tokens", 1.0),
                             ("input_sha256", "x" * 64)):
            source = rows(1)
            source[0]["index"][field] = value
            with self.assertRaises(ValueError):
                p.smoke_plans(source, 1)


if __name__ == "__main__":
    unittest.main()
