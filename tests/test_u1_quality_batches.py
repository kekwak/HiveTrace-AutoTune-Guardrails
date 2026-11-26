"""CPU-only quality batch contracts; fake scores are not model quality evidence."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from u1_benchmark import quality_batches
from u1_benchmark.contracts import parse_config
from u1_benchmark.core import digest, lines, read, sha, write
from u1_benchmark.messages import UnsupportedInput, payload
from u1_benchmark.precision import PrecisionError


MAPPING = {"targets": {"benign": "negative", "harmful": "positive"},
           "verdicts": {"safe": "negative", "unsafe": "positive"},
           "labels": ["negative", "positive"], "positive_label": "positive"}
CFG = {"seed": 17, "label_mapping": MAPPING}


def quality_rows(count=7):
    rows = []
    for i in range(count):
        text = f"PRIVATE quality input {i}: never persist this content"
        row = {"id": f"q{i}", "target": "harmful" if i % 2 else "benign",
               "text_sha256": digest(text), "slices": {"language": "en"}}
        if i % 2:
            row.update(messages=[{"role": "user", "content": "PRIVATE user context"},
                                 {"role": "assistant", "content": text}],
                       target_role="assistant", target_index=1)
            row["content_sha256"] = digest(row["messages"])
        else:
            row["text"] = text
        rows.append(row)
    return rows


class PassthroughPrecision:
    def __init__(self):
        self.calls = []
        self.active = False

    def call(self, function, *args, **kwargs):
        self.calls.append(function.__name__)
        self.active = True
        try:
            return function(*args, **kwargs)
        finally:
            self.active = False


class OutOfMemoryError(RuntimeError):
    pass


class UnsupportedBatch(RuntimeError):
    pass


class UnsupportedExperiment(RuntimeError):
    pass


class QualityAdapter:
    """Already batch-capable adapter; batch inference never delegates to singles."""
    limit = 32

    def __init__(self, rows, precision, candidate=False):
        self.precision = precision
        self.rows = {digest(payload(row)): row for row in rows}
        self.lengths = {row["id"]: i + 1 for i, row in enumerate(rows)}
        self.metadata = {"backend": "cpu-fixture", "quantized": candidate}
        self.candidate = candidate
        self.calls = []
        self.index_errors = {}
        self.index_overrides = {}
        self.fail_batch = None
        self.backend_error = None
        self.batch_count = 0

    def record(self, name, inputs):
        if not self.precision.active:
            raise AssertionError(f"{name} bypassed precision.call")
        self.calls.append((name, copy.deepcopy(inputs)))

    def index(self, value):
        self.record("index", value)
        row = self.rows[digest(value)]
        if row["id"] in self.index_errors:
            raise self.index_errors[row["id"]]
        index = {"user_tokens": 1, "total_tokens": self.lengths[row["id"]],
                 "input_sha256": digest(value), "ids_sha256": digest([1, 2]),
                 "mask_sha256": digest([1, 1]), "target_role": row.get("target_role", "user"),
                 "target_index": row.get("target_index", 0), "context_indices": [],
                 "source_mode": "user+assistant" if "messages" in row else "user-only",
                 "source_sha256": digest(value), "rendered_prompt": value,
                 "input_ids": [123456789], "attention_mask": [1]}
        return {**index, **self.index_overrides.get(row["id"], {})}

    def prepare_backend(self, inputs):
        self.record("prepare_backend", inputs)

    def prove_backend(self, value):
        self.record("prove_backend", value)

    def prepare_batch_backend(self, groups):
        self.record("prepare_batch_backend", groups)
        if self.backend_error:
            raise self.backend_error

    def prove_batch_backend(self, inputs):
        self.record("prove_batch_backend", inputs)

    def score(self, value, size):
        unsafe = self.rows[digest(value)]["target"] == "harmful"
        if self.candidate and size > 1:
            unsafe = not unsafe
        return {"label": "unsafe" if unsafe else "safe"}

    def check_expected(self, values, expected):
        if len(values) != len(expected):
            raise AssertionError("Index count changed")
        for value, index in zip(values, expected, strict=True):
            if index["input_sha256"] != digest(value):
                raise AssertionError("Index/input order changed")

    def verify(self, value, expected):
        self.record("verify", value)
        self.check_expected([value], [expected])
        return self.score(value, 1)

    def verify_batch(self, inputs, expected):
        self.record("verify_batch", inputs)
        self.check_expected(inputs, expected)
        self.batch_count += 1
        if self.fail_batch == self.batch_count:
            raise OutOfMemoryError("PRIVATE failure detail")
        return [self.score(value, len(inputs)) for value in inputs]


class QualityBatchConfigTests(unittest.TestCase):
    def test_default_and_arbitrary_positive_sizes_preserve_order(self):
        self.assertEqual(quality_batches.batch_sizes({}), [1])
        self.assertEqual(quality_batches.batch_sizes({"quality_batch_sizes": [37, 3, 1, 1025]}),
                         [37, 3, 1, 1025])

    def test_invalid_sizes_rejected_by_helper_and_config_parser(self):
        for sizes in (None, [], (), (1, 3), "1", 1, {}, [0], [-1], [1, 1],
                      [True], [False], [1, True], [1.0], ["3"], [None], [[1]]):
            with self.subTest(sizes=sizes):
                for validate in (quality_batches.batch_sizes, parse_config):
                    with self.assertRaisesRegex(ValueError, "quality_batch_sizes"):
                        validate({"schema_version": 3, "model": "gliner",
                                  "quality_batch_sizes": sizes})

    def test_parse_quality_config_accepts_non_speed_grid_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "manifest.json", {"fixture": True})
            cfg = parse_config({"schema_version": 3, "model": "gliner", "mode": "quality",
                                "environment": str(root / "env"), "baseline": {"backend": "eager"},
                                "quality_data": {"path": tmp, "manifest_sha256": sha(root / "manifest.json")},
                                "quality_batch_sizes": [3, 37, 1], "label_mapping": MAPPING})
            self.assertEqual(quality_batches.batch_sizes(cfg), [3, 37, 1])


class QualityPlanTests(unittest.TestCase):
    def test_order_is_seeded_without_mutating_rows_or_global_random(self):
        rows = quality_rows(20)
        original, state = copy.deepcopy(rows), random.getstate()
        expected = [row["id"] for row in rows]
        random.Random(17).shuffle(expected)
        self.assertEqual(quality_batches.ordered_ids(rows, 17), expected)
        self.assertEqual(quality_batches.ordered_ids(rows, 17), expected)
        self.assertNotEqual(quality_batches.ordered_ids(rows, 42), expected)
        self.assertEqual(rows, original)
        self.assertEqual(random.getstate(), state)

    def test_duplicate_ids_rejected(self):
        rows = quality_rows(1) * 2
        with self.assertRaisesRegex(ValueError, "Duplicate quality IDs"):
            quality_batches.ordered_ids(rows, 17)
        precision = PassthroughPrecision()
        with self.assertRaisesRegex(ValueError, "Duplicate quality IDs"):
            quality_batches.prepare_plan(QualityAdapter(rows, precision), rows, 3, 17, precision)

    def test_deterministic_plan_keeps_tail_and_evidence_hashes(self):
        rows, precision = quality_rows(), PassthroughPrecision()
        adapter = QualityAdapter(rows, precision)
        plan = quality_batches.prepare_plan(adapter, rows, 3, 17, precision)
        self.assertEqual(plan, quality_batches.prepare_plan(adapter, rows, 3, 17, precision))
        self.assertEqual([b["actual_batch_size"] for b in plan["batches"]], [3, 3, 1])
        self.assertEqual([key for b in plan["batches"] for key in b["ids"]], plan["ordered_ids"])
        self.assertEqual(plan["ordered_ids"], quality_batches.ordered_ids(rows, 17))
        self.assertEqual(plan["excluded"], {})
        for batch in plan["batches"]:
            expected = [plan["indices"][key] for key in batch["ids"]]
            self.assertEqual(batch["lengths"], [ix["total_tokens"] for ix in expected])
            self.assertEqual(batch["padded_length"], max(batch["lengths"]))
            self.assertEqual(batch["inputs_sha256"], digest(expected))
        self.assertTrue(all(name == "index" for name in precision.calls))

    def test_exclusions_remove_only_ineligible_rows_and_keep_order(self):
        rows, precision = quality_rows(9), PassthroughPrecision()
        adapter = QualityAdapter(rows, precision)
        adapter.index_overrides = {"q0": {"total_tokens": 0}, "q1": {"target_role": "user"},
                                   "q2": {"total_tokens": 33}, "q6": {"total_tokens": 32}}
        adapter.index_errors = {"q3": UnsupportedInput("PRIVATE unsupported input"),
                                "q4": ValueError("PRIVATE indexing failure")}
        plan = quality_batches.prepare_plan(adapter, rows, 3, 17, precision)
        self.assertEqual(plan["excluded"], {"q0": "over_limit", "q1": "target_mismatch",
                                            "q2": "over_limit", "q3": "not_applicable", "q4": "error"})
        eligible = [key for key in plan["ordered_ids"] if key not in plan["excluded"]]
        self.assertEqual([key for b in plan["batches"] for key in b["ids"]], eligible)
        self.assertEqual([b["actual_batch_size"] for b in plan["batches"]], [3, 1])
        self.assertIn("q6", eligible)

    def test_precision_error_is_not_downgraded_to_row_exclusion(self):
        rows, precision = quality_rows(), PassthroughPrecision()
        adapter = QualityAdapter(rows, precision)
        adapter.index_errors["q0"] = PrecisionError("precision changed")
        with self.assertRaises(PrecisionError):
            quality_batches.prepare_plan(adapter, rows, 3, 17, precision)


class QualityEvaluationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.rows, self.precision = quality_rows(), PassthroughPrecision()
        self.adapter = QualityAdapter(self.rows, self.precision)
        self.result = {}

    def evaluate(self, size=3):
        quality_batches.evaluate(self.adapter, self.rows, CFG, size, self.root,
                                 self.result, self.precision)
        return lines(self.root / "predictions.jsonl")

    def test_size_one_uses_single_prepare_proof_and_verify(self):
        predictions = self.evaluate(1)
        self.assertEqual(self.precision.calls, ["index"] * 7 + ["prepare_backend", "prove_backend"] + ["verify"] * 7)
        self.assertEqual(len(predictions), 7)
        self.assertTrue(all(p["status"] == "passed" for p in predictions))
        self.assertEqual(self.result["quality"]["overall"]["accuracy"], 1)

    def test_batch_path_uses_no_single_calls_even_for_tail(self):
        with patch("u1_benchmark.batch_adapters.wrap", side_effect=AssertionError("already wrapped")), \
                patch.object(self.adapter, "verify", side_effect=AssertionError("single fallback")):
            predictions = self.evaluate()
        self.assertEqual(self.precision.calls, ["index"] * 7 + ["prepare_batch_backend", "prove_batch_backend"] + ["verify_batch"] * 3)
        calls = [inputs for name, inputs in self.adapter.calls if name == "verify_batch"]
        self.assertEqual([len(inputs) for inputs in calls], [3, 3, 1])
        plan = read(self.root / "quality_plan.json")
        self.assertEqual([p["id"] for p in predictions], plan["ordered_ids"])
        by_id = {row["id"]: row for row in self.rows}
        self.assertEqual(calls, [[payload(by_id[key]) for key in batch["ids"]] for batch in plan["batches"]])
        self.assertEqual(self.result["quality_plan_sha256"], digest(plan))
        self.assertEqual(self.result["quality_batch_size"], 3)
        self.assertTrue(self.result["quality_evaluated"])
        self.assertIsNone(self.result["quality_preserved"])
        self.assertEqual(self.result["prepared_shapes"], sorted({(b["actual_batch_size"], b["padded_length"]) for b in plan["batches"]}))

    def test_quantization_style_candidate_outputs_change_by_actual_size(self):
        results, predictions = {}, {}

        def launch(phase, variant, directory, measurement):
            self.assertEqual(phase, "quality")
            precision = PassthroughPrecision()
            adapter = QualityAdapter(self.rows, precision, candidate=variant == "candidate")
            result = {"status": "passed"}
            size = measurement["quality_batch_size"]
            quality_batches.evaluate(adapter, self.rows, CFG, size, directory, result, precision)
            write(directory / "result.json", result)
            results[size, variant] = result
            predictions[size, variant] = lines(directory / "predictions.jsonl")

        quality_batches.execute_matrix({**CFG, "comparison": "quantization", "candidate": {"fixture": True},
                                        "quality_batch_sizes": [1, 3]}, self.root, launch)
        for key in ((1, "baseline"), (3, "baseline"), (1, "candidate")):
            self.assertEqual(results[key]["quality"]["overall"]["accuracy"], 1)
        self.assertAlmostEqual(results[3, "candidate"]["quality"]["overall"]["accuracy"], 1 / 7)
        self.assertNotEqual(predictions[1, "candidate"], predictions[3, "candidate"])
        self.assertTrue(all(result["quality_preserved"] is None for result in results.values()))

    def test_oom_saves_completed_failed_and_unrun_rows_before_reraising(self):
        self.adapter.fail_batch = 2
        with self.assertRaises(OutOfMemoryError):
            self.evaluate()
        predictions = lines(self.root / "predictions.jsonl")
        self.assertEqual([p["status"] for p in predictions], ["passed"] * 3 + ["oom"] * 3 + ["not_run"])
        self.assertTrue(all(p.get("error_type") == "OutOfMemoryError" for p in predictions[3:6]))
        self.assertTrue(all("score" not in p for p in predictions[3:]))
        self.assertEqual(self.result["quality"]["overall"]["evaluated"], 3)
        self.assertEqual(self.result["quality"]["overall"]["failures"], {"oom": 3, "not_run": 1})
        self.assertNotIn("PRIVATE", (self.root / "predictions.jsonl").read_text())

    def test_unsupported_backend_does_not_fall_back_to_single_inference(self):
        for error in (UnsupportedBatch, UnsupportedExperiment):
            with self.subTest(error=error.__name__):
                self.adapter.backend_error = error("PRIVATE backend detail")
                with self.assertRaises(error):
                    self.evaluate()
                self.assertTrue(all(p["status"] == "not_run" for p in lines(self.root / "predictions.jsonl")))
                self.assertFalse(self.result["quality_evaluated"])
                self.assertFalse(any(name in ("verify", "verify_batch", "prepare_backend") for name, _ in self.adapter.calls))

    def test_invalid_batch_response_is_not_recorded_as_success(self):
        for responses in ([], [{"label": "safe"}] * 2, [{"label": "safe", "text": "PRIVATE output"}] * 3):
            with self.subTest(responses=responses):
                with patch.object(self.adapter, "verify_batch", return_value=responses) as verify:
                    verify.__name__ = "verify_batch"
                    with self.assertRaises(ValueError):
                        self.evaluate()
                predictions = lines(self.root / "predictions.jsonl")
                self.assertEqual([p["status"] for p in predictions], ["error"] * 3 + ["not_run"] * 4)
                self.assertFalse(any("score" in p for p in predictions))

    def test_all_excluded_skips_backend_and_inference(self):
        self.adapter.index_errors = {row["id"]: UnsupportedInput("PRIVATE") for row in self.rows}
        predictions = self.evaluate()
        self.assertTrue(all(p["status"] == "not_applicable" for p in predictions))
        self.assertEqual(self.precision.calls, ["index"] * len(self.rows))
        self.assertEqual(read(self.root / "quality_plan.json")["batches"], [])
        self.assertFalse(self.result["quality_evaluated"])

    def test_private_text_and_tokenized_payloads_are_not_saved(self):
        self.evaluate()
        write(self.root / "result.json", self.result)
        for path in self.root.iterdir():
            content = path.read_text()
            for forbidden in ("PRIVATE", "rendered_prompt", "input_ids", "attention_mask", "123456789"):
                self.assertNotIn(forbidden, content, path.name)
        for prediction in lines(self.root / "predictions.jsonl"):
            row = next(row for row in self.rows if row["id"] == prediction["id"])
            self.assertEqual(prediction["text_sha256"], row["text_sha256"])
            if "content_sha256" in row:
                self.assertEqual(prediction["content_sha256"], row["content_sha256"])


class QualityMatrixTests(unittest.TestCase):
    def test_truncated_failed_result_does_not_abort_later_matrix_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls = Path(tmp), []

            def launch(phase, variant, directory, measurement):
                size = measurement["quality_batch_size"]
                calls.append(size)
                if size == 3:
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "result.json").write_text('{"status":')
                    raise RuntimeError("worker interrupted while writing result")
                write(directory / "result.json", {"status": "passed"})

            state = quality_batches.execute_matrix({"quality_batch_sizes": [1, 3, 37]}, root, launch)
            self.assertEqual(calls, [1, 3, 37])
            self.assertEqual(state["3/baseline"]["status"], "failed")
            self.assertEqual(state["37/baseline"]["status"], "passed")
            self.assertEqual(read(root / "quality_status.json"), state)

    def test_separate_launch_for_every_size_variant_and_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls = Path(tmp), []

            def launch(phase, variant, directory, measurement):
                calls.append((phase, variant, directory, measurement))
                write(directory / "result.json", {"status": "passed"})

            state = quality_batches.execute_matrix({"candidate": {"fixture": True}, "quality_batch_sizes": [3, 1, 37]}, root, launch)
            self.assertEqual([(p, v, d, m) for p, v, d, m in calls],
                             [("quality", variant, root / "quality" / str(size) / variant,
                               {"quality_batch_size": size}) for size in (3, 1, 37) for variant in ("baseline", "candidate")])
            self.assertEqual(len({id(c[3]) for c in calls}), 6)
            self.assertEqual(state, {f"{size}/{variant}": {"status": "passed"} for size in (3, 1, 37) for variant in ("baseline", "candidate")})
            self.assertEqual(read(root / "quality_status.json"), state)

    def test_missing_sizes_defaults_to_single_baseline_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def launch(phase, variant, directory, measurement):
                calls.append((phase, variant, measurement))
                write(directory / "result.json", {"status": "passed"})

            quality_batches.execute_matrix({}, Path(tmp), launch)
            self.assertEqual(calls, [("quality", "baseline", {"quality_batch_size": 1})])

    def test_failed_cell_does_not_skip_other_variants_or_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def launch(phase, variant, directory, measurement):
                key = measurement["quality_batch_size"], variant
                calls.append(key)
                if key == (3, "baseline"):
                    raise RuntimeError("failed cell")
                write(directory / "result.json", {"status": "unsupported" if key == (1, "candidate") else "passed"})

            state = quality_batches.execute_matrix({"quality_batch_sizes": [3, 1], "candidate": {"fixture": True}}, Path(tmp), launch)
            self.assertEqual(calls, [(3, "baseline"), (3, "candidate"), (1, "baseline"), (1, "candidate")])
            self.assertEqual(state["3/baseline"], {"status": "failed", "error_type": "RuntimeError"})
            self.assertEqual(state["1/candidate"]["status"], "unsupported")
            self.assertEqual(state["3/candidate"]["status"], "passed")

    def test_timeout_saves_partial_matrix_and_stops_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls = Path(tmp), []

            def launch(phase, variant, directory, measurement):
                calls.append(measurement["quality_batch_size"])
                if len(calls) == 2:
                    raise TimeoutError("deadline")
                write(directory / "result.json", {"status": "passed"})

            with self.assertRaises(TimeoutError):
                quality_batches.execute_matrix({"quality_batch_sizes": [1, 3, 37]}, root, launch)
            self.assertEqual(calls, [1, 3])
            self.assertEqual(read(root / "quality_status.json"), {
                "1/baseline": {"status": "passed"},
                "3/baseline": {"status": "timeout", "error_type": "TimeoutError"}})


if __name__ == "__main__":
    unittest.main()
