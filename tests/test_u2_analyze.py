"""Offline paired analysis of U1 quality predictions."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from u2_quant import analyze


def rows(n, target="unsafe", technique="t", start=0):
    return [{"id": f"r{start + i}", "target": target, "slices": {"technique": technique}} for i in range(n)]


def preds(items, label, scores=None):
    out = {}
    for row in items:
        score = {"label": label}
        if scores is not None:
            score.update(scores=scores(row), score_type="probabilities")
        out[row["id"]] = {"id": row["id"], "status": "passed", "score": score}
    return out


class Compare(unittest.TestCase):
    def test_identical_predictions(self):
        data = rows(50)
        table = analyze.compare(data, preds(data, "unsafe"), preds(data, "unsafe"))
        pooled = table.loc["pooled"]
        self.assertEqual(pooled["flips"], 0)
        self.assertEqual(pooled["p_mcnemar"], 1.0)
        self.assertAlmostEqual(pooled["zero_flip_bound"], 3 / 50)

    def test_known_worse_flips(self):
        data = rows(100)
        cand = preds(data, "unsafe")
        for row in data[:10]:
            cand[row["id"]]["score"]["label"] = "safe"
        pooled = analyze.compare(data, preds(data, "unsafe"), cand).loc["pooled"]
        self.assertEqual((pooled["worse"], pooled["better"]), (10, 0))
        self.assertLess(pooled["p_mcnemar"], 0.01)
        self.assertAlmostEqual(pooled["d_fnr"], 0.10)

    def test_directional_shift_detected_per_side(self):
        # FP8 сдвигает в сторону safe: на вредных — пропуски (хуже), на безопасных — меньше ложных тревог (лучше).
        harm, safe = rows(243), rows(243, target="safe", start=243)
        base, cand = preds(harm, "unsafe"), preds(harm, "unsafe")
        base.update(preds(safe, "unsafe")); cand.update(preds(safe, "unsafe"))
        for r in harm[:22]:
            cand[r["id"]]["score"]["label"] = "safe"
        for r in safe[:16]:
            cand[r["id"]]["score"]["label"] = "safe"
        pooled = analyze.compare(harm + safe, base, cand).loc["pooled"]
        self.assertGreater(pooled["p_mcnemar"], 0.05)          # смешанный тест сдвиг не видит
        self.assertEqual((pooled["fnr_worse"], pooled["fnr_better"]), (22, 0))
        self.assertLess(pooled["p_fnr"], 1e-5)
        self.assertEqual((pooled["fpr_worse"], pooled["fpr_better"]), (0, 16))
        self.assertLess(pooled["p_fpr"], 1e-3)

    def test_holm_per_side_monotone(self):
        data = rows(60, technique="a") + rows(60, technique="b", start=60) + rows(60, target="safe", technique="b", start=120)
        cand = preds(data, "unsafe")
        for row in data[:6] + data[60:62]:
            cand[row["id"]]["score"]["label"] = "safe"
        table = analyze.compare(data, preds(data, "unsafe"), cand).drop(index="pooled")
        for _, row in table.iterrows():
            self.assertGreaterEqual(row["p_holm_fnr"], row["p_fnr"])
            self.assertLessEqual(row["p_holm_fnr"], 1.0)

    def test_holm_monotone(self):
        data = rows(60, technique="a") + rows(60, technique="b", start=60) + rows(60, technique="c", start=120)
        cand = preds(data, "unsafe")
        for row in data[:6] + data[60:62]:
            cand[row["id"]]["score"]["label"] = "safe"
        table = analyze.compare(data, preds(data, "unsafe"), cand).drop(index="pooled")
        for _, row in table.iterrows():
            self.assertGreaterEqual(row["p_holm"], row["p_mcnemar"])
            self.assertLessEqual(row["p_holm"], 1.0)

    def test_mde_matches_readme(self):
        self.assertEqual(round(analyze.mde(243), 3), 0.040)
        self.assertEqual(round(analyze.mde(97), 3), 0.088)
        self.assertEqual(round(analyze.mde(944), 3), 0.015)

    def test_mde_separate_for_fnr_and_fpr(self):
        data = rows(243) + rows(97, target="safe", start=243)
        pooled = analyze.compare(data, preds(data, "unsafe"), preds(data, "unsafe")).loc["pooled"]
        self.assertEqual(round(pooled["mde_fnr"], 3), 0.040)
        self.assertEqual(round(pooled["mde_fpr"], 3), 0.088)

    def test_analyze_uses_common_ids_and_reports_coverage(self):
        data = rows(40)
        cand = preds(data, "unsafe")
        for row in data[:5]:
            del cand[row["id"]]
        pooled = analyze.compare(data, preds(data, "unsafe"), cand).loc["pooled"]
        self.assertEqual((pooled["n_total"], pooled["n_common"]), (40, 35))
        self.assertAlmostEqual(pooled["coverage"], 35 / 40)

    def test_analyze_counts_failures_separately(self):
        data = rows(30)
        cand = preds(data, "unsafe")
        cand[data[0]["id"]] = {"id": data[0]["id"], "status": "error"}
        pooled = analyze.compare(data, preds(data, "unsafe"), cand).loc["pooled"]
        self.assertEqual(pooled["failures_cand"], 1)
        self.assertEqual(pooled["flips"], 0)
        self.assertEqual(pooled["n_common"], 29)

    def test_auroc_only_with_scores(self):
        data = rows(20) + rows(20, target="safe", start=20)
        plain = analyze.compare(data, preds(data, "unsafe"), preds(data, "unsafe")).loc["pooled"]
        self.assertTrue(math.isnan(plain["auroc_base"]))
        score = lambda r: {"safe": 0.2, "unsafe": 0.8} if r["target"] == "unsafe" else {"safe": 0.7, "unsafe": 0.3}
        scored = analyze.compare(data, preds(data, "unsafe", score), preds(data, "unsafe", score)).loc["pooled"]
        self.assertAlmostEqual(scored["auroc_base"], 1.0)

    def test_main_writes_outputs(self):
        data = rows(10)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name, items in (("corpus.jsonl", data), ("base.jsonl", preds(data, "unsafe").values()),
                                ("cand.jsonl", preds(data, "safe").values())):
                (root / name).write_text("".join(json.dumps(x) + "\n" for x in items))
            analyze.main(["--corpus", str(root / "corpus.jsonl"), "--baseline", str(root / "base.jsonl"),
                          "--candidate", str(root / "cand.jsonl"), "--out", str(root / "out")])
            self.assertTrue((root / "out/analysis_technique.csv").exists())
            self.assertIn("pooled", (root / "out/analysis_technique.md").read_text())


if __name__ == "__main__":
    unittest.main()
