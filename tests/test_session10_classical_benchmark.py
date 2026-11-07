"""Artificial benchmark isolation, measurements, selection and immutable output."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session10_classical_benchmark import METHODS, ROOT, make_scenes, pipelines, readiness, run, score, summarize


class BenchmarkTests(unittest.TestCase):
    def test_deterministic_fixtures_and_groups(self):
        first, second = make_scenes(), make_scenes()
        self.assertEqual(len(first), 10)
        for a, b in zip(first, second):
            for key in ("gray", "truth", "roi", "thin"):
                np.testing.assert_array_equal(a[key], b[key])
        development = {s["source_group"] for s in first if s["role"] == "development"}
        validation = {s["source_group"] for s in first if s["role"] == "validation"}
        self.assertFalse(development & validation)
        for group in development | validation:
            pair = [s for s in first if s["source_group"] == group]
            self.assertEqual(len(pair), 2)
            np.testing.assert_array_equal(pair[0]["truth"], pair[1]["truth"])

    def test_pipeline_preservation_and_binary_outputs(self):
        gray = make_scenes()[0]["gray"]
        before = gray.copy()
        result = pipelines(gray)
        self.assertEqual(tuple(result), METHODS)
        np.testing.assert_array_equal(gray, before)
        for mask in result.values():
            self.assertEqual(mask.shape, gray.shape)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)) <= {0, 1})

    def test_roi_metric_arithmetic(self):
        truth = np.array([[1, 1], [0, 0]], np.uint8)
        prediction = np.array([[1, 0], [1, 1]], np.uint8)
        roi = np.array([[True, True], [True, False]])
        result = score(truth, prediction, roi, np.zeros_like(truth))
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 0))
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertEqual(result["signed_area_error_pp"], 0)
        self.assertIsNone(result["thin_feature_recall"])

    def test_thin_feature_loss(self):
        scene = make_scenes()[0]
        predictions = pipelines(scene["gray"])
        before = score(scene["truth"], predictions["Otsu"], scene["roi"], scene["thin"])
        after = score(scene["truth"], predictions["Otsu + opening"], scene["roi"], scene["thin"])
        self.assertEqual(before["thin_feature_recall"], 1)
        self.assertLess(after["thin_feature_recall"], 1)

    def test_invalid_inputs(self):
        scene = make_scenes()[0]
        with self.assertRaises(ValueError):
            score(scene["truth"], scene["truth"], np.zeros(scene["roi"].shape, bool), scene["thin"])
        with self.assertRaises(ValueError):
            score(scene["truth"], scene["truth"].astype(float), scene["roi"], scene["thin"])

    def test_selection_reproducible_and_validation_only(self):
        rows = []
        for scene in make_scenes():
            for method, prediction in pipelines(scene["gray"]).items():
                rows.append({"scene_id": scene["scene_id"], "role": scene["role"], "method": method,
                             **score(scene["truth"], prediction, scene["roi"], scene["thin"])})
        summary, winner = summarize(rows)
        self.assertEqual(winner, summarize(list(reversed(rows)))[1])
        altered = [dict(row, iou=0 if row["method"] == winner else 1) if row["role"] == "development" else row for row in rows]
        self.assertEqual(winner, summarize(altered)[1])
        self.assertNotEqual(summary["validation"][winner]["scene_macro_iou"], summary["validation"][winner]["pooled_foreground_iou"])

    def test_test_only_manifest_rejected_without_reading_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            path.write_text("sample_id,split,source_group_id\na,test,UNKNOWN\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test-only"):
                readiness(path)

    def test_output_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "existing"
            output.mkdir()
            report = output / "session10_summary.json"
            report.write_text("original")
            with patch("session10_classical_benchmark.make_scenes", side_effect=AssertionError("Should reject first")):
                with self.assertRaises(FileExistsError):
                    run(output)
            self.assertEqual(report.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
