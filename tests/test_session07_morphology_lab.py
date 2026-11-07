"""Geometry checks and measurement safeguards for binary morphology."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session07_morphology_lab import components, make_synthetic, morphologies, read_gray, summarize


class MorphologyTests(unittest.TestCase):
    def test_mask_and_kernel_validation(self):
        for mask in (np.zeros((0, 3), np.uint8), np.full((3, 3), 255, np.uint8), np.ones((3, 3), float)):
            with self.assertRaises(ValueError):
                morphologies(mask)
        for kernel in (np.ones((2, 2), np.uint8), np.zeros((3, 3), np.uint8), np.ones((3, 3), float)):
            with self.assertRaises(ValueError):
                morphologies(np.ones((5, 5), np.uint8), kernel)

    def test_erosion_dilation_and_source_preservation(self):
        mask = np.zeros((9, 9), np.uint8)
        mask[3:6, 3:6] = 1
        before = mask.copy()
        results = morphologies(mask, np.ones((3, 3), np.uint8))
        self.assertEqual(int(results["Erosion"].sum()), 1)
        self.assertEqual(int(results["Dilation"].sum()), 25)
        np.testing.assert_array_equal(mask, before)
        for result in results.values():
            self.assertEqual(result.shape, mask.shape)
            self.assertTrue(set(np.unique(result)) <= {0, 1})

    def test_opening_removes_speck_and_closing_fills_hole(self):
        mask = np.zeros((21, 21), np.uint8)
        mask[5:16, 5:16] = 1
        mask[10, 10] = 0
        mask[1, 1] = 1
        results = morphologies(mask)
        self.assertEqual(int(results["Opening"][1, 1]), 0)
        self.assertEqual(int(results["Closing"][10, 10]), 1)

    def test_thin_line_lost_despite_global_improvement(self):
        truth, candidate, line = make_synthetic()
        other_truth, other_candidate, _ = make_synthetic()
        np.testing.assert_array_equal(truth, other_truth)
        np.testing.assert_array_equal(candidate, other_candidate)
        results = morphologies(candidate)
        before = summarize(candidate, truth, line)
        after = summarize(results["Opening"], truth, line)
        self.assertEqual(before["thin_feature_retained_pixels"], 40)
        self.assertEqual(after["thin_feature_retained_pixels"], 0)
        self.assertGreater(after["metrics"]["per_class"]["1"]["iou"], before["metrics"]["per_class"]["1"]["iou"])

    def test_connectivity_stats_and_empty_background(self):
        diagonal = np.eye(3, dtype=np.uint8)
        self.assertEqual(len(components(diagonal, 4)), 3)
        merged = components(diagonal, 8)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["area_px"], 3)
        self.assertEqual(merged[0]["bbox_xywh_px"], [0, 0, 3, 3])
        self.assertEqual(merged[0]["centroid_xy_px"], [1.0, 1.0])
        self.assertEqual(components(np.zeros((3, 3), np.uint8)), [])

    def test_metric_and_signed_area_error(self):
        truth = np.zeros((4, 4), np.uint8)
        truth[:2, :2] = 1
        prediction = truth.copy()
        prediction[0, 0] = 0
        result = summarize(prediction, truth)
        self.assertEqual(result["signed_area_error_px"], -1)
        self.assertEqual(result["metrics"]["per_class"]["1"]["iou"], .75)
        self.assertAlmostEqual(result["metrics"]["per_class"]["1"]["dice"], 6 / 7)

    def test_16bit_safeguard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "16bit.png"
            Image.fromarray(np.zeros((32, 32), np.uint16)).save(path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "8-bit"):
                read_gray(path)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
