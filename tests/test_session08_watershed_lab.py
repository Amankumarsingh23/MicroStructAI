"""Known-instance separation, ID-invariant evaluation and source safeguards."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session08_watershed_lab import ROOT, distance_markers, instance_metrics, make_synthetic, read_gray, run, separate, statistics


class WatershedTests(unittest.TestCase):
    def test_deterministic_fixture(self):
        for first, second in zip(make_synthetic(), make_synthetic()):
            np.testing.assert_array_equal(first, second)

    def test_distance_and_markers(self):
        square = np.zeros((7, 7), np.uint8)
        square[1:6, 1:6] = 1
        distance, _ = distance_markers(square, threshold_abs=0)
        self.assertEqual(distance[3, 3], 3)
        self.assertTrue(np.all(distance[square == 0] == 0))
        _, mask, _ = make_synthetic()
        _, markers = distance_markers(mask)
        self.assertEqual(np.count_nonzero(markers), 3)
        self.assertTrue(np.all(mask[markers > 0] == 1))

    def test_touching_separation_and_boundaries(self):
        _, mask, truth = make_synthetic()
        before = mask.copy()
        result = separate(mask)
        summary = statistics(mask, result)
        self.assertEqual(summary["binary_components_8"], 2)
        self.assertEqual(summary["instance_count"], 3)
        self.assertGreater(instance_metrics(truth, result["labels"])["matched_mean_instance_iou"], .95)
        self.assertGreater(summary["watershed_line_pixels_inside_candidate"], 0)
        self.assertFalse(np.any(result["labels"][mask == 0]))
        np.testing.assert_array_equal(mask, before)

    def test_missing_marker_undersegments(self):
        _, mask, _ = make_synthetic()
        _, markers = distance_markers(mask)
        markers[125, 90] = 0
        result = separate(mask, markers)
        self.assertEqual(statistics(mask, result)["instance_count"], 2)

    def test_extra_marker_oversegments(self):
        _, mask, _ = make_synthetic()
        _, markers = distance_markers(mask)
        markers[125, 70] = 4
        result = separate(mask, markers)
        self.assertEqual(statistics(mask, result)["instance_count"], 4)

    def test_invalid_and_unseeded_masks(self):
        for mask in (np.ones((3, 3), np.uint8), np.full((3, 3), 255, np.uint8), np.zeros((3, 3), float)):
            with self.assertRaises(ValueError):
                separate(mask)
        empty = np.zeros((7, 7), np.uint8)
        self.assertEqual(statistics(empty, separate(empty))["instance_count"], 0)
        small = empty.copy()
        small[2:5, 2:5] = 1
        result = separate(small)  # No peak exceeds height 7.
        self.assertEqual(int(result["watershed_lines"].sum()), 0)
        self.assertEqual(int(result["unseeded_foreground"].sum()), 9)
        _, mask, _ = make_synthetic()
        markers = np.zeros(mask.shape, np.int32)
        markers[0, 0] = 1
        with self.assertRaises(ValueError):
            separate(mask, markers)

    def test_matching_id_invariance_and_unmatched_penalty(self):
        truth = np.array([[1, 1, 0, 2, 2]], dtype=np.int32)
        swapped = np.array([[8, 8, 0, 4, 4]], dtype=np.int32)
        self.assertEqual(instance_metrics(truth, swapped)["matched_mean_instance_iou"], 1)
        swapped[swapped == 4] = 0
        self.assertEqual(instance_metrics(truth, swapped)["matched_mean_instance_iou"], .5)
        self.assertEqual(instance_metrics(truth, swapped)["unmatched_truth"], 1)
        self.assertIsNone(instance_metrics(np.zeros_like(truth), np.zeros_like(truth))["matched_mean_instance_iou"])

    def test_source_and_16bit_safeguards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "16bit.png"
            Image.fromarray(np.zeros((32, 32), np.uint16)).save(path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                read_gray(path)
            self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "Reserved"):
            run(ROOT / "data/raw/figshare_orientation/test/images/1.png")


if __name__ == "__main__":
    unittest.main()
