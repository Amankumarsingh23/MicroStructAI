"""Threshold correctness, reproducibility and scientific input safeguards."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session06_threshold_lab import make_synthetic, otsu_numpy, read_gray, segment, run, ROOT
from session05_protocol import evaluate


class ThresholdTests(unittest.TestCase):
    def test_otsu_matches_opencv(self):
        scenes, _ = make_synthetic()
        for gray in scenes.values():
            threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            self.assertEqual(otsu_numpy(gray), int(threshold))
        # A score plateau chooses its first threshold.
        self.assertEqual(otsu_numpy(np.array([[55, 55, 195, 195]], dtype=np.uint8)), 55)

    def test_determinism_and_source_integrity(self):
        first, truth = make_synthetic()
        second, other_truth = make_synthetic()
        np.testing.assert_array_equal(truth, other_truth)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            before = first[name].copy()
            masks, _ = segment(first[name])
            again, _ = segment(first[name])
            np.testing.assert_array_equal(first[name], before)
            for method in masks:
                np.testing.assert_array_equal(masks[method], again[method])

    def test_binary_and_fixed_boundary_and_metrics(self):
        gray = np.array([[119, 120, 121], [200, 20, 50]], dtype=np.uint8)
        masks, _ = segment(gray)
        np.testing.assert_array_equal(masks["Fixed T=120"], [[1, 1, 0], [0, 1, 1]])
        for mask in masks.values():
            self.assertTrue(set(np.unique(mask)) <= {0, 1})
            self.assertEqual(mask.dtype, np.uint8)
        scenes, truth = make_synthetic()
        masks, _ = segment(scenes["even"])
        self.assertEqual(evaluate(truth, masks["Otsu"])["per_class"]["1"]["iou"], 1)

    def test_invalid_input_and_reserved_data(self):
        for gray in (np.zeros((4, 4), np.uint8), np.zeros((0, 4), np.uint8),
                     np.zeros((4, 4), np.uint16), np.zeros((4, 4, 3), np.uint8)):
            with self.assertRaises(ValueError):
                otsu_numpy(gray)
        scenes, _ = make_synthetic()
        for size in (1, 2, 4):
            with self.assertRaises(ValueError):
                segment(scenes["even"], block_size=size)
        with self.assertRaisesRegex(ValueError, "Reserved"):
            run(ROOT / "data/raw/figshare_orientation/test/images/1.png")

    def test_16bit_rejected_and_file_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "16bit.png"
            Image.fromarray(np.full((32, 32), 1024, dtype=np.uint16)).save(path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "8-bit"):
                read_gray(path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
