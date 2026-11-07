"""Check numerical correctness and scientifically important failure cases."""

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session04_preprocessing import illumination_correct, mae, preprocess, read_gray, run, synthetic_experiment, to_u8
from session03_dataset_audit import ROOT


class PreprocessingTests(unittest.TestCase):
    def test_read_rejects_16bit_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "16bit.png"
            Image.fromarray(np.full((32, 32), 1024, dtype=np.uint16)).save(path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "8-bit L/RGB"):
                read_gray(path)
            self.assertEqual(before, path.read_bytes())

    def test_round_and_clip_without_uint8_wraparound(self):
        np.testing.assert_array_equal(to_u8([-20, 0, 1.6, 255, 300]), [0, 0, 2, 255, 255])
        self.assertEqual(mae(np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8)), 255)

    def test_source_arrays_preserved(self):
        pixels = np.arange(1024, dtype=np.uint8).reshape(32, 32)
        before = pixels.copy()
        for result in preprocess(pixels).values():
            self.assertEqual(result.shape, pixels.shape)
            self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(pixels, before)

    def test_illumination_zero_constant_and_bad_sigma(self):
        for value in (0, 100, 255):
            image = np.full((32, 32), value, dtype=np.uint8)
            np.testing.assert_array_equal(illumination_correct(image), image)
        with self.assertRaises(ValueError):
            illumination_correct(image, sigma=0)

    def test_reproducible_noise_reduction_can_damage_thin_feature(self):
        _, metrics, line_metrics = synthetic_experiment()
        _, repeated, _ = synthetic_experiment()
        self.assertEqual(metrics, repeated)
        noisy = "Mixed Gaussian + impulse noise"
        gaussian = "Gaussian-filtered noisy image"
        median = "Median-filtered noisy image"
        self.assertLess(metrics[gaussian], metrics[noisy])
        self.assertLess(metrics[median], metrics[noisy])
        self.assertLess(metrics["Heuristic corrected shading"], metrics["Known multiplicative shading"])
        # Lower global error does not imply faithful preservation of a small feature.
        self.assertGreater(line_metrics[gaussian], line_metrics[noisy])

    def test_median_can_erase_a_genuine_one_pixel_line(self):
        image = np.full((21, 21), 200, dtype=np.uint8)
        image[:, 10] = 50
        filtered = cv2.medianBlur(image, 3)
        self.assertEqual(int(filtered[10, 10]), 200)

    def test_reserved_test_data_and_output_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            run(ROOT / "data/raw/figshare_orientation/test/images/1.png")
        with self.assertRaisesRegex(ValueError, "output paths"):
            run(ROOT / "outputs/session04_synthetic.png")


if __name__ == "__main__":
    unittest.main()
