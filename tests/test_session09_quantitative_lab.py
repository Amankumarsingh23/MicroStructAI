"""Measurement denominators, truncation, scale and segmentation error propagation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session09_quantitative_lab import ROOT, area_fraction, make_synthetic, measure_objects, measurement_bias, read_gray, run


class QuantitativeTests(unittest.TestCase):
    def test_valid_denominator_and_empty_roi(self):
        labels = np.array([[1, 1], [0, 0]], np.int32)
        roi = np.array([[True, False], [True, True]])
        self.assertAlmostEqual(area_fraction(labels, roi), 1 / 3)
        with self.assertRaises(ValueError):
            area_fraction(labels, np.zeros_like(roi))
        with self.assertRaises(ValueError):
            area_fraction(labels, np.ones(labels.shape, np.uint8))

    def test_equivalent_diameter_exact(self):
        labels = np.ones((10, 10), np.int32)
        feature = measure_objects(labels, np.ones(labels.shape, bool))[0]
        self.assertAlmostEqual(feature["equivalent_diameter_px"], np.sqrt(400 / np.pi))
        self.assertEqual(feature["area_px2"], 100)

    def test_linear_length_quadratic_area_scaling(self):
        labels = np.ones((10, 10), np.int32)
        roi = np.ones(labels.shape, bool)
        first = measure_objects(labels, roi, .25)[0]
        second = measure_objects(labels, roi, .5)[0]
        self.assertEqual(first["hypothetical_area_um2"], 6.25)
        self.assertEqual(second["hypothetical_area_um2"], 4 * first["hypothetical_area_um2"])
        self.assertEqual(second["hypothetical_equivalent_diameter_um"], 2 * first["hypothetical_equivalent_diameter_um"])
        self.assertNotIn("hypothetical_area_um2", measure_objects(labels, roi)[0])
        for scale in (-1, 0, float("nan")):
            with self.assertRaises(ValueError):
                measure_objects(labels, roi, scale)

    def test_border_flag(self):
        _, labels, roi = make_synthetic()
        features = measure_objects(labels, roi)
        self.assertEqual([r["instance_id"] for r in features if r["touches_image_border"]], [4])

    def test_roi_cut_and_boundary_flags(self):
        labels = np.zeros((10, 10), np.int32)
        labels[2:8, 2:8] = 1
        roi = np.ones(labels.shape, bool)
        roi[:, 5] = False
        feature = measure_objects(labels, roi)[0]
        self.assertTrue(feature["touches_roi_boundary"])
        self.assertTrue(feature["clipped_by_roi"])
        self.assertEqual(feature["area_px2"], 30)
        self.assertEqual(feature["observed_fragments_8"], 2)

    def test_dilation_measurement_bias_without_count_change(self):
        import cv2
        from scipy import ndimage as ndi
        _, truth, roi = make_synthetic()
        mask = cv2.dilate((truth > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
        prediction = ndi.label(mask, np.ones((3, 3)))[0]
        result = measurement_bias(truth, prediction, roi)
        self.assertEqual(result["truth_object_count"], 4)
        self.assertEqual(result["prediction_object_count"], 4)
        self.assertGreater(result["signed_fraction_error"], 0)
        self.assertEqual(result["semantic_metrics"]["per_class"]["1"]["fn"], 0)
        self.assertTrue(all(row["signed_equivalent_diameter_error_px"] > 0 for row in result["matched_measurement_errors"]))

    def test_reproducibility_and_input_preservation(self):
        first, labels, roi = make_synthetic()
        second, other_labels, other_roi = make_synthetic()
        np.testing.assert_array_equal(first, second)
        before, before_roi = labels.copy(), roi.copy()
        measure_objects(labels, roi)
        np.testing.assert_array_equal(labels, before)
        np.testing.assert_array_equal(roi, before_roi)
        np.testing.assert_array_equal(labels, other_labels)
        np.testing.assert_array_equal(roi, other_roi)

    def test_source_safeguards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "16bit.png"
            Image.fromarray(np.zeros((32, 32), np.uint16)).save(path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                read_gray(path)
            self.assertEqual(before, path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Reserved"):
            run(ROOT / "data/raw/figshare_orientation/test/images/1.png")


if __name__ == "__main__":
    unittest.main()
