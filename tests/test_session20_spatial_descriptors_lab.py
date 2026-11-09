"""Eight focused tests for synthetic Session 20 spatial descriptors."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from session20_spatial_descriptors_lab import (
    validate_binary, component_table, validate_points, nearest_neighbor_summary,
    quadrat_summary, csr_monte_carlo, artificial_patterns, artificial_component_mask, run,
)


class SpatialDescriptorTests(unittest.TestCase):
    def test_binary_and_roi_validation(self):
        mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        roi = np.ones_like(mask, bool)
        validate_binary(mask, roi)
        with self.assertRaises(ValueError):
            validate_binary(np.array([[0, 9]]), np.ones((1, 2), bool))
        with self.assertRaises(ValueError):
            validate_binary(mask, np.zeros_like(mask, bool))
        with self.assertRaises(ValueError):
            validate_binary(mask.astype(float), roi)

    def test_components_count_centroid_and_boundary_flags(self):
        roi = np.ones((16, 16), dtype=bool)
        mask = np.zeros_like(roi, dtype=np.uint8)
        mask[4:6, 5:7] = 1  # 4 pixels, centroid 4.5, 5.5; interior
        mask[12:16, 12:16] = 1  # image-border censored
        result = component_table(mask, roi)
        self.assertEqual((result['component_count'], result['complete_count'], result['censored_count']), (2, 1, 1))
        self.assertEqual(result['objects'][0]['area_px2'], 4)
        self.assertEqual((result['objects'][0]['centroid_row_px'], result['objects'][0]['centroid_col_px']), (4.5, 5.5))
        self.assertAlmostEqual(result['objects'][0]['ecd_px'], np.sqrt(16 / np.pi))
        self.assertTrue(result['objects'][1]['boundary_censored'])

    def test_excluded_roi_creates_censored_fragments(self):
        roi = np.ones((12, 12), bool)
        roi[4:8, 6] = False
        mask = np.zeros_like(roi, dtype=np.uint8)
        mask[5:7, 4:9] = 1
        result = component_table(mask, roi)
        self.assertEqual(result['component_count'], 2)
        self.assertEqual(result['censored_count'], 2)
        with self.assertRaises(ValueError):
            component_table(mask, roi, connectivity=6)

    def test_nearest_neighbor_distances_and_window_boundary(self):
        roi = np.ones((30, 30), dtype=bool)
        points = np.array([[10., 10.], [10., 15.], [25., 25.]])
        result = nearest_neighbor_summary(points, roi)
        self.assertEqual(result['count'], 3)
        self.assertAlmostEqual(result['nearest_distances_px'][0], 5)
        self.assertAlmostEqual(result['nearest_distances_px'][1], 5)
        self.assertGreater(result['edge_risk_count'], 0)
        with self.assertRaises(ValueError):
            nearest_neighbor_summary(np.array([[2., 2.]]), roi)
        with self.assertRaises(ValueError):
            validate_points(np.array([[2., 2.], [2., 2.]]), roi)

    def test_quadrats_equal_area_and_roi_exclusion(self):
        roi = np.ones((32, 32), bool)
        points = np.array([[2, 2], [4, 4], [20, 20], [20, 24]], dtype=float)
        result = quadrat_summary(points, roi, cell_size=16)
        self.assertEqual(result['eligible_quadrats'], 4)
        self.assertEqual(result['counts'], [2, 0, 0, 2])
        self.assertAlmostEqual(result['mean_count'], 1)
        self.assertAlmostEqual(result['dispersion_index'], 4/3)
        roi[0, 15] = False
        result = quadrat_summary(points, roi, cell_size=16)
        self.assertEqual(result['eligible_quadrats'], 3)
        self.assertEqual(result['points_in_eligible_quadrats'], 2)
        with self.assertRaises(ValueError):
            quadrat_summary(points, roi, cell_size=0)

    def test_monte_carlo_reproducibility_and_safe_null(self):
        roi = np.ones((32, 32), bool)
        points = np.array([[3, 3], [3, 6], [20, 20], [20, 22]], dtype=float)
        a = csr_monte_carlo(points, roi, repetitions=45, seed=7)
        b = csr_monte_carlo(points, roi, repetitions=45, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a['repetitions'], 45)
        self.assertLessEqual(a['null_05_95_px'][0], a['null_05_95_px'][1])
        with self.assertRaises(ValueError):
            csr_monte_carlo(points, roi, repetitions=5)

    def test_artificial_patterns_and_boundary_mask(self):
        a, b = artificial_patterns(seed=20), artificial_patterns(seed=20)
        np.testing.assert_array_equal(a['roi'], b['roi'])
        for name in ('random', 'regular', 'clustered'):
            np.testing.assert_array_equal(a['patterns'][name], b['patterns'][name])
            self.assertEqual(len(a['patterns'][name]), 30)
        mask = artificial_component_mask(a['roi'])
        result = component_table(mask, a['roi'])
        self.assertEqual(result['component_count'], 4)
        self.assertGreaterEqual(result['censored_count'], 2)

    def test_cli_outputs_protected_and_json_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'lab'
            result = run(dest, seed=20, repetitions=30, cell_size=16)
            self.assertEqual(set(result['patterns']), {'random', 'regular', 'clustered'})
            for name in ('session20_components.csv','session20_patterns.csv','session20_report.json','session20_spatial_patterns.png'):
                self.assertTrue((dest/name).is_file(), name)
            with self.assertRaises(FileExistsError):
                run(dest, seed=20, repetitions=30)


if __name__ == '__main__':
    unittest.main()
