"""Session 19 artificial stereology checks; no research images are accessed."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session19_stereology_sampling_lab import (area_fraction, systematic_points, synthetic_fields,
    compare_fields, bootstrap_fields, artificial_bands, boundary_line_scan, run)


class StereologySamplingTests(unittest.TestCase):
    def test_exact_roi_denominator_and_input_rejections(self):
        m = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        r = np.array([[True, False], [True, True]], dtype=bool)
        result = area_fraction(m, r)
        self.assertEqual((result["positive_pixels"], result["valid_pixels"]), (1, 3))
        self.assertAlmostEqual(result["area_fraction"], 1/3)
        with self.assertRaises(ValueError):
            area_fraction(np.array([[0, 9]]), np.ones((1, 2), bool))
        with self.assertRaises(ValueError):
            area_fraction(m, np.zeros_like(r))

    def test_systematic_grid_exact_fixture_and_roi_exclusion(self):
        m = np.zeros((8, 8), np.uint8)
        m[::2, ::2] = 1
        r = np.ones_like(m, bool)
        r[0, 0] = False
        p = systematic_points(m, r, step=2, offset=(0, 0))
        self.assertEqual((p["valid_points"], p["hits"]), (15, 15))
        self.assertEqual(p["point_fraction"], 1)
        with self.assertRaises(ValueError):
            systematic_points(m, r, 2, (2, 0))
        with self.assertRaises(ValueError):
            systematic_points(m, r, 20, (19, 19))

    def test_random_offset_changes_sampling_estimate(self):
        m = np.indices((12,12))[0].astype(np.uint8)%2
        roi = np.ones_like(m, bool)
        one = systematic_points(m,roi,2,(0,0))
        two = systematic_points(m,roi,2,(1,0))
        self.assertEqual((one["point_fraction"],two["point_fraction"]),(0,1))
        self.assertEqual(area_fraction(m,roi)["area_fraction"],.5)

    def test_generated_fields_deterministic_and_annotations_excluded(self):
        a,b = synthetic_fields(), synthetic_fields()
        self.assertEqual(len(a),8)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x["mask"], y["mask"])
            np.testing.assert_array_equal(x["image"], y["image"])
            np.testing.assert_array_equal(x["roi"], y["roi"])
            self.assertTrue((x["mask"] <= 1).all())
            self.assertFalse(x["roi"][2, -10])

    def test_per_field_grid_and_error_units(self):
        fields = synthetic_fields(n=3)
        r = compare_fields(fields,step=12,seed=19)
        self.assertEqual(len(r),3)
        for row in r:
            self.assertAlmostEqual(row["point_fraction"],row["hits"]/row["valid_points"])
            self.assertAlmostEqual(row["signed_point_error_pp"],100*(row["point_fraction"]-row["area_fraction"]))
        self.assertEqual(r,compare_fields(fields,step=12,seed=19))

    def test_bootstrap_resamples_fields_deterministically(self):
        x = [.10, .20, .30, .40]
        a,b = bootstrap_fields(x,replicates=300,seed=9),bootstrap_fields(x,replicates=300,seed=9)
        self.assertAlmostEqual(a["mean_field_fraction"],.25)
        np.testing.assert_array_equal(a["bootstrap_means"],b["bootstrap_means"])
        self.assertLessEqual(a["percentile_95_interval"][0],.25)
        self.assertGreaterEqual(a["percentile_95_interval"][1],.25)
        with self.assertRaises(ValueError):
            bootstrap_fields([.3])

    def test_directional_band_crossings_and_complete_chords(self):
        mask, roi = artificial_bands()
        h = boundary_line_scan(mask,roi,"horizontal",12,3)
        v = boundary_line_scan(mask,roi,"vertical",12,3)
        self.assertEqual(h["boundary_crossings"],0)
        self.assertGreater(v["boundary_crossings"],0)
        self.assertGreater(v["boundary_crossing_density_per_px"],h["boundary_crossing_density_per_px"])
        self.assertTrue(all(ch==8 for ch in v["complete_foreground_chords_px"]))
        with self.assertRaises(ValueError):
            boundary_line_scan(mask,roi,"diagonal")

    def test_end_to_end_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"synthetic_eval"
            r=run(root)
            self.assertTrue((root/"session19_fields.csv").is_file())
            self.assertTrue((root/"session19_report.json").is_file())
            self.assertTrue((root/"session19_sampling.png").is_file())
            self.assertEqual(r["synthetic_fields"],8)
            with self.assertRaises(FileExistsError):
                run(root)


if __name__=="__main__":
    unittest.main()
