"""Regression checks for conformal ranks, partition leakage and residual diagnostics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from session25_calibration_reliability_lab import (
    cohort, split_cohort, validate_partitions, fit_area_only, predict_area_only,
    conformal_radius, evaluate, diagnostics, run,
)


class CalibrationReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = cohort()
        cls.fit, cls.cal, cls.evaluation = split_cohort(cls.rows)

    def test_generator_reproducibility_and_pooled_exposure(self):
        self.assertEqual(self.rows, cohort())
        self.assertEqual(len(self.rows), 90)
        self.assertEqual(len({r['specimen_id'] for r in self.rows}), 90)
        for row in self.rows:
            self.assertAlmostEqual(row['area_fraction'], row['foreground_pixels']/row['valid_pixels'])
            self.assertEqual(row['n_fields'], 3)

    def test_partitions_disjoint_and_batch_balanced(self):
        self.assertEqual([len(p) for p in (self.fit,self.cal,self.evaluation)], [45,24,21])
        validate_partitions(self.fit,self.cal,self.evaluation)
        self.assertEqual(sum(r['batch_id']=='B1' for r in self.cal), 8)
        with self.assertRaises(ValueError):
            validate_partitions(self.fit,self.fit[:1])

    def test_finite_sample_rank_and_small_calibration(self):
        result = conformal_radius(np.arange(1,25), np.zeros(24), .1)
        self.assertEqual((result['rank'],result['radius_au']), (23,23))
        self.assertEqual(conformal_radius([2,4,5,7,9], [0]*5, .2)['radius_au'],9)
        with self.assertRaises(ValueError):
            conformal_radius([1]*8,[0]*8,.1)

    def test_invalid_calibration_arrays_and_alpha(self):
        for a,p,alpha in [([1],[1,2],.1),([float('nan')],[0],.1),([],[],.1),([1],[0],0),([1],[0],1)]:
            with self.assertRaises(ValueError):
                conformal_radius(a,p,alpha)

    def test_fit_scaling_and_no_calibration_target_leakage(self):
        before=copy.deepcopy(self.fit)
        model=fit_area_only(self.fit)
        self.assertAlmostEqual(model['mean'],np.mean([r['area_fraction'] for r in self.fit]))
        changed=[dict(r,response_au=r['response_au']+1e8) for r in self.cal]
        np.testing.assert_array_equal(predict_area_only(model,self.cal),predict_area_only(model,changed))
        self.assertEqual(self.fit,before)
        constant=[dict(r,area_fraction=.2) for r in self.fit]
        np.testing.assert_allclose(predict_area_only(fit_area_only(constant),constant),np.mean([r['response_au'] for r in constant]))

    def test_residual_sign_coverage_and_width(self):
        rows=[{'residual_au':e,'covered':c,'lower_au':0,'upper_au':10} for e,c in [(-2,True),(4,False),(3,True)]]
        metrics=diagnostics(rows)
        self.assertEqual(metrics['mae_au'],3)
        self.assertAlmostEqual(metrics['rmse_au'],np.sqrt(29/3))
        self.assertAlmostEqual(metrics['bias_actual_minus_prediction_au'],5/3)
        self.assertEqual(metrics['coverage'],2/3)
        self.assertEqual(metrics['mean_width_au'],10)

    def test_fixed_model_intervals_include_endpoints(self):
        model=fit_area_only(self.fit)
        point=float(predict_area_only(model,self.evaluation[:1])[0])
        row=dict(self.evaluation[0],response_au=point+4)
        result=evaluate(model,[row],4,'check')[0]
        self.assertTrue(result['covered'])
        self.assertEqual(result['residual_au'],4)
        self.assertEqual(result['upper_au']-result['lower_au'],8)

    def test_outputs_shift_partition_and_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'run'
            report=run(path)
            self.assertEqual(len(list(path.iterdir())),6)
            self.assertEqual(report,json.loads((path/'session25_report.json').read_text()))
            shift=report['shift']
            self.assertEqual([len(shift[k]) for k in ['fit_ids','calibration_ids','evaluation_ids']],[30,30,30])
            self.assertTrue(all(s.startswith('B3') for s in shift['evaluation_ids']))
            self.assertFalse(set(shift['fit_ids'])&set(shift['calibration_ids']))
            with self.assertRaises(FileExistsError):
                run(path)


if __name__=='__main__':
    unittest.main()
