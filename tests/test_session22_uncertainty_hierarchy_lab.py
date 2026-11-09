"""Eight focused tests for Session 22 synthetic measurement/error reliability lab."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from session21_feature_property_lab import synthetic_tables, aggregate_specimens, specimen_split
from session22_uncertainty_hierarchy_lab import (
    expected_fraction_bias, perturb_counts, noisy_specimen_features,
    fit_fraction_ridge, predict_fraction_ridge, leave_batch_out,
    error_metrics, evaluate_split, bootstrap_grouped_mean,
    pooled_and_within_batch_correlation, perturbation_sensitivity, run,
)


class UncertaintyHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields, cls.properties, _ = synthetic_tables(seed=21)
        cls.specimens = aggregate_specimens(cls.fields, cls.properties)
        cls.training, cls.validation = specimen_split(cls.specimens, seed=21)

    def test_count_conservation_and_exact_edge_cases(self):
        rng = np.random.default_rng(22)
        none = perturb_counts(100, 30, 0.0, 0.0, rng)
        self.assertEqual((none['fp'], none['fn'], none['measured_positive_pixels']), (0, 0, 30))
        all_flip = perturb_counts(100, 30, 1.0, 1.0, rng)
        self.assertEqual((all_flip['fp'], all_flip['fn'], all_flip['measured_positive_pixels']), (70, 30, 70))
        self.assertAlmostEqual(all_flip['signed_error_fraction'], 0.4)
        for kwargs in ({'valid': 0, 'foreground': 0}, {'valid': 10, 'foreground': 11}):
            with self.assertRaises(ValueError):
                perturb_counts(fpr=0.1, fnr=0.1, rng=rng, **kwargs)
        with self.assertRaises(ValueError):
            perturb_counts(5, 3, -.1, .1, rng)

    def test_expected_bias_and_empirical_reproducibility(self):
        self.assertAlmostEqual(expected_fraction_bias(.3, .04, .10), .028-.03)
        self.assertAlmostEqual(expected_fraction_bias(.25, .06, .14), .045-.035)
        self.assertAlmostEqual(expected_fraction_bias(.3, 0, 0), 0)
        rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
        a=[perturb_counts(1000, 300, .04, .10, rng1) for _ in range(150)]
        b=[perturb_counts(1000, 300, .04, .10, rng2) for _ in range(150)]
        self.assertEqual(a,b)
        empirical = np.mean([r['signed_error_fraction'] for r in a])
        self.assertLess(abs(empirical-expected_fraction_bias(.3,.04,.10)), .008)
        with self.assertRaises(ValueError):
            expected_fraction_bias(1.2, .1, .1)

    def test_roi_weighted_noisy_aggregation_and_original_preservation(self):
        snapshot=[dict(r) for r in self.fields]
        a, spec_a=noisy_specimen_features(self.fields,.02,.06,seed=33)
        b, spec_b=noisy_specimen_features(self.fields,.02,.06,seed=33)
        self.assertEqual((a,spec_a),(b,spec_b))
        self.assertEqual(self.fields, snapshot)
        self.assertEqual((len(a),len(spec_a)),(54,18))
        lookup={r['specimen_id']:r for r in spec_a}
        for row in a:
            self.assertEqual(row['measured_positive_pixels'],row['true_positive_pixels']+row['fp']-row['fn'])
        for specimen in self.specimens:
            other=lookup[specimen['specimen_id']]
            self.assertAlmostEqual(other['true_fraction'],specimen['roi_weighted_area_fraction'])
            self.assertEqual(other['measured_positive_pixels'],other['true_positive_pixels']+other['fp']-other['fn'])

    def test_fraction_ridge_fixed_train_scaling_and_rank_deficiency(self):
        training=[{'specimen_id':'A','roi_weighted_area_fraction':.25,'synthetic_response_au':10.},
                  {'specimen_id':'B','roi_weighted_area_fraction':.25,'synthetic_response_au':14.}]
        fit=fit_fraction_ridge(training,alpha=0)
        self.assertEqual(fit['weight'],0)
        self.assertAlmostEqual(float(predict_fraction_ridge(fit,[.2,.8])[0]),12)
        model=fit_fraction_ridge(self.training,alpha=2)
        self.assertEqual(model['training_specimen_ids'],[r['specimen_id'] for r in self.training])
        self.assertAlmostEqual(model['training_mean'],np.mean([r['roi_weighted_area_fraction'] for r in self.training]))
        self.assertEqual(len(predict_fraction_ridge(model,[.2,.5,.8])),3)
        with self.assertRaises(ValueError):
            fit_fraction_ridge(training,-.1)

    def test_grouped_and_batch_holdout_protect_partition(self):
        train,val=leave_batch_out(self.specimens,'B3')
        self.assertEqual((len(train),len(val)),(12,6))
        self.assertTrue(all(r['batch_id']=='B3' for r in val))
        self.assertFalse({r['specimen_id'] for r in train}&{r['specimen_id'] for r in val})
        first=evaluate_split(self.training,self.validation)
        shifted=evaluate_split(train,val)
        self.assertEqual((first['n_train'],first['n_validation']),(12,6))
        self.assertEqual((shifted['n_train'],shifted['n_validation']),(12,6))
        self.assertEqual(set(first['metrics']),{'train_mean','area_only_ridge','three_feature_ridge'})
        self.assertAlmostEqual(first['metrics']['three_feature_ridge']['mae_au'],6.6817,places=3)
        with self.assertRaises(ValueError):
            evaluate_split(self.training,self.training[:1])
        self.assertEqual(error_metrics([1,3],[2,4])['mae_au'],1)

    def test_batch_and_specimen_bootstraps_different_units(self):
        a=bootstrap_grouped_mean(self.specimens,'specimen_id',repetitions=200,seed=4)
        b=bootstrap_grouped_mean(self.specimens,'batch_id',repetitions=200,seed=4)
        self.assertEqual((a['n_groups'],b['n_groups']),(18,3))
        self.assertEqual(len(a['draws']),200)
        self.assertEqual(len(b['draws']),200)
        self.assertAlmostEqual(a['observed_mean'],b['observed_mean'])
        c=bootstrap_grouped_mean(self.specimens,'batch_id',repetitions=200,seed=4)
        np.testing.assert_array_equal(b['draws'],c['draws'])
        self.assertGreater(np.std(b['draws']),np.std(a['draws']))
        with self.assertRaises(ValueError):
            bootstrap_grouped_mean(self.specimens,'field_id')

    def test_conditional_prediction_sensitivity_and_confounding(self):
        fit=fit_fraction_ridge(self.training,2)
        sid=self.validation[0]['specimen_id']
        result=perturbation_sensitivity(self.fields,fit,sid,repetitions=100,seed=5)
        self.assertEqual(result['specimen_id'],sid)
        self.assertEqual(len(result['scenarios']),3)
        for scen in result['scenarios']:
            self.assertEqual(len(scen['measured_fraction_draws']),100)
            expected_shift = fit['weight'] / fit['training_std'] * (
                scen['measured_fraction_draws'] - result['unperturbed_fraction'])
            np.testing.assert_allclose(
                scen['prediction_draws'] - result['unperturbed_predicted_au'],
                expected_shift, atol=1e-12, rtol=1e-12)
            self.assertAlmostEqual(scen['unperturbed_predicted_au'],result['unperturbed_predicted_au'])
            self.assertLess(abs(scen['mean_empirical_error_fraction']-scen['expected_error_fraction']),.006)
        corr=pooled_and_within_batch_correlation(self.specimens)
        self.assertGreater(corr['pooled_pearson'],0)
        self.assertLess(abs(corr['within_batch_demeaned_pearson']),abs(corr['pooled_pearson']))

    def test_end_to_end_outputs_safeguards(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'s22'
            report=run(output,seed=22,repetitions=30)
            self.assertEqual((report['fields'],report['specimens']),(54,18))
            self.assertEqual(report['grouped_specimen_validation']['n_validation'],6)
            self.assertEqual(report['held_out_batch_validation']['n_validation'],6)
            for name in ('session22_report.json','session22_uncertainty.png',
                         'session22_perturbed_fields.csv','session22_perturbed_specimens.csv',
                         'session22_grouped_predictions.csv','session22_batch_holdout_predictions.csv'):
                self.assertTrue((output/name).is_file(),name)
            with (output/'session22_perturbed_fields.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),54)
            data=json.loads((output/'session22_report.json').read_text())
            self.assertIn('No steel micrographs',data['scope'])
            with self.assertRaises(FileExistsError):
                run(output,seed=22,repetitions=30)


if __name__=='__main__':
    unittest.main()
