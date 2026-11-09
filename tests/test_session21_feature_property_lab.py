"""Eight reproducible tests for synthetic specimen-level feature/target integrity."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from session21_feature_property_lab import (
    FEATURES, _component_summary, synthetic_tables, validate_tables,
    aggregate_specimens, specimen_split, fit_specimen_ridge,
    predict_specimen_ridge, identity_leakage_demo,
    bootstrap_specimen_mean, run,
)


class SpecimenFeatureTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields, cls.props, cls.samples = synthetic_tables(seed=21)
        cls.specimens = aggregate_specimens(cls.fields, cls.props)

    def test_synthetic_generation_reproducible_and_roi_semantics(self):
        fields2, props2, samples2 = synthetic_tables(seed=21)
        self.assertEqual(self.fields, fields2)
        self.assertEqual(self.props, props2)
        self.assertEqual(len(self.fields), 54)
        self.assertEqual(len(self.props), 18)
        self.assertEqual(len({r['field_id'] for r in self.fields}), 54)
        for (mask, roi, fid), (mask2, roi2, fid2) in zip(self.samples, samples2):
            self.assertEqual(fid, fid2)
            np.testing.assert_array_equal(mask, mask2)
            np.testing.assert_array_equal(roi, roi2)
            self.assertTrue(np.all(mask[~roi] == 0))
        self.assertTrue(all(r['source_kind'] == 'artificial_mask' for r in self.fields))

    def test_observed_component_identity_and_boundary_censoring(self):
        binary = np.zeros((12, 12), dtype=np.uint8)
        roi = np.ones((12, 12), dtype=bool)
        binary[3:5, 4:6] = 1
        binary[10:12, 9:12] = 1
        summary = _component_summary(binary, roi)
        self.assertEqual(summary['observed_components'], 2)
        self.assertEqual(summary['boundary_censored_components'], 1)
        self.assertTrue(summary['mean_observed_centroid_nn_px'] > 0)
        with self.assertRaises(ValueError):
            _component_summary(np.full((12,12), 9), roi)

    def test_join_rejects_duplicates_unknown_and_bad_exposures(self):
        validate_tables(self.fields, self.props)
        with self.assertRaisesRegex(ValueError, 'Duplicate field'):
            validate_tables(self.fields + [self.fields[0]], self.props)
        with self.assertRaisesRegex(ValueError, 'Duplicate specimen'):
            validate_tables(self.fields, self.props + [self.props[0]])
        bad = [dict(x) for x in self.fields]
        bad[0]['batch_id'] = 'UNKNOWN'
        with self.assertRaises(ValueError):
            validate_tables(bad, self.props)
        bad = [dict(x) for x in self.fields]
        bad[0]['foreground_pixels'] = bad[0]['roi_pixels'] + 1
        with self.assertRaises(ValueError):
            validate_tables(bad, self.props)

    def test_roi_weighted_fraction_differs_from_naive_field_mean(self):
        props = [{'specimen_id': 'S1', 'batch_id': 'B1',
                  'synthetic_response_au': 123.,
                  'target_provenance': 'simulated_specimen_target_not_measured'}]
        fields = []
        for idx, (valid, positive) in enumerate(((10, 1), (100, 50))):
            fields.append({'field_id': str(idx), 'specimen_id': 'S1', 'source_id': 'S1',
                           'batch_id': 'B1', 'roi_pixels': valid,
                           'foreground_pixels': positive, 'area_fraction': positive/valid,
                           'observed_components': 2, 'boundary_censored_components': 0,
                           'observed_component_density_px2': 2/valid,
                           'mean_observed_centroid_nn_px': 5.,
                           'calibration_um_per_px': None, 'source_kind': 'artificial_mask',
                           'mask_semantics': 'synthetic'})
        result = aggregate_specimens(fields, props)[0]
        self.assertAlmostEqual(result['roi_weighted_area_fraction'], 51/110)
        self.assertAlmostEqual(result['unweighted_mean_field_area_fraction'], .3)
        self.assertAlmostEqual(result['observed_component_density_px2'], 4/110)
        self.assertEqual(result['n_fields'], 2)
        self.assertEqual(result['synthetic_response_au'], 123.)

    def test_specimen_holdout_preserves_batches_and_prevents_field_overlap(self):
        tr, va = specimen_split(self.specimens, seed=21)
        self.assertEqual((len(tr), len(va)), (12, 6))
        ids_tr = {x['specimen_id'] for x in tr}
        ids_va = {x['specimen_id'] for x in va}
        self.assertFalse(ids_tr & ids_va)
        self.assertEqual({r['batch_id'] for r in tr}, {'B1','B2','B3'})
        self.assertEqual({r['batch_id'] for r in va}, {'B1','B2','B3'})
        self.assertTrue(all(x['specimen_id'] in ids_tr or x['specimen_id'] in ids_va for x in self.fields))
        for batch in ('B1','B2','B3'):
            self.assertEqual(sum(r['batch_id']==batch for r in va), 2)

    def test_train_only_scaling_and_predictions_cannot_fit_validation(self):
        train, validation = specimen_split(self.specimens, seed=21)
        fit = fit_specimen_ridge(train, alpha=2.)
        self.assertEqual(tuple(fit['feature_names']), FEATURES)
        for i,key in enumerate(FEATURES):
            self.assertAlmostEqual(fit['train_mean'][i], np.mean([row[key] for row in train]))
        pred1 = predict_specimen_ridge(fit, validation)
        pred2 = predict_specimen_ridge(fit, validation)
        self.assertEqual(pred1.shape, (6,))
        self.assertTrue(np.isfinite(pred1).all())
        np.testing.assert_allclose(pred1, pred2, atol=0, rtol=0)
        altered = [dict(x) for x in validation]
        for r in altered:
            r['synthetic_response_au'] += 1000000
        fit2 = fit_specimen_ridge(train, alpha=2.)
        self.assertEqual(fit, fit2)
        np.testing.assert_array_equal(predict_specimen_ridge(fit, altered), pred1)
        constant = [dict(row, **{key: 1.0 for key in FEATURES}) for row in train]
        unpenalized = fit_specimen_ridge(constant, alpha=0.)
        np.testing.assert_allclose(predict_specimen_ridge(unpenalized, constant),
                                   np.mean([row['synthetic_response_au'] for row in constant]))
        with self.assertRaises(ValueError):
            fit_specimen_ridge(train, alpha=-1)

    def test_leaky_specimen_id_lookup_cannot_generalize_to_unseen_specimens(self):
        train, val = specimen_split(self.specimens, seed=21)
        demonstration = identity_leakage_demo(self.fields, self.props, train, val)
        self.assertEqual(demonstration['bad_field_row_split_id_memorizer_mae_au'], 0.)
        self.assertEqual(demonstration['proper_specimen_split_unseen_ids'], len(val))
        self.assertGreater(demonstration['proper_specimen_split_id_memorizer_fallback_mae_au'], 0)
        boot = bootstrap_specimen_mean(self.specimens, 300, seed=21)
        self.assertEqual(boot['sampling_unit'], 'generated_specimen')
        self.assertLess(boot['percentile_interval_95'][0], boot['observed_mean'])
        self.assertGreater(boot['percentile_interval_95'][1], boot['observed_mean'])

    def test_end_to_end_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'session21'
            report = run(path, seed=21, alpha=2., bootstrap_repetitions=100)
            self.assertEqual(report['specimens'], 18)
            self.assertEqual(report['fields'], 54)
            self.assertEqual((report['training_field_count'],report['validation_field_count']), (36,18))
            for name in ('session21_fields.csv','session21_properties.csv',
                         'session21_specimens.csv','session21_predictions.csv',
                         'session21_feature_property.png','session21_report.json'):
                self.assertTrue((path/name).is_file(), name)
            with (path/'session21_specimens.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertEqual(len({r['specimen_id'] for r in rows}),18)
            with (path/'session21_report.json').open(encoding='utf-8') as stream:
                copy = json.load(stream)
            self.assertEqual(copy['property_target_level'], report['property_target_level'])
            with self.assertRaises(FileExistsError):
                run(path, bootstrap_repetitions=100)


if __name__ == '__main__':
    unittest.main()
