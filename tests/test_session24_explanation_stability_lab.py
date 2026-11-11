"""Session 24 selection boundaries and numerical regression checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from session21_feature_property_lab import FEATURES, synthetic_tables, aggregate_specimens, specimen_split
from session24_explanation_stability_lab import (
    fit_subset, predict_subset, inner_folds, select_model,
    selection_stability, coefficient_stability, run,
)


class ExplanationStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fields, props, _ = synthetic_tables(21)
        cls.train, cls.val = specimen_split(aggregate_specimens(fields, props), 21)

    def test_inner_folds_unique_complete_and_disjoint(self):
        folds = inner_folds(self.train)
        self.assertEqual(sorted(i for fold in folds for i in fold), list(range(12)))
        self.assertEqual([len(fold) for fold in folds], [4, 4, 4])
        self.assertEqual(folds, inner_folds(self.train))
        with self.assertRaises(ValueError):
            inner_folds(self.train + [self.train[0]])

    def test_rank_deficient_zero_penalty_and_excluded_features(self):
        rows = [dict(r, **{f: 1.0 for f in FEATURES}) for r in self.train]
        fit = fit_subset(rows, FEATURES[:2], 0)
        np.testing.assert_allclose(predict_subset(fit, rows), np.mean([r['synthetic_response_au'] for r in rows]))
        altered = [dict(r, **{FEATURES[2]: 1e8}) for r in rows]
        np.testing.assert_array_equal(predict_subset(fit, rows), predict_subset(fit, altered))
        with self.assertRaises(ValueError):
            fit_subset(rows, (), 2)

    def test_fold_preprocessing_uses_only_fit_rows(self):
        heldout = inner_folds(self.train)[0]
        fitting = [r for i,r in enumerate(self.train) if i not in heldout]
        model = fit_subset(fitting, FEATURES, 2)
        np.testing.assert_allclose(model['train_mean'],
                                   np.mean([[r[f] for f in FEATURES] for r in fitting], axis=0))
        self.assertEqual(set(model['training_specimen_ids']), {r['specimen_id'] for r in fitting})

    def test_selection_all_candidates_and_outer_targets_unused(self):
        snapshot = copy.deepcopy(self.train)
        best, candidates, _ = select_model(self.train)
        self.assertEqual(len(candidates), 21)
        self.assertEqual(best['inner_mae_au'], min(r['inner_mae_au'] for r in candidates))
        changed = copy.deepcopy(self.val)
        for row in changed:
            row['synthetic_response_au'] += 1e8
        self.assertEqual(best, select_model(self.train)[0])
        self.assertEqual(snapshot, self.train)

    def test_subsamples_unique_training_only_and_repeatable(self):
        choices, frequencies = selection_stability(self.train, 3, 24)
        self.assertEqual((choices, frequencies), selection_stability(self.train, 3, 24))
        for row in choices:
            self.assertEqual(len(row['specimen_ids']), 9)
            self.assertEqual(len(set(row['specimen_ids'])), 9)
            self.assertTrue(set(row['specimen_ids']) <= {r['specimen_id'] for r in self.train})
        self.assertTrue(all(0 <= v <= 1 for v in frequencies.values()))

    def test_bootstrap_fixed_specification_and_input_preservation(self):
        fit = fit_subset(self.train, FEATURES[:2], 0)
        before = copy.deepcopy(fit)
        summaries = coefficient_stability(self.train, fit, 20, 24)
        self.assertEqual(summaries, coefficient_stability(self.train, fit, 20, 24))
        self.assertEqual(fit, before)
        self.assertEqual([r['feature'] for r in summaries], list(FEATURES[:2]))
        self.assertTrue(all(r['p10'] <= r['bootstrap_median'] <= r['p90'] for r in summaries))

    def test_inner_score_matches_manual_fold_predictions(self):
        _, candidates, folds = select_model(self.train)
        candidate = candidates[0]
        errors = []
        for heldout in folds:
            fitting = [r for i,r in enumerate(self.train) if i not in heldout]
            validation = [self.train[i] for i in heldout]
            model = fit_subset(fitting, candidate['features'], candidate['alpha'])
            errors.extend(abs(predict_subset(model, validation)-[r['synthetic_response_au'] for r in validation]))
        self.assertAlmostEqual(candidate['inner_mae_au'], float(np.mean(errors)))

    def test_outputs_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lab'
            report = run(path, subsamples=2, bootstraps=10, permutations=5)
            self.assertEqual(len(list(path.iterdir())), 6)
            self.assertEqual(report, json.loads((path/'session24_report.json').read_text()))
            self.assertFalse(set(report['training_specimens']) & set(report['validation_specimens']))
            with self.assertRaises(FileExistsError):
                run(path)


if __name__ == '__main__':
    unittest.main()
