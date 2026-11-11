"""Eight targeted checks for Session 23 artificial explainability lab."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from session21_feature_property_lab import (
    FEATURES, synthetic_tables, aggregate_specimens,
    specimen_split, fit_specimen_ridge, predict_specimen_ridge,
)
from session23_explainability_lab import (
    _mae,validate_specimen_holdout,linear_feature_contributions,
    permutation_importance,correlation_summary,correlated_proxy_demo,run,
)


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        field, prop, _=synthetic_tables(seed=21)
        cls.specimens=aggregate_specimens(field,prop)
        cls.train,cls.val=specimen_split(cls.specimens,seed=21)
        cls.fit=fit_specimen_ridge(cls.train,alpha=2)

    def test_specimen_split_and_validation_safeguards(self):
        validate_specimen_holdout(self.train,self.val)
        self.assertEqual((len(self.train),len(self.val)),(12,6))
        self.assertFalse(set(r['specimen_id'] for r in self.train)&set(r['specimen_id'] for r in self.val))
        with self.assertRaises(ValueError):
            validate_specimen_holdout(self.train,self.train[:1])
        with self.assertRaises(ValueError):
            validate_specimen_holdout(self.train,[])
        with self.assertRaises(ValueError):
            validate_specimen_holdout(self.train+[self.train[0]],self.val)

    def test_linear_contributions_reconstruct_every_prediction(self):
        out=linear_feature_contributions(self.fit,self.val)
        self.assertEqual(out['contributions'].shape,(6,3))
        pred=predict_specimen_ridge(self.fit,self.val)
        np.testing.assert_allclose(out['baseline']+out['contributions'].sum(axis=1),pred,atol=1e-9,rtol=0)
        self.assertEqual(out['features'],list(FEATURES))
        self.assertIn('not conditional SHAP',out['note'])

    def test_training_mean_baseline_zero_contributions(self):
        synthetic=self.val[0].copy()
        for name,mu in zip(FEATURES,self.fit['train_mean']):
            synthetic[name]=mu
        out=linear_feature_contributions(self.fit,[synthetic])
        np.testing.assert_allclose(out['contributions'],0,atol=1e-12)
        self.assertAlmostEqual(out['predictions'][0],self.fit['target_train_mean'],places=10)

    def test_permutation_deterministic_and_heldout_unchanged(self):
        before=[row.copy() for row in self.val]
        a=permutation_importance(self.fit,self.val,repeats=20,seed=19)
        b=permutation_importance(self.fit,self.val,repeats=20,seed=19)
        self.assertEqual(a,b)
        self.assertEqual(self.val,before)
        self.assertEqual(len(a['feature_importance']),3)
        actual=np.array([r['synthetic_response_au'] for r in self.val])
        expected=_mae(actual,predict_specimen_ridge(self.fit,self.val))
        self.assertAlmostEqual(a['baseline_mae_au'],expected)

    def test_group_permutation_contract_and_invalid_inputs(self):
        r=permutation_importance(self.fit,self.val,repeats=3,seed=4,groups=((0,1),(2,)))
        self.assertEqual(r['feature_importance'][0]['features'],list(FEATURES[:2]))
        for options in ({'repeats':0},{'seed':-1},{'groups':((3,),)},{'groups':((0,0),)},{'groups':((),)},{'groups':()}):
            with self.assertRaises(ValueError):
                permutation_importance(self.fit,self.val,**options)
        with self.assertRaises(ValueError):
            permutation_importance(self.fit,self.val[:1])

    def test_training_only_correlations(self):
        info=correlation_summary(self.train)
        self.assertEqual(np.array(info['training_feature_pearson']).shape,(3,3))
        np.testing.assert_allclose(np.diag(info['training_feature_pearson']),1,atol=1e-12)
        self.assertGreaterEqual(info['train_fraction_target_pearson'],-1)
        self.assertLessEqual(info['train_fraction_target_pearson'],1)

    def test_highly_correlated_proxies_and_joint_shuffle(self):
        info=correlated_proxy_demo(seed=23,repeats=60)
        self.assertGreater(info['training_correlation'],.99)
        d=info['mean_delta_mae_au']
        self.assertGreater(d['both_proxies'],d['proxy_A'])
        self.assertGreater(d['both_proxies'],d['proxy_B'])
        self.assertEqual((info['train_count'],info['validation_count']),(130,50))

    def test_end_to_end_outputs_protected_and_no_physical_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'s23'
            report=run(out,seed=23,repeats=15)
            expected={'session23_predictions.csv','session23_linear_contributions.csv',
                      'session23_permutation_importance.csv','session23_report.json',
                      'session23_explainability.png'}
            self.assertEqual({p.name for p in out.iterdir()},expected)
            self.assertEqual(len(report['training_specimens']),12)
            self.assertEqual(len(report['validation_specimens']),6)
            self.assertLess(report['validation_mae_ridge_au'],report['validation_mae_train_mean_au'])
            self.assertIn('no real steel',report['scope'].lower())
            with self.assertRaises(FileExistsError):
                run(out,seed=23,repeats=15)
            self.assertEqual(report['code_sha256'],json.loads((out/'session23_report.json').read_text())['code_sha256'])


if __name__=='__main__':
    unittest.main()
