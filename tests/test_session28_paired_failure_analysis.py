"""Session 28 paired source comparison and safeguards regression checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from session28_paired_failure_analysis import (
    METHODS, CONDITIONS, seal, verify, binary_mask, metrics, paired_sources,
    bootstrap, sign_flip, generate, preflight, run,
)


class PairedFailureTests(unittest.TestCase):
    def test_labels_and_thin_diagnostic(self):
        original=np.array([[0,1,8,9]],np.uint8)
        before=original.copy()
        np.testing.assert_array_equal(binary_mask(original),[[1,1,1,0]])
        result=metrics(original,np.array([[1,0,0,1]],np.uint8))
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,2))
        self.assertEqual(result['thin_recall'],0)
        self.assertEqual(result['signed_area_error_pp'],-25)
        np.testing.assert_array_equal(original,before)
        with self.assertRaises(ValueError):
            binary_mask(np.array([[10]],np.uint8))

    def test_metric_boundaries_and_failure_taxonomy(self):
        original=np.full((12,12),9,np.uint8); original[5,5]=8
        prediction=np.zeros((12,12),np.uint8); prediction[5,6]=1
        self.assertEqual(metrics(original,prediction)['boundary_f1'],1)
        self.assertEqual(metrics(original,np.zeros_like(prediction))['failure'],'undersegmentation')
        self.assertEqual(metrics(original,np.ones_like(prediction))['failure'],'oversegmentation')
        self.assertEqual(metrics(original,binary_mask(original))['failure'],'no_pixel_errors')

    def test_pairing_within_sources_then_equal_sources(self):
        rows=[]
        for source,delta in [('A',.1),('B',-.05)]:
            for condition in CONDITIONS:
                rows.extend([{'source_id':source,'condition':condition,'method':METHODS[0],'iou':.5+delta},
                             {'source_id':source,'condition':condition,'method':METHODS[1],'iou':.5}])
        pairs=paired_sources(rows)
        self.assertAlmostEqual(np.mean([r['paired_iou_difference'] for r in pairs]),.025)
        with self.assertRaises(ValueError):
            paired_sources(rows[:-1])
        with self.assertRaises(ValueError):
            paired_sources(rows+[rows[0]])

    def test_group_bootstrap_deterministic_and_bounded(self):
        a=bootstrap([-.1,.2,.3],100,28)
        np.testing.assert_array_equal(a,bootstrap([-.1,.2,.3],100,28))
        self.assertTrue(np.all((a>=-.1-1e-12)&(a<=.3+1e-12)))
        with self.assertRaises(ValueError):
            bootstrap([.2])

    def test_exact_sign_flip_enumeration(self):
        result=sign_flip([1.,1.])
        self.assertEqual(result['configurations'],4)
        self.assertEqual(result['two_sided_p'],.5)
        self.assertEqual(sign_flip([0.,0.])['two_sided_p'],1)
        self.assertEqual(sign_flip([1.]*12)['configurations'],4096)

    def test_seal_tampering(self):
        frozen=seal({'methods':list(METHODS),'seed':28})
        verify(frozen)
        frozen['protocol']['seed']=29
        with self.assertRaises(ValueError):
            verify(frozen)

    def test_preflight_protected_and_source_overlap_before_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate(root)
            protected=copy.deepcopy(rows); protected[-1]['split']='test'
            with patch('PIL.Image.open') as opened:
                with self.assertRaises(ValueError):
                    preflight(protected,root)
                opened.assert_not_called()
            rows[-1]['source_id']=rows[0]['source_id']
            with self.assertRaisesRegex(ValueError,'source lineage'):
                preflight(rows,root)

    def test_reproducible_output_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b'
            first=run(a,repetitions=10)
            self.assertEqual(first,run(b,repetitions=10))
            self.assertEqual(first,json.loads((a/'session28_report.json').read_text()))
            self.assertEqual(first['validation_images'],36)
            self.assertEqual(first['validation_source_groups'],12)
            with self.assertRaises(FileExistsError):
                run(a)


if __name__=='__main__':
    unittest.main()
