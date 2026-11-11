"""Session 27 protocol, metrics, preflight and reproducibility regression checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from session26_provenance_audit import generate_fixture, ProtectedDataError, binary_from_schema
from session27_benchmark_protocol import (
    make_protocol, protocol_hash, seal_protocol, verify_protocol,
    predict, pair_metrics, aggregate, evaluate_manifest, run,
)


class BenchmarkProtocolTests(unittest.TestCase):
    def test_seal_canonical_hash_and_tampering(self):
        protocol=make_protocol()
        self.assertEqual(protocol_hash(protocol),protocol_hash(dict(reversed(list(protocol.items())))))
        sealed=seal_protocol(protocol)
        self.assertEqual(verify_protocol(sealed),protocol)
        changed=copy.deepcopy(sealed)
        changed['protocol']['boundary_tolerance_px']=2
        with self.assertRaises(ValueError):
            verify_protocol(changed)
        with self.assertRaises(ValueError):
            verify_protocol(seal_protocol(changed['protocol']))

    def test_original_zero_is_foreground_and_threshold_strict(self):
        np.testing.assert_array_equal(binary_from_schema(np.array([[0,8,9]],np.uint8),'original10'),[[1,1,0]])
        gray=np.array([[119,120,164,165]],np.uint8)
        np.testing.assert_array_equal(predict(gray,'fixed_dark_120'),[[1,0,0,0]])
        np.testing.assert_array_equal(predict(gray,'fixed_dark_165'),[[1,1,1,0]])

    def test_metric_arithmetic_and_area_error(self):
        truth=np.array([[1,1,0,0]],np.uint8)
        prediction=np.array([[1,0,1,0]],np.uint8)
        metrics=pair_metrics(truth,prediction)
        self.assertEqual([metrics[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
        self.assertAlmostEqual(metrics['iou'],1/3)
        self.assertEqual(metrics['dice'],.5)
        self.assertEqual(metrics['signed_area_error_pp'],0)

    def test_pooled_and_macro_have_distinct_weights(self):
        rows=[{'tp':80,'fp':10,'fn':10,'tn':0,'iou':.8,'boundary_f1':1.,'signed_area_error_pp':0},
              {'tp':2,'fp':3,'fn':5,'tn':0,'iou':.2,'boundary_f1':.5,'signed_area_error_pp':-20}]
        result=aggregate(rows)
        self.assertAlmostEqual(result['scene_macro_foreground_iou'],.5)
        self.assertAlmostEqual(result['pooled_foreground_iou'],82/110)

    def test_absent_class_and_boundary_tolerance(self):
        empty=np.zeros((12,12),np.uint8)
        metrics=pair_metrics(empty,empty)
        self.assertIsNone(metrics['iou'])
        self.assertIsNone(metrics['boundary_f1'])
        a=empty.copy(); a[5,5]=1
        b=empty.copy(); b[5,6]=1
        self.assertEqual(pair_metrics(a,b)['boundary_f1'],1)
        self.assertEqual(pair_metrics(a,empty)['boundary_f1'],0)
        b=empty.copy(); b[6,6]=1
        self.assertEqual(pair_metrics(a,b)['boundary_f1'],0)

    def test_protected_preflight_zero_image_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate_fixture(root)
            rows[-1]['split']='test'
            with patch('PIL.Image.open') as opened:
                with self.assertRaises(ProtectedDataError):
                    evaluate_manifest(rows,root,seal_protocol(make_protocol()))
                opened.assert_not_called()

    def test_source_reuse_blocks_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate_fixture(root)
            rows[-1]['source_id']=rows[0]['source_id']
            with patch('session27_benchmark_protocol.predict') as predictor:
                with self.assertRaisesRegex(ValueError,'Blocking audit'):
                    evaluate_manifest(rows,root,seal_protocol(make_protocol()))
                predictor.assert_not_called()

    def test_reproducible_reports_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b'
            first=run(a,repetitions=10)
            second=run(b,repetitions=10)
            self.assertEqual(first,second)
            self.assertEqual(first,json.loads((a/'session27_report.json').read_text()))
            self.assertEqual(len(first['summaries']),6)
            self.assertEqual([r['group_count'] for r in first['bootstrap']],[4,4,4,2,2,2])
            with self.assertRaises(FileExistsError):
                run(a)


if __name__=='__main__':
    unittest.main()
