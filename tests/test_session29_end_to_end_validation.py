"""Session 29 end-to-end traceability, audit and calibration regression checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from session29_end_to_end_validation import (
    original_to_binary, seal, verify, generate, preflight, audit, aggregate,
    fit_model, predict, calibrate, run,
)


class PipelineValidationTests(unittest.TestCase):
    def test_labels_and_sealed_protocol(self):
        mask=np.array([[0,8,9]],np.uint8)
        np.testing.assert_array_equal(original_to_binary(mask),[[1,1,0]])
        frozen=seal({'seed':29}); verify(frozen)
        frozen['protocol']['seed']=30
        with self.assertRaises(ValueError):
            verify(frozen)
        with self.assertRaises(ValueError):
            original_to_binary(np.array([[10]],np.uint8))

    def test_role_counts_and_protected_preflight_no_image_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows,targets=generate(root,29)
            self.assertEqual(len(rows),108)
            self.assertEqual([sum(r['role']==role for r in targets) for role in ('fit','calibration','evaluation')],[18,9,9])
            rows[-1]['split']='test'
            with patch('PIL.Image.open') as opened:
                with self.assertRaises(ValueError):
                    audit(rows,root)
                opened.assert_not_called()

    def test_cross_role_lineage_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'; rows,_=generate(root,29)
            fitting=next(r for r in rows if r['role']=='fit')
            other=next(r for r in rows if r['role']=='evaluation')
            for key in ('source_id','texture_id','template_id','specimen_id'):
                altered=copy.deepcopy(rows)
                target=next(r for r in altered if r['field_id']==other['field_id'])
                target[key]=fitting[key]
                with self.assertRaisesRegex(ValueError,'Cross-role'):
                    preflight(altered,root)

    def test_path_traversal_and_resolved_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'; rows,_=generate(root,29)
            bad=copy.deepcopy(rows); bad[-1]['image_rel']='../outside.png'
            with self.assertRaises(ValueError):
                preflight(bad,root)
            original=Path.resolve
            def resolving(path,*args,**kwargs):
                if path.name==rows[0]['image_rel']:
                    return Path(folder)/'outside.png'
                return original(path,*args,**kwargs)
            with patch.object(Path,'resolve',resolving):
                with self.assertRaises(ValueError):
                    preflight(rows,root)

    def test_duplicate_decoded_images_block_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'; rows,_=generate(root,29)
            with Image.open(root/rows[0]['image_rel']) as image:
                image.save(root/rows[-1]['image_rel'])
            with self.assertRaisesRegex(ValueError,'Duplicate decoded'):
                audit(rows,root)

    def test_roi_pooling_unique_property_join(self):
        targets=[{'specimen_id':'S','batch_id':'B','role':'fit','target_au':10}]
        fields=[{'field_id':str(i),'specimen_id':'S','batch_id':'B','role':'fit',
                 'valid_pixels':n,'foreground_pixels':p,'components':1}
                for i,(n,p) in enumerate(((2000,500),(4000,1600),(1000,0)))]
        result=aggregate(fields,targets)[0]
        self.assertAlmostEqual(result['area_fraction'],2100/7000)
        self.assertAlmostEqual(result['component_density_px2'],3/7000)
        with self.assertRaises(ValueError):
            aggregate(fields,targets+targets)

    def test_fitting_only_scaling_and_calibration_isolation(self):
        fitting=[{'specimen_id':str(i),'area_fraction':.1+i*.05,'target_au':100+i*3} for i in range(18)]
        model=fit_model(fitting); before=copy.deepcopy(model)
        calibration=[{'specimen_id':'C'+str(i),'area_fraction':.2,'target_au':110+i} for i in range(9)]
        result=calibrate(model,calibration,.1)
        self.assertEqual(result['rank'],9)
        changed=[dict(r,target_au=r['target_au']+1000) for r in calibration]
        np.testing.assert_array_equal(predict(model,changed),predict(model,calibration))
        self.assertEqual(model,before)
        self.assertAlmostEqual(model['mean'],np.mean([r['area_fraction'] for r in fitting]))
        with self.assertRaises(ValueError):
            calibrate(model,calibration,.01)

    def test_reproducible_pipeline_and_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b'
            first=run(a)
            self.assertEqual(first,run(b))
            self.assertEqual(first,json.loads((a/'session29_report.json').read_text()))
            self.assertEqual(len(first['trace']),9)
            self.assertTrue(all(len(r['field_ids'])==3 for r in first['trace']))
            self.assertEqual(first['readiness']['scientific_status'],'NO-GO for real-material claims')
            self.assertEqual(len(list((a/'segmentation_masks').glob('*.png'))),216)
            with self.assertRaises(FileExistsError):
                run(a)


if __name__=='__main__':
    unittest.main()
