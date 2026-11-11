"""Session 26 fail-closed preflight and dataset integrity regression tests."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from session26_provenance_audit import (
    ProtectedDataError, preflight, safe_development_path, binary_from_schema,
    generate_fixture, audit, run,
)


class ProvenanceAuditTests(unittest.TestCase):
    def test_deterministic_fixture_and_clean_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b'
            rows=generate_fixture(a)
            self.assertEqual(rows,generate_fixture(b))
            for row in rows:
                self.assertEqual((a/row['image_rel']).read_bytes(),(b/row['image_rel']).read_bytes())
            report=audit(rows,a)
            self.assertEqual(report['status'],'manifest_consistent')
            self.assertEqual(len(report['warnings']),6)
            self.assertEqual(len(report['pairs']),6)

    def test_original_and_derived_semantics_preserve_arrays(self):
        mask=np.array([[0,1,8,9]],np.uint8)
        before=mask.copy()
        np.testing.assert_array_equal(binary_from_schema(mask,'original10'),[[1,1,1,0]])
        np.testing.assert_array_equal(mask,before)
        np.testing.assert_array_equal(binary_from_schema(np.array([[0,1]],np.uint8),'derived_binary'),[[0,1]])
        for bad in (np.array([[10]],np.uint8),np.array([[1.]],float)):
            with self.assertRaises(ValueError):
                binary_from_schema(bad,'original10')

    def test_declared_shared_lineage_blocks_cross_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate_fixture(root)
            rows[-1]['texture_id']=rows[0]['texture_id']
            self.assertTrue(any('texture_id' in s for s in audit(rows,root)['blockers']))

    def test_pixel_duplicate_detected_across_encodings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate_fixture(root)
            with Image.open(root/rows[0]['image_rel']) as img:
                img.save(root/'encoded.bmp')
            rows[-1]['image_rel']='encoded.bmp'
            report=audit(rows,root)
            self.assertTrue(any('image_pixel_sha256' in s for s in report['blockers']))
            shutil.copyfile(root/rows[0]['image_rel'],root/'copy.png')
            rows[-1]['image_rel']='copy.png'
            self.assertTrue(any('image_sha256' in s for s in audit(rows,root)['blockers']))

    def test_protected_row_stops_all_image_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate_fixture(root)
            rows[-1]['split']='test'
            with patch('session26_provenance_audit.Image.open') as opened:
                with self.assertRaises(ProtectedDataError):
                    audit(rows,root)
                opened.assert_not_called()

    def test_traversal_absolute_and_resolved_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate_fixture(root)
            for name in ('../image.png',r'C:\outside\image.png','official_test/image.png'):
                with self.assertRaises(ProtectedDataError):
                    safe_development_path(root,name)
            original=Path.resolve
            outside=Path(folder)/'outside.png'
            def resolving(path,*args,**kwargs):
                if path.name=='image_0.png':
                    return outside
                return original(path,*args,**kwargs)
            with patch.object(Path,'resolve',resolving):
                with self.assertRaises(ProtectedDataError):
                    safe_development_path(root,'image_0.png')

    def test_invalid_mask_dimensions_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dev'
            rows=generate_fixture(root)
            Image.fromarray(np.full((64,64),10,np.uint8)).save(root/rows[0]['mask_rel'])
            self.assertTrue(audit(rows,root)['blockers'])
            Image.fromarray(np.full((10,10),9,np.uint8)).save(root/rows[0]['mask_rel'])
            self.assertTrue(any('dimensions' in s for s in audit(rows,root)['blockers']))
            rows[0]['calibration_um_per_px']='-1'
            with self.assertRaises(ValueError):
                preflight(rows,root)

    def test_report_outputs_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'output'
            report=run(output)
            self.assertEqual(report,json.loads((output/'session26_report.json').read_text()))
            self.assertTrue((output/'session26_domain_diagnostics.png').is_file())
            self.assertTrue((output/'artificial_development/development_manifest.csv').is_file())
            with self.assertRaises(FileExistsError):
                run(output)
            with self.assertRaises(ValueError):
                run(Path(folder)/'invalid',data_root=folder)


if __name__=='__main__':
    unittest.main()
