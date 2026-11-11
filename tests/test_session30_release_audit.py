"""Session 30 independent saved-record and release-integrity regression tests."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from session29_end_to_end_validation import run as generate_pipeline
from session30_release_audit import audit, run, verify_release, safe_path, digest


class ReleaseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=tempfile.TemporaryDirectory()
        cls.base=Path(cls.fixture.name)/'base'
        generate_pipeline(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'ledger'
        shutil.copytree(self.base,self.root)

    def change_json(self,name,change):
        path=self.root/name
        value=json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value),encoding='utf-8')

    def change_csv(self,name,change):
        path=self.root/name
        with path.open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        change(rows)
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def test_actual_recomputation_and_read_only_input(self):
        before={str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()}
        result=audit(self.root)
        self.assertEqual(result['checksum_entries_verified'],540)
        self.assertEqual(result['role_counts'],{'fit':18,'calibration':9,'evaluation':9})
        self.assertAlmostEqual(result['recomputed_mae_au'],4.202506495900435)
        self.assertEqual(result['recomputed_coverage'],1)
        self.assertEqual(before,{str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()})

    def test_file_byte_tampering(self):
        image=next((self.root/'segmentation_masks').glob('*.png'))
        with image.open('ab') as handle:
            handle.write(b'tampered')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            audit(self.root)

    def test_protocol_tampering(self):
        self.change_json('session29_frozen_protocol.json',lambda r:r['protocol'].update(alpha=.2))
        with self.assertRaisesRegex(ValueError,'Protocol seal'):
            audit(self.root)

    def test_protected_paths_and_resolved_escape_before_image_reads(self):
        self.change_csv('session29_source_manifest.csv',lambda rows:rows[-1].update(split='test'))
        with patch('PIL.Image.open') as opened:
            with self.assertRaises(ValueError):
                audit(self.root)
            opened.assert_not_called()
        with self.assertRaises(ValueError):
            safe_path(self.root,'../outside.png')
        original=Path.resolve
        def resolve(path,*args,**kwargs):
            if path.name=='session29_report.json':
                return self.root.parent/'outside.json'
            return original(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolve):
            with self.assertRaises(ValueError):
                safe_path(self.root,'session29_report.json')

    def test_cross_role_source_reuse(self):
        def overlap(rows):
            fit=next(r for r in rows if r['role']=='fit')
            other=next(r for r in rows if r['role']=='evaluation')
            other['texture_id']=fit['texture_id']
        self.change_csv('session29_source_manifest.csv',overlap)
        with self.assertRaisesRegex(ValueError,'Cross-role'):
            audit(self.root)

    def test_saved_statistics_and_field_counts_recomputed(self):
        self.change_json('session29_report.json',lambda r:r.update(evaluation_mae_au=0))
        with self.assertRaisesRegex(ValueError,'reported MAE'):
            audit(self.root)
        shutil.copyfile(self.base/'session29_report.json',self.root/'session29_report.json')
        self.change_csv('session29_field_features.csv',lambda rows:rows[0].update(foreground_pixels=0))
        with self.assertRaisesRegex(ValueError,'field foreground'):
            audit(self.root)

    def test_broken_specimen_trace(self):
        self.change_json('session29_report.json',lambda r:r['trace'][0].update(field_ids=['invalid']*3))
        with self.assertRaisesRegex(ValueError,'Broken field'):
            audit(self.root)

    def test_release_integrity_and_overwrite(self):
        output=Path(self.temp.name)/'release'
        run(self.root,output)
        self.assertTrue(verify_release(output))
        self.assertEqual(len(list(output.iterdir())),5)
        with self.assertRaises(FileExistsError):
            run(self.root,output)
        with (output/'session30_project_summary.md').open('a') as handle:
            handle.write('changed')
        with self.assertRaisesRegex(ValueError,'artifact checksum'):
            verify_release(output)


if __name__=='__main__':
    unittest.main()
