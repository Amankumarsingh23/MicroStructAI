"""Controlled failure cases for scientific dataset validation."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session03_dataset_audit import audit, inspect_pair, make_demo


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "dataset"
        self.manifest = self.root.parent / "manifest.csv"
        for split in ("train", "test"):
            for kind in ("images", "masks"):
                (self.root / split / kind).mkdir(parents=True)
        self.image = self.root / "train/images/a.png"
        self.mask = self.root / "train/masks/a.png"
        Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(self.image)
        Image.fromarray(np.full((4, 4), 9, dtype=np.uint8)).save(self.mask)

    def run_audit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit(self.root, self.manifest)

    def test_manifest_reproducibility_and_unknowns(self):
        rows = self.run_audit()
        original = self.manifest.read_bytes()
        self.run_audit()
        self.assertEqual(original, self.manifest.read_bytes())
        self.assertEqual(rows[0]["specimen_id"], "UNKNOWN")
        self.assertEqual(rows[0]["pixel_scale_um"], "")
        self.assertEqual(rows[0]["split"], "train")

    def test_invalid_id_does_not_replace_last_successful_manifest(self):
        self.run_audit()
        original = self.manifest.read_bytes()
        Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(self.mask)
        with self.assertRaisesRegex(ValueError, "Invalid class IDs.*255"):
            self.run_audit()
        self.assertEqual(original, self.manifest.read_bytes())

    def test_dimensions_must_match(self):
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(self.mask)
        with self.assertRaisesRegex(ValueError, "Mismatched dimensions"):
            self.run_audit()

    def test_missing_pair_and_orphan_mask(self):
        self.mask.unlink()
        with self.assertRaisesRegex(ValueError, "Unpaired filenames"):
            self.run_audit()
        self.image.unlink()
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(self.mask)
        with self.assertRaisesRegex(ValueError, "missing images"):
            self.run_audit()

    def test_exact_cross_split_duplicate(self):
        (self.root / "test/images/a.png").write_bytes(self.image.read_bytes())
        (self.root / "test/masks/a.png").write_bytes(self.mask.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate across splits"):
            self.run_audit()

    def test_palette_indices_are_preserved(self):
        mask = Image.fromarray(np.full((4, 4), 9, dtype=np.uint8)).convert("P")
        mask.putpalette([255, 255, 255] * 256)
        mask.save(self.mask)
        self.assertEqual(inspect_pair(self.image, self.mask)["classes"], "9")

    def test_rgb_mask_rejected(self):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(self.mask)
        with self.assertRaisesRegex(ValueError, "Unsupported mask"):
            self.run_audit()

    def test_16bit_image_rejected(self):
        Image.fromarray(np.full((4, 4), 1024, dtype=np.uint16)).save(self.image)
        with self.assertRaisesRegex(ValueError, "Unsupported image"):
            self.run_audit()

    def test_corrupt_image_rejected(self):
        self.image.write_bytes(b"not a PNG")
        with self.assertRaises(OSError):
            self.run_audit()

    def test_demo_cannot_overwrite_research_directory(self):
        with self.assertRaisesRegex(ValueError, "never a research directory"):
            make_demo(self.root)


if __name__ == "__main__":
    unittest.main()
