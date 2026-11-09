"""Paired label safety, train normalization and reproducible robustness runs."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session13_training_lab import GroupedToyDataset
from session15_augmentation_robustness import NormalizedToyDataset, normalization_statistics, paired_transform, run, stress_image


class AugmentationTests(unittest.TestCase):
    def test_paired_geometry_alignment_and_source_preservation(self):
        mask = np.zeros((12, 12), np.uint8)
        mask[2:5, 7:11] = 1
        image = mask.astype(np.float32)
        before_image, before_mask = image.copy(), mask.copy()
        transformed, labels = paired_transform(image, mask, np.random.default_rng(5), photometric=False)
        np.testing.assert_array_equal(transformed, labels)
        np.testing.assert_array_equal(image, before_image)
        np.testing.assert_array_equal(mask, before_mask)
        self.assertTrue(transformed.flags.c_contiguous and labels.flags.c_contiguous)

    def test_photometric_mask_preservation(self):
        image = np.full((12, 12), .5, np.float32)
        mask = np.zeros(image.shape, np.int64)
        mask[2:5, 3:6] = 1
        changed, labels = paired_transform(image, mask, np.random.default_rng(10), geometry=False)
        np.testing.assert_array_equal(mask, labels)
        self.assertFalse(np.array_equal(image, changed))
        self.assertTrue(np.isfinite(changed).all())
        self.assertTrue(np.all((changed >= 0) & (changed <= 1)))

    def test_original_orientation_semantics_guard(self):
        image = np.full((10, 10), .5, np.float32)
        labels = np.arange(100, dtype=np.int64).reshape(10, 10) % 10
        with self.assertRaisesRegex(ValueError, "angle remapping"):
            paired_transform(image, labels, np.random.default_rng(0), semantics="orientation")
        _, unchanged = paired_transform(image, labels, np.random.default_rng(0), semantics="orientation", geometry=False)
        np.testing.assert_array_equal(labels, unchanged)
        with self.assertRaises(ValueError):
            paired_transform(image, labels, np.random.default_rng(0), semantics="binary")

    def test_training_only_normalization(self):
        training = [(torch.full((1, 2, 2), .25), torch.zeros((2, 2), dtype=torch.long)),
                    (torch.full((1, 2, 2), .75), torch.zeros((2, 2), dtype=torch.long))]
        statistics = normalization_statistics(training)
        self.assertEqual(statistics["mean"], .5)
        self.assertEqual(statistics["std"], .25)
        holdout = [(torch.ones(1, 2, 2), torch.zeros((2, 2), dtype=torch.long))]
        validation = NormalizedToyDataset(holdout, statistics)
        self.assertTrue(torch.equal(validation[0][0], torch.full((1, 2, 2), 2.)))
        self.assertEqual(statistics["mean"], .5)

    def test_epoch_dependent_determinism_and_fixed_validation(self):
        training = GroupedToyDataset(range(8))
        statistics = normalization_statistics(training)
        augmented = NormalizedToyDataset(training, statistics, augmented=True)
        augmented.set_epoch(1)
        first = augmented[0]
        repeated = augmented[0]
        self.assertTrue(torch.equal(first[0], repeated[0]) and torch.equal(first[1], repeated[1]))
        augmented.set_epoch(2)
        self.assertFalse(torch.equal(first[0], augmented[0][0]))
        validation = NormalizedToyDataset(GroupedToyDataset(range(8, 12)), statistics)
        before = validation[0]
        validation.set_epoch(5)
        self.assertTrue(torch.equal(before[0], validation[0][0]) and torch.equal(before[1], validation[0][1]))

    def test_stress_conditions_and_invalid_input(self):
        image = np.full((8, 8), .5, np.float32)
        self.assertAlmostEqual(float(stress_image(image, "dim", 0).mean()), .25)
        self.assertAlmostEqual(float(stress_image(image, "bright", 0).mean()), .65, places=6)
        np.testing.assert_array_equal(stress_image(image, "noisy", 2), stress_image(image, "noisy", 2))
        with self.assertRaises(ValueError):
            stress_image(image, "unknown", 0)
        with self.assertRaises(ValueError):
            stress_image(np.full((8, 8), np.nan), "dim", 0)

    def test_comparison_initial_states_and_checkpoint_restoration(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(Path(folder) / "run", epochs=1)
            first = report["results"]["unaugmented"]
            second = report["results"]["augmented"]
            self.assertEqual(first["initial_state_sha256"], second["initial_state_sha256"])
            for result in (first, second):
                self.assertTrue(result["restoration_verified"])
                self.assertEqual(result["stresses"]["nominal"], result["selected_validation"])
            self.assertFalse(set(report["train_groups"]) & set(report["validation_groups"]))

    def test_output_and_configuration_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(folder)
            with self.assertRaises(ValueError):
                run(Path(folder) / "invalid", epochs=0)
            self.assertFalse((Path(folder) / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
