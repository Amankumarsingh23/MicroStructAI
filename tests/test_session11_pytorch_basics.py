"""Tensor contracts, label semantics and actual optimizer behavior."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session11_pytorch_basics import TinyPixelNet, ToyMicrographDataset, binary_from_orientation, one_step, run, toy_pair, validate_targets


class PytorchTests(unittest.TestCase):
    def test_orientation_mapping_preserves_original(self):
        original = np.arange(10, dtype=np.uint8)[None, :]
        before = original.copy()
        result = binary_from_orientation(original)
        np.testing.assert_array_equal(result, [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
        np.testing.assert_array_equal(original, before)

    def test_invalid_labels_and_dataset_indices(self):
        for labels in (np.array([[255]], np.uint8), np.array([[1.5]]), np.array([[-1]])):
            with self.assertRaises(ValueError):
                binary_from_orientation(labels)
        with self.assertRaises(IndexError):
            ToyMicrographDataset()[8]

    def test_deterministic_data(self):
        for a, b in zip(toy_pair(0), toy_pair(0)):
            np.testing.assert_array_equal(a, b)
        dataset = ToyMicrographDataset()
        image, target = dataset[0]
        image.fill_(0)
        self.assertGreater(float(dataset[0][0].sum()), 0)

    def test_loader_shapes_and_types(self):
        images, targets = next(iter(DataLoader(ToyMicrographDataset(), batch_size=2, num_workers=0)))
        self.assertEqual(tuple(images.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(targets.shape), (2, 64, 64))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.long)
        self.assertGreaterEqual(float(images.min()), 0)
        self.assertLessEqual(float(images.max()), 1)

    def test_convolution_dimensions_and_parameter_count(self):
        model = TinyPixelNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 50)
        self.assertEqual(tuple(model(torch.zeros(2, 1, 64, 64)).shape), (2, 2, 64, 64))
        self.assertEqual(tuple(nn.Conv2d(1, 4, 3)(torch.zeros(2, 1, 64, 64)).shape), (2, 4, 62, 62))

    def test_cross_entropy_contract_and_ten_class_outputs(self):
        logits = TinyPixelNet(10)(torch.zeros(2, 1, 64, 64))
        targets = torch.full((2, 64, 64), 9, dtype=torch.long)
        validate_targets(logits, targets)
        self.assertTrue(torch.isfinite(nn.CrossEntropyLoss()(logits, targets)))
        for invalid in (targets.float(), targets.unsqueeze(1), targets + 1):
            with self.assertRaises(ValueError):
                validate_targets(logits, invalid)

    def test_optimizer_update_gradients_and_reproducibility(self):
        report, arrays = one_step()
        other, other_arrays = one_step()
        self.assertTrue(report["parameters_changed"])
        self.assertGreater(report["sum_gradient_norms"], 0)
        self.assertLess(report["same_batch_loss_after_step"], report["initial_loss"])
        self.assertEqual(report["initial_loss"], other["initial_loss"])
        np.testing.assert_array_equal(arrays["probability"], other_arrays["probability"])

    def test_no_output_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(folder)


if __name__ == "__main__":
    unittest.main()
