"""Grouped training, differentiable objectives, validation and saved-state selection."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session12_unet_architecture import SmallUNet
from session13_training_lab import GroupedToyDataset, foreground_soft_dice_loss, is_better, restore_checkpoint, run, training_class_weights, validate


class TrainingTests(unittest.TestCase):
    def test_grouped_geometry_partition(self):
        train, validation = GroupedToyDataset(range(8)), GroupedToyDataset(range(8, 12))
        self.assertEqual((len(train), len(validation)), (16, 8))
        self.assertFalse(set(train.groups) & set(validation.groups))
        for index in range(0, len(train), 2):
            self.assertTrue(torch.equal(train[index][1], train[index + 1][1]))
        self.assertTrue(torch.equal(train[0][0], train[0][0]))

    def test_tensor_shapes_and_dtypes(self):
        images, targets = next(iter(DataLoader(GroupedToyDataset([0, 1]), batch_size=4)))
        self.assertEqual(tuple(images.shape), (4, 1, 48, 48))
        self.assertEqual(tuple(targets.shape), (4, 48, 48))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.long)
        self.assertTrue(0 <= images.min() <= images.max() <= 1)

    def test_training_only_class_weights(self):
        data = [(torch.zeros(1, 2, 2), torch.tensor([[0, 0], [0, 1]], dtype=torch.long))]
        weights, counts = training_class_weights(data)
        self.assertEqual(counts.tolist(), [3, 1])
        self.assertEqual(weights.tolist(), [1., 3.])
        weights, _ = training_class_weights(GroupedToyDataset(range(8)))
        self.assertEqual(weights.tolist(), [1., 8.])

    def test_differentiable_dice_and_invalid_targets(self):
        logits = torch.zeros(2, 2, 8, 8, requires_grad=True)
        targets = torch.zeros(2, 8, 8, dtype=torch.long)
        targets[:, 2:6, 2:6] = 1
        loss = foreground_soft_dice_loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()), 0)
        with self.assertRaises(ValueError):
            foreground_soft_dice_loss(logits, targets.float())

    def test_validation_does_not_update_model(self):
        model = SmallUNet().train()
        before = {name: p.clone() for name, p in model.state_dict().items()}
        result = validate(model, DataLoader(GroupedToyDataset([8]), batch_size=2))
        self.assertTrue(model.training)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, before[name]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(result['tp'] + result['fp'] + result['fn'] + result['tn'], 2 * 48 * 48)

    def test_checkpoint_selection_and_ties(self):
        best = {"pooled_foreground_iou": .7, "unweighted_ce": .2}
        self.assertTrue(is_better({"pooled_foreground_iou": .8, "unweighted_ce": .9}, best))
        self.assertTrue(is_better({"pooled_foreground_iou": .7, "unweighted_ce": .1}, best))
        self.assertFalse(is_better(best.copy(), best))
        self.assertFalse(is_better({"pooled_foreground_iou": .6, "unweighted_ce": .1}, best))

    def test_training_checkpoint_restoration(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(output, epochs=1)
            restored, checkpoint = restore_checkpoint(output / "session13_best_state.pt")
            self.assertTrue(report["restoration_verified"])
            self.assertEqual(checkpoint["epoch"], report["best_epoch"])
            self.assertIn("optimizer_state_dict", checkpoint)
            actual = validate(restored, DataLoader(GroupedToyDataset(range(8, 12)), batch_size=4))
            self.assertEqual(actual, report["restored_validation"])

    def test_overwrite_and_config_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(folder)
            with self.assertRaises(ValueError):
                run(Path(folder) / "bad", epochs=0)
            self.assertFalse((Path(folder) / "bad").exists())


if __name__ == "__main__":
    unittest.main()
