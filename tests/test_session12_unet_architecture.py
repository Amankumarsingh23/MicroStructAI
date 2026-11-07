"""Architecture contracts, skips, alignment and end-to-end differentiation."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session12_unet_architecture import DoubleConv, SmallUNet, UpBlock, run, smoke_step


class UNetTests(unittest.TestCase):
    def test_double_conv_spatial_shape(self):
        self.assertEqual(tuple(DoubleConv(1, 8)(torch.zeros(2, 1, 64, 64)).shape), (2, 8, 64, 64))

    def test_skip_concatenation(self):
        result, trace = UpBlock(32, 16, 16)(torch.zeros(2, 32, 16, 16), torch.zeros(2, 16, 32, 32))
        self.assertEqual(trace["concatenated_shape"], [2, 32, 32, 32])
        self.assertFalse(trace["interpolated"])
        self.assertEqual(tuple(result.shape), (2, 16, 32, 32))

    def test_exact_parameters_and_binary_output(self):
        model = SmallUNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 29330)
        logits, trace = model(torch.zeros(2, 1, 64, 64), True)
        self.assertEqual(tuple(logits.shape), (2, 2, 64, 64))
        self.assertEqual(trace["bottleneck"], [2, 32, 16, 16])

    def test_ten_class_output_and_loss(self):
        logits = SmallUNet(10)(torch.zeros(1, 1, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 10, 64, 64))
        loss = torch.nn.CrossEntropyLoss()(logits, torch.full((1, 64, 64), 9, dtype=torch.long))
        self.assertTrue(torch.isfinite(loss))

    def test_odd_spatial_alignment(self):
        source = torch.randn(1, 1, 65, 67)
        before = source.clone()
        logits, trace = SmallUNet()(source, True)
        self.assertEqual(tuple(logits.shape), (1, 2, 65, 67))
        self.assertTrue(trace["up2"]["interpolated"])
        self.assertTrue(trace["up1"]["interpolated"])
        self.assertTrue(torch.equal(source, before))

    def test_gradients_updates_and_deterministic_smoke(self):
        report, _ = smoke_step()
        repeat, _ = smoke_step()
        self.assertEqual(report["initial_loss"], repeat["initial_loss"])
        self.assertLess(report["same_batch_loss_after_step"], report["initial_loss"])
        for block in ("enc1", "enc2", "bottleneck", "up2", "up1", "head"):
            self.assertGreater(report["block_gradient_norm_sums"][block], 0)
        self.assertTrue(any(v > 0 for v in report["parameter_update_norms"].values()))
        self.assertTrue(report["source_tensors_preserved"])

    def test_invalid_inputs_and_class_count(self):
        for source in (torch.zeros(1, 3, 64, 64), torch.zeros(1, 1, 3, 64), torch.zeros(1, 1, 64, 64).long()):
            with self.assertRaises(ValueError):
                SmallUNet()(source)
        with self.assertRaises(ValueError):
            SmallUNet(1)

    def test_immutable_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(folder)


if __name__ == "__main__":
    unittest.main()
