"""Atrous sampling, ASPP shapes, separable weights and end-to-end gradients."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session11_pytorch_basics import validate_targets
from session16_deeplab_architecture import DepthwiseSeparableConv, MiniASPP, MiniDeepLabV3Plus, run, smoke_step


class DeepLabTests(unittest.TestCase):
    def test_dilation_sampling_and_parameter_count(self):
        image = torch.zeros(1, 1, 25, 25)
        image[0, 0, 12, 12] = 1
        for rate in (1, 2, 3, 4):
            layer = nn.Conv2d(1, 1, 3, dilation=rate, padding=rate, bias=False)
            with torch.no_grad():
                layer.weight.fill_(1)
            output = layer(image)
            self.assertEqual(tuple(output.shape), (1, 1, 25, 25))
            positions = torch.nonzero(output[0, 0])
            self.assertEqual(len(positions), 9)
            self.assertEqual(layer.weight.numel(), 9)
            self.assertEqual(int(positions[:, 0].max() - positions[:, 0].min() + 1), 2 * rate + 1)

    def test_separable_weight_savings(self):
        block = DepthwiseSeparableConv(16, 32)
        spatial = block.depthwise.weight.numel()
        pointwise = block.pointwise.weight.numel()
        self.assertEqual((spatial, pointwise, spatial + pointwise), (144, 512, 656))
        self.assertEqual(nn.Conv2d(16, 32, 3, bias=False).weight.numel(), 4608)
        self.assertIsNone(block.depthwise.bias)

    def test_aspp_branches_and_global_pool(self):
        result, trace = MiniASPP()(torch.randn(2, 32, 12, 12), return_trace=True)
        self.assertEqual(trace["branch_shapes"], [[2, 8, 12, 12]] * 5)
        self.assertEqual(trace["global_pooled_shape"], [2, 8, 1, 1])
        self.assertEqual(trace["concatenated_shape"], [2, 40, 12, 12])
        self.assertEqual(tuple(result.shape), (2, 16, 12, 12))

    def test_exact_parameters_and_decoder_shapes(self):
        model = MiniDeepLabV3Plus()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10642)
        output, trace = model(torch.zeros(2, 1, 48, 48), return_trace=True)
        self.assertEqual(trace["low_encoder"], [2, 8, 24, 24])
        self.assertEqual(trace["high_encoder"], [2, 32, 12, 12])
        self.assertEqual(trace["decoder_concatenation"], [2, 20, 24, 24])
        self.assertEqual(tuple(output.shape), (2, 2, 48, 48))

    def test_odd_sizes_and_ten_class_targets(self):
        model = MiniDeepLabV3Plus()
        self.assertEqual(tuple(model(torch.zeros(1, 1, 65, 67)).shape), (1, 2, 65, 67))
        logits = MiniDeepLabV3Plus(10)(torch.zeros(1, 1, 48, 48))
        targets = torch.full((1, 48, 48), 9, dtype=torch.long)
        validate_targets(logits, targets)
        self.assertTrue(torch.isfinite(nn.CrossEntropyLoss()(logits, targets)))
        self.assertEqual(tuple(logits.shape), (1, 10, 48, 48))
        with self.assertRaises(ValueError):
            validate_targets(model(torch.zeros(1, 1, 48, 48)), targets)

    def test_finite_gradients_updates_and_source_preservation(self):
        report, _ = smoke_step()
        repeated, _ = smoke_step()
        self.assertEqual(report["initial_loss"], repeated["initial_loss"])
        self.assertLess(report["same_batch_loss_after_step"], report["initial_loss"])
        self.assertTrue(report["source_tensors_preserved"])
        for value in report["block_gradient_norm_sums"].values():
            self.assertGreater(value, 0)
        self.assertTrue(any(value > 0 for value in report["parameter_update_norms"].values()))

    def test_invalid_input_and_dilation(self):
        for image in (torch.zeros(1, 2, 48, 48), torch.zeros(1, 1, 3, 48), torch.zeros(1, 1, 48, 48).long()):
            with self.assertRaises(ValueError):
                MiniDeepLabV3Plus()(image)
        with self.assertRaises(ValueError):
            DepthwiseSeparableConv(16, 32, dilation=0)
        with self.assertRaises(ValueError):
            MiniDeepLabV3Plus(classes=1)

    def test_output_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(folder)


if __name__ == "__main__":
    unittest.main()
