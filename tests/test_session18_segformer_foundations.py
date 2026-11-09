"""Eight tests for the independent educational SegFormer-inspired Session 18 lab."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session18_segformer_foundations import (OverlapPatchEmbed, ReducedSelfAttention,
    MixFeedForward, MiniSegFormer, original_to_binary, artificial_batch, smoke_step, run)


class SegFormerFoundationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_overlap_patch_embedding_and_odd_shapes(self):
        p = OverlapPatchEmbed(1, 16, 7, 4)
        tokens, h, w = p(torch.zeros(2, 1, 48, 48))
        self.assertEqual((h, w), (12, 12))
        self.assertEqual(tuple(tokens.shape), (2, 144, 16))
        tokens, h, w = p(torch.zeros(1, 1, 65, 67))
        self.assertEqual((h, w), (17, 17))
        self.assertEqual(tuple(tokens.shape), (1, 289, 16))
        self.assertEqual(p.proj.kernel_size, (7, 7))
        self.assertEqual(p.proj.stride, (4, 4))

    def test_spatially_reduced_attention_valid_probabilities(self):
        layer = ReducedSelfAttention(16, heads=2, reduction=2)
        x = torch.randn(2, 144, 16)
        y, attn, grid = layer(x, 12, 12, return_attention=True)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(tuple(attn.shape), (2, 2, 144, 36))
        self.assertEqual(grid, (6, 6))
        torch.testing.assert_close(attn.sum(-1), torch.ones(2, 2, 144), atol=1e-5, rtol=0)
        with self.assertRaises(ValueError):
            layer(x, 12, 11)
        with self.assertRaises(ValueError):
            ReducedSelfAttention(17, heads=2, reduction=2)

    def test_mix_ffn_local_depthwise_and_spatial_contract(self):
        ffn = MixFeedForward(16, 2)
        tokens = torch.randn(1, 12 * 12, 16, requires_grad=True)
        y = ffn(tokens, 12, 12)
        self.assertEqual(tuple(y.shape), tuple(tokens.shape))
        self.assertEqual(ffn.dwconv.groups, 32)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        with self.assertRaises(ValueError):
            ffn(tokens.detach(), 10, 10)

    def test_hierarchical_shape_and_decoder_trace(self):
        net = MiniSegFormer()
        logits, trace, attention = net(torch.rand(2, 1, 48, 48), return_trace=True)
        self.assertEqual(tuple(logits.shape), (2, 2, 48, 48))
        self.assertEqual(trace["first_tokens"], [2, 144, 16])
        self.assertEqual(trace["low_features"], [2, 16, 12, 12])
        self.assertEqual(trace["second_tokens"], [2, 36, 32])
        self.assertEqual(trace["high_features"], [2, 32, 6, 6])
        self.assertEqual(trace["attention"], [2, 2, 144, 36])
        self.assertEqual(trace["decoder_concatenation"], [2, 32, 12, 12])
        self.assertEqual(tuple(attention.shape), (6, 6))
        self.assertAlmostEqual(float(attention.detach().sum()), 1.0, places=5)
        self.assertFalse(any("pos_embed" in n or "position_embedding" in n
                             for n, _ in net.named_parameters()))

    def test_binary_ten_class_odd_and_input_guard(self):
        self.assertEqual(tuple(MiniSegFormer(10)(torch.rand(1, 1, 48, 48)).shape), (1, 10, 48, 48))
        self.assertEqual(tuple(MiniSegFormer()(torch.rand(1, 1, 65, 67)).shape), (1, 2, 65, 67))
        with self.assertRaises(ValueError):
            MiniSegFormer(classes=1)
        with self.assertRaises(ValueError):
            MiniSegFormer()(torch.rand(1, 3, 48, 48))
        with self.assertRaises(ValueError):
            MiniSegFormer()(torch.rand(1, 1, 15, 48))
        with self.assertRaises(ValueError):
            MiniSegFormer()(torch.full((1, 1, 48, 48), float('nan')))

    def test_original_ten_class_map_preserves_zero_and_source(self):
        old = np.array([[0, 1, 8, 9], [9, 0, 8, 5]], np.uint8)
        snapshot = old.copy()
        converted = original_to_binary(old)
        np.testing.assert_array_equal(converted, [[1, 1, 1, 0], [0, 1, 1, 1]])
        self.assertEqual(converted.dtype, np.uint8)
        np.testing.assert_array_equal(old, snapshot)
        with self.assertRaises(ValueError):
            original_to_binary(np.array([[0, 10]], np.uint8))

    def test_artificial_fixture_and_one_step_gradients(self):
        images, masks = artificial_batch()
        self.assertEqual(tuple(images.shape), (2, 1, 48, 48))
        self.assertEqual(tuple(masks.shape), (2, 48, 48))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(masks.dtype, torch.long)
        self.assertTrue(torch.equal(masks[0], masks[1]))
        report, arrays = smoke_step(seed=18)
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertLess(report["same_batch_loss_after_step"], report["initial_loss"])
        self.assertTrue(report["source_preserved"])
        self.assertTrue(all(v > 0 for v in report["gradient_norms_by_module"].values()))
        self.assertAlmostEqual(float(arrays["attention"].sum()), 1.0, places=5)

    def test_reporting_and_prevent_output_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "session18"
            report = run(path, seed=18)
            self.assertTrue((path / "session18_transformer_smoke.png").exists())
            self.assertTrue((path / "session18_report.json").exists())
            self.assertEqual(report["ten_class_output"], [1, 10, 48, 48])
            with self.assertRaises(FileExistsError):
                run(path)


if __name__ == "__main__":
    unittest.main()
