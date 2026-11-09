"""Controlled model contracts, shared data, restored states and CPU measurements."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session13_training_lab import GroupedToyDataset, is_better, save_checkpoint, training_class_weights
from session14_evaluation_lab import scene_metrics, thin_diagnostic
from session15_augmentation_robustness import NormalizedToyDataset, normalization_statistics
from session17_architecture_comparison import build_model, cpu_latency, restore_model, run


class ArchitectureComparisonTests(unittest.TestCase):
    def test_both_architecture_shapes_and_parameters(self):
        for name, expected in (("unet", 29330), ("mini_deeplab", 10642)):
            model = build_model(name)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(tuple(model(torch.zeros(1, 1, 48, 48)).shape), (1, 2, 48, 48))
        with self.assertRaises(ValueError):
            build_model("unknown")

    def test_shared_grouped_normalization_contract(self):
        training = GroupedToyDataset(range(8))
        validation = GroupedToyDataset(range(8, 12))
        statistics = normalization_statistics(training)
        first = NormalizedToyDataset(validation, statistics)
        second = NormalizedToyDataset(validation, statistics)
        self.assertEqual(first.groups, validation.groups)
        self.assertFalse(set(training.groups) & set(first.groups))
        for index in range(len(first)):
            self.assertTrue(torch.equal(first[index][0], second[index][0]))
            self.assertTrue(torch.equal(first[index][1], second[index][1]))
        self.assertEqual(first.statistics, statistics)
        self.assertEqual(training_class_weights(training)[0].tolist(), [1., 8.])

    def test_thin_recall_and_overlap_metrics(self):
        truth, prediction, thin = thin_diagnostic()
        metrics = scene_metrics(truth, prediction, np.ones(truth.shape, bool), thin)
        self.assertAlmostEqual(metrics["iou"], 3000 / 3040)
        self.assertEqual(metrics["thin_feature_recall"], 0)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 40)

    def test_model_specific_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("unet", "mini_deeplab"):
                model = build_model(name).eval()
                optimizer = torch.optim.Adam(model.parameters(), lr=.005)
                path = Path(folder) / f"{name}.pt"
                sample = torch.ones(1, 1, 48, 48)
                with torch.no_grad():
                    expected = model(sample)
                save_checkpoint(path, model, optimizer, 2, {"pooled_foreground_iou": .5}, {"architecture": name})
                with torch.no_grad():
                    next(model.parameters()).add_(1)
                restored, checkpoint = restore_model(path, name)
                self.assertEqual(checkpoint["epoch"], 2)
                with torch.no_grad():
                    torch.testing.assert_close(restored(sample), expected, rtol=0, atol=0)
                other = "unet" if name == "mini_deeplab" else "mini_deeplab"
                with self.assertRaises(ValueError):
                    restore_model(path, other)

    def test_validation_selection_tie_policy(self):
        best = {"pooled_foreground_iou": .7, "unweighted_ce": .2}
        self.assertTrue(is_better({"pooled_foreground_iou": .8, "unweighted_ce": .4}, best))
        self.assertTrue(is_better({"pooled_foreground_iou": .7, "unweighted_ce": .1}, best))
        self.assertFalse(is_better(dict(best), best))

    def test_warmed_cpu_latency_contract(self):
        model = torch.nn.Conv2d(1, 2, 1).eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        metrics = cpu_latency(model, torch.zeros(1, 1, 48, 48))
        self.assertEqual(metrics["repetitions"], 15)
        self.assertEqual(metrics["warmup"], 5)
        self.assertGreater(metrics["median_ms_per_image"], 0)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))
        with self.assertRaises(ValueError):
            cpu_latency(model, torch.zeros(2, 1, 48, 48))

    def test_end_to_end_restoration_for_both_models(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(output, epochs=1, seeds=(17,))
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(len(list(output.glob("*.pt"))), 2)
            self.assertEqual(report["fixed_120_baseline"]["per_class"]["1"]["iou"], 1)
            for result in report["runs"]:
                self.assertTrue(result["restoration_verified"])
                self.assertEqual(result["best_epoch"], 1)
                self.assertEqual(len(result["per_scene"]), 8)
                self.assertAlmostEqual(result["selected_validation"]["pooled_foreground_iou"],
                                       result["evaluation"]["per_class"]["1"]["iou"])
                self.assertEqual(result["cpu_latency"]["cpu_threads"], 1)

    def test_output_and_seed_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileExistsError):
                run(root)
            for seeds in ((), (17, 17), (-1,)):
                with self.assertRaises(ValueError):
                    run(root / "invalid", seeds=seeds)
            with self.assertRaises(ValueError):
                run(root / "invalid", epochs=0)
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
