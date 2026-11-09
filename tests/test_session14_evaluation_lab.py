"""Evaluation arithmetic, geometry, group resampling and checkpoint preservation."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session12_unet_architecture import SmallUNet
from session13_training_lab import GroupedToyDataset, validate
from session14_evaluation_lab import boundary_metrics, class_metrics, confusion_matrix, group_bootstrap, infer_scenes, run, scene_metrics, summarize, thin_diagnostic


class EvaluationTests(unittest.TestCase):
    def test_confusion_and_multiclass_metrics(self):
        matrix = np.array([[8900, 100], [200, 800]], dtype=np.int64)
        metrics = class_metrics(matrix)
        foreground = metrics["per_class"]["1"]
        self.assertAlmostEqual(metrics["pixel_accuracy"], .97)
        self.assertAlmostEqual(foreground["precision"], 8 / 9)
        self.assertAlmostEqual(foreground["recall"], .8)
        self.assertAlmostEqual(foreground["iou"], 8 / 11)
        self.assertAlmostEqual(foreground["dice"], 16 / 19)
        truth = np.array([[0, 9]], dtype=np.uint8)
        matrix = confusion_matrix(truth, truth, np.ones(truth.shape, bool), 10)
        self.assertEqual(matrix[9, 9], 1)
        self.assertIsNone(class_metrics(matrix)["per_class"]["8"]["iou"])
        self.assertEqual(class_metrics(matrix)["macro_class_iou"], 1)

    def test_roi_and_invalid_labels(self):
        truth = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        roi = np.array([[True, True], [False, False]])
        matrix = confusion_matrix(truth, np.zeros_like(truth), roi)
        np.testing.assert_array_equal(matrix, [[1, 0], [1, 0]])
        with self.assertRaises(ValueError):
            confusion_matrix(truth, truth.astype(float), roi)
        with self.assertRaises(ValueError):
            confusion_matrix(truth, truth, np.zeros_like(roi))
        with self.assertRaises(ValueError):
            confusion_matrix(truth, np.full_like(truth, 255), roi)

    def test_scene_macro_and_pooled_difference(self):
        rows = []
        cases = [(np.array([[1, 0, 0, 0]], np.uint8), np.array([[1, 0, 0, 0]], np.uint8)),
                 (np.array([[1, 1, 1, 0]], np.uint8), np.array([[1, 0, 0, 1]], np.uint8))]
        for index, (truth, prediction) in enumerate(cases):
            rows.append({"method": "toy", "group_id": index, **scene_metrics(truth, prediction, np.ones(truth.shape, bool))})
        summary = summarize(rows)["toy"]
        self.assertAlmostEqual(summary["scene_macro_foreground_iou"], .625)
        self.assertAlmostEqual(summary["per_class"]["1"]["iou"], .4)

    def test_boundary_tolerance_and_empty_cases(self):
        truth = np.zeros((20, 20), np.uint8)
        truth[5:15, 5:15] = 1
        prediction = np.roll(truth, 1, axis=1)
        roi = np.ones(truth.shape, bool)
        exact = boundary_metrics(truth, prediction, roi, tolerance=0)
        tolerant = boundary_metrics(truth, prediction, roi, tolerance=1.5)
        self.assertLess(exact["boundary_f1"], tolerant["boundary_f1"])
        self.assertEqual(tolerant["boundary_f1"], 1)
        empty = np.zeros_like(truth)
        self.assertIsNone(boundary_metrics(empty, empty, roi)["boundary_f1"])
        self.assertEqual(boundary_metrics(truth, empty, roi)["boundary_f1"], 0)

    def test_thin_diagnostic(self):
        truth, prediction, thin = thin_diagnostic()
        result = scene_metrics(truth, prediction, np.ones(truth.shape, bool), thin)
        self.assertEqual(int(thin.sum()), 40)
        self.assertAlmostEqual(result["iou"], 3000 / 3040)
        self.assertEqual(result["thin_feature_recall"], 0)

    def test_group_bootstrap_preserves_views(self):
        # Each whole group has mean .5, while individual views differ.
        rows = [{"group_id": group, "iou": value} for group in range(4) for value in (0., 1.)]
        first = group_bootstrap(rows)
        self.assertEqual(first, group_bootstrap(rows))
        self.assertEqual(first["scene_macro_iou_percentile_95"], [.5, .5])
        self.assertEqual(first["defined_replicates"], 2000)

    def test_threshold_monotonicity_and_parameter_preservation(self):
        model = SmallUNet()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        scenes = infer_scenes(model, GroupedToyDataset([8]))
        for scene in scenes:
            low = scene["probability"] > .3
            high = scene["probability"] > .7
            self.assertTrue(np.all(~high | low))
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))

    def test_checkpoint_evaluation_and_output_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = SmallUNet()
            validation_groups = list(range(8, 12))
            metrics = validate(model, DataLoader(GroupedToyDataset(validation_groups), batch_size=4))
            checkpoint_path = root / "state.pt"
            torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "validation": metrics,
                        "config": {"train_groups": list(range(8)), "validation_groups": validation_groups}}, checkpoint_path)
            before = checkpoint_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(root / "evaluation", checkpoint_path)
            self.assertTrue(report["saved_iou_reproduced"])
            self.assertEqual(before, checkpoint_path.read_bytes())
            self.assertEqual(len(report["methods"]), 3)
            self.assertEqual(report["methods"]["U-Net"]["group_bootstrap"]["group_count"], 4)
            with self.assertRaises(FileExistsError):
                run(root / "evaluation", checkpoint_path)


if __name__ == "__main__":
    unittest.main()
