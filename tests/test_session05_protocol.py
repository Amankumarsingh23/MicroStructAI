"""Evaluate split independence, protocol preservation and exact metric arithmetic."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session05_protocol import demo_records, evaluate, split_records, write_protocol


class ProtocolTests(unittest.TestCase):
    def test_deterministic_order_independent_groups_and_frozen_tests(self):
        records = demo_records()
        before = copy.deepcopy(records)
        result = split_records(records)
        self.assertEqual(result, split_records(list(reversed(records))))
        self.assertEqual(result["role_counts"], {"development_train": 45, "validation": 15, "frozen_test": 3})
        self.assertFalse(set(result["training_groups"]) & set(result["validation_groups"]))
        self.assertEqual(records, before)
        expected = {row["sample_id"] for row in records if row["split"] == "test"}
        self.assertEqual(expected, {row["sample_id"] for row in result["records"] if row["role"] == "frozen_test"})
        for row in result["records"]:
            original = next(r for r in before if r["sample_id"] == row["sample_id"])
            self.assertEqual({k: v for k, v in row.items() if k != "role"}, original)

    def test_unknown_training_groups_and_test_only_rejected(self):
        records = demo_records()
        records[0]["source_group_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "Unknown source_group"):
            split_records(records)
        with self.assertRaisesRegex(ValueError, "test-only"):
            split_records([row for row in records if row["split"] == "test"])

    def test_duplicate_ids_hashes_and_shared_test_lineage_rejected(self):
        records = demo_records()
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            split_records(records + [records[0]])
        records[0]["image_sha256"] = "same-file-hash"
        records[-1]["image_sha256"] = "same-file-hash"
        with self.assertRaisesRegex(ValueError, "hash shared"):
            split_records(records)
        del records[-1]["image_sha256"]
        records[-1]["source_group_id"] = records[0]["source_group_id"]
        with self.assertRaisesRegex(ValueError, "source group shared"):
            split_records(records)

    def test_small_group_and_fraction_boundaries(self):
        records = [{"sample_id": "a", "source_group_id": "A", "split": "train"},
                   {"sample_id": "b", "source_group_id": "B", "split": "train"}]
        for fraction in (0.001, 0.999):
            result = split_records(records, validation_fraction=fraction)
            self.assertEqual(len(result["validation_groups"]), 1)
            self.assertEqual(len(result["training_groups"]), 1)
        for fraction in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                split_records(records, validation_fraction=fraction)
        with self.assertRaisesRegex(ValueError, "two verified"):
            split_records(records[:1])

    def test_toy_metric_arithmetic(self):
        truth = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        prediction = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
        result = evaluate(truth, prediction)
        self.assertAlmostEqual(result["pixel_accuracy"], 10 / 12)
        self.assertAlmostEqual(result["per_class"]["0"]["iou"], 7 / 9)
        self.assertAlmostEqual(result["per_class"]["1"]["iou"], 0.6)
        self.assertAlmostEqual(result["per_class"]["1"]["dice"], 0.75)
        self.assertAlmostEqual(result["macro_iou"], (7 / 9 + 0.6) / 2)

    def test_absent_classes_false_positives_and_ten_class_schema(self):
        truth = np.zeros((2, 2), dtype=np.uint8)
        result = evaluate(truth, truth, labels=range(10))
        self.assertIsNone(result["per_class"]["9"]["iou"])
        self.assertEqual(result["macro_iou"], 1)
        result = evaluate(truth, np.ones_like(truth))
        self.assertEqual(result["per_class"]["1"]["iou"], 0)
        self.assertEqual(result["macro_iou"], 0)
        with self.assertRaisesRegex(ValueError, "schema"):
            evaluate(truth, np.full_like(truth, 255))
        with self.assertRaisesRegex(ValueError, "matching 2D"):
            evaluate(truth, truth[:1])
        with self.assertRaisesRegex(ValueError, "integer"):
            evaluate(truth.astype(float), truth)

    def test_immutable_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            write_protocol({"version": 1}, path)
            with self.assertRaises(FileExistsError):
                write_protocol({"version": 2}, path)
            self.assertEqual(json.loads(path.read_text()), {"version": 1})


if __name__ == "__main__":
    unittest.main()
