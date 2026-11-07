"""Versioned grouped development splits and categorical segmentation metrics."""

import argparse
import csv
import json
from pathlib import Path
import platform
import random
import subprocess

import numpy as np

from session03_dataset_audit import ROOT, checksum

VERSION = "session05-v1"
UNKNOWN = {"", "unknown", "none", "null", "n/a"}


def split_records(records, seed=42, validation_fraction=0.25):
    """Require supplied lineage for development records; preserve official tests."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be strictly between 0 and 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Seed must be an integer")
    records = [dict(row) for row in records]
    if not records:
        raise ValueError("Manifest has no records")
    ids, seen_hashes = set(), {}
    for row in records:
        sample = row.get("sample_id", "").strip()
        if not sample or sample in ids:
            raise ValueError(f"Missing or duplicate sample_id: {sample}")
        ids.add(sample)
        if row.get("split") not in {"train", "test"}:
            raise ValueError("Original split must be train or test; do not rewrite official membership")
        if "role" in row:
            raise ValueError("Input already has a role; supply the original manifest")
        digest = row.get("image_sha256", "").strip()
        if digest and digest.casefold() not in UNKNOWN:
            previous = seen_hashes.get(digest)
            if previous is not None and previous != row["split"]:
                raise ValueError("Exact image hash shared across official train/test")
            seen_hashes[digest] = row["split"]
    development = [row for row in records if row["split"] == "train"]
    if not development:
        raise ValueError("No official training records; a test-only manifest cannot form development splits")
    for row in development:
        group = row.get("source_group_id", "").strip()
        if group.casefold() in UNKNOWN:
            raise ValueError(f"Unknown source_group_id for training record {row['sample_id']}; verified lineage is required")
        if group != row["source_group_id"]:
            raise ValueError("Group identifiers must not have surrounding whitespace")
    groups = sorted({row["source_group_id"] for row in development})
    if len(groups) < 2:
        raise ValueError("At least two verified development groups are required")
    test_groups = {row.get("source_group_id", "").strip() for row in records
                   if row["split"] == "test" and row.get("source_group_id", "").strip().casefold() not in UNKNOWN}
    if set(groups) & test_groups:
        raise ValueError("Known source group shared across official train/test")
    random.Random(seed).shuffle(groups)
    # Python round uses ties-to-even; clamp to keep both roles nonempty.
    n_validation = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:n_validation])
    training_groups = set(groups[n_validation:])
    assert not validation_groups & training_groups
    for row in records:
        row["role"] = ("frozen_test" if row["split"] == "test" else
                       "validation" if row["source_group_id"] in validation_groups else "development_train")
    records.sort(key=lambda row: row["sample_id"])
    return {
        "seed": seed, "validation_fraction_of_groups_requested": validation_fraction,
        "rounding": "round (ties to even), clamped to [1, number_of_groups - 1]",
        "training_groups": sorted(training_groups), "validation_groups": sorted(validation_groups),
        "role_counts": {role: sum(row["role"] == role for row in records)
                        for role in ("development_train", "validation", "frozen_test")},
        "records": records,
        "lineage_limit": "Groups are accepted from supplied metadata, not independently verified by this program. "
                         "Unknown test lineage prevents a claim of source independence across official splits.",
    }


def _validate_masks(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or not truth.size:
        raise ValueError("Masks must be nonempty matching 2D arrays")
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("Categorical masks must contain integer class IDs")
    return truth, prediction


def class_metrics(truth, prediction, label):
    truth, prediction = _validate_masks(truth, prediction)
    actual, predicted = truth == label, prediction == label
    tp = int(np.count_nonzero(actual & predicted))
    fp = int(np.count_nonzero(~actual & predicted))
    fn = int(np.count_nonzero(actual & ~predicted))
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
            "iou": tp / union if union else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if union else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def evaluate(truth, prediction, labels=(0, 1)):
    truth, prediction = _validate_masks(truth, prediction)
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, (int, np.integer)) for label in labels):
        raise ValueError("Provide distinct integer class labels")
    if (set(np.unique(truth)) | set(np.unique(prediction))) - set(labels):
        raise ValueError("Mask contains IDs outside the declared label schema; no implicit ignore labels")
    per_class = {str(label): class_metrics(truth, prediction, label) for label in labels}
    valid = [result for result in per_class.values() if result["iou"] is not None]
    return {"pixel_accuracy": float(np.mean(truth == prediction)), "per_class": per_class,
            "macro_iou": float(np.mean([result["iou"] for result in valid])),
            "macro_dice": float(np.mean([result["dice"] for result in valid])),
            "absent_class_policy": "Absent in both truth and prediction: null, excluded from macro averages. "
                                   "Present in only one: zero overlap, included.",
            "aggregation": "Metrics on the supplied mask pair; not mean-per-image or specimen aggregation."}


def demo_records():
    records = [{"sample_id": f"demo-{group:02d}-{image}", "split": "train",
                "source_group_id": f"ARTIFICIAL-group-{group:02d}"}
               for group in range(12) for image in range(5)]
    records.extend({"sample_id": f"ARTIFICIAL-test-{i}", "split": "test", "source_group_id": "UNKNOWN"}
                   for i in range(3))
    return records


def build_protocol(records, seed=42, validation_fraction=0.25, source=None):
    split = split_records(records, seed, validation_fraction)
    truth = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return {"protocol_version": VERSION, "source": source or "ARTIFICIAL metadata only",
            "split": split, "toy_binary_metrics": evaluate(truth, prediction),
            "labels": {"original": "IDs 0..8: pearlite geometric orientation bins; 9: ferrite",
                       "derived_binary": "0=ferrite (original 9); 1=pearlite (original 0..8); preserve originals"},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "platform": platform.platform(), "hardware": "NOT_RECORDED; no model training"},
            "code": {"git_commit": git.stdout.strip() if git.returncode == 0 else "UNKNOWN",
                     "working_tree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
                     "script_sha256": checksum(Path(__file__))},
            "benchmark_status": "Engineering demonstration only. No real grouped split or model benchmark established.",
            "preprocessing": "NOT_SELECTED; Session 4 methods remain exploratory",
            "model_and_training": "NOT_APPLICABLE; no model or optimizer used",
            "future_evaluation": "Per-class/macro IoU and Dice, constituent metrics, feature/boundary review, "
                                 "repeated-seed variation; select on validation only; final official test once frozen."}


def write_protocol(protocol, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silent overwrite, including input/output collisions.
    payload = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/session05_protocol_demo.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    args = parser.parse_args()
    try:
        source = None
        if args.manifest:
            with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            source = {"manifest": str(args.manifest.resolve()), "sha256": checksum(args.manifest)}
        else:
            records = demo_records()
        protocol = build_protocol(records, args.seed, args.validation_fraction, source)
        write_protocol(protocol, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\nProtocol not written. Use a new output filename for a revision.\n")
    print("Role counts:")
    for role, count in protocol["split"]["role_counts"].items():
        print(f"{role}: {count}")
    print("Toy metrics:", json.dumps(protocol["toy_binary_metrics"], sort_keys=True))
    print("Protocol:", args.output)


if __name__ == "__main__":
    main()
