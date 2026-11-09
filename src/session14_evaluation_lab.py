"""Evaluate the selected artificial checkpoint with class, boundary and group metrics."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import torch
from torch.utils.data import DataLoader

from session03_dataset_audit import ROOT, checksum
from session13_training_lab import GroupedToyDataset, restore_checkpoint


def confusion_matrix(truth, prediction, roi, classes=2):
    if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
        raise ValueError("Class count must be an integer of at least two")
    for labels in (truth, prediction):
        if (not isinstance(labels, np.ndarray) or labels.ndim != 2 or not labels.size
                or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels >= classes)):
            raise ValueError("Labels must be nonempty 2D integer arrays within the class schema")
    if (truth.shape != prediction.shape or not isinstance(roi, np.ndarray)
            or roi.shape != truth.shape or roi.dtype != bool or not roi.any()):
        raise ValueError("Label shapes must match a nonempty boolean valid ROI")
    encoded = truth[roi].astype(np.int64) * classes + prediction[roi].astype(np.int64)
    return np.bincount(encoded, minlength=classes**2).reshape(classes, classes)


def class_metrics(matrix):
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] < 2
            or matrix.shape[0] != matrix.shape[1] or not np.issubdtype(matrix.dtype, np.integer)
            or np.any(matrix < 0)):
        raise ValueError("Confusion counts must form a nonnegative integer square matrix")
    result = {}
    for label in range(matrix.shape[0]):
        tp = int(matrix[label, label])
        fp = int(matrix[:, label].sum()) - tp
        fn = int(matrix[label, :].sum()) - tp
        union = tp + fp + fn
        result[str(label)] = {
            "tp": tp, "fp": fp, "fn": fn,
            "iou": tp / union if union else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if union else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    defined = [row["iou"] for row in result.values() if row["iou"] is not None]
    total = int(matrix.sum())
    return {"per_class": result, "macro_class_iou": float(np.mean(defined)) if defined else None,
            "pixel_accuracy": float(np.trace(matrix) / total) if total else None}


def boundary_metrics(truth, prediction, roi, tolerance=1.5):
    confusion_matrix(truth, prediction, roi)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    # Compute boundaries before ROI restriction, avoiding artificial ROI-cut interfaces.
    safe_roi = ndi.binary_erosion(roi, structure=np.ones((3, 3)), border_value=1)
    boundaries = []
    for mask in (truth > 0, prediction > 0):
        boundary = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
        boundaries.append(boundary & safe_roi)
    actual, predicted = boundaries
    n_actual, n_predicted = int(actual.sum()), int(predicted.sum())
    matched_predicted = int((predicted & (ndi.distance_transform_edt(~actual) <= tolerance)).sum()) if n_actual else 0
    matched_actual = int((actual & (ndi.distance_transform_edt(~predicted) <= tolerance)).sum()) if n_predicted else 0
    precision = matched_predicted / n_predicted if n_predicted else None
    recall = matched_actual / n_actual if n_actual else None
    f1 = None
    if n_actual or n_predicted:
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1,
            "true_boundary_pixels": n_actual, "predicted_boundary_pixels": n_predicted,
            "matched_predicted_boundary_pixels": matched_predicted, "matched_true_boundary_pixels": matched_actual}


def scene_metrics(truth, prediction, roi, thin=None):
    matrix = confusion_matrix(truth, prediction, roi)
    metrics = class_metrics(matrix)
    foreground = metrics["per_class"]["1"]
    thin_recall = None
    if thin is not None:
        if thin.shape != truth.shape or thin.dtype != bool or np.any(thin & (truth != 1)):
            raise ValueError("Thin annotation must be a boolean foreground subset")
        count = int((thin & roi).sum())
        if count:
            thin_recall = float((thin & roi & (prediction == 1)).sum() / count)
    return {"tn": int(matrix[0, 0]), **foreground,
            "pixel_accuracy": metrics["pixel_accuracy"], "macro_class_iou": metrics["macro_class_iou"],
            "signed_area_error_pp": 100 * (foreground["fp"] - foreground["fn"]) / int(roi.sum()),
            "thin_feature_recall": thin_recall, **boundary_metrics(truth, prediction, roi)}


def group_bootstrap(rows, repetitions=2000, seed=14):
    if not rows or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Bootstrap needs scene rows and positive repetitions")
    groups = sorted({row["group_id"] for row in rows})
    sums, counts = [], []
    for group in groups:
        values = [row["iou"] for row in rows if row["group_id"] == group and row["iou"] is not None]
        sums.append(sum(values))
        counts.append(len(values))
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(repetitions, len(groups)))
    totals = np.asarray(sums)[draws].sum(axis=1)
    denominators = np.asarray(counts)[draws].sum(axis=1)
    estimates = totals[denominators > 0] / denominators[denominators > 0]
    interval = np.percentile(estimates, [2.5, 97.5]).tolist() if estimates.size else [None, None]
    return {"group_count": len(groups), "repetitions": repetitions, "seed": seed,
            "defined_replicates": int(estimates.size), "scene_macro_iou_percentile_95": interval,
            "policy": "Resample whole source groups with replacement and include every defined scene of each draw",
            "scope": "Four artificial groups reused for model selection. Interval reflects this fixture's resampling variability. Selection optimism, source lineage and real-domain uncertainty require separate evidence."}


def summarize(rows):
    methods = sorted({row["method"] for row in rows})
    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        pooled = np.array([[sum(r["tn"] for r in selected), sum(r["fp"] for r in selected)],
                           [sum(r["fn"] for r in selected), sum(r["tp"] for r in selected)]], dtype=np.int64)
        ious = [r["iou"] for r in selected if r["iou"] is not None]
        f1s = [r["boundary_f1"] for r in selected if r["boundary_f1"] is not None]
        summary[method] = {"confusion_matrix": pooled.tolist(), **class_metrics(pooled),
                           "scene_macro_foreground_iou": float(np.mean(ious)) if ious else None,
                           "scene_macro_boundary_f1": float(np.mean(f1s)) if f1s else None,
                           "mean_signed_area_error_pp": float(np.mean([r["signed_area_error_pp"] for r in selected])),
                           "group_bootstrap": group_bootstrap(selected)}
    return summary


def infer_scenes(model, dataset):
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    scenes = []
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)):
            probabilities = model(images).softmax(1)[:, 1]
            for offset in range(images.shape[0]):
                index = batch_index * 4 + offset
                scenes.append({"scene_id": f"group-{dataset.groups[index // 2]}-view-{index % 2}",
                               "group_id": dataset.groups[index // 2], "image": images[offset, 0].numpy().copy(),
                               "truth": targets[offset].numpy().copy(), "probability": probabilities[offset].numpy().copy()})
    if any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()):
        raise RuntimeError("Evaluation changed model parameters")
    return scenes


def thin_diagnostic():
    truth = np.zeros((100, 100), dtype=np.uint8)
    truth[10:60, 10:70] = 1
    thin = np.zeros(truth.shape, dtype=bool)
    thin[55:95, 85] = True
    truth[thin] = 1
    prediction = truth.copy()
    prediction[thin] = 0
    return truth, prediction, thin


def run(output, checkpoint_path=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation output exists. Choose a fresh directory")
    truth, diagnostic_prediction, thin = thin_diagnostic()
    diagnostic = scene_metrics(truth, diagnostic_prediction, np.ones(truth.shape, bool), thin)
    report = {"boundary_policy": "Foreground inner boundary from 3x3 erosion, image exterior is background, Euclidean tolerance 1.5 pixels. Exclude boundary pixels beside invalid ROI. Both-empty boundary F1 null, one-empty zero.",
              "class_policy": "True rows, predicted columns. Both-absent classes have null overlap and are excluded from macro-class IoU.",
              "probability_policy": "Foreground probability strictly greater than threshold. Default 0.5 agrees with binary argmax including class-0 ties.",
              "thin_diagnostic": diagnostic}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    report["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}
    report["code"] = {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
                      "script_sha256": checksum(Path(__file__))}
    report["implementation"] = "Reconstructed from handout requirements. Referenced package was unavailable."
    scenes, rows = [], []
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        source_hash = checksum(checkpoint_path)
        model, checkpoint = restore_checkpoint(checkpoint_path)
        config = checkpoint["config"]
        training_groups = set(config["train_groups"])
        validation_groups = config["validation_groups"]
        if not validation_groups or training_groups & set(validation_groups):
            raise ValueError("Checkpoint development groups must be disjoint")
        dataset = GroupedToyDataset(validation_groups)
        scenes = infer_scenes(model, dataset)
        sensitivity = {}
        for threshold in (.3, .4, .5, .6, .7):
            threshold_rows = []
            for scene in scenes:
                roi = np.ones(scene["truth"].shape, bool)
                prediction = (scene["probability"] > threshold).astype(np.uint8)
                row = {"scene_id": scene["scene_id"], "group_id": scene["group_id"], "method": "U-Net",
                       **scene_metrics(scene["truth"], prediction, roi)}
                threshold_rows.append(row)
                if threshold == .5:
                    rows.append(row)
            pooled_tp = sum(r["tp"] for r in threshold_rows)
            pooled_fp = sum(r["fp"] for r in threshold_rows)
            pooled_fn = sum(r["fn"] for r in threshold_rows)
            union = pooled_tp + pooled_fp + pooled_fn
            sensitivity[str(threshold)] = {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn,
                                           "foreground_iou": pooled_tp / union if union else None,
                                           "precision": pooled_tp / (pooled_tp + pooled_fp) if pooled_tp + pooled_fp else None,
                                           "recall": pooled_tp / (pooled_tp + pooled_fn) if pooled_tp + pooled_fn else None}
        for scene in scenes:
            for method, prediction in (("Fixed 120", (np.rint(scene["image"] * 255) <= 120).astype(np.uint8)),
                                       ("All background", np.zeros(scene["truth"].shape, np.uint8))):
                rows.append({"scene_id": scene["scene_id"], "group_id": scene["group_id"], "method": method,
                             **scene_metrics(scene["truth"], prediction, np.ones(scene["truth"].shape, bool))})
        summary = summarize(rows)
        actual_iou = summary["U-Net"]["per_class"]["1"]["iou"]
        expected_iou = checkpoint["validation"]["pooled_foreground_iou"]
        if not np.isclose(actual_iou, expected_iou, rtol=0, atol=1e-8):
            raise RuntimeError("Default evaluation does not reproduce saved checkpoint IoU")
        if checksum(checkpoint_path) != source_hash:
            raise RuntimeError("Checkpoint bytes changed during evaluation")
        report.update({"checkpoint_path": str(checkpoint_path.resolve()),
                       "checkpoint_sha256": source_hash, "checkpoint_epoch": checkpoint["epoch"],
                       "dataset_script_sha256": checksum(Path(__file__).with_name("session13_training_lab.py")),
                       "saved_validation_foreground_iou": expected_iou,
                       "saved_iou_reproduced": True, "model_parameters_preserved": True,
                       "validation_groups": validation_groups, "methods": summary,
                       "threshold_sensitivity": sensitivity,
                       "selection_policy": "Default threshold remains 0.5. Alternative thresholds are validation sensitivity measurements.",
                       "scope": "Evaluation of the same artificial groups used for Session 13 selection"})
    output.mkdir(parents=True, exist_ok=False)
    if rows:
        with (output / "session14_per_scene.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        scene = scenes[0]
        predictions = [("U-Net default", (scene["probability"] > .5).astype(np.uint8)),
                       ("Fixed 120", (np.rint(scene["image"] * 255) <= 120).astype(np.uint8)),
                       ("All background", np.zeros(scene["truth"].shape, np.uint8))]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
        panels = [("Artificial input", scene["image"]), ("Known mask", scene["truth"]),
                  ("P(class 1)", scene["probability"]), *predictions]
        try:
            for ax, (title, pixels) in zip(axes.ravel(), panels):
                ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(title)
                ax.set_axis_off()
            fig.suptitle("Session 14 artificial validation comparison")
            fig.savefig(output / "session14_comparison.png", dpi=120)
        finally:
            plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    try:
        for ax, (title, pixels) in zip(axes, (("Known truth", truth), ("Constructed missing line", diagnostic_prediction), ("Annotated thin pixels", thin))):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle(f"Controlled diagnostic, IoU {diagnostic['iou']:.5f}, thin recall {diagnostic['thin_feature_recall']:.0%}")
        fig.savefig(output / "session14_thin_diagnostic.png", dpi=120)
    finally:
        plt.close(fig)
    (output / "session14_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session14_v1")
    args = parser.parse_args()
    try:
        run(args.output_dir, args.checkpoint)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
