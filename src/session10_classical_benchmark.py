"""Artificial development/validation benchmark; never reads research image data."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from session03_dataset_audit import ROOT, checksum
from session04_preprocessing import to_u8
from session05_protocol import split_records
from session06_threshold_lab import segment
from session07_morphology_lab import validate_mask

METHODS = ("Fixed 120", "Otsu", "Adaptive mean", "Adaptive Gaussian", "Gaussian + Otsu", "Otsu + opening", "Otsu + closing")
PARTITIONS = {"compact": "development", "elongated": "development", "crowded": "development",
              "particles": "validation", "bands": "validation"}


def make_scenes():
    scenes = []
    for group_index, (group, role) in enumerate(PARTITIONS.items()):
        truth = np.zeros((192, 256), np.uint8)
        thin = np.zeros_like(truth)
        if group == "compact":
            cv2.circle(truth, (65, 65), 30, 1, -1)
            cv2.circle(truth, (170, 110), 36, 1, -1)
            thin[25:65, 218] = 1
            thin[110:150, 32:35] = 1
            truth[thin > 0] = 1
        elif group == "elongated":
            cv2.ellipse(truth, (70, 65), (48, 13), 25, 0, 360, 1, -1)
            cv2.ellipse(truth, (175, 125), (46, 17), -30, 0, 360, 1, -1)
        elif group == "crowded":
            for y in (45, 90, 135):
                for x in (50, 94, 138, 182):
                    cv2.circle(truth, (x, y), 24, 1, -1)
        elif group == "particles":
            for x, y, radius in ((35, 30, 5), (80, 40, 8), (140, 45, 6), (215, 38, 10),
                                  (40, 110, 6), (110, 120, 9), (180, 128, 5), (225, 165, 7)):
                cv2.circle(truth, (x, y), radius, 1, -1)
        else:
            truth[25:48, 20:230] = 1
            truth[75:96, 35:240] = 1
            truth[125:151, 15:218] = 1
        roi = np.ones(truth.shape, bool)
        roi[175:, :25] = False
        noise = np.random.default_rng(100 + group_index).normal(0, 5, truth.shape)
        clean = np.where(truth, 55., 200.)
        for lighting in ("even", "shaded"):
            field = 1 if lighting == "even" else np.linspace(.6, 1.2, truth.shape[1])[None, :]
            gray = to_u8(clean * field + noise)
            scenes.append({"scene_id": f"{group}:{lighting}", "source_group": group, "role": role,
                           "gray": gray, "truth": truth.copy(), "roi": roi.copy(), "thin": thin.copy()})
    return scenes


def pipelines(gray):
    before = gray.copy()
    basic, _ = segment(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    _, filtered = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    results = {"Fixed 120": basic["Fixed T=120"], "Otsu": basic["Otsu"],
               "Adaptive mean": basic["Adaptive mean"], "Adaptive Gaussian": basic["Adaptive Gaussian"],
               "Gaussian + Otsu": filtered // 255,
               "Otsu + opening": cv2.morphologyEx(basic["Otsu"], cv2.MORPH_OPEN, kernel),
               "Otsu + closing": cv2.morphologyEx(basic["Otsu"], cv2.MORPH_CLOSE, kernel)}
    assert np.array_equal(gray, before)
    return results


def score(truth, prediction, roi, thin):
    validate_mask(truth)
    validate_mask(prediction)
    validate_mask(thin)
    if (truth.shape != prediction.shape or thin.shape != truth.shape or roi.shape != truth.shape
            or roi.dtype != bool or not roi.any() or np.any((thin > 0) & (truth == 0))):
        raise ValueError("Masks/ROI must match, ROI must be valid bool, and thin feature must lie inside truth")
    t, p = truth.astype(bool), prediction.astype(bool)
    tp, fp = int(np.count_nonzero(t & p & roi)), int(np.count_nonzero(~t & p & roi))
    fn, tn = int(np.count_nonzero(t & ~p & roi)), int(np.count_nonzero(~t & ~p & roi))
    union = tp + fp + fn
    true_count = int(cv2.connectedComponents((t & roi).astype(np.uint8), connectivity=8)[0] - 1)
    pred_count = int(cv2.connectedComponents((p & roi).astype(np.uint8), connectivity=8)[0] - 1)
    thin_count = int(np.count_nonzero((thin > 0) & roi))
    signed = 100 * (fp - fn) / int(roi.sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / union if union else None, "dice": 2 * tp / (2 * tp + fp + fn) if union else None,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "signed_area_error_pp": signed, "absolute_area_error_pp": abs(signed),
            "true_components_8": true_count, "predicted_components_8": pred_count,
            "signed_component_count_error": pred_count - true_count,
            "absolute_component_count_error": abs(pred_count - true_count),
            "thin_pixels": thin_count,
            "thin_feature_recall": int(np.count_nonzero((thin > 0) & p & roi)) / thin_count if thin_count else None}


def summarize(rows):
    summary = {}
    for role in ("development", "validation"):
        summary[role] = {}
        for method in METHODS:
            selected = [row for row in rows if row["role"] == role and row["method"] == method]
            means = {}
            for metric in ("iou", "dice", "absolute_area_error_pp", "absolute_component_count_error", "thin_feature_recall"):
                values = [row[metric] for row in selected if row[metric] is not None]
                means["scene_macro_" + metric] = float(np.mean(values)) if values else None
            totals = {key: sum(row[key] for row in selected) for key in ("tp", "fp", "fn", "tn")}
            denominator = totals["tp"] + totals["fp"] + totals["fn"]
            means["pooled_foreground_iou"] = totals["tp"] / denominator if denominator else None
            worst = min(selected, key=lambda row: row["iou"] if row["iou"] is not None else float("inf"))
            summary[role][method] = {**means, "scene_count": len(selected), "worst_scene": worst["scene_id"],
                                    "worst_scene_iou": worst["iou"], "pooled_confusion_counts": totals}
    # Frozen illustrative selection rule. No real/test inference is permitted.
    winner = min(METHODS, key=lambda method: (
        -summary["validation"][method]["scene_macro_iou"],
        summary["validation"][method]["scene_macro_absolute_area_error_pp"], method))
    return summary, winner


def readiness(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    return split_records(records)  # Metadata only, strict lineage check; no images read.


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Benchmark output directory already exists; use a fresh directory")
    scenes = make_scenes()
    rows = []
    comparison = None
    for scene in scenes:
        predictions = pipelines(scene["gray"])
        for method, prediction in predictions.items():
            rows.append({"scene_id": scene["scene_id"], "source_group": scene["source_group"],
                         "role": scene["role"], "method": method,
                         **score(scene["truth"], prediction, scene["roi"], scene["thin"])})
        if scene["scene_id"] == "particles:shaded":
            comparison = (scene, predictions)
    summary, winner = summarize(rows)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True)
    report = {"protocol_version": "session10-artificial-v1", "partitions": PARTITIONS, "summary": summary,
              "selected_pipeline": winner, "selection_rule": "Validation scene-macro foreground IoU descending, then absolute area error ascending, then name",
              "selection_scope": "Illustrative validation selection; not an unbiased final test score",
              "parameters": {"seeds": list(range(100, 105)), "fixed_threshold": 120, "adaptive_block_size": 41,
                             "adaptive_C": 6, "gaussian_kernel": [5, 5], "gaussian_sigma": 1,
                             "morphology_kernel": "3x3 ellipse/cross, one iteration, OpenCV default neutral border",
                             "noise_sigma": 5, "intensities": [55, 200], "shading": [.6, 1.2]},
              "aggregation": "Scene-macro means exclude undefined values; no thin annotation yields null, not perfect recall. Pooled foreground IoU reported separately.",
              "scope": "Reconstructed artificial fixture; code package not attached. No research images, official tests or educational images read.",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
              "code": {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
                       "script_sha256": checksum(Path(__file__))},
              "tuning_budget": "Seven fixed pipelines, one configuration each; no parameter search",
              "real_data_dependency": "Official training archive unaudited; source-texture lineage unknown; no real benchmark claim"}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "session10_results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "session10_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scene, predictions = comparison
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), layout="constrained")
    try:
        for ax, (title, pixels) in zip(axes.ravel(), [("Grayscale (synthetic)", scene["gray"]),
                                                    ("Known truth", scene["truth"]), *predictions.items()]):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=255 if title.startswith("Grayscale") else 1,
                      interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle("Artificial validation: particles/shaded (not a real micrograph)")
        fig.savefig(output / "session10_validation_comparison.png", dpi=120)
    finally:
        plt.close(fig)
    print("Comparisons:", len(rows), "Selected illustrative pipeline:", winner)
    for method in METHODS:
        print(method, json.dumps(summary["validation"][method]))
    return report, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session10")
    parser.add_argument("--readiness-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.readiness_manifest:
            readiness(args.readiness_manifest)
            print("Metadata passes strict grouped-development checks; this does not independently verify lineage or file contents.")
        else:
            run(args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
