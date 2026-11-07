"""Distance-marker watershed hypotheses, evaluated only on artificial instances."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import skimage
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, watershed

from session03_dataset_audit import ROOT, checksum
from session04_preprocessing import read_gray, to_u8
from session06_threshold_lab import segment
from session07_morphology_lab import validate_mask


def make_synthetic():
    rows, cols = np.indices((240, 320))
    centers = [(125, 90, 44), (125, 150, 42), (65, 250, 27)]
    distances = np.stack([(rows-r)**2 + (cols-c)**2 for r, c, _ in centers])
    eligible = np.stack([distances[i] <= radius**2 for i, (_, _, radius) in enumerate(centers)])
    candidate = eligible.any(axis=0).astype(np.uint8)
    nearest = np.argmin(np.where(eligible, distances, np.inf), axis=0) + 1
    truth = np.where(candidate, nearest, 0).astype(np.int32)
    gray = to_u8(np.where(candidate, 55, 205) + np.random.default_rng(31).normal(0, 4, candidate.shape))
    return gray, candidate, truth


def distance_markers(mask, min_distance=16, threshold_abs=7.0):
    validate_mask(mask)
    if not np.any(mask == 0):
        raise ValueError("Distance transform requires background in this lab")
    if not isinstance(min_distance, int) or min_distance < 1:
        raise ValueError("min_distance must be a positive integer")
    if not np.isfinite(threshold_abs) or threshold_abs < 0:
        raise ValueError("threshold_abs must be finite and nonnegative")
    distance = ndi.distance_transform_edt(mask.astype(bool))
    peaks = peak_local_max(distance, min_distance=min_distance, threshold_abs=threshold_abs,
                           labels=mask, exclude_border=False)
    # Stable IDs by coordinate order; instance metrics must not depend on ID numbering.
    peaks = sorted(map(tuple, peaks.tolist()))
    markers = np.zeros(mask.shape, dtype=np.int32)
    for label, (row, col) in enumerate(peaks, 1):
        markers[row, col] = label
    return distance, markers


def separate(mask, markers=None, min_distance=16, threshold_abs=7.0):
    distance, detected = distance_markers(mask, min_distance, threshold_abs)
    if markers is None:
        markers = detected
    if (not isinstance(markers, np.ndarray) or markers.shape != mask.shape
            or not np.issubdtype(markers.dtype, np.integer) or np.any(markers < 0)
            or np.any((markers > 0) & (mask == 0))):
        raise ValueError("Markers must be nonnegative integer IDs inside the candidate mask")
    before_mask, before_markers = mask.copy(), markers.copy()
    labels = (watershed(-distance, markers=markers, mask=mask.astype(bool), connectivity=1,
                        watershed_line=True).astype(np.int32) if np.any(markers) else np.zeros_like(markers))
    regions, _ = ndi.label(mask)  # Same 4-neighbor connectivity as watershed.
    seeded_ids = np.unique(regions[markers > 0])
    seeded = np.isin(regions, seeded_ids) & (mask > 0)
    lines = seeded & (labels == 0)
    unseeded = (mask > 0) & ~seeded
    assert np.array_equal(mask, before_mask) and np.array_equal(markers, before_markers)
    assert not np.any(labels[mask == 0])
    return {"distance": distance, "markers": markers.copy(), "labels": labels,
            "watershed_lines": lines, "unseeded_foreground": unseeded}


def instance_metrics(truth, prediction):
    for values in (truth, prediction):
        if (not isinstance(values, np.ndarray) or values.ndim != 2 or not values.size
                or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0)):
            raise ValueError("Instance labels must be nonempty nonnegative 2D integer arrays")
    if truth.shape != prediction.shape:
        raise ValueError("Instance shapes must match")
    actual = np.unique(truth[truth > 0])
    predicted = np.unique(prediction[prediction > 0])
    scores = np.zeros((len(actual), len(predicted)))
    for row, a in enumerate(actual):
        for col, p in enumerate(predicted):
            intersection = np.count_nonzero((truth == a) & (prediction == p))
            union = np.count_nonzero((truth == a) | (prediction == p))
            scores[row, col] = intersection / union
    rows, cols = linear_sum_assignment(-scores)
    matches = [{"truth_id": int(actual[r]), "prediction_id": int(predicted[c]), "iou": float(scores[r, c])}
               for r, c in zip(rows, cols) if scores[r, c] > 0]
    denominator = max(len(actual), len(predicted))
    return {"matched_mean_instance_iou": sum(item["iou"] for item in matches) / denominator if denominator else None,
            "truth_count": len(actual), "prediction_count": len(predicted), "matches": matches,
            "unmatched_truth": len(actual) - len(matches), "unmatched_prediction": len(predicted) - len(matches),
            "policy": "Hungarian maximum-total-IoU matching; sum IoU / max(truth count, prediction count). "
                      "Unmatched objects contribute zero. Both empty: null. Introductory metric, not AP/PQ."}


def statistics(mask, result):
    return {"binary_components_8": int(cv2.connectedComponents(mask, connectivity=8)[0] - 1),
            "marker_count": int(len(np.unique(result["markers"][result["markers"] > 0]))),
            "instance_count": int(len(np.unique(result["labels"][result["labels"] > 0]))),
            "watershed_line_pixels_inside_candidate": int(result["watershed_lines"].sum()),
            "unseeded_foreground_pixels": int(result["unseeded_foreground"].sum())}


def save_comparison(gray, mask, truth, result, path, title):
    panels = [("Intensity input", gray, "gray"), ("Candidate binary mask", mask, "gray"),
              ("Distance transform (pixels)", result["distance"], "magma"),
              ("Marker IDs", result["markers"], "nipy_spectral"),
              ("Watershed instance IDs", result["labels"], "nipy_spectral"),
              ("Known artificial truth" if truth is not None else "Watershed lines inside candidate",
               truth if truth is not None else result["watershed_lines"], "nipy_spectral" if truth is not None else "gray")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    try:
        for ax, (name, values, cmap) in zip(axes.ravel(), panels):
            ax.imshow(values, cmap=cmap, interpolation="nearest", vmin=0)
            ax.set_title(name)
            ax.set_axis_off()
        fig.suptitle(title)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def save_edges(gray, truth, result, path):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 40, 100)
    panels = [("Intensity image", gray), ("Canny image edges", edges),
              ("Known instance boundaries", find_boundaries(truth, mode="inner")),
              ("Watershed lines inside candidate", result["watershed_lines"])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    try:
        for ax, (name, values) in zip(axes.ravel(), panels):
            ax.imshow(values, cmap="gray", interpolation="nearest", vmin=0)
            ax.set_title(name)
            ax.set_axis_off()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def run(image_path=None):
    output = ROOT / "outputs/session08"
    if image_path is not None:
        image_path = Path(image_path).resolve()
        if ((ROOT / "data/raw/figshare_orientation").resolve() in image_path.parents
                or "masks" in {part.casefold() for part in image_path.parts}):
            raise ValueError("Reserved Figshare data and mask folders are excluded")
        if output.resolve() in image_path.parents:
            raise ValueError("Input cannot be a generated output")
        original_hash = checksum(image_path)
        educational = read_gray(image_path)
    output.mkdir(parents=True, exist_ok=True)
    gray, mask, truth = make_synthetic()
    result = separate(mask)
    fewer = result["markers"].copy()
    # Remove a seed in the touching component; the other seed then claims both lobes.
    touching_seed = np.argwhere((fewer > 0) & (np.indices(mask.shape)[0] > 100))[0]
    fewer[tuple(touching_seed)] = 0
    extra = result["markers"].copy()
    extra[125, 70] = int(extra.max()) + 1
    variants = {"normal": result, "removed_marker": separate(mask, fewer), "extra_marker": separate(mask, extra)}
    report = {"parameters": {"seed": 31, "min_distance": 16, "threshold_abs_px": 7.0,
                              "watershed_connectivity": 4, "watershed_line": True,
                              "canny": [40, 100], "canny_preblur_kernel": [5, 5], "canny_preblur_sigma": 1},
              "versions": {"scipy": scipy.__version__, "skimage": skimage.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
              "fixture_note": "Reconstructed synthetic fixture; source package not attached. Nearest eligible center resolves overlap, lower ID breaks ties.",
              "synthetic": {name: {**statistics(mask, variant), **instance_metrics(truth, variant["labels"])}
                            for name, variant in variants.items()}}
    save_comparison(gray, mask, truth, result, output / "session08_synthetic.png", "Synthetic touching-object separation: known instance truth")
    save_edges(gray, truth, result, output / "session08_edges.png")
    print(json.dumps(report["synthetic"], indent=2))
    if image_path is not None:
        candidates, thresholds = segment(educational)
        educational_result = separate(candidates["Otsu"])
        report["educational"] = {**statistics(candidates["Otsu"], educational_result),
                                 "otsu_thresholds": thresholds, "source_sha256": original_hash,
                                 "instance_iou": None, "semantics": "Unverified candidates, not grains or particles"}
        save_comparison(educational, candidates["Otsu"], None, educational_result,
                        output / "session08_educational.png", "Educational watershed: algorithmic hypotheses only")
        assert checksum(image_path) == original_hash, "Source image changed"
        print("Educational:", report["educational"])
    (output / "session08_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    try:
        run(args.image)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
