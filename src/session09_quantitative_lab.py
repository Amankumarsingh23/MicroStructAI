"""ROI-aware 2D descriptors; calibration is hypothetical for artificial data only."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

from session03_dataset_audit import ROOT, checksum
from session04_preprocessing import read_gray, to_u8
from session05_protocol import evaluate
from session06_threshold_lab import segment
from session08_watershed_lab import instance_metrics


def validate_inputs(labels, valid_roi):
    if (not isinstance(labels, np.ndarray) or labels.ndim != 2 or not labels.size
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError("Expected nonempty nonnegative 2D integer instance labels")
    if (not isinstance(valid_roi, np.ndarray) or valid_roi.dtype != bool
            or valid_roi.shape != labels.shape or not valid_roi.any()):
        raise ValueError("ROI must be matching boolean array with at least one valid pixel")


def area_fraction(labels, valid_roi):
    validate_inputs(labels, valid_roi)
    return float(np.count_nonzero((labels > 0) & valid_roi) / np.count_nonzero(valid_roi))


def measure_objects(labels, valid_roi, demo_um_per_pixel=None):
    validate_inputs(labels, valid_roi)
    if demo_um_per_pixel is not None and (not np.isfinite(demo_um_per_pixel) or demo_um_per_pixel <= 0):
        raise ValueError("Hypothetical scale must be finite and positive")
    # Keep IDs unchanged; descriptors refer to visible in-ROI fragments, not full objects.
    observed = np.where(valid_roi, labels, 0)
    roi_boundary = valid_roi & ~ndi.binary_erosion(valid_roi, border_value=1)
    features = []
    for region in regionprops(observed):
        area = float(region.area)
        perimeter = float(region.perimeter_crofton)
        major, minor = float(region.axis_major_length), float(region.axis_minor_length)
        coords = region.coords
        original = labels == region.label
        touches_image = bool(np.any((coords[:, 0] == 0) | (coords[:, 0] == labels.shape[0] - 1)
                                    | (coords[:, 1] == 0) | (coords[:, 1] == labels.shape[1] - 1)))
        clipped = bool(np.any(original & ~valid_roi))
        touches_roi = clipped or bool(np.any(roi_boundary[coords[:, 0], coords[:, 1]]))
        feature = {"instance_id": int(region.label), "area_px2": area,
                   "perimeter_crofton_px": perimeter, "equivalent_diameter_px": float(np.sqrt(4 * area / np.pi)),
                   "axis_major_length_px": major, "axis_minor_length_px": minor,
                   "circularity": float(4 * np.pi * area / perimeter**2) if perimeter > 0 else None,
                   "aspect_ratio": major / minor if minor > 0 else None,
                   "centroid_row_px": float(region.centroid[0]), "centroid_col_px": float(region.centroid[1]),
                   "touches_image_border": touches_image, "touches_roi_boundary": touches_roi,
                   "clipped_by_roi": clipped,
                   "observed_fragments_8": int(ndi.label(observed == region.label, np.ones((3, 3)))[1])}
        if demo_um_per_pixel is not None:
            feature.update({"hypothetical_area_um2": area * demo_um_per_pixel**2,
                            "hypothetical_equivalent_diameter_um": feature["equivalent_diameter_px"] * demo_um_per_pixel,
                            "hypothetical_perimeter_um": perimeter * demo_um_per_pixel})
        features.append(feature)
    return features


def make_synthetic():
    labels = np.zeros((240, 320), np.int32)
    cv2.circle(labels, (70, 60), 22, 1, -1)
    cv2.ellipse(labels, (205, 60), (36, 14), 30, 0, 360, 2, -1)
    labels[145:192, 35:103] = 3
    cv2.circle(labels, (315, 183), 20, 4, -1)
    roi = np.ones(labels.shape, bool)
    roi[220:, :80] = False
    gray = to_u8(np.where(labels > 0, 55, 205) + np.random.default_rng(41).normal(0, 3, labels.shape))
    return gray, labels, roi


def measurement_bias(truth, prediction, roi):
    validate_inputs(truth, roi)
    validate_inputs(prediction, roi)
    before = measure_objects(truth, roi)
    after = measure_objects(prediction, roi)
    matching = instance_metrics(np.where(roi, truth, 0), np.where(roi, prediction, 0))
    before_by_id = {r["instance_id"]: r for r in before}
    after_by_id = {r["instance_id"]: r for r in after}
    errors = []
    for pair in matching["matches"]:
        a, b = before_by_id[pair["truth_id"]], after_by_id[pair["prediction_id"]]
        errors.append({**pair, "signed_area_error_px2": b["area_px2"] - a["area_px2"],
                       "signed_equivalent_diameter_error_px": b["equivalent_diameter_px"] - a["equivalent_diameter_px"]})
    return {"truth_fraction": area_fraction(truth, roi), "prediction_fraction": area_fraction(prediction, roi),
            "signed_fraction_error": area_fraction(prediction, roi) - area_fraction(truth, roi),
            "signed_fraction_error_percentage_points": 100 * (area_fraction(prediction, roi) - area_fraction(truth, roi)),
            "semantic_metrics": evaluate((truth[roi] > 0).astype(np.uint8)[None, :],
                                         (prediction[roi] > 0).astype(np.uint8)[None, :]),
            "instance_matching": matching, "matched_measurement_errors": errors,
            "truth_object_count": len(before), "prediction_object_count": len(after)}


def write_objects(features, path):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0]) if features else ["instance_id"])
        writer.writeheader()
        writer.writerows(features)


def size_distribution(features):
    values = np.array([f["equivalent_diameter_px"] for f in features])
    counts, edges = np.histogram(values, bins=5)
    return {"counts": counts.tolist(), "bin_edges_px": edges.tolist(),
            "policy": "Number-weighted observed-region equivalent diameters; includes flagged truncated objects; not population statistics"}


def save_synthetic(gray, truth, roi, prediction, objects, bias, path):
    error = np.zeros(truth.shape, np.uint8)
    error[(truth > 0) & (prediction > 0) & roi] = 1
    error[(truth == 0) & (prediction > 0) & roi] = 2
    error[(truth > 0) & (prediction == 0) & roi] = 3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    try:
        for ax, pixels, title, cmap in (
            (axes[0, 0], gray, "Artificial grayscale observation", "gray"),
            (axes[0, 1], truth, "Known artificial instance IDs (gray=excluded ROI)", "nipy_spectral"),
            (axes[0, 2], prediction > 0, "Dilated candidate region", "gray"),
            (axes[1, 0], error, "Pixel errors: TP=blue, FP=orange, FN=red", ListedColormap(["black", "royalblue", "orange", "red"]))):
            if cmap != "gray":
                pixels = np.ma.array(pixels, mask=~roi)
            display_cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
            display_cmap.set_bad("lightgray")
            ax.imshow(pixels, cmap=display_cmap, interpolation="nearest", vmin=0,
                      vmax=3 if ax is axes[1, 0] else None)
            if cmap != "gray":
                ax.set_facecolor("lightgray")
            ax.set_title(title)
            ax.set_axis_off()
        axes[1, 1].bar([str(r["instance_id"]) for r in objects], [r["area_px2"] for r in objects])
        axes[1, 1].set(title="Known observed region areas", xlabel="Instance ID", ylabel="Area (pixels squared)")
        axes[1, 2].bar(["Known", "Dilated"], [bias["truth_fraction"], bias["prediction_fraction"]])
        axes[1, 2].set(title="Dilation-induced measurement bias", ylabel="Valid-ROI foreground fraction", ylim=(0, .5))
        fig.suptitle("Session 9 | Artificial measurements, not material data")
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def run(image_path=None, demo_um_per_pixel=None):
    if demo_um_per_pixel is not None and (not np.isfinite(demo_um_per_pixel) or demo_um_per_pixel <= 0):
        raise ValueError("Hypothetical scale must be finite and positive")
    output = ROOT / "outputs/session09"
    if image_path is not None:
        image_path = Path(image_path).resolve()
        if ((ROOT / "data/raw/figshare_orientation").resolve() in image_path.parents
                or "masks" in {p.casefold() for p in image_path.parts}):
            raise ValueError("Reserved Figshare data and mask folders are excluded")
        if output.resolve() in image_path.parents:
            raise ValueError("Input cannot be a generated output")
        source_hash = checksum(image_path)
        educational = read_gray(image_path)
    output.mkdir(parents=True, exist_ok=True)
    gray, labels, roi = make_synthetic()
    before_labels, before_roi = labels.copy(), roi.copy()
    dilated = cv2.dilate((labels > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    prediction = ndi.label(dilated, np.ones((3, 3)))[0].astype(np.int32)
    objects = measure_objects(labels, roi, demo_um_per_pixel)
    bias = measurement_bias(labels, prediction, roi)
    report = {"synthetic": {"valid_pixels": int(roi.sum()), "objects": objects, "bias": bias,
                             "size_distribution": size_distribution(objects)},
              "calibration": {"synthetic_um_per_pixel": demo_um_per_pixel, "status": "HYPOTHETICAL, artificial data only" if demo_um_per_pixel else "No physical calibration"},
              "parameters": {"seed": 41, "dilation_kernel": [5, 5], "connectivity": 8,
                             "perimeter": "regionprops Crofton (4 directions), unclamped circularity"},
              "fixture_note": "Reconstructed from handout description; source package not attached. Not ASTM-compliant or an unbiased stereological estimator."}
    write_objects(objects, output / "session09_objects.csv")
    save_synthetic(gray, labels, roi, prediction, objects, bias, output / "session09_synthetic.png")
    assert np.array_equal(labels, before_labels) and np.array_equal(roi, before_roi)
    print("Synthetic objects:", json.dumps(objects, indent=2))
    print("Synthetic bias:", json.dumps(bias, indent=2))
    if image_path is not None:
        masks, thresholds = segment(educational)
        candidate_labels = ndi.label(masks["Otsu"], np.ones((3, 3)))[0].astype(np.int32)
        real_roi = np.ones(educational.shape, bool)
        # Deliberately do not pass the hypothetical demo scale to educational data.
        real_objects = measure_objects(candidate_labels, real_roi)
        write_objects(real_objects, output / "session09_educational_objects.csv")
        report["educational"] = {"otsu_thresholds": thresholds, "candidate_fraction": area_fraction(candidate_labels, real_roi),
                                 "candidate_component_count": len(real_objects), "source_sha256": source_hash,
                                 "calibration": None, "size_distribution": size_distribution(real_objects),
                                 "roi_policy": "Full image, annotation not excluded: exploratory only, not a constituent measurement"}
        assert checksum(image_path) == source_hash, "Source file changed"
        print("Educational:", report["educational"])
    (output / "session09_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--demo-um-per-pixel", type=float)
    args = parser.parse_args()
    try:
        run(args.image, args.demo_um_per_pixel)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
