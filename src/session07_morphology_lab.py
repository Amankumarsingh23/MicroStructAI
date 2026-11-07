"""Binary morphology and component measurements without metallurgical claims."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from session03_dataset_audit import ROOT, checksum
from session04_preprocessing import read_gray
from session05_protocol import evaluate
from session06_threshold_lab import segment


def validate_mask(mask):
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8
            or mask.ndim != 2 or not mask.size or not set(np.unique(mask)) <= {0, 1}):
        raise ValueError("Expected nonempty 2D uint8 binary mask with IDs 0/1")


def validate_kernel(kernel):
    if (not isinstance(kernel, np.ndarray) or kernel.dtype != np.uint8 or kernel.ndim != 2
            or not kernel.size or any(size % 2 != 1 for size in kernel.shape)
            or not set(np.unique(kernel)) <= {0, 1}
            or kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] != 1
            or not np.array_equal(kernel, kernel[::-1, ::-1])):
        raise ValueError("Kernel must be odd-sized, symmetric uint8 0/1 with an active center")


def morphologies(mask, kernel=None):
    validate_mask(mask)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    validate_kernel(kernel)
    original = mask.copy()
    # OpenCV default morphology border uses neutral values (erosion high, dilation low).
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    results = {
        "Candidate": mask.copy(), "Erosion": cv2.erode(mask, kernel),
        "Dilation": cv2.dilate(mask, kernel), "Opening": opened,
        "Closing": cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel),
        "Opening -> Closing": cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel),
    }
    assert np.array_equal(mask, original), "Input mask changed"
    for result in results.values():
        validate_mask(result)
        assert result.shape == mask.shape
    return results


def components(mask, connectivity=8):
    validate_mask(mask)
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    return [{"component_id": label, "area_px": int(stats[label, cv2.CC_STAT_AREA]),
             "bbox_xywh_px": [int(value) for value in stats[label, :4]],
             "centroid_xy_px": [float(value) for value in centroids[label]]}
            for label in range(1, count)]


def make_synthetic():
    truth = np.zeros((192, 256), dtype=np.uint8)
    truth[30:112, 25:85] = 1
    cv2.circle(truth, (172, 80), 29, 1, -1)
    line = np.zeros_like(truth)
    line[136:176, 218] = 1
    truth[line == 1] = 1
    candidate = truth.copy()
    # Keep added speckles away from truth so line survival is independently interpretable.
    allowed = cv2.dilate(truth, np.ones((9, 9), dtype=np.uint8)) == 0
    positions = np.flatnonzero(allowed)
    chosen = np.random.default_rng(23).choice(positions, size=120, replace=False)
    candidate.ravel()[chosen] = 1
    candidate[65, 55] = 0
    candidate[80, 172] = 0
    return truth, candidate, line


def summarize(mask, truth=None, line=None):
    objects = components(mask)
    result = {"foreground_pixels": int(mask.sum()), "foreground_fraction": float(mask.mean()),
              "connectivity": 8, "component_count": len(objects), "components": objects}
    if truth is not None:
        result["metrics"] = evaluate(truth, mask)
        result["signed_area_error_px"] = int(mask.sum()) - int(truth.sum())
        result["absolute_area_error_px"] = abs(result["signed_area_error_px"])
        result["area_error_fraction_of_truth"] = result["signed_area_error_px"] / int(truth.sum())
    if line is not None:
        result["thin_feature_retained_pixels"] = int(np.count_nonzero((line == 1) & (mask == 1)))
        result["thin_feature_truth_pixels"] = int(line.sum())
    return result


def figure(reference, results, path, title, educational=False):
    panels = {"Input grayscale" if educational else "Known artificial truth": reference,
              **{name: results[name] for name in ("Candidate", "Erosion", "Dilation", "Opening", "Opening -> Closing")}}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    try:
        for ax, (name, mask) in zip(axes.ravel(), panels.items()):
            ax.imshow(mask, cmap="gray", vmin=0, vmax=255 if name == "Input grayscale" else 1,
                      interpolation="nearest")
            ax.set_title(name)
            ax.set_axis_off()
        fig.suptitle(title)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def run(image_path=None):
    output = ROOT / "outputs/session07"
    if image_path is not None:
        image_path = Path(image_path).resolve()
        if ((ROOT / "data/raw/figshare_orientation").resolve() in image_path.parents
                or "masks" in {part.casefold() for part in image_path.parts}):
            raise ValueError("Reserved Figshare data/mask folders cannot be used for exploration")
        if output.resolve() in image_path.parents:
            raise ValueError("Input cannot be a generated output")
        original_hash = checksum(image_path)
        gray = read_gray(image_path)
    output.mkdir(parents=True, exist_ok=True)
    truth, candidate, line = make_synthetic()
    results = morphologies(candidate)
    report = {"opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "parameters": {"seed": 23, "kernel": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                             "iterations": 1, "connectivity": 8, "added_speckles": 120,
                             "removed_interior_pixels": 2, "border": "OpenCV neutral morphology default"},
              "fixture_note": "Reconstructed fixture; referenced source package was not attached. Not steel.",
              "synthetic": {name: summarize(mask, truth, line) for name, mask in results.items()}}
    diagonal = np.eye(3, dtype=np.uint8)
    report["diagonal_connectivity_counts"] = {str(c): len(components(diagonal, c)) for c in (4, 8)}
    figure(truth, results, output / "session07_synthetic.png", "Morphology: shape changes on controlled masks")
    for name, stats in report["synthetic"].items():
        print(f"{name}: IoU={stats['metrics']['per_class']['1']['iou']:.6f}, "
              f"area error={stats['signed_area_error_px']:+d} px, line={stats['thin_feature_retained_pixels']}/40")
    print("Diagonal component counts:", report["diagonal_connectivity_counts"])
    if image_path is not None:
        masks, thresholds = segment(gray)
        real_results = morphologies(masks["Otsu"])
        report["educational"] = {"source_sha256": original_hash, "image_path": str(image_path),
                                 "otsu_thresholds": thresholds,
                                 "results": {name: summarize(mask) for name, mask in real_results.items()},
                                 "supervised_metrics": None}
        figure(gray, real_results, output / "session07_educational.png",
               "Educational Otsu candidates: geometry changes only; no validated phase/particle labels", True)
        assert checksum(image_path) == original_hash, "Source file changed"
        for name, stats in report["educational"]["results"].items():
            print(f"Educational {name}: fraction={stats['foreground_fraction']:.6f}, components={stats['component_count']}")
    (output / "session07_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    try:
        run(args.image)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
