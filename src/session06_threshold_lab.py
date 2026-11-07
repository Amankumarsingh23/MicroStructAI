"""Threshold candidate dark regions; real-image outputs are not phase labels."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from session03_dataset_audit import ROOT, checksum
from session04_preprocessing import read_gray, to_u8
from session05_protocol import evaluate

PARAMETERS = {"seed": 17, "fixed_threshold": 120, "block_size": 41, "C": 6,
              "noise_sigma": 4, "shading_range": [0.55, 1.25]}


def validate_gray(gray):
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError("Expected a nonempty 2D uint8 image")


def otsu_numpy(gray):
    """Maximize between-class variance; choose the first maximizing threshold."""
    validate_gray(gray)
    if gray.min() == gray.max():
        raise ValueError("Otsu is undefined for a constant image")
    counts = np.bincount(gray.ravel(), minlength=256)
    p = counts.astype(np.float64) / gray.size
    w0 = np.cumsum(p)
    w1 = 1 - w0
    cumulative = np.cumsum(p * np.arange(256))
    valid = (w0 > 0) & (w1 > 1e-12)
    scores = np.full(256, -np.inf)
    scores[valid] = ((cumulative[-1] * w0[valid] - cumulative[valid]) ** 2
                     / (w0[valid] * w1[valid]))
    return int(np.argmax(scores))


def segment(gray, fixed_threshold=120, block_size=41, offset=6):
    validate_gray(gray)
    if not isinstance(block_size, int) or block_size <= 1 or block_size % 2 != 1:
        raise ValueError("block_size must be an odd integer greater than 1")
    if not np.isfinite(fixed_threshold) or not 0 <= fixed_threshold <= 255 or not np.isfinite(offset):
        raise ValueError("Invalid threshold or offset")
    before = gray.copy()
    numpy_t = otsu_numpy(gray)
    otsu_t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    masks = {
        f"Fixed T={fixed_threshold:g}": (gray <= fixed_threshold).astype(np.uint8),
        "Otsu": (otsu // 255).astype(np.uint8),
        "Adaptive mean": cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, block_size, offset) // 255,
        "Adaptive Gaussian": cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, block_size, offset) // 255,
    }
    assert np.array_equal(before, gray), "Input array changed"
    return masks, {"numpy": numpy_t, "opencv": float(otsu_t)}


def make_synthetic():
    truth = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(truth, (85, 90), 39, 1, -1)
    cv2.ellipse(truth, (218, 148), (61, 29), 25, 0, 360, 1, -1)
    truth[189:201, 56:164] = 1
    clean = np.where(truth == 1, 55, 200).astype(np.uint8)
    noise = np.random.default_rng(PARAMETERS["seed"]).normal(0, PARAMETERS["noise_sigma"], clean.shape)
    even = to_u8(clean.astype(float) + noise)
    shading = np.linspace(*PARAMETERS["shading_range"], clean.shape[1])[None, :]
    shaded = to_u8(clean.astype(float) * shading + noise)
    return {"even": even, "shaded": shaded}, truth


def save_figure(gray, truth, masks, path, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    try:
        panels = {"Input intensity": gray,
                  "Known synthetic dark-region mask" if truth is not None else "Input histogram": truth,
                  **masks}
        for ax, (name, pixels) in zip(axes.ravel(), panels.items()):
            if pixels is None:
                ax.hist(gray.ravel(), bins=np.arange(257) - 0.5, color="slategray")
                ax.set(xlabel="Intensity", ylabel="Pixel count")
            else:
                ax.imshow(pixels, cmap="gray", vmin=0, vmax=255 if name == "Input intensity" else 1,
                          interpolation="nearest")
                ax.set_axis_off()
            ax.set_title(name)
        fig.suptitle(title)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def run(image_path=None):
    output = ROOT / "outputs/session06"
    if image_path is not None:
        image_path = Path(image_path).resolve()
        if ((ROOT / "data/raw/figshare_orientation").resolve() in image_path.parents
                or "masks" in {part.casefold() for part in image_path.parts}):
            raise ValueError("Reserved Figshare data and mask folders are excluded from exploration")
        if output.resolve() in image_path.parents:
            raise ValueError("Input must be outside the generated output directory")
        gray_real = read_gray(image_path)
        original_hash = checksum(image_path)
    output.mkdir(parents=True, exist_ok=True)
    scenes, truth = make_synthetic()
    report = {"parameters": PARAMETERS, "opencv_version": cv2.__version__,
              "numpy_version": np.__version__, "synthetic": {},
              "fixture_note": "Reconstructed fixture: the referenced code package was not attached; not an exact reproduction.",
              "semantics": "1=candidate dark region; educational predictions are not validated phase labels"}
    for name, gray in scenes.items():
        masks, thresholds = segment(gray)
        metrics = {method: evaluate(truth, mask) for method, mask in masks.items()}
        report["synthetic"][name] = {"otsu_thresholds": thresholds, "metrics": metrics}
        save_figure(gray, truth, masks, output / f"session06_{name}.png", f"Synthetic thresholding: {name}")
        print(name, "Otsu thresholds:", thresholds)
        for method, result in metrics.items():
            print(f"  {method}: dark-region IoU={result['per_class']['1']['iou']:.6f}")
    if image_path is not None:
        masks, thresholds = segment(gray_real)
        fractions = {method: float(mask.mean()) for method, mask in masks.items()}
        report["educational"] = {"image_path": str(image_path), "source_sha256": original_hash,
                                 "otsu_thresholds": thresholds, "candidate_dark_fractions": fractions,
                                 "supervised_metrics": None}
        save_figure(gray_real, None, masks, output / "session06_educational.png",
                    "Educational steel: candidate intensity regions only; no validated phase labels")
        assert checksum(image_path) == original_hash, "Source file changed"
        print("Educational Otsu:", thresholds, "candidate dark fractions:", fractions)
    (output / "session06_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved results:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    try:
        run(args.image)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
