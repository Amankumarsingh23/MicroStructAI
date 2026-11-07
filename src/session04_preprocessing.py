"""Compare independent preprocessing methods; never infer phase labels from contrast."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from session03_dataset_audit import ROOT, checksum

PARAMETERS = {
    "seed": 42, "noise_sigma": 10.0, "impulse_probability": 0.02,
    "gaussian_sigma": 1.0, "gaussian_kernel": [5, 5], "median_kernel": 3,
    "clahe_clip_limit": 2.0, "clahe_tile_grid": [8, 8],
    "illumination_sigma": 55.0, "illumination_epsilon": 1.0,
}


def read_gray(path):
    """Reject unsupported types before any potentially lossy conversion."""
    with Image.open(path) as image:
        pixels = np.asarray(image)
        if image.mode not in {"L", "RGB"} or pixels.dtype != np.uint8:
            raise ValueError("Expected 8-bit L/RGB image")
        return np.asarray(image.convert("L")).copy()


def to_u8(values):
    """Round and clip, avoiding wraparound from direct uint8 casting."""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _validate_gray(gray):
    if gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError("Expected a nonempty 2D uint8 intensity image")


def illumination_correct(gray, sigma=55):
    """Heuristic shading estimate, NOT calibrated blank/dark-field correction."""
    _validate_gray(gray)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Illumination sigma must be finite and positive")
    observed = gray.astype(np.float32)
    field = cv2.GaussianBlur(observed, (0, 0), sigmaX=sigma,
                             borderType=cv2.BORDER_REFLECT_101)
    reference = float(field.mean())
    return to_u8(observed * reference / np.maximum(field, PARAMETERS["illumination_epsilon"]))


def preprocess(gray):
    """Independent comparisons plus one explicitly named correction+CLAHE variant."""
    _validate_gray(gray)
    before = gray.copy()
    clahe = cv2.createCLAHE(clipLimit=PARAMETERS["clahe_clip_limit"],
                            tileGridSize=tuple(PARAMETERS["clahe_tile_grid"]))
    corrected = illumination_correct(gray, PARAMETERS["illumination_sigma"])
    results = {
        "Original": gray.copy(),
        "Gaussian (sigma=1 px)": cv2.GaussianBlur(
            gray, tuple(PARAMETERS["gaussian_kernel"]), sigmaX=PARAMETERS["gaussian_sigma"], borderType=cv2.BORDER_REFLECT_101),
        "Median (3 x 3)": cv2.medianBlur(gray, PARAMETERS["median_kernel"]),
        "CLAHE (clip=2, tiles=8 x 8)": clahe.apply(gray),
        "Heuristic illumination correction": corrected,
        "Correction + CLAHE": clahe.apply(corrected),
    }
    assert np.array_equal(gray, before), "Preprocessing modified the source array"
    return results


def mae(reference, result):
    if reference.shape != result.shape:
        raise ValueError("MAE requires matching shapes")
    # Convert BEFORE subtraction: uint8 subtraction would wrap around.
    return float(np.abs(reference.astype(np.float64) - result.astype(np.float64)).mean())


def synthetic_experiment():
    """Deterministic toy shapes, mixed noise and a separate shading corruption."""
    clean = np.full((256, 256), 200, dtype=np.uint8)
    cv2.circle(clean, (70, 75), 28, 60, thickness=-1)
    cv2.circle(clean, (180, 160), 40, 80, thickness=-1)
    clean[35:220, 126:128] = 50  # A deliberately fragile, two-pixel-wide feature.
    source = clean.copy()
    rng = np.random.default_rng(PARAMETERS["seed"])
    noisy = to_u8(clean.astype(np.float64) + rng.normal(0, PARAMETERS["noise_sigma"], clean.shape))
    impulses = rng.random(clean.shape)
    noisy[impulses < PARAMETERS["impulse_probability"] / 2] = 0
    noisy[impulses > 1 - PARAMETERS["impulse_probability"] / 2] = 255
    variants = preprocess(noisy)
    gaussian = variants["Gaussian (sigma=1 px)"]
    median = variants["Median (3 x 3)"]
    # Known illumination is used only to generate corruption, not for estimation.
    illumination = np.linspace(0.6, 1.2, clean.shape[1])[None, :]
    shaded = to_u8(clean.astype(np.float64) * illumination)
    corrected = illumination_correct(shaded)
    panels = {"Clean artificial reference": clean, "Mixed Gaussian + impulse noise": noisy,
              "Gaussian-filtered noisy image": gaussian, "Median-filtered noisy image": median,
              "Known multiplicative shading": shaded, "Heuristic corrected shading": corrected}
    metrics = {name: mae(clean, result) for name, result in list(panels.items())[1:]}
    line_metrics = {name: mae(clean[35:220, 126:128], result[35:220, 126:128])
                    for name, result in list(panels.items())[1:]}
    assert np.array_equal(clean, source), "Synthetic clean reference was modified"
    return panels, metrics, line_metrics


def save_panels(panels, path, title, metrics=None):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    try:
        for ax, (name, pixels) in zip(axes.ravel(), panels.items()):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            suffix = f"\nMAE to clean: {metrics[name]:.3f}" if metrics and name in metrics else ""
            ax.set_title(name + suffix)
            ax.set_axis_off()
        fig.suptitle(title)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def run(image_path=None):
    # Prevent this exploratory lab from becoming an official-test tuning tool.
    if image_path is not None:
        image_path = Path(image_path).resolve()
        reserved = (ROOT / "data/raw/figshare_orientation").resolve()
        if reserved in image_path.parents or "masks" in {p.casefold() for p in image_path.parts}:
            raise ValueError("Use educational/training images, not reserved Figshare data or masks.")
        if image_path in {ROOT / "outputs" / name for name in (
            "session04_synthetic.png", "session04_real_comparison.png", "session04_metrics.json"
        )}:
            raise ValueError("Input must differ from all generated output paths")
        gray = read_gray(image_path)
        original_hash = checksum(image_path)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    panels, metrics, line_metrics = synthetic_experiment()
    save_panels(panels, output / "session04_synthetic.png",
                "Session 4 | Artificial numerical reference (not steel)\nMixed-noise and shading experiments are separate", metrics)
    for name, value in metrics.items():
        print(f"{name}: MAE = {value:.3f}")
    report = {"parameters": PARAMETERS, "opencv_version": cv2.__version__,
              "numpy_version": np.__version__, "synthetic_mae": metrics,
              "thin_line_roi_mae": line_metrics,
              "scope": "Own deterministic fixture; handout's truncated generator cannot be reproduced exactly."}
    if image_path is not None:
        variants = preprocess(gray)
        save_panels(variants, output / "session04_real_comparison.png",
                    "Session 4 | Educational micrograph: qualitative comparison only\nNo clean reference; no phase accuracy or physical calibration established")
        if checksum(image_path) != original_hash:
            raise RuntimeError("Source file changed during processing")
        report["real_image"] = {"path": str(image_path), "sha256": original_hash,
                                "shape": list(gray.shape), "source_preserved": True}
        print("Saved: outputs/session04_real_comparison.png; source SHA-256 unchanged")
    (output / "session04_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved: outputs/session04_synthetic.png and outputs/session04_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional educational 8-bit L/RGB micrograph")
    args = parser.parse_args()
    try:
        run(args.image)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
