"""Session 2: inspect sampling, intensity statistics and interpolation."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def mask_experiment():
    """Show why interpolating categorical IDs can create a nonexistent class."""
    toy_mask = np.array([[0, 0, 2, 2], [0, 0, 2, 2]], dtype=np.uint8)
    image = Image.fromarray(toy_mask)
    nearest = np.asarray(image.resize((11, 2), Image.Resampling.NEAREST))
    bilinear = np.asarray(image.resize((11, 2), Image.Resampling.BILINEAR))
    print("Original classes:", np.unique(toy_mask))
    print("Nearest classes:", np.unique(nearest))
    print("Bilinear classes:", np.unique(bilinear))
    assert set(np.unique(nearest)) == {0, 2}
    assert 1 in np.unique(bilinear)


def run_lab(image_path: Path, output_path: Path):
    if image_path.resolve() == output_path.resolve():
        raise ValueError("Output must differ from the input image.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}. Use --image to select a micrograph.")
    with Image.open(image_path) as image:
        original_mode = image.mode
        raw = np.asarray(image)
        if raw.dtype != np.uint8:
            raise ValueError(
                f"Input uses {raw.dtype}. Do not silently reduce scientific images to 8-bit."
            )
        # Palette indices, CMYK and transparent images need explicit handling.
        if original_mode not in {"L", "RGB"}:
            raise ValueError(f"Input mode {original_mode} is unsupported; use an 8-bit L or RGB image.")
        gray = np.asarray(image.convert("L"))

    height, width = gray.shape
    if height < 240 or width < 360:
        raise ValueError("This exercise's fixed crop requires at least 240 rows and 360 columns.")
    print("Original mode:", original_mode)
    print("Raw data type:", raw.dtype)
    print("Raw shape:", raw.shape)
    print("Grayscale shape:", gray.shape)
    print("Total pixels per channel:", gray.size)
    print("Min / max:", gray.min(), gray.max())
    print("Mean:", round(float(gray.mean()), 2))
    print("Std deviation:", round(float(gray.std()), 2))
    p5, p50, p95 = np.percentile(gray, [5, 50, 95])
    print("5th percentile:", p5)
    print("Median:", p50)
    print("95th percentile:", p95)

    crop = gray[40:240, 60:360]
    pil_gray = Image.fromarray(gray)
    new_size = (width // 2, height // 2)  # Pillow uses (width, height).
    nearest = np.asarray(pil_gray.resize(new_size, Image.Resampling.NEAREST))
    lanczos = np.asarray(pil_gray.resize(new_size, Image.Resampling.LANCZOS))
    print("Crop shape:", crop.shape)
    print("Nearest shape:", nearest.shape)
    print("Lanczos shape:", lanczos.shape)
    print("Same-field sampling multipliers (x, y):", width / new_size[0], height / new_size[1])
    assert crop.shape == (200, 300)
    assert nearest.shape == (height // 2, width // 2)
    assert lanczos.shape == nearest.shape
    mask_experiment()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    try:
        for ax, pixels, title in (
            (axes[0, 0], gray, "Original grayscale"),
            (axes[0, 1], crop, "Crop: rows 40:240, columns 60:360"),
            (axes[1, 0], nearest, "Half-size nearest neighbor"),
            (axes[1, 1], lanczos, "Half-size Lanczos"),
        ):
            # Disable display smoothing so it does not hide resampling differences.
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        axes[0, 2].hist(gray.ravel(), bins=np.arange(257) - 0.5, color="slategray")
        axes[0, 2].set(title="Full-image intensity histogram", xlabel="Intensity",
                       ylabel="Pixel count", xlim=(-0.5, 255.5))
        axes[1, 2].plot(np.arange(width), gray[height // 2, :], linewidth=0.8)
        axes[1, 2].set(title=f"Center-row intensity profile (row {height // 2})",
                       xlabel="Column (pixel)", ylabel="Intensity", ylim=(0, 255))
        fig.suptitle("MicroStructAI | Session 2\nUnverified physical calibration; intensities are not phase labels")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
    print("Saved:", output_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "data/raw/steel_01.png")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/session02_image_lab.png")
    args = parser.parse_args()
    try:
        run_lab(args.image, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
