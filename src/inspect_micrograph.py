"""Inspect image intensities; these measurements do not identify steel phases."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop plotting window.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def inspect_micrograph(image_path: Path, output_path: Path):
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Missing image: {image_path}. Place a micrograph there or use --image."
        )

    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
        gray = np.array(image.convert("L"))

    height, width = gray.shape
    row, column = height // 2, width // 2
    # Slice stops are exclusive. Ensure that even a 1-pixel image has a crop.
    top, left = height // 4, width // 4
    bottom, right = max(top + 1, 3 * height // 4), max(left + 1, 3 * width // 4)
    crop = gray[top:bottom, left:right]

    print("RGB shape:", rgb.shape)
    print("Grayscale shape:", gray.shape)
    print("Data type:", gray.dtype)
    print("Minimum intensity:", gray.min())
    print("Maximum intensity:", gray.max())
    print("Average intensity:", gray.mean())
    print(f"Center pixel intensity at [{row}, {column}]:", gray[row, column])
    print(f"Crop rows [{top}:{bottom}], columns [{left}:{right}]")
    print("Crop shape:", crop.shape)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    try:
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("Original (RGB)")
        axes[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("Grayscale (8-bit)")
        axes[1, 0].imshow(crop, cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title("Central crop (approximately half each dimension)")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_axis_off()
        axes[1, 1].hist(gray.ravel(), bins=np.arange(257) - 0.5, color="slategray")
        axes[1, 1].set(xlim=(-0.5, 255.5), xlabel="Pixel intensity", ylabel="Pixel count",
                       title="Intensity histogram (256 bins)")
        fig.suptitle(f"MicroStructAI | {image_path.name}\nBrightness does not establish phase identity")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
    print("Saved inspection:", output_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "data/raw/steel_01.png")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/steel_01_inspection.png")
    args = parser.parse_args()
    if args.image.resolve() == args.output.resolve():
        parser.error("Output must differ from the input image.")
    try:
        inspect_micrograph(args.image, args.output)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
