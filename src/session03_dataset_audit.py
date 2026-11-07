"""Validate normalized train/test PNG pairs without changing source files."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/demo/session03"
LABELS = set(range(10))
SCHEMA = "figshare-30175549-v2:0-8=pearlite-orientation;9=ferrite"


def checksum(path):
    """Stream a file's SHA-256 digest (not a visual-similarity test)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_demo(folder=DEMO):
    """Create deterministic artificial fixtures only in the dedicated demo path."""
    folder = Path(folder).resolve()
    if folder != DEMO.resolve():
        raise ValueError("--demo may only write to data/demo/session03, never a research directory.")
    marker = folder / ".microstructai-demo"
    if folder.exists() and any(folder.iterdir()) and not marker.is_file():
        raise ValueError("Refusing to overwrite a nonempty folder without the demo marker.")
    folder.mkdir(parents=True, exist_ok=True)
    marker.write_text("Artificial Session 3 fixtures; not microscopy data.\n", encoding="utf-8")
    rng = np.random.default_rng(42)
    for split, names in {"train": ["a", "b"], "test": ["c"]}.items():
        for kind in ("images", "masks"):
            (folder / split / kind).mkdir(parents=True, exist_ok=True)
        for name in names:
            pixels = rng.integers(0, 256, (32, 32), dtype=np.uint8)
            mask = rng.integers(0, 10, (32, 32), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / split / "images" / f"{name}.png")
            Image.fromarray(mask).save(folder / split / "masks" / f"{name}.png")
    print("Created artificial test files.")


def inspect_pair(image_path, mask_path):
    """Read mask IDs directly, including palette indices; never convert masks."""
    with Image.open(image_path) as image:
        pixels = np.asarray(image)
        if image.mode not in {"L", "RGB"} or pixels.dtype != np.uint8:
            raise ValueError(f"Unsupported image mode/type: {image_path} ({image.mode}, {pixels.dtype})")
        height, width = pixels.shape[:2]
        image_mode = image.mode
    with Image.open(mask_path) as mask:
        labels = np.asarray(mask)
        if mask.mode not in {"L", "P"} or labels.ndim != 2 or labels.dtype != np.uint8:
            raise ValueError(f"Unsupported mask mode/type: {mask_path} ({mask.mode}, {labels.dtype})")
        if labels.shape != (height, width):
            raise ValueError(f"Mismatched dimensions: {image_path} {(height, width)} vs {mask_path} {labels.shape}")
        unique = {int(value) for value in np.unique(labels)}
        invalid = unique - LABELS
        if invalid:
            raise ValueError(f"Invalid class IDs {sorted(invalid)} in {mask_path}; expected 0..9.")
    return {"height": height, "width": width, "image_mode": image_mode,
            "mask_mode": mask.mode, "classes": "|".join(map(str, sorted(unique)))}


def _files(folder):
    if not folder.is_dir():
        raise ValueError(f"Missing directory: {folder}")
    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected entry (expected flat PNG files): {path}")
        key = path.name.casefold()
        if key in files:
            raise ValueError(f"Ambiguous filename: {path}")
        files[key] = path
    return files


def audit(data_root, manifest, source_dataset="UNKNOWN"):
    """Audit all pairs first, then atomically write a reproducible CSV manifest."""
    data_root, manifest = Path(data_root).resolve(), Path(manifest).resolve()
    if manifest == data_root or data_root in manifest.parents:
        raise ValueError("Manifest must be outside the dataset directory.")
    if not data_root.is_dir():
        raise ValueError(f"Missing dataset root: {data_root}")
    unexpected = {p.name for p in data_root.iterdir()} - {"train", "test", ".microstructai-demo"}
    if unexpected:
        raise ValueError(f"Unexpected dataset entries: {sorted(unexpected)}")
    rows, seen = [], {}
    counts = {"train": 0, "test": 0}
    for split in counts:
        images = _files(data_root / split / "images")
        masks = _files(data_root / split / "masks")
        if images.keys() != masks.keys():
            raise ValueError(f"Unpaired filenames in {split}: missing masks {sorted(images.keys() - masks.keys())}; "
                             f"missing images {sorted(masks.keys() - images.keys())}")
        for name, image_path in images.items():
            mask_path = masks[name]
            info = inspect_pair(image_path, mask_path)
            image_hash = checksum(image_path)
            if image_hash in seen and seen[image_hash][0] != split:
                raise ValueError(f"Exact image duplicate across splits: {seen[image_hash][1]} and {image_path}")
            seen[image_hash] = (split, image_path)
            rows.append({
                "sample_id": f"{split}/{image_path.stem}", "split": split,
                "image_path": image_path.relative_to(data_root).as_posix(),
                "mask_path": mask_path.relative_to(data_root).as_posix(),
                "source_dataset": source_dataset, "specimen_id": "UNKNOWN",
                "source_group_id": "UNKNOWN", "acquisition_method": "UNKNOWN",
                "pixel_scale_um": "", "label_schema": SCHEMA, **info,
                "image_sha256": image_hash, "mask_sha256": checksum(mask_path),
            })
            counts[split] += 1
    if not rows:
        raise ValueError("No image/mask pairs found.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=manifest.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, manifest)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    print(f"PASS: {len(rows)} image/mask pairs audited")
    print("Manifest:", manifest)
    for split, count in counts.items():
        print(f"{split}: {count} pairs")
    print("Note: Exact duplicates checked.\nSource-lineage independence remains unverified.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Generate only the dedicated artificial fixtures")
    parser.add_argument("--data-root", type=Path, default=DEMO)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/session03_manifest.csv")
    parser.add_argument("--source-dataset", default="UNKNOWN", help="Verified source DOI/name; do not guess")
    args = parser.parse_args()
    try:
        if args.demo:
            make_demo(args.data_root)
        source = "ARTIFICIAL-session03" if args.data_root.resolve() == DEMO.resolve() else args.source_dataset
        audit(args.data_root, args.manifest, source)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\nAudit failed; any existing manifest was NOT refreshed.\n")


if __name__ == "__main__":
    main()
