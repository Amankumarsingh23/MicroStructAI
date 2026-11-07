"""Fetch version-pinned metadata and a small official-test subset; no training data."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from session03_dataset_audit import ROOT, audit, checksum

API = "https://api.figshare.com/v2/articles/30175549/versions/2"
DOI = "10.6084/m9.figshare.30175549.v2"


def prepare(pairs):
    if not 1 <= pairs <= 499:
        raise ValueError("Choose 1..499 test pairs.")
    with urlopen(API, timeout=30) as response:
        metadata = json.load(response)
    metadata_dir = ROOT / "data/metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "figshare-30175549-v2.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if metadata.get("version") != 2 or metadata.get("doi") != DOI:
        raise ValueError("Unexpected dataset identity/version; inspect saved metadata.")
    if metadata.get("license", {}).get("name") != "CC BY 4.0":
        raise ValueError("License changed; inspect saved metadata before downloading.")
    archive_info = next(item for item in metadata["files"] if item["name"] == "Testing.zip")
    archive = ROOT / "data/downloads/Testing.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        with urlopen(archive_info["download_url"], timeout=60) as response, partial.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        partial.replace(archive)
    if archive.stat().st_size != archive_info["size"]:
        raise ValueError("Archive byte count differs from metadata.")
    with archive.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if md5 != archive_info["computed_md5"]:
        raise ValueError("Archive MD5 differs from Figshare metadata.")

    destination = ROOT / "data/raw/figshare_orientation"
    for split in ("train", "test"):
        for kind in ("images", "masks"):
            (destination / split / kind).mkdir(parents=True, exist_ok=True)
    records = []
    with ZipFile(archive) as source:
        images = {Path(n).name: n for n in source.namelist()
                  if n.startswith("Testing/Synthetic Micrographs/") and n.lower().endswith(".png")}
        masks = {Path(n).name: n for n in source.namelist()
                 if n.startswith("Testing/Ground Truth/") and n.lower().endswith(".png")}
        if not images or images.keys() != masks.keys():
            raise ValueError("Archive layout or image/mask filenames differ from expectations.")
        selected = sorted(images, key=lambda name: int(Path(name).stem))[:pairs]
        expected = set(selected)
        for kind in ("images", "masks"):
            existing = {p.name for p in (destination / "test" / kind).iterdir()}
            if existing - expected:
                raise ValueError("Destination has extra files; use the existing audit or a fresh checkout.")
        for name in selected:
            record = {"filename": name, "official_split": "test"}
            for kind, entries in (("images", images), ("masks", masks)):
                payload = source.read(entries[name])  # Explicit members, no bulk path extraction.
                target = destination / "test" / kind / name
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError(f"Refusing to replace modified source data: {target}")
                if not target.exists():
                    target.write_bytes(payload)
                record[kind + "_archive_member"] = entries[name]
                record[kind + "_sha256"] = checksum(target)
            records.append(record)
    provenance = {
        "doi": DOI, "metadata_url": API, "license": metadata["license"],
        "authors": metadata["authors"], "archive": archive_info,
        "archive_sha256": checksum(archive), "archive_image_count": len(images),
        "archive_mask_count": len(masks), "selected_pairs": records,
        "scope": "Official-test subset only; train folders intentionally empty. No split reassignment.",
    }
    (metadata_dir / "session03_real_subset.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Archive contains {len(images)} images and {len(masks)} masks.")
    audit(destination, ROOT / "outputs/session03_real_manifest.csv", DOI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    try:
        prepare(args.pairs)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"Error: {error}\n")
