"""MicroStructAI Session 19: synthetic stereology sampling and directional interfaces.

NO research imagery is accessed. Pixel labels are synthetic binary regions, not
ferrite/pearlite labels. This is NOT an ASTM-compliant measurement procedure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _validate(mask: np.ndarray, roi: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size:
        raise ValueError("Mask must be a nonempty 2D binary NumPy array")
    if mask.dtype.kind not in "biu" or np.any((mask != 0) & (mask != 1)):
        raise ValueError("Mask must have binary (0/1) integer or bool labels")
    if not isinstance(roi, np.ndarray) or roi.shape != mask.shape or roi.dtype != bool or not np.any(roi):
        raise ValueError("ROI must be a same-shape nonempty boolean array")


def area_fraction(mask: np.ndarray, roi: np.ndarray) -> dict:
    """Exact 2D pixel-area fraction within a valid ROI (not automatically 3D)."""
    _validate(mask, roi)
    positive = int(np.count_nonzero(mask.astype(bool) & roi))
    valid = int(np.count_nonzero(roi))
    return {"positive_pixels": positive, "valid_pixels": valid, "area_fraction": positive / valid}


def systematic_points(mask: np.ndarray, roi: np.ndarray, step: int = 12,
                      offset: tuple[int, int] = (0, 0)) -> dict:
    """Sample valid ROI grid positions with explicitly supplied grid phase/offset."""
    _validate(mask, roi)
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    if (not isinstance(offset, tuple) or len(offset) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 or v >= step for v in offset
    )):
        raise ValueError("offset must be a 2-tuple of grid phases in [0, step)")
    rows = np.arange(offset[0], mask.shape[0], step)
    cols = np.arange(offset[1], mask.shape[1], step)
    if not len(rows) or not len(cols):
        raise ValueError("grid does not intersect image")
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    keep = roi[rr, cc]
    rr, cc = rr[keep], cc[keep]
    if not len(rr):
        raise ValueError("no grid points fall inside valid ROI")
    hits = int(np.count_nonzero(mask[rr, cc]))
    return {"hits": hits, "valid_points": int(len(rr)), "point_fraction": hits / len(rr),
            "row_coordinates": rr, "column_coordinates": cc}


def synthetic_fields(n: int = 8, size: int = 128, seed: int = 19) -> list[dict]:
    """Deterministic *artificial* heterogeneous 2D scenes and annotation exclusion."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 2 or not isinstance(size, int) or size < 48:
        raise ValueError("Need >=2 synthetic fields of size >=48")
    fields = []
    rr, cc = np.indices((size, size))
    for i in range(n):
        rng = np.random.default_rng(seed + 1009 * i)
        mask = np.zeros((size, size), dtype=np.uint8)
        for _ in range(7 + 2 * i):
            cr = rng.integers(0, size)
            cl = rng.integers(0, size)
            radius = int(rng.integers(6, 16))
            mask[((rr - cr) ** 2 + (cc - cl) ** 2) <= radius ** 2] = 1
        roi = np.ones((size, size), dtype=bool)
        roi[2:16, size - 32:size - 3] = False  # fictitious legend/annotation exclusion
        gray = np.where(mask != 0, 70, 195).astype(np.float32)
        gray += rng.normal(0, 7, gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        fields.append({"group_id": i, "mask": mask, "roi": roi, "image": gray})
    return fields


def compare_fields(fields: list[dict], step: int = 12, seed: int = 19) -> list[dict]:
    """One phase-randomized systematic grid per artificial field."""
    rng = np.random.default_rng(seed)
    results = []
    for field in fields:
        offset = (int(rng.integers(0, step)), int(rng.integers(0, step)))
        truth = area_fraction(field["mask"], field["roi"])
        sampled = systematic_points(field["mask"], field["roi"], step, offset)
        results.append({"group_id": int(field["group_id"]), "area_fraction": truth["area_fraction"],
                        "point_fraction": float(sampled["point_fraction"]),
                        "signed_point_error_pp": 100 * (sampled["point_fraction"] - truth["area_fraction"]),
                        "hits": sampled["hits"], "valid_points": sampled["valid_points"],
                        "offset_row": offset[0], "offset_col": offset[1],
                        "valid_pixels": truth["valid_pixels"]})
    return results


def bootstrap_fields(fractions, replicates: int = 2000, seed: int = 19) -> dict:
    """Resample *fields*, not correlated points, as an uncertainty demonstration."""
    values = np.asarray(fractions, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Need at least two finite, valid field fractions")
    if not isinstance(replicates, int) or replicates < 100:
        raise ValueError("Need at least 100 bootstrap replicates")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"mean_field_fraction": float(values.mean()),
            "percentile_95_interval": [float(lo), float(hi)],
            "replicates": replicates, "n_artificial_fields": len(values),
            "bootstrap_means": samples}


def boundary_line_scan(mask: np.ndarray, roi: np.ndarray, direction: str,
                       spacing: int = 12, offset: int = 3) -> dict:
    """Count 0/1 *constituent interface crossings*, NOT true grain boundaries.

    Horizontal lines traverse columns at selected rows; vertical lines traverse rows
    at selected columns. Count only adjacent sampled positions inside the ROI.
    Complete foreground chords require valid background on both sides of the run.
    """
    _validate(mask, roi)
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be horizontal or vertical")
    if not isinstance(spacing, int) or spacing < 1 or not isinstance(offset, int) or not (0 <= offset < spacing):
        raise ValueError("invalid scan spacing or offset")
    limit = mask.shape[0] if direction == "horizontal" else mask.shape[1]
    positions = list(range(offset, limit, spacing))
    if not positions:
        raise ValueError("no valid transects")
    total_pairs = crossing_count = 0
    chords = []
    for k in positions:
        labels = (mask[k, :] if direction == "horizontal" else mask[:, k]).astype(bool)
        valid = roi[k, :] if direction == "horizontal" else roi[:, k]
        eligible = valid[:-1] & valid[1:]
        total_pairs += int(eligible.sum())
        crossing_count += int(np.count_nonzero((labels[:-1] != labels[1:]) & eligible))
        # Foreground runs explicitly bracketed by valid background, not field/ROI edges.
        idx = 0
        while idx < len(labels):
            if not (valid[idx] and labels[idx]):
                idx += 1
                continue
            start = idx
            while idx < len(labels) and valid[idx] and labels[idx]:
                idx += 1
            end = idx  # exclusive
            if (start > 0 and end < len(labels) and valid[start - 1] and valid[end]
                    and not labels[start - 1] and not labels[end]):
                chords.append(end - start)
    if not total_pairs:
        raise ValueError("no eligible ROI-adjacent line positions")
    return {"direction": direction, "line_count": len(positions),
            "boundary_crossings": crossing_count, "eligible_adjacent_length_px": total_pairs,
            "boundary_crossing_density_per_px": crossing_count / total_pairs,
            "complete_foreground_chords_px": chords,
            "mean_complete_chord_px": float(np.mean(chords)) if chords else None}


def artificial_bands(size: int = 96, period: int = 24, thickness: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if min(size, period, thickness) < 1 or thickness >= period:
        raise ValueError("invalid artificial stripe dimensions")
    r = np.arange(size)[:, None]
    mask = np.broadcast_to(((r % period) < thickness).astype(np.uint8), (size, size)).copy()
    return mask, np.ones_like(mask, dtype=bool)


def run(output_dir: Path, grid_step: int = 12, seed: int = 19) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Output directory exists; select a fresh path")
    if grid_step < 1:
        raise ValueError("grid_step must be positive")
    fields = synthetic_fields(seed=seed)
    rows = compare_fields(fields, grid_step, seed)
    area = np.array([r["area_fraction"] for r in rows])
    points = np.array([r["point_fraction"] for r in rows])
    boot = bootstrap_fields(area, seed=seed)
    band_mask, band_roi = artificial_bands()
    horizontal = boundary_line_scan(band_mask, band_roi, "horizontal")
    vertical = boundary_line_scan(band_mask, band_roi, "vertical")
    if horizontal["boundary_crossings"] != 0 or vertical["boundary_crossings"] <= 0:
        raise RuntimeError("Directional synthetic fixture did not exhibit anisotropic interfaces")
    output_dir.mkdir(parents=True)
    try:
        with (output_dir / "session19_fields.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        report = {"seed": seed, "grid_step_px": grid_step, "synthetic_fields": len(fields),
                  "equal_field_area_fraction_mean": float(area.mean()),
                  "equal_field_point_fraction_mean": float(points.mean()),
                  "pooled_sampled_point_fraction": float(sum(r["hits"] for r in rows) / sum(r["valid_points"] for r in rows)),
                  "mean_abs_field_sampling_error_pp": float(np.mean(100 * np.abs(points - area))),
                  "area_fraction_field_bootstrap": {k:v for k,v in boot.items() if k != "bootstrap_means"},
                  "directional_binary_interface_probe": {"horizontal": horizontal, "vertical": vertical},
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "data_use": "Deterministic artificial binary masks only; no real steel, no Figshare official test, no micrometre calibration",
                  "interpretation_limits": [
                    "Sampled grid points within an artificial ROI approximate its known pixel area fraction; not independent observations",
                    "Only fields are resampled for illustration; generated scenes are not physical specimens or a representative probability sample",
                    "Directional 0/1 boundary crossings are not grain-boundary intercepts, grain size or a valid ASTM E112 measurement",
                    "Illustrates principles related to systematic point counting but makes no ASTM E562 compliance claim",
                    "Pearlite constituent fraction is not cementite phase fraction; no 3D inference from these artificial scenes"
                  ]}
        (output_dir / "session19_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        f = fields[0]
        axes[0, 0].imshow(f["image"], cmap="gray", vmin=0, vmax=255)
        axes[0, 0].contour(~f["roi"], levels=[.5], colors="yellow", linewidths=1)
        sampled = systematic_points(f["mask"], f["roi"], grid_step,
                                     (rows[0]["offset_row"], rows[0]["offset_col"]))
        hit = f["mask"][sampled["row_coordinates"], sampled["column_coordinates"]].astype(bool)
        for target, color in ((True, "#e27e49"), (False, "#2b92c9")):
            axes[0, 0].scatter(sampled["column_coordinates"][hit==target],
                              sampled["row_coordinates"][hit==target],
                              s=20, color=color, edgecolors="white", linewidths=.3)
        axes[0, 0].set_title("Synthetic field 0: phase-randomized point grid")
        axes[0, 0].set_axis_off()
        x = np.arange(len(rows))
        axes[0, 1].bar(x-.18, area, width=.36, label="Exact ROI pixel fraction")
        axes[0, 1].bar(x+.18, points, width=.36, label="Systematic point fraction")
        axes[0, 1].set(xlabel="Artificial field ID", ylabel="Fraction", title="Point count versus exact 2D area")
        axes[0, 1].set_xticks(x)
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].imshow(band_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for y in range(3, band_mask.shape[0], 12):
            axes[1, 0].axhline(y, color="#e5a43f", alpha=.8, lw=.7)
        for xpos in range(3, band_mask.shape[1], 12):
            axes[1, 0].axvline(xpos, color="#1c8fca", alpha=.8, lw=.7)
        axes[1, 0].set_title("Synthetic horizontal bands: direction matters")
        axes[1, 0].set_axis_off()
        axes[1, 1].hist(boot["bootstrap_means"], bins=25, alpha=.8)
        axes[1, 1].axvline(float(area.mean()), color="black", ls="--", label="Field-mean area fraction")
        axes[1, 1].axvline(boot["percentile_95_interval"][0], color="red", ls=":")
        axes[1, 1].axvline(boot["percentile_95_interval"][1], color="red", ls=":")
        axes[1, 1].set(xlabel="Mean fraction over resampled artificial fields", ylabel="Bootstrap repetitions",
                        title="Resample fields, not individual pixels")
        axes[1, 1].legend(fontsize=8)
        fig.suptitle("Session 19: artificial stereological sampling — NOT real metallography", fontsize=13)
        fig.savefig(output_dir / "session19_sampling.png", dpi=150)
        plt.close(fig)
        return report
    except Exception:
        # preserve output for debugging; avoid claiming success
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/session19_v1"))
    parser.add_argument("--grid-step", type=int, default=12)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    result = run(args.output_dir, args.grid_step, args.seed)
    print("Synthetic field mean area fraction:", round(result["equal_field_area_fraction_mean"], 6))
    print("Mean sampled point fraction:", round(result["equal_field_point_fraction_mean"], 6))
    print("Per-field bootstrap interval:", result["area_fraction_field_bootstrap"]["percentile_95_interval"])
    print("Direction-interface counts:", result["directional_binary_interface_probe"]["horizontal"]["boundary_crossings"],
          result["directional_binary_interface_probe"]["vertical"]["boundary_crossings"])
