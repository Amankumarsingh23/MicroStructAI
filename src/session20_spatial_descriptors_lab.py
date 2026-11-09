"""MicroStructAI Session 20: synthetic spatial descriptors and validity checks.

NO official research-test data or physical steel images. The point patterns are
artificial; connected components are mask regions, NOT verified grains/particles.
Distances are in pixels unless separately calibrated. This is not a point-process
inference standard, ASTM method, or independent-specimen experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def validate_binary(mask: np.ndarray, roi: np.ndarray) -> None:
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0
            or mask.dtype.kind not in 'biu' or np.any((mask != 0) & (mask != 1))):
        raise ValueError('mask must contain only 0/1 integer or bool labels in a nonempty 2D array')
    if (not isinstance(roi, np.ndarray) or roi.shape != mask.shape
            or roi.dtype != bool or not roi.any()):
        raise ValueError('roi must be a same-shape nonempty boolean valid-area mask')


def component_table(mask: np.ndarray, roi: np.ndarray, connectivity: int = 8) -> dict:
    """Measure observed 2D connected components. Censored objects are flagged, not silently sized as complete."""
    validate_binary(mask, roi)
    if connectivity not in (4, 8):
        raise ValueError('connectivity must be 4 or 8')
    structure = ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, number = ndi.label((mask == 1) & roi, structure=structure)
    # A pixel in this band is close to either the image frame or an excluded ROI.
    safe = ndi.binary_erosion(roi, structure=np.ones((3, 3), dtype=bool), border_value=0)
    risky = roi & ~safe
    objects = []
    for ident in range(1, number + 1):
        rr, cc = np.nonzero(labels == ident)
        area = int(len(rr))
        censored = bool(np.any(risky[rr, cc]))
        objects.append({
            'component_id': ident,
            'area_px2': area,
            'centroid_row_px': float(rr.mean()),
            'centroid_col_px': float(cc.mean()),
            'ecd_px': float(np.sqrt(4 * area / np.pi)),
            'boundary_censored': censored,
        })
    return {
        'connectivity': connectivity,
        'objects': objects,
        'component_count': number,
        'complete_count': sum(not obj['boundary_censored'] for obj in objects),
        'censored_count': sum(obj['boundary_censored'] for obj in objects),
        'positive_roi_pixels': int(np.count_nonzero((mask == 1) & roi)),
        'roi_area_px2': int(roi.sum()),
        'observed_component_density_per_px2': number / int(roi.sum()),
        'complete_component_density_per_px2': sum(not obj['boundary_censored'] for obj in objects) / int(roi.sum()),
        'limitations': 'Connected regions are label-dependent 2D components, not confirmed physical particles or grain instances. Censored regions are flagged, not corrected. Density is per ROI pixel area.'
    }


def validate_points(points: np.ndarray, roi: np.ndarray) -> np.ndarray:
    if not isinstance(roi, np.ndarray) or roi.dtype != bool or roi.ndim != 2 or not roi.any():
        raise ValueError('roi must be a nonempty 2D bool array')
    if (not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2
            or not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all()):
        raise ValueError('points must be a finite numeric [N,2] array of row/column coordinates')
    pts = points.astype(float)
    if len(pts) > 1 and np.unique(pts, axis=0).shape[0] != len(pts):
        raise ValueError('duplicate point centers are not allowed in this teaching lab')
    if np.any(pts < 0) or np.any(pts[:, 0] >= roi.shape[0]) or np.any(pts[:, 1] >= roi.shape[1]):
        raise ValueError('a point lies outside the image')
    rr, cc = np.floor(pts).astype(int).T
    if not np.all(roi[rr, cc]):
        raise ValueError('all point centers must be inside the valid ROI')
    return pts


def nearest_neighbor_summary(points: np.ndarray, roi: np.ndarray) -> dict:
    """Nearest other CENTER distance with conservative boundary-risk flags.

    Boundary risk: a nearest-neighbor search circle extends beyond the observation
    window. Interior-only averaging is descriptive, not an unbiased edge correction.
    """
    pts = validate_points(points, roi)
    if len(pts) < 2:
        raise ValueError('at least two centers are required for nearest-neighbor analysis')
    nearest = cKDTree(pts).query(pts, k=2)[0][:, 1]
    padded = np.pad(roi, 1, constant_values=False)
    distance_to_invalid_pixel = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
    pixel_indices = np.floor(pts).astype(int)
    edge_distances = distance_to_invalid_pixel[pixel_indices[:, 0], pixel_indices[:, 1]]
    risk = nearest >= edge_distances
    roi_area = int(roi.sum())
    density = len(pts) / roi_area
    expected_infinite_plane = 1.0 / (2.0 * np.sqrt(density))
    return {
        'count': len(pts),
        'roi_area_px2': roi_area,
        'centroid_density_per_px2': float(density),
        'nearest_distances_px': nearest.tolist(),
        'mean_nearest_px': float(nearest.mean()),
        'median_nearest_px': float(np.median(nearest)),
        'edge_risk_flags': risk.tolist(),
        'edge_risk_count': int(np.sum(risk)),
        'interior_only_mean_nearest_px': float(nearest[~risk].mean()) if np.any(~risk) else None,
        'naive_infinite_plane_csr_expected_px': float(expected_infinite_plane),
        'naive_clark_evans_ratio': float(nearest.mean() / expected_infinite_plane),
        'limitations': 'Centers only; no object-size adjustment. Naive Clark-Evans reference is an unbounded continuous-Poisson approximation, not corrected for ROI boundaries or finite hard-core bodies; interpret with CSR Monte Carlo and caution.'
    }


def quadrat_summary(points: np.ndarray, roi: np.ndarray, cell_size: int = 16) -> dict:
    """Counts centers in equal-area fully valid quadrats; discards partially valid cells."""
    pts = validate_points(points, roi)
    if isinstance(cell_size, bool) or not isinstance(cell_size, (int, np.integer)) or cell_size < 1:
        raise ValueError('cell_size must be a positive integer')
    h, w = roi.shape
    n_rows, n_cols = h // cell_size, w // cell_size
    valid_cells = np.zeros((n_rows, n_cols), dtype=bool)
    counts = np.zeros((n_rows, n_cols), dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            block = roi[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
            valid_cells[r, c] = bool(block.all())
    for r, c in pts:
        ri, ci = int(np.floor(r)) // cell_size, int(np.floor(c)) // cell_size
        if ri < n_rows and ci < n_cols and valid_cells[ri, ci]:
            counts[ri, ci] += 1
    eligible_counts = counts[valid_cells]
    if eligible_counts.size < 2:
        raise ValueError('at least two fully valid quadrats are required')
    average = float(np.mean(eligible_counts))
    variance = float(np.var(eligible_counts, ddof=1))
    return {
        'cell_size_px': int(cell_size),
        'eligible_quadrats': int(eligible_counts.size),
        'excluded_partial_or_invalid_quadrats': int(valid_cells.size - eligible_counts.size),
        'points_in_eligible_quadrats': int(eligible_counts.sum()),
        'counts': eligible_counts.tolist(),
        'mean_count': average,
        'sample_variance_count': variance,
        'dispersion_index': float(variance / average) if average > 0 else None,
        'limitations': 'Variance/mean is descriptive; this statistic depends on quadrat size and origin. Valid full cells only; incomplete/ROI-cut cells excluded, and any points in them are omitted.'
    }


def csr_monte_carlo(points: np.ndarray, roi: np.ndarray, repetitions: int = 300, seed: int = 20) -> dict:
    """Reference for point-center NN under uniformly random discrete-pixel centers in the SAME ROI.

    Conditional on observed N; sampling without replacement creates a discrete-site
    null model, not an exact continuous Poisson or hard-core particle null.
    """
    pts = validate_points(points, roi)
    if len(pts) < 2:
        raise ValueError('at least two centers required')
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 20:
        raise ValueError('repetitions must be an integer >=20')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be nonnegative int')
    eligible = np.column_stack(np.nonzero(roi))
    if len(pts) > len(eligible):
        raise ValueError('more point centers than valid pixels')
    rng = np.random.default_rng(seed)
    observed = nearest_neighbor_summary(pts, roi)['mean_nearest_px']
    reference = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        sample = eligible[rng.choice(len(eligible), size=len(pts), replace=False)].astype(float)
        reference[i] = cKDTree(sample).query(sample, k=2)[0][:, 1].mean()
    return {
        'null': 'Uniform random unique pixel-center positions conditional on N and the same valid ROI',
        'seed': seed,
        'repetitions': repetitions,
        'observed_mean_nearest_px': float(observed),
        'null_mean_px': float(reference.mean()),
        'null_05_95_px': [float(x) for x in np.quantile(reference, [0.05, 0.95])],
        'fraction_simulations_le_observed': float((1 + np.count_nonzero(reference <= observed)) / (repetitions + 1)),
        'fraction_simulations_ge_observed': float((1 + np.count_nonzero(reference >= observed)) / (repetitions + 1)),
        'limitations': 'Exploratory finite-ROI point-center reference; not a confirmatory p-value, not a specimen-level uncertainty interval. Objects have spatial extent, so a discrete uniform point null can be physically inappropriate.'
    }


def artificial_patterns(seed: int = 20, side: int = 128, count: int = 30) -> dict:
    if side != 128 or count != 30 or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('this fixed teaching fixture uses side=128, count=30 and nonnegative integer seed')
    roi = np.ones((side, side), dtype=bool)
    roi[:22, 98:] = False  # mimic an excluded overlay / invalid window segment
    rng = np.random.default_rng(seed)
    eligible = np.column_stack(np.nonzero(roi))
    random_points = eligible[rng.choice(len(eligible), size=count, replace=False)].astype(float)
    r_grid = np.linspace(28, 104, 5)
    c_grid = np.linspace(16, 106, 6)
    regular = np.array([(r, c) for r in r_grid for c in c_grid], dtype=float)
    clustered = []
    for center in ((35, 35), (66, 77), (101, 104)):
        accepted = 0
        while accepted < 10:
            candidate = np.rint(rng.normal(center, 4.5, size=2)).astype(int)
            if (np.all((candidate >= 0) & (candidate < side))
                    and roi[candidate[0], candidate[1]]
                    and tuple(candidate) not in clustered):
                clustered.append(tuple(candidate))
                accepted += 1
    patterns = {'random': random_points, 'regular': regular, 'clustered': np.asarray(clustered, dtype=float)}
    for points in patterns.values():
        validate_points(points, roi)
    return {'roi': roi, 'patterns': patterns, 'seed': seed}


def artificial_component_mask(roi: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(roi, dtype=np.uint8)
    mask[14:22, 12:22] = 1      # complete object
    mask[60:66, 74:82] = 1      # complete object
    mask[85:90, 123:128] = 1   # image boundary censored
    mask[17:29, 94:103] = 1    # intersects invalid ROI, censored
    return mask


def run(output_dir: Path, seed: int = 20, repetitions: int = 300, cell_size: int = 16) -> dict:
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError('Output directory exists; choose a new one')
    if not (isinstance(seed, int) and seed >= 0 and isinstance(repetitions, int)
            and repetitions >= 20 and isinstance(cell_size, int) and cell_size >= 1):
        raise ValueError('Invalid seed, repetitions, or quadrat size')
    fixture = artificial_patterns(seed=seed)
    roi, patterns = fixture['roi'], fixture['patterns']
    mask = artificial_component_mask(roi)
    objects = component_table(mask, roi)
    results = {}
    for name, points in patterns.items():
        results[name] = {
            'nearest_neighbor': nearest_neighbor_summary(points, roi),
            'quadrats': quadrat_summary(points, roi, cell_size),
            'csr_reference': csr_monte_carlo(points, roi, repetitions, seed + {'random': 0, 'regular': 1, 'clustered': 2}[name]),
            'centers_rc_px': points.tolist(),
        }
    report = {
        'session': 20,
        'seed': seed,
        'roi_area_px2': int(roi.sum()),
        'excluded_roi_px2': int(roi.size - roi.sum()),
        'coordinate_system': '(row,column), image-plane pixels, no physical calibration',
        'classes': 'Artificial 0/1 regions only; NOT ferrite/pearlite or grain identities',
        'component_measurements': objects,
        'patterns': results,
        'independence': 'Three toy point patterns are generated by construction. They are not independent physical specimens.',
        'methodology': '8-connected masks; conservative boundary flags; centroid NN by KDTree; complete equal-area quadrats; Monte Carlo uniform unique pixel positions conditional on N and same ROI.',
        'interpretation': 'Descriptor demonstrations only. Do not infer physical particle distribution, grain size, or industrial clustering from this synthetic exercise. Official Figshare test partition is not accessed.',
    }
    path.mkdir(parents=True, exist_ok=False)
    with (path / 'session20_components.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['component_id', 'area_px2', 'centroid_row_px', 'centroid_col_px', 'ecd_px', 'boundary_censored'])
        writer.writeheader()
        writer.writerows(objects['objects'])
    with (path / 'session20_patterns.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['pattern', 'n_centers', 'mean_nn_px', 'edge_risk_count', 'dispersion_index', 'csr_05_px', 'csr_95_px'])
        for name, result in results.items():
            nn = result['nearest_neighbor']; quad = result['quadrats']; csr = result['csr_reference']
            writer.writerow([name, nn['count'], nn['mean_nearest_px'], nn['edge_risk_count'], quad['dispersion_index'], *csr['null_05_95_px']])
    (path / 'session20_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    try:
        for ax, (name, points) in zip(axes[0], patterns.items()):
            ax.imshow(roi, cmap='gray', vmin=0, vmax=1, interpolation='nearest', alpha=.30)
            ax.scatter(points[:, 1], points[:, 0], s=26, edgecolors='black', linewidth=.4)
            ax.set(xlim=(-.5, 127.5), ylim=(127.5, -.5), title=f'{name.capitalize()}: {len(points)} synthetic centers', xlabel='Column (px)', ylabel='Row (px)')
        axes[1, 0].imshow(np.where(roi, mask, np.nan), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        for obj in objects['objects']:
            axes[1, 0].text(obj['centroid_col_px'], obj['centroid_row_px'], 'C' if obj['boundary_censored'] else 'I', ha='center', va='center', fontsize=10, bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
        axes[1, 0].set(title='Binary components: C=censored, I=interior', xlabel='Column (px)', ylabel='Row (px)')
        names = list(results)
        means = [results[k]['nearest_neighbor']['mean_nearest_px'] for k in names]
        axes[1, 1].bar(names, means)
        for i, name in enumerate(names):
            low, high = results[name]['csr_reference']['null_05_95_px']
            axes[1, 1].plot([i, i], [low, high], 'k-', lw=3, alpha=.65)
        axes[1, 1].set(title='Mean centroid NN (bars); CSR 5–95% interval (lines)', ylabel='Distance (pixels)')
        disp = [results[k]['quadrats']['dispersion_index'] for k in names]
        axes[1, 2].bar(names, disp)
        axes[1, 2].axhline(1, color='black', linestyle='--', linewidth=1)
        axes[1, 2].set(title='Quadrat variance / mean (descriptive)', ylabel='Dispersion index')
        fig.suptitle('Session 20 — Synthetic spatial measurements; no physical specimen inference', fontsize=12)
        fig.savefig(path / 'session20_spatial_patterns.png', dpi=135)
    finally:
        plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/session20_v1'))
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--repetitions', type=int, default=300)
    parser.add_argument('--cell-size', type=int, default=16)
    args = parser.parse_args()
    try:
        result = run(args.output_dir, args.seed, args.repetitions, args.cell_size)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')
    for name, desc in result['patterns'].items():
        print(name, 'mean NN:', round(desc['nearest_neighbor']['mean_nearest_px'], 3),
              'dispersion:', round(desc['quadrats']['dispersion_index'], 3),
              'CSR 5–95:', [round(x, 3) for x in desc['csr_reference']['null_05_95_px']])
    print('components:', result['component_measurements']['component_count'],
          'complete:', result['component_measurements']['complete_count'],
          'censored:', result['component_measurements']['censored_count'])
