"""Session 26 development-only provenance audit and artificial fixtures."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLUMNS = ('pair_id', 'split', 'image_rel', 'mask_rel', 'schema',
           'texture_id', 'template_id', 'source_id', 'calibration_um_per_px')


class ProtectedDataError(ValueError):
    """Manifest or path is outside the permitted development scope."""


def protected(path):
    return any(re.search(r'(^|[^a-z])(test|testing|holdout)([^a-z]|$)', part.lower())
               for part in Path(path).parts)


def safe_development_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    windows = PureWindowsPath(relative)
    if (not relative or rel.is_absolute() or windows.is_absolute() or windows.drive
            or '..' in rel.parts or '..' in windows.parts or protected(rel) or protected(root)):
        raise ProtectedDataError('Only relative development paths are permitted')
    target = (root/rel).resolve()
    if not target.is_relative_to(root) or protected(target):
        raise ProtectedDataError('Resolved path escapes development root or names protected data')
    if not target.is_file():
        raise ValueError(f'Missing development file {relative}')
    return target


def preflight(rows, root):
    """Validate the whole manifest before any image content is accessed."""
    if not rows:
        raise ValueError('Manifest is empty')
    checked, ids, paths = [], set(), set()
    for row in rows:
        if not set(COLUMNS) <= set(row):
            raise ValueError('Missing manifest columns')
        if row['split'] not in ('train', 'val'):
            raise ProtectedDataError('Only train and val partitions permitted')
        if not row['pair_id'] or row['pair_id'] in ids:
            raise ValueError('Pair IDs must be nonempty and unique')
        ids.add(row['pair_id'])
        if row['schema'] not in ('original10', 'derived_binary'):
            raise ValueError('Unknown mask schema')
        scale = row['calibration_um_per_px']
        if scale not in ('', None):
            if not np.isfinite(float(scale)) or float(scale) <= 0:
                raise ValueError('Calibration must be positive finite or absent')
        image = safe_development_path(root, row['image_rel'])
        mask = safe_development_path(root, row['mask_rel'])
        if image == mask or image in paths or mask in paths:
            raise ValueError('Manifest reuses an image/mask path')
        paths.update((image, mask))
        checked.append((row, image, mask))
    return checked


def binary_from_schema(mask, schema):
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or not mask.size:
        raise ValueError('Mask must be a nonempty 2D integer array')
    if schema == 'original10':
        if np.any(mask < 0) or np.any(mask > 9):
            raise ValueError('Original labels must be IDs 0..9')
        return (mask != 9).astype(np.uint8)
    if schema == 'derived_binary' and np.isin(mask, (0, 1)).all():
        return mask.astype(np.uint8, copy=True)
    raise ValueError('Invalid derived labels or unknown schema')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_hash(array):
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def audit(rows, root):
    checked = preflight(rows, root)
    measurements, blockers, warnings = [], [], []
    for row, image_path, mask_path in checked:
        try:
            with Image.open(image_path) as opened:
                image = np.array(opened.convert('RGB'))
            with Image.open(mask_path) as opened:
                mask = np.array(opened)
            binary = binary_from_schema(mask, row['schema'])
            if image.shape[:2] != mask.shape:
                raise ValueError('Image/mask dimensions differ')
            if row['calibration_um_per_px'] in ('', None):
                warnings.append(f"{row['pair_id']}: physical calibration unavailable")
            for key in ('source_id', 'texture_id', 'template_id'):
                if not row[key] or row[key].strip().lower() in ('unknown', 'unverified'):
                    warnings.append(f"{row['pair_id']}: {key} unknown")
            measurements.append({**row, 'height': mask.shape[0], 'width': mask.shape[1],
                                 'labels': np.unique(mask).tolist(),
                                 'image_sha256': file_hash(image_path),
                                 'mask_sha256': file_hash(mask_path),
                                 'image_pixel_sha256': pixel_hash(image),
                                 'mask_pixel_sha256': pixel_hash(mask),
                                 'mean_intensity': float(image.mean()),
                                 'foreground_code_fraction': float(binary.mean())})
        except (ValueError, OSError) as error:
            blockers.append(f"{row['pair_id']}: {error}")
    for key in ('source_id', 'texture_id', 'template_id', 'image_sha256', 'image_pixel_sha256'):
        grouped = {}
        for row in measurements:
            value = row[key]
            if value and str(value).strip().lower() not in ('unknown', 'unverified'):
                grouped.setdefault(value, set()).add(row['split'])
        for value, splits in grouped.items():
            if len(splits) > 1:
                blockers.append(f'Cross-partition overlap for {key}: {value}')
    summaries = {}
    for split in ('train', 'val'):
        subset = [r for r in measurements if r['split'] == split]
        if subset:
            summaries[split] = {'pairs': len(subset),
                                'mean_intensity': float(np.mean([r['mean_intensity'] for r in subset])),
                                'mean_foreground_code_fraction': float(np.mean([r['foreground_code_fraction'] for r in subset]))}
        else:
            blockers.append(f'No valid {split} pairs')
    return {'status': 'manifest_consistent' if not blockers else 'blocked',
            'pairs': measurements, 'blockers': blockers, 'warnings': warnings,
            'split_summaries': summaries,
            'source_independence': 'unverified; declared identifiers cannot certify actual lineage',
            'limits': ['Hash equality detects exact duplicates only',
                       'RGB-decoded hashes compare normalized image pixels; mask hashes retain categorical arrays',
                       'Intensity differences are descriptive and do not establish model failure',
                       'Authorized root and declared path metadata cannot identify disguised protected copies',
                       'Artificial schema fixtures establish code behavior, not physical validity']}


def generate_fixture(root, seed=26):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rr, cc = np.indices((64, 64))
    rows = []
    for i in range(6):
        split = 'train' if i < 4 else 'val'
        mask = np.full((64, 64), 9, np.uint8)
        center = rng.integers(18, 46, size=2)
        radius = int(rng.integers(9, 14))
        circle = (rr-center[0])**2+(cc-center[1])**2 <= radius**2
        mask[circle] = (cc[circle] % 2).astype(np.uint8)
        base = 200 if split == 'train' else 150
        gray = np.clip(np.rint(np.where(mask == 9, base, base-110)+rng.normal(0, 5, mask.shape)), 0, 255).astype(np.uint8)
        image_name, mask_name = f'image_{i}.png', f'mask_{i}.png'
        Image.fromarray(gray).save(root/image_name)
        Image.fromarray(mask).save(root/mask_name)
        rows.append(dict(zip(COLUMNS, (f'pair_{i}', split, image_name, mask_name,
                                      'original10', f'artificial_texture_{i}',
                                      f'artificial_template_{i}', f'artificial_source_{i}', ''))))
    with (root/'development_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(output, seed=26, data_root=None, manifest=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    if (data_root is None) != (manifest is None):
        raise ValueError('Supply both data root and manifest')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    if data_root is None:
        output.mkdir(parents=True, exist_ok=False)
        data_root = output/'artificial_development'
        rows = generate_fixture(data_root, seed)
        fixture = True
    else:
        with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        preflight(rows, data_root)
        output.mkdir(parents=True, exist_ok=False)
        fixture = False
    report = audit(rows, data_root)
    report.update({'seed': seed, 'artificial_fixture': fixture,
                   'implementation': 'Reconstructed from Session 26 handout; source package unavailable locally',
                   'manifest_sha256': file_hash(Path(data_root)/'development_manifest.csv' if fixture else Path(manifest)),
                   'source_sha256': file_hash(Path(__file__))})
    (output/'session26_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    try:
        for axis, key, title in zip(axes, ('mean_intensity', 'mean_foreground_code_fraction'),
                                     ('Mean image intensity', 'Mean foreground-code fraction')):
            names = list(report['split_summaries'])
            axis.bar(names, [report['split_summaries'][n][key] for n in names])
            axis.set_title(title)
        fig.suptitle('Session 26 development audit diagnostics')
        fig.savefig(output/'session26_domain_diagnostics.png', dpi=135)
    finally:
        plt.close(fig)
    print(report['status'], 'pairs', len(report['pairs']), 'blockers', len(report['blockers']), 'warnings', len(report['warnings']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/session26_v1'))
    parser.add_argument('--seed', type=int, default=26)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    report = run(args.output_dir, args.seed, args.data_root, args.manifest)
    raise SystemExit(1 if report['blockers'] else 0)
