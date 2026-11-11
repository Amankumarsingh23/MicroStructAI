"""Session 27 frozen artificial-development benchmark protocol."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from session26_provenance_audit import generate_fixture, audit, preflight, binary_from_schema, file_hash
from session14_evaluation_lab import scene_metrics, boundary_metrics, class_metrics, group_bootstrap

METHODS = ('fixed_dark_120', 'fixed_dark_165', 'all_ferrite')


def make_protocol(seed=27, repetitions=300):
    if type(seed) is not int or seed < 0 or type(repetitions) is not int or repetitions < 1:
        raise ValueError('Nonnegative seed and positive repetitions required')
    return {'version': 1, 'fixture': 'Session 26 artificial development generator',
            'seed': seed, 'repetitions': repetitions, 'partitions': ['train', 'val'],
            'methods': list(METHODS), 'prediction': 'Pillow L grayscale; foreground intensity strictly below fixed threshold',
            'original_labels': '0..8 pearlite orientation codes; 9 ferrite',
            'derived_labels': '0 ferrite; 1 pearlite-coded foreground',
            'metrics': ['iou', 'dice', 'precision', 'recall', 'pixel_accuracy', 'boundary_f1', 'signed_area_error_pp'],
            'boundary': 'Foreground inner boundary from 3x3 erosion, exterior background; Euclidean tolerance 1 pixel; safe ROI excludes invalid-neighbor boundary pixels',
            'boundary_tolerance_px': 1, 'empty_class': 'Both-absent IoU/Dice null, exclude from macro; one-empty overlap zero',
            'empty_boundary': 'Both-empty F1 null; one-empty F1 zero',
            'aggregation': 'Pooled confusion counts and equal-image macro foreground IoU; equal-image boundary F1 and area error',
            'bootstrap': 'Resample declared source groups within partition; descriptive scene-macro IoU percentile span',
            'selection': 'Fixed methods; no fitting, threshold tuning or checkpoint selection',
            'official_test_policy': 'never_access', 'real_data_status': 'NO-GO; verified lineage and physical validation unresolved'}


def protocol_hash(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def seal_protocol(protocol):
    return {'protocol': protocol, 'sha256': protocol_hash(protocol)}


def verify_protocol(sealed):
    if sealed['sha256'] != protocol_hash(sealed['protocol']):
        raise ValueError('Protocol hash mismatch')
    protocol = sealed['protocol']
    if protocol != make_protocol(protocol['seed'], protocol['repetitions']):
        raise ValueError('Unsupported benchmark specification')
    return protocol


def predict(gray, method):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('Expected uint8 grayscale image')
    if method == 'all_ferrite':
        return np.zeros_like(gray)
    if method not in METHODS:
        raise ValueError('Unknown method')
    threshold = 120 if method == 'fixed_dark_120' else 165
    return (gray < threshold).astype(np.uint8)


def pair_metrics(truth, prediction):
    roi = np.ones(truth.shape, bool)
    metrics = scene_metrics(truth, prediction, roi)
    metrics.update(boundary_metrics(truth, prediction, roi, tolerance=1))
    return metrics


def aggregate(rows):
    if not rows:
        raise ValueError('Nonempty metric rows required')
    counts = {key: sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn', 'tn')}
    matrix = np.array([[counts['tn'], counts['fp']], [counts['fn'], counts['tp']]], dtype=np.int64)
    classes = class_metrics(matrix)
    def mean_defined(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    return {**counts, 'pairs': len(rows),
            **{'pooled_foreground_'+k: v for k,v in classes['per_class']['1'].items() if k not in ('tp','fp','fn')},
            'pooled_pixel_accuracy': classes['pixel_accuracy'],
            'scene_macro_foreground_iou': mean_defined('iou'),
            'scene_macro_boundary_f1': mean_defined('boundary_f1'),
            'mean_signed_area_error_pp': mean_defined('signed_area_error_pp')}


def evaluate_manifest(rows, root, sealed):
    protocol = verify_protocol(sealed)
    checked = preflight(rows, root)
    audit_report = audit(rows, root)
    if audit_report['blockers']:
        raise ValueError('Blocking audit findings prevent benchmark evaluation')
    metrics = []
    for row, image, mask in checked:
        with Image.open(image) as opened:
            gray = np.array(opened.convert('L'))
        with Image.open(mask) as opened:
            truth = binary_from_schema(np.array(opened), row['schema'])
        for method in protocol['methods']:
            metrics.append({'pair_id': row['pair_id'], 'split': row['split'], 'method': method,
                            'group_id': row['source_id'], **pair_metrics(truth, predict(gray, method))})
    return metrics, audit_report


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, seed=27, repetitions=300):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    sealed = seal_protocol(make_protocol(seed, repetitions))
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = output/'session27_frozen_protocol.json'
    protocol_path.write_text(json.dumps(sealed, indent=2)+'\n', encoding='utf-8')
    root = output/'artificial_development'
    rows = generate_fixture(root, seed)
    # Verify the persisted seal immediately before auditing and evaluating images.
    persisted = json.loads(protocol_path.read_text(encoding='utf-8'))
    metrics, audit_report = evaluate_manifest(rows, root, persisted)
    summaries, bootstraps = [], []
    for split in ('train', 'val'):
        for method in METHODS:
            selected = [r for r in metrics if r['split'] == split and r['method'] == method]
            summaries.append({'split': split, 'method': method, **aggregate(selected)})
            boot = group_bootstrap(selected, repetitions=repetitions, seed=seed)
            bootstraps.append({'split': split, 'method': method, 'group_count': boot['group_count'],
                               'repetitions': repetitions, 'defined_replicates': boot['defined_replicates'],
                               'p025': boot['scene_macro_iou_percentile_95'][0],
                               'p975': boot['scene_macro_iou_percentile_95'][1],
                               'scope': 'Descriptive source resampling of generated fixtures; few groups'})
    report = {'implementation': 'Reconstructed from Session 27 handout; source package unavailable locally',
              'protocol_sha256': persisted['sha256'], 'manifest_sha256': file_hash(root/'development_manifest.csv'),
              'source_sha256': file_hash(Path(__file__)),
              'dependencies_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
                                      ('session26_provenance_audit.py', 'session14_evaluation_lab.py')},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': Image.__version__, 'matplotlib': matplotlib.__version__},
              'audit_status': audit_report['status'], 'audit_warnings': audit_report['warnings'],
              'input_pairs': audit_report['pairs'], 'summaries': summaries, 'bootstrap': bootstraps,
              'scientific_status': 'NO-GO for real-data benchmarking; source independence and physical validity unresolved',
              'limits': ['Runtime sealing detects changes and does not establish independent preregistration',
                         'Two declared artificial validation sources cannot support material-population inference',
                         'Fixed threshold scores describe this generated fixture only',
                         'No trained architecture comparison or real-data evaluation performed']}
    write_csv(output/'session27_pair_metrics.csv', metrics)
    write_csv(output/'session27_summary.csv', summaries)
    write_csv(output/'session27_group_bootstrap.csv', bootstraps)
    (output/'session27_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    try:
        for axis, key, title in zip(axes, ('pooled_foreground_iou', 'scene_macro_boundary_f1', 'mean_signed_area_error_pp'),
                                     ('Pooled foreground IoU', 'Scene-macro boundary F1', 'Signed area error (pp)')):
            x = np.arange(len(METHODS))
            for offset, split in ((-.18, 'train'), (.18, 'val')):
                values = [next(r[key] for r in summaries if r['split']==split and r['method']==method) for method in METHODS]
                axis.bar(x+offset, values, width=.36, label=split)
            axis.set_xticks(x, ['Dark 120', 'Dark 165', 'All ferrite'], rotation=15)
            axis.set_title(title)
            axis.legend()
        fig.suptitle('Session 27 frozen artificial benchmark')
        fig.savefig(output/'session27_comparison.png', dpi=135)
    finally:
        plt.close(fig)
    print('Protocol', persisted['sha256'])
    for summary in summaries:
        print(summary['split'], summary['method'], 'pooled IoU', summary['pooled_foreground_iou'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/session27_v1'))
    parser.add_argument('--seed', type=int, default=27)
    parser.add_argument('--repetitions', type=int, default=300)
    args = parser.parse_args()
    run(args.output_dir, args.seed, args.repetitions)
