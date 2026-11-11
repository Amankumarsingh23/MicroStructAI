"""Session 25 synthetic regression reliability and split conformal calibration."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


def cohort(seed=25, per_batch=30):
    if type(seed) is not int or seed < 0 or type(per_batch) is not int or per_batch < 20:
        raise ValueError('Need nonnegative seed and at least 20 specimens per batch')
    rng = np.random.default_rng(seed)
    records = []
    for batch in range(3):
        for index in range(per_batch):
            latent = float(np.clip(.18 + .055*batch + rng.normal(0, .04), .06, .6))
            target = float(103 + 130*latent + 9*batch + rng.normal(0, 4))
            total = positive = 0
            for field in range(3):
                roi = np.ones((64, 64), bool)
                roi[:8, 49:] = False
                if field:
                    roi[-field*3:, :] = False
                texture = ndi.gaussian_filter(rng.normal(size=roi.shape), 1.75)
                fraction = float(np.clip(latent+rng.normal(0, .025), .03, .7))
                threshold = np.quantile(texture[roi], 1-fraction)
                positive += int(np.count_nonzero((texture >= threshold) & roi))
                total += int(roi.sum())
            records.append({'specimen_id': f'B{batch+1}_S{index+1:03d}',
                            'batch_id': f'B{batch+1}', 'n_fields': 3,
                            'valid_pixels': total, 'foreground_pixels': positive,
                            'area_fraction': positive/total, 'response_au': target,
                            'provenance': 'generated masks and simulated response',
                            'calibration_um_per_px': None})
    return records


def split_cohort(rows, seed=25):
    rng = np.random.default_rng(seed)
    fitting, calibration, evaluation = [], [], []
    for batch in sorted({r['batch_id'] for r in rows}):
        group = [r for r in rows if r['batch_id'] == batch]
        ordered = [group[i] for i in rng.permutation(len(group))]
        nfit, ncal = len(group)//2, int(round(len(group)*8/30))
        fitting.extend(ordered[:nfit])
        calibration.extend(ordered[nfit:nfit+ncal])
        evaluation.extend(ordered[nfit+ncal:])
    validate_partitions(fitting, calibration, evaluation)
    return fitting, calibration, evaluation


def validate_partitions(*parts):
    ids = [r['specimen_id'] for part in parts for r in part]
    if any(not part for part in parts) or len(ids) != len(set(ids)):
        raise ValueError('Partitions need nonempty disjoint unique specimen IDs')


def fit_area_only(rows):
    x = np.array([r['area_fraction'] for r in rows], float)
    y = np.array([r['response_au'] for r in rows], float)
    if not rows or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite fitting observations required')
    mean, scale = float(x.mean()), float(x.std())
    scale = scale if scale > 1e-12 else 1.0
    z = (x-mean)/scale
    weight = float(np.linalg.lstsq(z[:, None], y-y.mean(), rcond=None)[0][0])
    return {'mean': mean, 'std': scale, 'target_mean': float(y.mean()),
            'weight': weight, 'training_ids': [r['specimen_id'] for r in rows]}


def predict_area_only(model, rows):
    x = np.array([r['area_fraction'] for r in rows], float)
    if not np.isfinite(x).all():
        raise ValueError('Finite fractions required')
    return model['target_mean']+model['weight']*(x-model['mean'])/model['std']


def conformal_radius(actual, predicted, alpha=.1):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if (a.ndim != 1 or p.shape != a.shape or not a.size or not np.isfinite(a).all()
            or not np.isfinite(p).all() or isinstance(alpha, bool)
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('Matching finite vectors and alpha in (0,1) required')
    ordered = np.sort(abs(a-p))
    rank = math.ceil((len(a)+1)*(1-alpha))
    if rank > len(a):
        raise ValueError('Insufficient calibration specimens for a finite radius')
    return {'radius_au': float(ordered[rank-1]), 'rank': rank, 'n_calibration': len(a),
            'alpha': float(alpha), 'ordered_absolute_residuals': ordered.tolist()}


def diagnostics(rows):
    residual = np.array([r['residual_au'] for r in rows])
    if not len(residual):
        raise ValueError('Nonempty diagnostic rows required')
    return {'n': len(rows), 'mae_au': float(np.mean(abs(residual))),
            'rmse_au': float(np.sqrt(np.mean(residual**2))),
            'bias_actual_minus_prediction_au': float(residual.mean()),
            'covered': sum(r['covered'] for r in rows),
            'coverage': float(np.mean([r['covered'] for r in rows])),
            'mean_width_au': float(np.mean([r['upper_au']-r['lower_au'] for r in rows]))}


def evaluate(model, rows, radius, protocol):
    records = []
    for row, predicted in zip(rows, predict_area_only(model, rows)):
        actual = row['response_au']
        lower, upper = float(predicted-radius), float(predicted+radius)
        records.append({'protocol': protocol, 'specimen_id': row['specimen_id'],
                        'batch_id': row['batch_id'], 'actual_au': actual,
                        'predicted_au': float(predicted), 'lower_au': lower, 'upper_au': upper,
                        'residual_au': float(actual-predicted), 'covered': bool(lower <= actual <= upper)})
    return records


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, seed=25, alpha=.1, per_batch=30):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    rows = cohort(seed, per_batch)
    fitting, calibration, evaluation = split_cohort(rows, seed)
    model = fit_area_only(fitting)
    calibrated = conformal_radius([r['response_au'] for r in calibration], predict_area_only(model, calibration), alpha)
    training_radius = conformal_radius([r['response_au'] for r in fitting], predict_area_only(model, fitting), alpha)
    proper = evaluate(model, evaluation, calibrated['radius_au'], 'independent_calibration')
    invalid = evaluate(model, evaluation, training_radius['radius_au'], 'training_residuals_invalid')
    shift_fit = [r for r in fitting if r['batch_id'] != 'B3']
    shift_cal = [r for r in calibration+evaluation if r['batch_id'] != 'B3']
    shift_eval = [r for r in rows if r['batch_id'] == 'B3']
    validate_partitions(shift_fit, shift_cal, shift_eval)
    shift_model = fit_area_only(shift_fit)
    shift_radius = conformal_radius([r['response_au'] for r in shift_cal], predict_area_only(shift_model, shift_cal), alpha)
    shifted = evaluate(shift_model, shift_eval, shift_radius['radius_au'], 'unseen_B3')
    all_predictions = proper+invalid+shifted
    batches = []
    for protocol in ('independent_calibration', 'training_residuals_invalid', 'unseen_B3'):
        for batch in sorted({r['batch_id'] for r in all_predictions if r['protocol'] == protocol}):
            group = [r for r in all_predictions if r['protocol'] == protocol and r['batch_id'] == batch]
            batches.append({'protocol': protocol, 'batch_id': batch, **diagnostics(group)})
    ordered = sorted(proper, key=lambda r: r['predicted_au'])
    bins = [{'bin': i, 'prediction_min_au': group[0]['predicted_au'],
             'prediction_max_au': group[-1]['predicted_au'], **diagnostics(group)}
            for i, indices in enumerate(np.array_split(np.arange(len(ordered)), 3))
            if (group := [ordered[int(j)] for j in indices])]
    report = {'scope': 'Reconstructed Session 25 artificial cohort; simulated arbitrary-unit targets',
              'seed': seed, 'per_batch': per_batch, 'nominal_coverage': 1-alpha,
              'partition_ids': {name: [r['specimen_id'] for r in part] for name, part in
                                [('fit', fitting), ('calibration', calibration), ('evaluation', evaluation)]},
              'model': model, 'calibration': calibrated,
              'proper': diagnostics(proper), 'invalid_training_calibration': diagnostics(invalid),
              'training_residual_radius': training_radius,
              'shift': {'fit_ids': [r['specimen_id'] for r in shift_fit],
                        'calibration_ids': [r['specimen_id'] for r in shift_cal],
                        'evaluation_ids': [r['specimen_id'] for r in shift_eval],
                        'model': shift_model, 'calibration': shift_radius, **diagnostics(shifted)},
              'limits': ['Generator reconstructed because source package unavailable locally',
                         'Finite-sample marginal conformal coverage requires exchangeability',
                         'Batch-stratified fixed composition is an illustrative design, not an iid guarantee',
                         'Subgroup coverage and unseen-batch coverage require separate assessment',
                         'Simulated targets and masks provide no physical hardness validation'],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in [('synthetic_specimens', rows), ('diagnostic_predictions', all_predictions),
                          ('batch_diagnostics', batches), ('residual_bins', bins)]:
        write_csv(output/f'session25_{name}.csv', records)
    (output/'session25_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    try:
        axes[0, 0].errorbar(range(len(proper)), [r['predicted_au'] for r in proper], yerr=calibrated['radius_au'], fmt='o', label='Prediction and interval')
        axes[0, 0].scatter(range(len(proper)), [r['actual_au'] for r in proper], marker='x', color='red', label='Simulated response')
        axes[0, 0].set(title='Independent calibration', xlabel='Evaluation specimen index', ylabel='Response (a.u.)')
        axes[0, 0].legend()
        ordinary = [r for r in batches if r['protocol'] == 'independent_calibration']
        axes[0, 1].bar([r['batch_id'] for r in ordinary], [r['mae_au'] for r in ordinary])
        axes[0, 1].set(title='Batch residual MAE', ylabel='Absolute error (a.u.)')
        axes[1, 0].bar([r['batch_id'] for r in ordinary], [r['coverage'] for r in ordinary])
        axes[1, 0].axhline(1-alpha, color='red', linestyle='--')
        axes[1, 0].set(title='Observed subgroup coverage', ylim=(0, 1.05))
        axes[1, 1].plot(range(1, calibrated['n_calibration']+1), calibrated['ordered_absolute_residuals'], 'o-')
        axes[1, 1].axvline(calibrated['rank'], color='red', linestyle='--')
        axes[1, 1].set(title='Finite-sample calibration cutoff', xlabel='Ordered residual rank', ylabel='Absolute residual (a.u.)')
        fig.suptitle('Session 25 synthetic prediction interval diagnostics')
        fig.savefig(output/'session25_calibration_reliability.png', dpi=135)
    finally:
        plt.close(fig)
    print('Proper calibration', report['proper'])
    print('Unseen batch', diagnostics(shifted))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/session25_v1'))
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--per-batch', type=int, default=30)
    args = parser.parse_args()
    run(args.output_dir, args.seed, args.alpha, args.per_batch)
