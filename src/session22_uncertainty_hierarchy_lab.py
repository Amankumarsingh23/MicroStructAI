"""Session 22: measurement uncertainty, hierarchical resampling and batch shift.

Uses the synthetic Session 21 tables. No genuine microscopy, real material-property
measurements, official Figshare test access or physically calibrated parameters.
Count-level binomial label errors demonstrate area-fraction uncertainty ONLY:
no spatial boundary or instance error is simulated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from session21_feature_property_lab import (
    synthetic_tables, aggregate_specimens, specimen_split,
    fit_specimen_ridge, predict_specimen_ridge,
)

SCENARIOS = {
    'mild': {'fpr': 0.02, 'fnr': 0.06},
    'missed_foreground': {'fpr': 0.01, 'fnr': 0.20},
    'false_positive': {'fpr': 0.12, 'fnr': 0.04},
}


def _prob(value, label):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{label} must be a finite probability in [0,1]')
    return float(value)


def expected_fraction_bias(true_fraction: float, fpr: float, fnr: float) -> float:
    """E[measured_fraction - true_fraction] under independent pixel flips."""
    f = _prob(true_fraction, 'fraction')
    a = _prob(fpr, 'FPR')
    b = _prob(fnr, 'FNR')
    return a * (1.0 - f) - b * f


def perturb_counts(valid: int, foreground: int, fpr: float, fnr: float,
                   rng: np.random.Generator) -> dict:
    """Independent FN/FP binomial draws from known artificial ROI label counts."""
    if (isinstance(valid, (bool, np.bool_)) or not isinstance(valid, (int, np.integer))
            or valid <= 0 or isinstance(foreground, (bool, np.bool_))
            or not isinstance(foreground, (int, np.integer)) or not 0 <= foreground <= valid):
        raise ValueError('Require positive integer ROI and valid integer foreground count')
    a, b = _prob(fpr, 'FPR'), _prob(fnr, 'FNR')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('Pass a NumPy random Generator')
    fn = int(rng.binomial(foreground, b))
    fp = int(rng.binomial(valid - foreground, a))
    measured = foreground - fn + fp
    return {'valid_pixels': int(valid), 'true_positive_pixels': int(foreground),
            'fp': fp, 'fn': fn, 'measured_positive_pixels': measured,
            'true_fraction': foreground / valid, 'measured_fraction': measured / valid,
            'signed_error_fraction': (fp - fn) / valid}


def noisy_specimen_features(fields: list[dict], fpr: float, fnr: float, seed: int):
    """Perturb each known field count, then pool counts within each specimen."""
    if not fields or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('Require fields and nonnegative integer seed')
    rng = np.random.default_rng(seed)
    field_rows, groups = [], {}
    for field in fields:
        result = perturb_counts(field['roi_pixels'], field['foreground_pixels'], fpr, fnr, rng)
        row = {'field_id': field['field_id'], 'specimen_id': field['specimen_id'],
               'batch_id': field['batch_id'], **result}
        field_rows.append(row)
        groups.setdefault(field['specimen_id'], []).append(row)
    specimen_rows = []
    for sid, records in sorted(groups.items()):
        batch_ids = {r['batch_id'] for r in records}
        if len(batch_ids) != 1:
            raise ValueError('Specimen records must agree on batch')
        valid = sum(r['valid_pixels'] for r in records)
        actual = sum(r['true_positive_pixels'] for r in records)
        predicted = sum(r['measured_positive_pixels'] for r in records)
        fp = sum(r['fp'] for r in records)
        fn = sum(r['fn'] for r in records)
        if predicted != actual + fp - fn:
            raise AssertionError('Confusion-count conservation failed')
        specimen_rows.append({'specimen_id': sid, 'batch_id': next(iter(batch_ids)),
                              'n_fields': len(records), 'roi_pixels': valid,
                              'true_positive_pixels': actual,
                              'measured_positive_pixels': predicted, 'fp': fp, 'fn': fn,
                              'true_fraction': actual / valid,
                              'measured_fraction': predicted / valid,
                              'signed_error_fraction': (fp - fn) / valid})
    return field_rows, specimen_rows


def fit_fraction_ridge(training: list[dict], alpha: float = 2.0) -> dict:
    """Train-only standardization, one-feature ridge, unpenalized intercept."""
    if not training or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Require training specimens and finite alpha >= 0')
    x = np.asarray([r['roi_weighted_area_fraction'] for r in training], dtype=float)
    y = np.asarray([r['synthetic_response_au'] for r in training], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((x < 0) | (x > 1)):
        raise ValueError('Invalid training fraction/target')
    mu = float(x.mean())
    scale = float(x.std())
    if scale < 1e-12:
        scale = 1.0
    z = (x - mu) / scale
    denom = float(z @ z + alpha)
    weight = float(z @ (y - y.mean()) / denom) if denom > 0 else 0.0
    return {'training_mean': mu, 'training_std': scale,
            'response_mean': float(y.mean()), 'weight': weight,
            'alpha': float(alpha),
            'training_specimen_ids': [r['specimen_id'] for r in training]}


def predict_fraction_ridge(fit: dict, fractions) -> np.ndarray:
    x = np.asarray(fractions, dtype=float)
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError('Prediction fractions must be finite and in [0,1]')
    return fit['response_mean'] + fit['weight'] * (x - fit['training_mean']) / fit['training_std']


def leave_batch_out(specimens: list[dict], batch_id: str = 'B3'):
    train = [r for r in specimens if r['batch_id'] != batch_id]
    val = [r for r in specimens if r['batch_id'] == batch_id]
    if not train or not val or set(r['specimen_id'] for r in train) & set(r['specimen_id'] for r in val):
        raise ValueError('Leave-one-batch-out requires disjoint nonempty specimens')
    if set(r['batch_id'] for r in train) & set(r['batch_id'] for r in val):
        raise AssertionError('Batch identity crossed leave-batch-out split')
    return train, val


def error_metrics(actual, predicted) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.ndim != 1 or p.shape != a.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Expected nonempty matching 1D finite actual/prediction arrays')
    e = p-a
    return {'mae_au': float(np.mean(abs(e))), 'rmse_au': float(np.sqrt(np.mean(e**2))),
            'mean_signed_error_au': float(np.mean(e)), 'n_specimens': int(len(a))}


def evaluate_split(training, validation) -> dict:
    if set(r['specimen_id'] for r in training) & set(r['specimen_id'] for r in validation):
        raise ValueError('Specimen overlap between training and validation')
    fit_one = fit_fraction_ridge(training, 2.0)
    fit_three = fit_specimen_ridge(training, alpha=2.0)
    actual = np.asarray([r['synthetic_response_au'] for r in validation], dtype=float)
    base = np.full_like(actual, float(np.mean([r['synthetic_response_au'] for r in training])))
    area = predict_fraction_ridge(fit_one, [r['roi_weighted_area_fraction'] for r in validation])
    three = predict_specimen_ridge(fit_three, validation)
    methods = {'train_mean': base, 'area_only_ridge': area, 'three_feature_ridge': three}
    results = {k: error_metrics(actual, v) for k, v in methods.items()}
    records = [{'specimen_id': r['specimen_id'], 'batch_id': r['batch_id'],
                'synthetic_response_au': float(y),
                **{f'pred_{name}_au': float(pred[i]) for name, pred in methods.items()}}
               for i, (r, y) in enumerate(zip(validation, actual))]
    return {'n_train': len(training), 'n_validation': len(validation),
            'train_specimens': [r['specimen_id'] for r in training],
            'validation_specimens': [r['specimen_id'] for r in validation],
            'metrics': results, 'one_feature_model': fit_one, 'predictions': records}


def pooled_and_within_batch_correlation(specimens: list[dict]) -> dict:
    """Within-batch correlation of demeaned variables is descriptive, not causal."""
    x = np.array([r['roi_weighted_area_fraction'] for r in specimens], float)
    y = np.array([r['synthetic_response_au'] for r in specimens], float)
    batches = [r['batch_id'] for r in specimens]
    xd, yd = x.copy(), y.copy()
    for b in sorted(set(batches)):
        ids = np.array([i for i, label in enumerate(batches) if label == b])
        xd[ids] -= x[ids].mean()
        yd[ids] -= y[ids].mean()
    def corr(a, b):
        if min(float(a.std()), float(b.std())) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    return {'pooled_pearson': corr(x, y), 'within_batch_demeaned_pearson': corr(xd, yd),
            'warning': 'Correlations on simulated specimens; neither statistic establishes a causal material relationship.'}


def bootstrap_grouped_mean(specimens: list[dict], group_key: str,
                           repetitions=1000, seed=22) -> dict:
    """Resample independent units/groups, never individual image pixels."""
    if group_key not in ('specimen_id', 'batch_id') or repetitions < 1 or not specimens:
        raise ValueError('Use specimen_id/batch_id and positive repetitions')
    groups = {}
    for row in specimens:
        groups.setdefault(row[group_key], []).append(float(row['roi_weighted_area_fraction']))
    keys = sorted(groups)
    values = [np.asarray(groups[k], float) for k in keys]
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions, float)
    for i in range(repetitions):
        chosen = rng.integers(0, len(values), size=len(values))
        boot[i] = float(np.concatenate([values[k] for k in chosen]).mean())
    return {'unit': group_key, 'n_groups': len(keys), 'n_rows': len(specimens),
            'observed_mean': float(np.mean([row['roi_weighted_area_fraction'] for row in specimens])),
            'percentile_95': [float(v) for v in np.quantile(boot, [.025, .975])],
            'repetitions': repetitions, 'draws': boot,
            'limitation': 'Synthetic population with few batches; cannot infer uncertainty for real materials.'}


def perturbation_sensitivity(fields: list[dict], fit: dict, specimen_id: str,
                              repetitions=300, seed=22) -> dict:
    """Conditional label-count error propagation to a FIXED area-only ridge model.

    Does not perturb shape, boundary, component count, centroid NN, target labels,
    model fitting or specimen selection. Percentile spans are NOT prediction intervals.
    """
    if repetitions < 2 or not isinstance(repetitions, int):
        raise ValueError('Need at least two repetitions')
    selected = [r for r in fields if r['specimen_id'] == specimen_id]
    if not selected:
        raise ValueError('Unknown specimen')
    total = sum(r['roi_pixels'] for r in selected)
    positive = sum(r['foreground_pixels'] for r in selected)
    truth = positive / total
    baseline_prediction = float(predict_fraction_ridge(fit, truth))
    records = []
    for j, (name, settings) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng(seed + 1009*j)
        observed = np.empty(repetitions)
        for i in range(repetitions):
            measured = sum(perturb_counts(r['roi_pixels'], r['foreground_pixels'],
                                          settings['fpr'], settings['fnr'], rng)['measured_positive_pixels']
                           for r in selected)
            observed[i] = measured / total
        predictions = predict_fraction_ridge(fit, observed)
        expected_error = expected_fraction_bias(truth, **settings)
        records.append({'scenario': name, 'fpr': settings['fpr'], 'fnr': settings['fnr'],
                        'n_fields': len(selected), 'true_fraction': truth,
                        'expected_error_fraction': expected_error,
                        'mean_empirical_error_fraction': float((observed - truth).mean()),
                        'measured_fraction_5_95': [float(v) for v in np.quantile(observed, [.05,.95])],
                        'prediction_au_5_95': [float(v) for v in np.quantile(predictions, [.05,.95])],
                        'mean_predicted_au': float(predictions.mean()),
                        'unperturbed_predicted_au': baseline_prediction,
                        'measured_fraction_draws': observed,
                        'prediction_draws': predictions})
    return {'specimen_id': specimen_id, 'valid_pixels': total,
            'unperturbed_fraction': truth, 'unperturbed_predicted_au': baseline_prediction,
            'scenarios': records,
            'scope': 'Independent label-count flips conditional on known artificial masks; NOT calibrated segmentation or property-prediction intervals.'}


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Cannot write empty CSV')
    keys = list(rows[0])
    if any(set(r) != set(keys) for r in rows):
        raise ValueError('Inconsistent CSV schema')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _finite_json(data):
    if isinstance(data, np.ndarray):
        return [_finite_json(x) for x in data.tolist()]
    if isinstance(data, dict):
        return {k: _finite_json(v) for k,v in data.items()}
    if isinstance(data, list):
        return [_finite_json(v) for v in data]
    if isinstance(data, (np.floating, np.integer)):
        return data.item()
    return data


def run(output: Path, seed=22, repetitions=300, holdout_batch='B3') -> dict:
    output=Path(output)
    if output.exists():
        raise FileExistsError('Output exists; choose a fresh directory')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or not isinstance(repetitions,int) or repetitions < 10:
        raise ValueError('Nonnegative integer seed and at least ten repetitions required')
    fields, properties, _ = synthetic_tables(seed=21)  # pin exact Session 21 fixture
    specimens = aggregate_specimens(fields, properties)
    train, val = specimen_split(specimens, seed=21)
    batch_train, batch_val = leave_batch_out(specimens, holdout_batch)
    grouped = evaluate_split(train, val)
    shifted = evaluate_split(batch_train, batch_val)
    noisy_fields, noisy_specimens = noisy_specimen_features(fields, .02, .06, seed)
    area_model = grouped['one_feature_model']
    selected_id = val[0]['specimen_id']
    sensitivity = perturbation_sensitivity(fields, area_model, selected_id, repetitions, seed)
    specimen_boot = bootstrap_grouped_mean(specimens, 'specimen_id', repetitions, seed)
    batch_boot = bootstrap_grouped_mean(specimens, 'batch_id', repetitions, seed)
    correlation = pooled_and_within_batch_correlation(specimens)
    for x, y in zip(sorted(specimens, key=lambda z:z['specimen_id']), noisy_specimens):
        if x['specimen_id'] != y['specimen_id'] or not np.isclose(x['roi_weighted_area_fraction'], y['true_fraction'], atol=1e-12):
            raise AssertionError('Original and measurement tables misaligned')
    report={'scope':'Artificial image fields and simulated property proxies only. No steel micrographs, measured hardness, Figshare train, or frozen official test.',
            'seed_for_measurement_errors':seed, 'session21_fixture_seed':21,
            'fields':len(fields), 'specimens':len(specimens),'batches':sorted(set(r['batch_id'] for r in specimens)),
            'noise_model':'Independent FP/FN counts conditional on known synthetic class totals; ignores spatial structure, class uncertainty dependence, and model retraining.',
            'nominal_error_rates':{'fpr':.02,'fnr':.06},
            'grouped_specimen_validation':{k:v for k,v in grouped.items() if k!='predictions'},
            'held_out_batch_validation':{k:v for k,v in shifted.items() if k!='predictions'},
            'correlations':correlation,
            'bootstrap':{'specimen':{k:v for k,v in specimen_boot.items() if k!='draws'},
                         'batch':{k:v for k,v in batch_boot.items() if k!='draws'}},
            'measurement_sensitivity':{k:v if k!='scenarios' else [{q:z for q,z in s.items() if not q.endswith('_draws')} for s in v]
                                        for k,v in sensitivity.items()},
            'limitations': ['Label-count perturbations do not capture boundary/object errors',
                            'Six held-out specimens and three artificial batches are insufficient for reliable population inference',
                            'Bootstrap spans are descriptive and not calibrated predictive coverage',
                            'Batch-specific distribution shift is synthetic, not experimental generalization',
                            'The physical and causal interpretation of synthetic associations is invalid.'],
            'provenance_sha256':{'source_session21':hashlib.sha256(Path(__file__).with_name('session21_feature_property_lab.py').read_bytes()).hexdigest(),
                                 'source_session22':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    output.mkdir(parents=True,exist_ok=False)
    _csv(output/'session22_perturbed_fields.csv',noisy_fields)
    _csv(output/'session22_perturbed_specimens.csv',noisy_specimens)
    _csv(output/'session22_grouped_predictions.csv',grouped['predictions'])
    _csv(output/'session22_batch_holdout_predictions.csv',shifted['predictions'])
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for b in sorted(set(r['batch_id'] for r in specimens)):
        rows=[r for r in specimens if r['batch_id']==b]
        axes[0,0].scatter([r['roi_weighted_area_fraction'] for r in rows],
                          [r['synthetic_response_au'] for r in rows],label=b)
    axes[0,0].legend()
    axes[0,0].set(xlabel='Artificial foreground fraction',ylabel='Synthetic target (a.u.)',
                  title='Pooled correlation can reflect processing batch')
    names=['train_mean','area_only_ridge','three_feature_ridge']
    pos=np.arange(len(names))
    axes[0,1].bar(pos-.18,[grouped['metrics'][n]['mae_au'] for n in names],width=.36,label='specimen holdout')
    axes[0,1].bar(pos+.18,[shifted['metrics'][n]['mae_au'] for n in names],width=.36,label=f'leave {holdout_batch} out')
    axes[0,1].set_xticks(pos,names,rotation=15,ha='right')
    axes[0,1].set(ylabel='MAE (artificial units)',title='Specimen versus batch shift')
    axes[0,1].legend()
    for scenario in sensitivity['scenarios']:
        axes[1,0].hist(scenario['measured_fraction_draws'],bins=14,alpha=.45,label=scenario['scenario'])
    axes[1,0].axvline(sensitivity['unperturbed_fraction'],linestyle='--',label='known synthetic truth')
    axes[1,0].legend(fontsize=8)
    axes[1,0].set(xlabel='Perturbed ROI fraction',ylabel='Repetitions',title='Label-count measurement sensitivity')
    axes[1,1].hist(specimen_boot['draws'],bins=22,alpha=.5,label='specimen bootstrap')
    axes[1,1].hist(batch_boot['draws'],bins=22,alpha=.5,label='batch bootstrap')
    axes[1,1].legend()
    axes[1,1].set(xlabel='Mean artificial foreground fraction',ylabel='Repetitions',
                  title='Resampling unit changes uncertainty')
    fig.suptitle('Session 22 — synthetic uncertainty and hierarchical prediction',fontsize=14)
    try:
        fig.savefig(output/'session22_uncertainty.png',dpi=140)
    finally:
        plt.close(fig)
    (output/'session22_report.json').write_text(json.dumps(_finite_json(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('specimen split MAE:', {k:round(v['mae_au'],4) for k,v in grouped['metrics'].items()})
    print('batch holdout MAE:', {k:round(v['mae_au'],4) for k,v in shifted['metrics'].items()})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=22)
    parser.add_argument('--repetitions',type=int,default=300)
    parser.add_argument('--holdout-batch',choices=['B1','B2','B3'],default='B3')
    args=parser.parse_args()
    run(args.output_dir,args.seed,args.repetitions,args.holdout_batch)
