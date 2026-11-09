"""MicroStructAI Session 21 — synthetic field/specimen feature tables and leakage lab.

NO real steel, Figshare training/test data, or measured material-property labels.
Masks and response values are generated independently from a shared synthetic
latent variable and batch effect. Responses are arbitrary units, NOT hardness HV.
All distances/counts are pixel-based and are NOT confirmed grains/precipitates.
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
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

FEATURES = ('roi_weighted_area_fraction', 'observed_component_density_px2',
            'mean_observed_centroid_nn_px')


def _python_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError('Cannot write an empty table')
    keys = list(records[0])
    if any(set(record) != set(keys) for record in records):
        raise ValueError('Nonuniform table schema')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _component_summary(binary: np.ndarray, roi: np.ndarray) -> dict:
    """Observed 8-connected mask regions, not physical particles or grains."""
    if (binary.shape != roi.shape or binary.ndim != 2 or roi.dtype != bool or
            not roi.any() or not np.isin(binary, [0, 1]).all()):
        raise ValueError('Require valid binary mask and same-shape nonempty ROI')
    labels, count = ndi.label(binary.astype(bool) & roi, structure=np.ones((3, 3), bool))
    boundary = roi & ~ndi.binary_erosion(roi, structure=np.ones((3, 3), bool), border_value=0)
    censored = 0
    centers = []
    for identifier in range(1, count + 1):
        region = labels == identifier
        censored += int(np.any(region & boundary))
        centers.append(np.argwhere(region).mean(axis=0))
    nearest = None
    if len(centers) >= 2:
        distances, _ = cKDTree(np.asarray(centers)).query(np.asarray(centers), k=2)
        nearest = float(distances[:, 1].mean())
    return {'observed_components': int(count), 'boundary_censored_components': int(censored),
            'mean_observed_centroid_nn_px': nearest}


def synthetic_tables(seed: int = 21, specimens: int = 18, fields_per_specimen: int = 3):
    """Build masks and independently generated specimen-response proxies.

    A synthetic latent structure variable influences BOTH field morphology and
    the synthetic response; a batch effect also influences both. This makes
    confounding visible. The response is not computed from the feature table.
    """
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or
            specimens < 9 or specimens % 3 != 0 or fields_per_specimen < 2):
        raise ValueError('Need nonnegative seed, >=9 specimens divisible by 3, and >=2 fields each')
    rng = np.random.default_rng(seed)
    field_records, properties, samples = [], [], []
    for index in range(specimens):
        specimen_id = f'S{index+1:03d}'
        batch_index = index // (specimens // 3)
        batch_id = f'B{batch_index+1}'
        latent = float(np.clip(.18 + .055 * batch_index + rng.normal(0, .038), .09, .48))
        synthetic_response = float(103 + 130 * latent + 9 * batch_index + rng.normal(0, 3.5))
        properties.append({'specimen_id': specimen_id, 'batch_id': batch_id,
                           'synthetic_response_au': synthetic_response,
                           'target_provenance': 'simulated_specimen_target_not_measured'})
        for field in range(fields_per_specimen):
            field_id = f'{specimen_id}_F{field+1:02d}'
            # Same specimen has multiple correlated image fields. The ROI varies.
            roi = np.ones((64, 64), dtype=bool)
            roi[:8, 49:] = False
            if field:  # Different field exposures: some bottom rows are excluded.
                roi[64 - (field * 3):, :] = False
            raw = rng.normal(size=(64, 64))
            texture = ndi.gaussian_filter(raw, sigma=1.75)
            field_fraction = float(np.clip(latent + rng.normal(0, .025), .05, .65))
            threshold = np.quantile(texture[roi], 1 - field_fraction)
            mask = ((texture >= threshold) & roi).astype(np.uint8)
            summary = _component_summary(mask, roi)
            valid = int(roi.sum())
            foreground = int(mask[roi].sum())
            field_records.append({
                'field_id': field_id, 'specimen_id': specimen_id, 'source_id': specimen_id,
                'batch_id': batch_id, 'roi_pixels': valid,
                'foreground_pixels': foreground,
                'area_fraction': foreground / valid,
                'observed_components': summary['observed_components'],
                'boundary_censored_components': summary['boundary_censored_components'],
                'observed_component_density_px2': summary['observed_components'] / valid,
                'mean_observed_centroid_nn_px': summary['mean_observed_centroid_nn_px'],
                'calibration_um_per_px': None, 'source_kind': 'artificial_mask',
                'mask_semantics': '0=generated_bg;1=generated_fg;not_metallurgical',
            })
            if len(samples) < 3:
                samples.append((mask.copy(), roi.copy(), field_id))
    validate_tables(field_records, properties)
    return field_records, properties, samples


def validate_tables(fields: list[dict], properties: list[dict]) -> None:
    """Enforce specimen→property one-to-many integrity and numeric validity."""
    if not fields or not properties:
        raise ValueError('Both tables must contain records')
    if len({x['field_id'] for x in fields}) != len(fields):
        raise ValueError('Duplicate field IDs')
    if len({x['specimen_id'] for x in properties}) != len(properties):
        raise ValueError('Duplicate specimen property rows')
    lookup = {x['specimen_id']: x for x in properties}
    if {x['specimen_id'] for x in fields} != set(lookup):
        raise ValueError('Unmatched specimen IDs between fields and targets')
    for record in fields:
        sid = record['specimen_id']
        if (record['source_id'] != sid or record['batch_id'] != lookup[sid]['batch_id']
                or not isinstance(record['roi_pixels'], (int, np.integer))
                or not isinstance(record['foreground_pixels'], (int, np.integer))
                or record['roi_pixels'] <= 0
                or not 0 <= record['foreground_pixels'] <= record['roi_pixels']
                or not np.isclose(record['area_fraction'], record['foreground_pixels']/record['roi_pixels'])
                or record['observed_components'] < record['boundary_censored_components']
                or record['boundary_censored_components'] < 0
                or not np.isclose(record['observed_component_density_px2'],
                                  record['observed_components']/record['roi_pixels'])):
            raise ValueError('Invalid field metadata or ROI/measurement consistency')
    for record in properties:
        if (not np.isfinite(record['synthetic_response_au']) or
                record['target_provenance'] != 'simulated_specimen_target_not_measured'):
            raise ValueError('Target must be a finite simulated specimen-level response')


def aggregate_specimens(fields: list[dict], properties: list[dict]) -> list[dict]:
    """One feature vector per specimen; the target is joined ONCE per specimen."""
    validate_tables(fields, properties)
    grouped = {p['specimen_id']: [] for p in properties}
    for field in fields:
        grouped[field['specimen_id']].append(field)
    table = []
    for property_row in properties:
        sid = property_row['specimen_id']
        rows = grouped[sid]
        valid = sum(row['roi_pixels'] for row in rows)
        foreground = sum(row['foreground_pixels'] for row in rows)
        comps = sum(row['observed_components'] for row in rows)
        censored = sum(row['boundary_censored_components'] for row in rows)
        nn_values = [row['mean_observed_centroid_nn_px'] for row in rows
                     if row['mean_observed_centroid_nn_px'] is not None]
        table.append({
            'specimen_id': sid, 'batch_id': property_row['batch_id'], 'n_fields': len(rows),
            'valid_roi_pixels': valid, 'foreground_pixels': foreground,
            'roi_weighted_area_fraction': foreground / valid,
            'unweighted_mean_field_area_fraction': float(np.mean([r['area_fraction'] for r in rows])),
            'observed_components': comps, 'boundary_censored_components': censored,
            'observed_component_density_px2': comps / valid,
            'mean_observed_centroid_nn_px': float(np.mean(nn_values)) if nn_values else None,
            'synthetic_response_au': property_row['synthetic_response_au'],
            'target_provenance': property_row['target_provenance'],
        })
    return table


def specimen_split(specimen_table: list[dict], seed: int = 21):
    """Hold out 2 specimens out of every batch of 6, never individual fields."""
    if not specimen_table:
        raise ValueError('Empty specimen table')
    grouped = {}
    for row in specimen_table:
        grouped.setdefault(row['batch_id'], []).append(row['specimen_id'])
    rng = np.random.default_rng(seed)
    holdout = set()
    for batch, ids in sorted(grouped.items()):
        if len(ids) < 3:
            raise ValueError(f'Need >=3 specimens in {batch}')
        ids = sorted(ids)
        holdout.update(rng.permutation(ids)[:max(1, len(ids)//3)].tolist())
    train = [row for row in specimen_table if row['specimen_id'] not in holdout]
    validation = [row for row in specimen_table if row['specimen_id'] in holdout]
    if set(x['specimen_id'] for x in train) & set(x['specimen_id'] for x in validation):
        raise AssertionError('Specimen leakage')
    return train, validation


def fit_specimen_ridge(training: list[dict], alpha: float = 2.):
    """Train-only z-score scaling and closed-form ridge on specimen features."""
    if not training or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Need training rows and nonnegative finite alpha')
    X = np.asarray([[row[key] for key in FEATURES] for row in training], dtype=float)
    y = np.asarray([row['synthetic_response_au'] for row in training], dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Cannot fit with unavailable numeric features')
    mu = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Z = (X - mu)/scale
    ymean = float(y.mean())
    design = np.vstack([Z, np.sqrt(alpha) * np.eye(Z.shape[1])])
    response = np.concatenate([y - ymean, np.zeros(Z.shape[1])])
    weights = np.linalg.lstsq(design, response, rcond=None)[0]
    return {'feature_names': FEATURES, 'train_mean': mu.tolist(),
            'train_std': scale.tolist(), 'target_train_mean': ymean,
            'weights_standardized': weights.tolist(), 'alpha': float(alpha),
            'training_specimen_ids': [x['specimen_id'] for x in training]}


def predict_specimen_ridge(fit: dict, specimens: list[dict]) -> np.ndarray:
    if tuple(fit['feature_names']) != FEATURES:
        raise ValueError('Incorrect feature specification')
    X = np.asarray([[row[key] for key in FEATURES] for row in specimens], dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURES) or not np.isfinite(X).all():
        raise ValueError('Invalid predictor inputs')
    Z = (X-np.asarray(fit['train_mean']))/np.asarray(fit['train_std'])
    return float(fit['target_train_mean']) + Z@np.asarray(fit['weights_standardized'])


def identity_leakage_demo(fields: list[dict], properties: list[dict], train: list[dict], val: list[dict]) -> dict:
    """Illustrate perfect target memorization when a specimen appears on both sides.

    This is NOT a legitimate model. It deliberately looks up specimen IDs, which
    would be unavailable for unseen specimens. Never report this as accuracy.
    """
    labels = {row['specimen_id']: row['synthetic_response_au'] for row in properties}
    # Naive row split: fields 1+2 from every specimen vs field 3 from SAME specimen.
    grouped = {}
    for row in fields:
        grouped.setdefault(row['specimen_id'], []).append(row)
    lookup = {}
    naive_val = []
    for sid, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r['field_id'])
        if len(rows) < 2:
            raise ValueError('Need at least two views per specimen')
        for _ in rows[:-1]:
            lookup[sid] = labels[sid]
        naive_val.append(rows[-1])
    leaked_mae = float(np.mean([abs(lookup[r['specimen_id']] - labels[r['specimen_id']])
                                for r in naive_val]))
    group_train_lookup = {row['specimen_id']: row['synthetic_response_au'] for row in train}
    unseen_specimens = [row['specimen_id'] for row in val if row['specimen_id'] not in group_train_lookup]
    if len(unseen_specimens) != len(val):
        raise AssertionError('Group holdout should have no known specimen identifiers')
    fallback_mean = float(np.mean(list(group_train_lookup.values())))
    fallback_mae = float(np.mean([abs(fallback_mean-row['synthetic_response_au']) for row in val]))
    return {'bad_field_row_split_id_memorizer_mae_au': leaked_mae,
            'proper_specimen_split_unseen_ids': len(unseen_specimens),
            'proper_specimen_split_id_memorizer_fallback_mae_au': fallback_mae,
            'warning': 'The bad split deliberately memorizes repeated specimen IDs; zero error is leakage, not prediction.'}


def bootstrap_specimen_mean(specimens: list[dict], repetitions=2000, seed=21):
    if len(specimens) < 2 or repetitions < 1:
        raise ValueError('Need >=2 specimens and >=1 repeats')
    values = np.asarray([row['roi_weighted_area_fraction'] for row in specimens], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return {'observed_mean': float(values.mean()),
            'percentile_interval_95': [float(x) for x in np.quantile(sampled, [.025, .975])],
            'sampling_unit': 'generated_specimen', 'repetitions': int(repetitions),
            'limitation': 'Generated specimens, not a probability sample from a real materials population; no true material CI.'}


def run(output, seed: int = 21, alpha: float = 2., bootstrap_repetitions: int = 2000):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output directory exists; choose a new path')
    if not isinstance(seed, int) or seed < 0 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Invalid seed/alpha')
    fields, properties, examples = synthetic_tables(seed)
    specimens = aggregate_specimens(fields, properties)
    training, validation = specimen_split(specimens, seed)
    fit = fit_specimen_ridge(training, alpha)
    predictions = predict_specimen_ridge(fit, validation)
    truth = np.asarray([r['synthetic_response_au'] for r in validation])
    baseline = float(np.mean([r['synthetic_response_au'] for r in training]))
    ridge_mae = float(np.mean(abs(truth-predictions)))
    baseline_mae = float(np.mean(abs(truth-baseline)))
    leakage = identity_leakage_demo(fields, properties, training, validation)
    leakage['valid_specimen_feature_ridge_mae_au'] = ridge_mae
    leakage['train_mean_only_baseline_mae_au'] = baseline_mae
    results = []
    for r, p in zip(validation, predictions):
        results.append({'specimen_id': r['specimen_id'], 'batch_id': r['batch_id'],
                        'actual_synthetic_response_au': r['synthetic_response_au'],
                        'predicted_synthetic_response_au': float(p),
                        'absolute_error_au': float(abs(p-r['synthetic_response_au']))})
    report = {
        'scope': 'Artificial masks; simulated material-like response in arbitrary units; NO real hardness, physical property measurements, Figshare research images, or frozen test access.',
        'seed': seed, 'fields': len(fields), 'specimens': len(specimens),
        'training_specimens': [row['specimen_id'] for row in training],
        'validation_specimens': [row['specimen_id'] for row in validation],
        'training_field_count': sum(r['n_fields'] for r in training),
        'validation_field_count': sum(r['n_fields'] for r in validation),
        'property_target_level': 'one synthetic response per specimen, never independent per field',
        'feature_semantics': 'mask area, count/density and observed centroid NN in pixels; components are NOT verified grains, precipitates, ferrite/pearlite labels',
        'source_lineage': 'each source_id equals specimen_id; generated fields grouped before splitting',
        'fit': fit, 'validation_ridge_mae_au': ridge_mae,
        'validation_train_mean_baseline_mae_au': baseline_mae,
        'leakage_demonstration': leakage,
        'specimen_bootstrap': bootstrap_specimen_mean(specimens, bootstrap_repetitions, seed),
        'no_claims': 'Synthetic batch effects and latent structure make correlation demonstrable but cannot establish physical or causal materials-property relationships. Validation is model-development data, not a final independent test.'
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output/'session21_fields.csv', fields)
    _write_csv(output/'session21_properties.csv', properties)
    _write_csv(output/'session21_specimens.csv', specimens)
    _write_csv(output/'session21_predictions.csv', results)
    (output/'session21_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    mask, roi, field_id = examples[0]
    image = np.where(~roi, np.nan, mask)
    axes[0,0].imshow(image, cmap='gray', vmin=0,vmax=1, interpolation='nearest')
    axes[0,0].set_title(f'Generated binary mask / ROI — {field_id}')
    axes[0,0].axis('off')
    batch_color = {'B1':'#5682ab', 'B2':'#d18b4c', 'B3':'#59a17a'}
    for batch in sorted(set(r['batch_id'] for r in specimens)):
        subset = [r for r in specimens if r['batch_id']==batch]
        axes[0,1].scatter([r['roi_weighted_area_fraction'] for r in subset],
                          [r['synthetic_response_au'] for r in subset], label=batch,
                          color=batch_color.get(batch), alpha=.8)
    axes[0,1].set(xlabel='ROI-weighted foreground fraction (artificial)', ylabel='Simulated response (a.u.)', title='Association with deliberate synthetic batch effects')
    axes[0,1].legend()
    axes[1,0].scatter(truth, predictions, color='#5682ab')
    lo, hi = min(truth.min(), predictions.min()), max(truth.max(), predictions.max())
    axes[1,0].plot([lo,hi], [lo,hi], '--', color='gray')
    axes[1,0].set(xlabel='Simulated true response (a.u.)', ylabel='Predicted response (a.u.)', title='Held-out specimen IDs: illustrative ridge')
    names = ['Train-mean\nbaseline','Specimen-level\nridge','Leaky ID\nlookup']
    values = [baseline_mae, ridge_mae, leakage['bad_field_row_split_id_memorizer_mae_au']]
    axes[1,1].bar(names, values, color=['#7d96a9','#5682ab','#cd8b64'])
    axes[1,1].set(ylabel='MAE (synthetic arbitrary units)', title='Different protocols: leaky ID lookup is INVALID')
    axes[1,1].text(2, .12*max(values), 'LEAKAGE', ha='center', color='#9a322b', fontsize=9, weight='bold')
    fig.suptitle('Session 21 — synthetic feature tables, specimen splits and target leakage', fontsize=14)
    try:
        fig.savefig(output/'session21_feature_property.png', dpi=135)
    finally:
        plt.close(fig)
    print(f'Artificial data: {len(fields)} fields, {len(specimens)} specimens; train {len(training)}, validation {len(validation)}')
    print(f'Ridge validation MAE={ridge_mae:.4f} a.u.; train-mean baseline MAE={baseline_mae:.4f} a.u.')
    print('Leaky specimen-ID memorizer MAE:', leakage['bad_field_row_split_id_memorizer_mae_au'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--alpha', type=float, default=2.)
    args = parser.parse_args()
    try:
        run(args.output_dir, seed=args.seed, alpha=args.alpha)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
