"""MicroStructAI Session 23: explainability on synthetic specimen-level ridge models.

Educational only. No microscopy archives, Figshare official test, physical steel,
real hardness, or genuine phase/constituent labels are read. Every response is
simulated in arbitrary units. Model attributions are not causal explanations.

Requires the user's existing src/session21_feature_property_lab.py.
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
    FEATURES, synthetic_tables, aggregate_specimens, specimen_split,
    fit_specimen_ridge, predict_specimen_ridge,
)


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.ndim != 1 or pred.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('MAE needs two finite, equal-length one-dimensional arrays')
    if y.size == 0:
        raise ValueError('MAE needs at least one specimen')
    return float(np.mean(np.abs(y-pred)))


def validate_specimen_holdout(training: list[dict], validation: list[dict]) -> None:
    """Never let repeated fields or the same specimen cross the held-out split."""
    if not training or not validation:
        raise ValueError('Training and validation must both contain specimens')
    train_ids = [r['specimen_id'] for r in training]
    val_ids = [r['specimen_id'] for r in validation]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError('One record per specimen is required')
    if set(train_ids).intersection(val_ids):
        raise ValueError('Specimen identifiers overlap across train and validation')
    for row in training + validation:
        if any(key not in row for key in (*FEATURES, 'synthetic_response_au', 'batch_id')):
            raise ValueError('Missing required feature/target/provenance columns')
        if not all(np.isscalar(row[key]) and row[key] is not None
                   and np.isfinite(row[key]) for key in (*FEATURES, 'synthetic_response_au')):
            raise ValueError('All specimen features and targets must be finite')


def linear_feature_contributions(fit: dict, specimens: list[dict]) -> dict:
    """Exact decomposition of this standardized additive ridge model.

    baseline = train-target mean; phi_j = standardized_coef_j *
    (x_j - training_feature_mean_j)/training_feature_std_j.
    Sum_j phi_j + baseline = prediction, to numerical precision.

    This equals the independent/interventional-background linear SHAP expression
    for a linear model, but it is NOT correlation-aware conditional SHAP and
    is NOT a causal effect of changing the physical microstructure.
    """
    if not specimens or tuple(fit['feature_names']) != tuple(FEATURES):
        raise ValueError('Need specimens and matching three-feature ridge specification')
    matrix = np.asarray([[r[k] for k in FEATURES] for r in specimens], dtype=float)
    mu = np.asarray(fit['train_mean'], dtype=float)
    scale = np.asarray(fit['train_std'], dtype=float)
    coef = np.asarray(fit['weights_standardized'], dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] != len(FEATURES)
            or not np.isfinite(matrix).all() or not np.isfinite(mu).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(coef).all()):
        raise ValueError('Invalid feature data or ridge normalization')
    contributions = (matrix-mu)/scale*coef
    baseline = float(fit['target_train_mean'])
    reconstructed = baseline + contributions.sum(axis=1)
    predictions = np.asarray(predict_specimen_ridge(fit, specimens), dtype=float)
    if not np.allclose(reconstructed, predictions, rtol=0, atol=1e-9):
        raise AssertionError('Feature contributions must exactly reconstruct the model')
    return {'baseline': baseline, 'features': list(FEATURES),
            'contributions': contributions, 'predictions': predictions,
            'specimen_ids': [r['specimen_id'] for r in specimens],
            'note': 'Independent-reference linear additive attributions; not conditional SHAP or causality.'}


def permutation_importance(fit: dict, heldout: list[dict], *, repeats: int=200,
                           seed: int=23, groups: tuple[tuple[int, ...], ...] | None=None) -> dict:
    """Model reliance: shuffled whole specimen rows on untouched validation targets.

    For each feature group, one shared row permutation moves all members of that
    group jointly. Shuffling only one of correlated predictors can create
    implausible combinations. Repeated shuffles quantify permutation variation,
    not independent-specimen confidence intervals or causal importance.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('Require positive integer shuffle count')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('Require nonnegative integer seed')
    if len(heldout) < 2:
        raise ValueError('At least two held-out specimens are needed')
    if groups is None:
        groups = tuple((i,) for i in range(len(FEATURES)))
    if not groups or any(not group or len(set(group)) != len(group) or
                         any(not isinstance(j, int) or isinstance(j, bool) or j < 0 or j >= len(FEATURES)
                             for j in group) for group in groups):
        raise ValueError('Invalid permutation feature groups')
    x = np.asarray([[r[k] for k in FEATURES] for r in heldout], dtype=float)
    y = np.asarray([r['synthetic_response_au'] for r in heldout], dtype=float)
    baseline = _mae(y, predict_specimen_ridge(fit, heldout))
    rng = np.random.default_rng(seed)
    results = []
    for group in groups:
        deltas = []
        for _ in range(repeats):
            order = rng.permutation(len(heldout))
            modified = [r.copy() for r in heldout]
            for col in group:
                for i, row in enumerate(modified):
                    row[FEATURES[col]] = float(x[order[i], col])
            shuffled_mae = _mae(y, predict_specimen_ridge(fit, modified))
            deltas.append(shuffled_mae-baseline)
        results.append({'features': [FEATURES[i] for i in group],
                        'delta_mae_mean_au': float(np.mean(deltas)),
                        'delta_mae_p10_p90_au': [float(v) for v in np.quantile(deltas, [.1,.9])],
                        'permutations': int(repeats)})
    return {'baseline_mae_au': baseline, 'feature_importance': results,
            'n_heldout_specimens': len(heldout),
            'warning': 'Permutation ΔMAE may be negative. This measures validation model reliance, not causal effects. Strongly correlated features can distort marginal permutations.'}


def correlation_summary(training: list[dict]) -> dict:
    """Training-only Pearson correlations, plus within-batch mean-centering."""
    if len(training) < 4:
        raise ValueError('Need at least four training specimens')
    x = np.asarray([[r[k] for k in FEATURES] for r in training], dtype=float)
    if not np.isfinite(x).all() or np.any(np.std(x, axis=0) < 1e-12):
        raise ValueError('Need varying, finite training features')
    corr = np.corrcoef(x, rowvar=False)
    fractions = x[:, 0]
    targets = np.asarray([r['synthetic_response_au'] for r in training], dtype=float)
    pooled = float(np.corrcoef(fractions, targets)[0,1])
    batches = [r['batch_id'] for r in training]
    centered_f, centered_y = fractions.copy(), targets.copy()
    for batch in set(batches):
        ids = np.asarray([i for i, name in enumerate(batches) if name == batch])
        centered_f[ids] -= fractions[ids].mean()
        centered_y[ids] -= targets[ids].mean()
    within = float(np.corrcoef(centered_f, centered_y)[0,1]) if (
        centered_f.std() > 1e-12 and centered_y.std() > 1e-12) else None
    return {'feature_names': list(FEATURES), 'training_feature_pearson': corr.tolist(),
            'train_fraction_target_pearson': pooled,
            'train_within_batch_centered_fraction_target_pearson': within,
            'warning': 'Correlations are descriptive on synthetic training specimens; no causal or physical claim.'}


def correlated_proxy_demo(seed: int=23, repeats: int=200) -> dict:
    """Two near-duplicate synthetic predictors demonstrate individual vs joint importance.

    The duplicated features carry overlapping information; a ridge model can
    distribute its weight across them. All generated numbers are unitless.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or repeats < 1:
        raise ValueError('Invalid correlated-proxy simulation options')
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=180)
    x = np.column_stack((latent, latent + rng.normal(0,.025,size=len(latent))))
    y = 4*latent + rng.normal(0,.2,size=len(latent))
    train_x, val_x = x[:130], x[130:]
    train_y, val_y = y[:130], y[130:]
    mu, sd = train_x.mean(0), train_x.std(0)
    z = (train_x - mu)/sd
    alpha = 2.0
    A = np.vstack((z, np.sqrt(alpha)*np.eye(2)))
    b = np.concatenate((train_y-train_y.mean(), np.zeros(2)))
    weights = np.linalg.lstsq(A,b,rcond=None)[0]
    def pred(matrix):
        return train_y.mean()+(matrix-mu)/sd@weights
    baseline = _mae(val_y,pred(val_x))
    differences = {}
    for label,indices in {'proxy_A':(0,), 'proxy_B':(1,), 'both_proxies':(0,1)}.items():
        deltas=[]
        for _ in range(repeats):
            order = rng.permutation(len(val_x))
            copy = val_x.copy()
            copy[:,list(indices)] = val_x[order][:,list(indices)]
            deltas.append(_mae(val_y,pred(copy))-baseline)
        differences[label] = float(np.mean(deltas))
    return {'training_correlation':float(np.corrcoef(train_x,rowvar=False)[0,1]),
            'baseline_mae_au':baseline, 'mean_delta_mae_au':differences,
            'train_count':130,'validation_count':50,
            'caveat':'Joint permutation preserves the A/B pairing across rows but is still not causal or a specimen-sampling CI.'}


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError('Cannot write an empty table')
    keys = list(records[0].keys())
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def run(output: Path, seed: int=23, repeats: int=200) -> dict:
    output=Path(output)
    if output.exists():
        raise FileExistsError('Output already exists; use a fresh directory')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed < 0:
        raise ValueError('Invalid seed')
    fields, properties, _ = synthetic_tables(seed=21)
    specimens = aggregate_specimens(fields,properties)
    training, validation = specimen_split(specimens,seed=21)
    validate_specimen_holdout(training,validation)
    fitted = fit_specimen_ridge(training,alpha=2.0)
    predicted = np.asarray(predict_specimen_ridge(fitted,validation))
    actual = np.asarray([r['synthetic_response_au'] for r in validation])
    baseline_mae = _mae(actual,np.full(len(actual),fitted['target_train_mean']))
    ridge_mae = _mae(actual,predicted)
    contributions = linear_feature_contributions(fitted,validation)
    permutations = permutation_importance(fitted,validation,repeats=repeats,seed=seed)
    correlations = correlation_summary(training)
    proxy = correlated_proxy_demo(seed=seed,repeats=repeats)
    output.mkdir(parents=True,exist_ok=False)
    pred_records = [{'specimen_id':r['specimen_id'], 'batch_id':r['batch_id'],
                     'target_simulated_au':float(y), 'prediction_simulated_au':float(p),
                     'absolute_error_au':float(abs(y-p))} for r,y,p in zip(validation,actual,predicted)]
    attribution_records=[]
    for r,row in zip(validation,contributions['contributions']):
        rec={'specimen_id':r['specimen_id'], 'batch_id':r['batch_id'],
             'baseline_au':float(contributions['baseline'])}
        for name,phi in zip(FEATURES,row):
            rec['phi_'+name+'_au']=float(phi)
        rec['prediction_au']=float(rec['baseline_au']+sum(row))
        attribution_records.append(rec)
    permutation_records=[]
    for row in permutations['feature_importance']:
        permutation_records.append({'feature_group':'+'.join(row['features']),
                                    'delta_mae_mean_au':row['delta_mae_mean_au'],
                                    'delta_mae_p10_au':row['delta_mae_p10_p90_au'][0],
                                    'delta_mae_p90_au':row['delta_mae_p10_p90_au'][1]})
    _write_csv(output/'session23_predictions.csv',pred_records)
    _write_csv(output/'session23_linear_contributions.csv',attribution_records)
    _write_csv(output/'session23_permutation_importance.csv',permutation_records)
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {'scope':'18 generated specimens, 54 artificial mask fields, simulated property targets (a.u.); no real steel hardness or Figshare data.',
              'seed_for_permutations':seed, 'repeats':repeats,
              'training_specimens':[r['specimen_id'] for r in training],
              'validation_specimens':[r['specimen_id'] for r in validation],
              'validation_mae_ridge_au':ridge_mae,'validation_mae_train_mean_au':baseline_mae,
              'ridge_fit':{'alpha':fitted['alpha'],'train_only_means':fitted['train_mean'],
                           'train_only_stds':fitted['train_std'],
                           'standardized_coefficients':fitted['weights_standardized'],
                           'baseline_au':fitted['target_train_mean']},
              'linear_contributions_max_abs_reconstruction_error':float(np.max(np.abs(predicted-(contributions['baseline'] + contributions['contributions'].sum(axis=1))))),
              'permutation_importance':permutations,'correlations':correlations,
              'correlated_proxy_demo':proxy,'code_sha256':script_sha,
              'scientific_limits':['Explanations describe a fitted numerical predictor, not physical causality.',
                                   'Validation has six synthetic specimens and is reused for explanations; no independent test inference.',
                                   'Permutation can break correlated-feature dependencies and return negative scores.',
                                   'Linear contributions use independent-reference baseline, not conditional SHAP.',
                                   'No actual labels, measured material properties, physical calibration, or official test data.']}
    (output/'session23_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    display_names=['Area fraction','Component density','Mean NN distance']
    axes[0,0].bar(display_names,np.asarray(fitted['weights_standardized']))
    axes[0,0].axhline(0,color='gray',lw=.8)
    axes[0,0].set(title='Standardized ridge coefficients\n(not causal effects)',ylabel='a.u. per 1 train SD')
    axes[0,0].tick_params(axis='x',rotation=15)
    imp=permutations['feature_importance']
    means=np.array([r['delta_mae_mean_au'] for r in imp])
    axes[0,1].bar(display_names,means)
    axes[0,1].axhline(0,color='gray',lw=.8)
    axes[0,1].set(title='Validation permutation model reliance\n(6 held-out synthetic specimens)',ylabel='Mean MAE increase (a.u.)')
    axes[0,1].tick_params(axis='x',rotation=15)
    phis=contributions['contributions'][0]
    base=contributions['baseline']
    axes[1,0].barh(['Train-target baseline']+display_names,[base]+list(phis))
    axes[1,0].axvline(0,color='gray',lw=.8)
    axes[1,0].set(title=f'Linear contribution values: {validation[0]["specimen_id"]}\nSum={predicted[0]:.2f} a.u. (not one additive bar chart)',xlabel='Baseline/individual contribution (a.u.)')
    px=proxy['mean_delta_mae_au']
    axes[1,1].bar(['Proxy A','Proxy B','Joint A+B'],[px['proxy_A'],px['proxy_B'],px['both_proxies']])
    axes[1,1].set(title=f'Correlated proxy stress test (r={proxy["training_correlation"]:.4f})',ylabel='Mean validation MAE increase (a.u.)')
    fig.suptitle('Session 23: synthetic model explanations, not physical mechanisms',fontsize=13)
    fig.savefig(output/'session23_explainability.png',dpi=135)
    plt.close(fig)
    print('Training specimens',len(training),'validation specimens',len(validation))
    print(f'Ridge MAE {ridge_mae:.4f}; train-mean MAE {baseline_mae:.4f}')
    print('Feature permutation delta MAE:',{r['features'][0]:round(r['delta_mae_mean_au'],3) for r in imp})
    print('Correlated proxy delta MAE:',{k:round(v,3) for k,v in px.items()})
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/session23_v1'))
    parser.add_argument('--seed',type=int,default=23)
    parser.add_argument('--repeats',type=int,default=200)
    args=parser.parse_args()
    try:
        run(args.output_dir,args.seed,args.repeats)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(1,f'Error: {exc}\n')
