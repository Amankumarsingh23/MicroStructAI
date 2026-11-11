"""Session 24 synthetic explanation stability with training-only selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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
from session23_explainability_lab import permutation_importance, validate_specimen_holdout


def fit_subset(rows, features, alpha):
    """Reuse Session 21 scaling and solver with excluded columns held constant."""
    features = tuple(features)
    if not features or len(set(features)) != len(features) or any(f not in FEATURES for f in features):
        raise ValueError('Choose a nonempty unique subset of declared features')
    adapted = [dict(row, **{f: 0.0 for f in FEATURES if f not in features}) for row in rows]
    fit = fit_specimen_ridge(adapted, alpha)
    fit['feature_names'] = list(fit['feature_names'])
    fit['selected_features'] = list(features)
    return fit


def predict_subset(fit, rows):
    adapted = [dict(row, **{f: 0.0 for f in FEATURES if f not in fit['selected_features']}) for row in rows]
    return predict_specimen_ridge(fit, adapted)


def inner_folds(rows, seed=24):
    """Three disjoint folds, stratified by artificial batch when possible."""
    ids = [r['specimen_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) < 6:
        raise ValueError('Need at least six unique specimens')
    rng = np.random.default_rng(seed)
    folds = [[], [], []]
    cursor = 0
    for batch in sorted({r['batch_id'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['batch_id'] == batch]
        for index in rng.permutation(indices):
            folds[cursor % 3].append(int(index))
            cursor += 1
    return folds


def select_model(rows, seed=24):
    folds = inner_folds(rows, seed)
    candidates = []
    for size in range(1, len(FEATURES) + 1):
        for features in itertools.combinations(FEATURES, size):
            for alpha in (0.0, 2.0, 20.0):
                errors = []
                for heldout in folds:
                    training = [r for i, r in enumerate(rows) if i not in heldout]
                    validation = [rows[i] for i in heldout]
                    fit = fit_subset(training, features, alpha)
                    errors.extend(abs(predict_subset(fit, validation) - np.array(
                        [r['synthetic_response_au'] for r in validation])))
                candidates.append({'features': list(features), 'alpha': alpha,
                                   'inner_mae_au': float(np.mean(errors))})
    # Enumeration order resolves exact ties by fewer features, declared order, then alpha.
    best = min(candidates, key=lambda row: row['inner_mae_au'])
    return best, candidates, folds


def selection_stability(training, repeats=24, seed=24):
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Positive subsample count required')
    rng = np.random.default_rng(seed)
    groups = [[r for r in training if r['batch_id'] == b]
              for b in sorted({r['batch_id'] for r in training})]
    choices = []
    for repeat in range(repeats):
        subset = []
        for group in groups:
            if len(group) < 3:
                raise ValueError('Need three specimens per batch for subsampling')
            subset.extend(group[i] for i in rng.choice(len(group), len(group)-1, replace=False))
        best, _, _ = select_model(subset, seed + repeat)
        choices.append({'repeat': repeat, 'features': best['features'], 'alpha': best['alpha'],
                        'inner_mae_au': best['inner_mae_au'],
                        'specimen_ids': [r['specimen_id'] for r in subset]})
    frequencies = {f: sum(f in r['features'] for r in choices)/repeats for f in FEATURES}
    return choices, frequencies


def coefficient_stability(training, fitted, repeats=200, seed=24):
    if type(repeats) is not int or repeats < 2:
        raise ValueError('At least two bootstrap refits required')
    rng = np.random.default_rng(seed)
    groups = [[r for r in training if r['batch_id'] == b]
              for b in sorted({r['batch_id'] for r in training})]
    draws = []
    for _ in range(repeats):
        sample = [group[i] for group in groups for i in rng.integers(0, len(group), len(group))]
        fit = fit_subset(sample, fitted['selected_features'], fitted['alpha'])
        draws.append(fit['weights_standardized'])
    values = np.array(draws)
    summaries = []
    for feature in fitted['selected_features']:
        j = FEATURES.index(feature)
        coef = fitted['weights_standardized'][j]
        summaries.append({'feature': feature, 'original_coefficient': coef,
                          'bootstrap_median': float(np.median(values[:, j])),
                          'p10': float(np.quantile(values[:, j], .1)),
                          'p90': float(np.quantile(values[:, j], .9)),
                          'sign_agreement': float(np.mean(np.sign(values[:, j]) == np.sign(coef)))})
    return summaries


def write_csv(path, rows):
    encoded = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in row.items()} for row in rows]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(encoded[0]))
        writer.writeheader()
        writer.writerows(encoded)


def run(output, seed=24, subsamples=24, bootstraps=200, permutations=100):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    fields, properties, _ = synthetic_tables(seed=21)
    specimens = aggregate_specimens(fields, properties)
    training, validation = specimen_split(specimens, seed=21)
    validate_specimen_holdout(training, validation)
    best, candidates, folds = select_model(training, seed)
    selected = fit_subset(training, best['features'], best['alpha'])
    fixed = fit_specimen_ridge(training, alpha=2.0)
    actual = np.array([r['synthetic_response_au'] for r in validation])
    predictions = {'train_mean': np.full(len(validation), fixed['target_train_mean']),
                   'fixed_ridge': predict_specimen_ridge(fixed, validation),
                   'inner_selected': predict_subset(selected, validation)}
    outer = [{'method': method, 'mae_au': float(np.mean(abs(pred-actual)))}
             for method, pred in predictions.items()]
    choices, frequencies = selection_stability(training, subsamples, seed)
    coefficients = coefficient_stability(training, selected, bootstraps, seed)
    reliance = permutation_importance(fixed, validation, repeats=permutations, seed=seed,
                                     groups=((0,), (1,), (2,), (1, 2)))
    report = {
        'scope': 'Artificial Session 21 specimens and simulated responses in arbitrary units',
        'implementation': 'Reconstructed from Session 24 handout; supplied code unavailable locally',
        'seed': seed, 'subsamples': subsamples, 'bootstraps': bootstraps, 'permutations': permutations,
        'training_specimens': [r['specimen_id'] for r in training],
        'validation_specimens': [r['specimen_id'] for r in validation],
        'inner_validation_specimens': [[training[i]['specimen_id'] for i in fold] for fold in folds],
        'selection': best, 'selected_fit': selected, 'outer_comparison': outer,
        'selection_frequencies': frequencies, 'coefficient_stability': coefficients,
        'fixed_model_group_permutation': reliance,
        'protocol': 'Batch-stratified three-fold inner CV; pooled out-of-fold MAE; exact ties follow candidate enumeration. Subsamples remove one unique specimen per batch and vary inner-fold seed. Bootstrap holds batch counts fixed and refits fixed selected specification.',
        'limits': ['One outer holdout, six synthetic validation specimens',
                   'Outer specimens were used for earlier sessions and are development data',
                   'Bootstrap conditions on selected features/penalty and uses refit-specific scaling',
                   'Selection frequencies depend on subsample and fold algorithms',
                   'Explanations concern fitted synthetic relationships; physical validation requires audited data'],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'session24_inner_candidates.csv', candidates)
    write_csv(output/'session24_subsample_choices.csv', choices)
    write_csv(output/'session24_coefficient_stability.csv', coefficients)
    write_csv(output/'session24_outer_comparison.csv', outer)
    (output/'session24_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    try:
        axes[0, 0].bar(np.arange(len(candidates)), [r['inner_mae_au'] for r in candidates])
        axes[0, 0].set(xlabel='Candidate enumeration index', ylabel='Inner CV MAE (a.u.)', title='21 training-only candidate specifications')
        axes[0, 1].bar([r['method'] for r in outer], [r['mae_au'] for r in outer])
        axes[0, 1].set(ylabel='Outer development MAE (a.u.)', title='Six held-out generated specimens')
        labels = ['Area fraction', 'Component density', 'NN distance']
        axes[1, 0].bar(labels, list(frequencies.values()))
        axes[1, 0].set(ylim=(0, 1.05), ylabel='Selection frequency', title='Training subsamples with inner reselection')
        for i, row in enumerate(coefficients):
            axes[1, 1].plot([i, i], [row['p10'], row['p90']], color='steelblue', linewidth=3)
            axes[1, 1].scatter(i, row['bootstrap_median'], color='black')
        axes[1, 1].set_xticks(range(len(coefficients)), [labels[FEATURES.index(r['feature'])] for r in coefficients])
        axes[1, 1].set(ylabel='Standardized coefficient', title='Fixed specification bootstrap median and 10-90% span')
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=12)
        fig.suptitle('Session 24 synthetic selection and explanation stability')
        fig.savefig(output/'session24_explanation_stability.png', dpi=135)
    finally:
        plt.close(fig)
    print('Selected', best)
    print('Outer comparison', outer)
    print('Selection frequencies', frequencies)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/session24_v1'))
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--subsamples', type=int, default=24)
    parser.add_argument('--bootstraps', type=int, default=200)
    parser.add_argument('--permutations', type=int, default=100)
    args = parser.parse_args()
    run(args.output_dir, args.seed, args.subsamples, args.bootstraps, args.permutations)
