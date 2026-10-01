"""Sparse, preregistered EMA-inclusive combinations; never fit an estimator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import TARGETS
from .ema_average_span import old_cache, task_frames
from .ema_evaluation_diagnostics import sha, write_new
from .v5_resolution import paired_summary
from .v7_periodic import digest

COMPONENTS = ('v36_time', 'n_time', 'v7_time', 'ema')
REFERENCE_WEIGHTS = (.20, .30, -.25, .75)
PROBES = {
    'V36_TO_N005': (.15, .35, -.25, .75),
    'N_TO_V36005': (.25, .25, -.25, .75),
    'V36_TO_V7005': (.15, .30, -.20, .75),
    'V7_TO_V36005': (.25, .30, -.30, .75),
    'N_TO_V7005': (.20, .25, -.20, .75),
    'V7_TO_N005': (.20, .35, -.30, .75),
}


def combine(parts, weights):
    arrays = [np.asarray(parts[k], float) for k in COMPONENTS]
    if (len(weights) != 4 or abs(sum(weights)-1) > 1e-12 or weights[3] != .75
            or arrays[0].ndim != 1 or any(a.shape != arrays[0].shape for a in arrays)
            or any(not np.isfinite(a).all() for a in arrays)):
        raise ValueError('Frozen EMA strength and finite aligned components required')
    result = sum(w*a for w, a in zip(weights, arrays))
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid affine combination; no clipping')
    return result


def score(y, iron, time):
    actual = np.asarray(y, float)
    prediction = np.column_stack((iron, time))
    if (actual.shape != prediction.shape or actual.ndim != 2 or actual.shape[1] != 2
            or not np.isfinite(actual).all() or not np.isfinite(prediction).all()
            or (prediction < 0).any() or (np.abs(actual).sum(axis=0) <= 0).any()):
        raise ValueError('Finite complete two-target package required')
    return float(100-50*np.sum(np.abs(actual-prediction).sum(axis=0)/np.abs(actual).sum(axis=0)))


def describe(y, baseline, candidate, folds, spouts, *, rng_seed):
    """Pooled seed gain; fold/group results and 322-row draws are descriptive."""
    y, baseline, candidate = [np.asarray(a, float) for a in (y, baseline, candidate)]
    if y.ndim != 1 or baseline.shape != y.shape or candidate.shape != y.shape:
        raise ValueError('Aligned full seed vectors required')
    improvement = np.abs(y-baseline)-np.abs(y-candidate)
    def metric(mask):
        denominator = np.abs(y[mask]).sum()
        if denominator <= 0:
            raise ValueError('Positive WMAPE denominator required')
        return dict(rows=int(mask.sum()), gain=float(50*improvement[mask].sum()/denominator))
    draws = []
    rng = np.random.default_rng(rng_seed)
    for _ in range(2000):
        ids = rng.choice(len(y), size=min(322, len(y)), replace=False)
        draws.append(float(50*improvement[ids].sum()/np.abs(y[ids]).sum()))
    return dict(**metric(np.ones(len(y), bool)),
        folds={str(f): metric(folds == f) for f in sorted(set(folds))},
        spouts={str(s): metric(spouts == s) for s in sorted(set(spouts))},
        improved_rows=int((improvement > 0).sum()), harmed_rows=int((improvement < 0).sum()),
        harm_absolute_sum=float(-improvement[improvement < 0].sum()),
        improvement_absolute_sum=float(improvement[improvement > 0].sum()),
        row_improvement_quantiles=np.quantile(improvement, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        sample_322=dict(replicates=2000, seed=rng_seed, negative_fraction=float(np.mean(np.asarray(draws) < 0)),
            quantiles=np.quantile(draws, [.025, .5, .975]).tolist(),
            scope='fixed_OOF_without_replacement_not_platform_failure_probability'))


def select_development(records):
    eligible = [name for name in PROBES if all(records[name][str(s)]['gain'] > 0 for s in (42, 3407))]
    return max(eligible, key=lambda n: np.mean([records[n][str(s)]['gain'] for s in (42, 3407)])) if eligible else None


def collect_parts(main, frame, folds, spec):
    """Close each design matrix inside one split seed, with physical cache identity."""
    main = Path(main)
    result = {}
    for seed, fv in folds.items():
        vectors = {k: np.full(len(frame), np.nan) for k in (*COMPONENTS, 'iron', 'q75')}
        for fold in range(5):
            training, query = task_frames(frame, fv, fold)
            mask = fv == fold
            if seed in (42, 3407):
                old = old_cache(dict(main_root=str(main), old_development=spec['old_development']),
                    seed, fold, training, query)
                path = main/spec['old_development']/f'reference-s{seed}-f{fold}/predictions.npz'
                with np.load(path, allow_pickle=False) as saved:
                    values = {k: saved[k].copy() for k in COMPONENTS[:-1]}
                values.update(ema=old['ema'], iron=old['iron'], q75=old['q75'])
            else:
                unit = main/spec['confirmation_cache']/f's{seed}-f{fold}'
                warm = json.loads((unit/'warm-complete.json').read_text())
                cold = json.loads((unit/'cold-complete.json').read_text())
                if (cold['status'] != 'passed' or cold['warm_receipt_sha256'] != sha(unit/'warm-complete.json')
                        or warm['predictions_sha256'] != sha(unit/'predictions.npz')
                        or warm['reference_metadata']['fit_ids_digest'] != digest(training.sample_id.tolist())):
                    raise ValueError('Closed matching confirmation cache required')
                with np.load(unit/'predictions.npz', allow_pickle=False) as saved:
                    if saved['query_ids'].tolist() != query.sample_id.tolist():
                        raise ValueError('Confirmation cache query identity')
                    values = {k: saved[k].copy() for k in COMPONENTS[:-1]}
                    values.update(ema=saved['old_ema'].copy(), iron=saved['iron'].copy(), q75=saved['q75'].copy())
            np.testing.assert_allclose(combine(values, REFERENCE_WEIGHTS), values['q75'], atol=1e-10, rtol=0)
            for key in vectors:
                vectors[key][mask] = values[key]
        if any(not np.isfinite(a).all() for a in vectors.values()):
            raise ValueError('Incomplete within-seed design matrix')
        result[seed] = vectors
    return result


def evaluate(main, frame, folds, spec, out):
    if spec['combination_probes'] != {k: list(v) for k, v in PROBES.items()}:
        raise ValueError('Preregistered sparse pool changed')
    parts = collect_parts(main, frame, folds, spec)
    records = {name: {} for name in PROBES}
    # Select on the two complete development seeds before revealing other metrics.
    order = [42, 3407, 271828, 314159]
    selected = None
    for position, seed in enumerate(order):
        fv, vectors = folds[seed], parts[seed]
        predictions = {}
        for name, weights in PROBES.items():
            candidate = combine(vectors, weights)
            records[name][str(seed)] = describe(frame.tap_time_len.to_numpy(), vectors['q75'], candidate,
                fv, frame.spout_no.to_numpy(), rng_seed=961045+seed)
            records[name][str(seed)].update(
                reference_score=score(frame[list(TARGETS)].to_numpy(), vectors['iron'], vectors['q75']),
                candidate_score=score(frame[list(TARGETS)].to_numpy(), vectors['iron'], candidate))
            predictions[name] = candidate
        with (Path(out)/f'combination-s{seed}.npz').open('xb') as stream:
            np.savez_compressed(stream, query_ids=frame.sample_id.to_numpy(str), folds=fv,
                actual=frame[list(TARGETS)].to_numpy(), **vectors, **predictions)
        if position == 1:
            selected = select_development(records)
            write_new(Path(out)/'combination-development-choice.json', dict(candidate=selected,
                scope='two_complete_development_seeds_only', all_development_gains={
                    n: {str(s): records[n][str(s)]['gain'] for s in (42, 3407)} for n in PROBES}))
    summary = {name: paired_summary([records[name][str(s)]['gain'] for s in order]) for name in PROBES}
    positive = bool(selected and summary[selected]['positive'] == 4 and summary[selected]['lcb95'] > 0)
    report = dict(status='completed_zero_fit_review', records=records, seed_summaries=summary,
        selected_on_development=selected, selected_four_seed_gate=positive,
        classification='platform_exploration_review_required' if positive else 'no_robust_weight_probe',
        formal_new_model_promotion=False, new_fits=0, new_split_seeds=0, packages=0, uploads=0,
        platform_gain=None, LCB_scope='split_stability_same_sample_pool_not_platform_guarantee')
    write_new(Path(out)/'combination-report.json', report)
    return report
