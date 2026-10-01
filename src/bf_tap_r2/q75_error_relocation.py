"""Identity-bound descriptive error maps; never fit, select or release models."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .data import TARGETS
from .ema_evaluation_diagnostics import load_saved, sha, verify_files, write_new

CANDIDATES = ('V32', 'EMA_TIME_Q50', 'EMA_TIME_Q100', 'SAM_TIME_Q50',
              'PTARL_TIME_Q20', 'EMA_IRON_EMA_TIME', 'DE3_IRON_Q75_RESERVE',
              'EMA_PTARL_ENDPOINT_HALF', 'EMA_PLUS_PTARL_INCREMENT_Q20')


def check_vectors(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if (y.ndim != 2 or y.shape[1] != 2 or y.shape != p.shape or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(p).all()
            or (y <= 0).any() or (p < 0).any()):
        raise ValueError('Finite positive targets and aligned nonnegative predictions required')
    return y, p


def concentration(values):
    a = np.asarray(values, float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Finite nonnegative contribution vector required')
    order = np.argsort(-a, kind='stable')
    total = float(a.sum())
    counts = {'top20': min(20, len(a)), 'top1pct': math.ceil(.01*len(a)),
              'top5pct': math.ceil(.05*len(a))}
    return {name: {'rows': n, 'share': float(a[order[:n]].sum()/total) if total else 0.}
            for name, n in counts.items()}


def input_regions(frame, folds, features):
    """Each held fold uses only the other folds' feature cutpoints."""
    folds = np.asarray(folds)
    if folds.shape != (len(frame),) or set(folds) != set(range(5)):
        raise ValueError('Five aligned complete folds required')
    regions = {'ALL': np.ones(len(frame), bool)}
    thresholds = {}
    for s in sorted(frame.spout_no.unique()):
        regions[f'spout={int(s)}'] = frame.spout_no.to_numpy() == s
    for feature in features:
        x = frame[feature].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite conditional input')
        bins = np.empty(len(frame), int)
        thresholds[feature] = {}
        for f in range(5):
            held = folds == f
            cuts = np.quantile(x[~held], [.25, .5, .75])
            thresholds[feature][str(f)] = cuts.tolist()
            bins[held] = np.searchsorted(cuts, x[held], side='right')
        for q in range(4):
            mask = bins == q
            regions[f'{feature}:Q{q+1}'] = mask
            for s in sorted(frame.spout_no.unique()):
                regions[f'{feature}:Q{q+1}:spout={int(s)}'] = mask & (frame.spout_no.to_numpy() == s)
    return regions, thresholds


def residual_summary(y, p, mask, denominators):
    if not mask.any():
        return {'rows': 0}
    r = y[mask]-p[mask]
    local_den = y[mask].sum(axis=0)
    error = np.abs(r).sum(axis=0)
    return dict(rows=int(mask.sum()), mean=r.mean(axis=0).tolist(), median=np.median(r, axis=0).tolist(),
                positive_fraction=(r > 0).mean(axis=0).tolist(), mae=np.abs(r).mean(axis=0).tolist(),
                wmape=(error/local_den).tolist(), score_loss_global_denominator=float(50*(error/denominators).sum()),
                structural_interpretation_allowed=bool(mask.sum() >= 100))


def movement_summary(y, ref, candidate, mask, denominators):
    if not mask.any():
        return {'rows': 0}
    reduction = np.abs(y[mask]-ref[mask])-np.abs(y[mask]-candidate[mask])
    score_rows = 50*(reduction/denominators).sum(axis=1)
    local_gain = 50*(reduction.sum(axis=0)/y[mask].sum(axis=0)).sum()
    positives = np.maximum(score_rows, 0)
    count = math.ceil(.01*len(score_rows))
    order = np.argsort(-positives, kind='stable')
    return dict(rows=int(mask.sum()), gain_global_denominator=float(score_rows.sum()),
                gain_region_denominator=float(local_gain), improvement_fraction=float((score_rows > 0).mean()),
                worsening_fraction=float((score_rows < 0).mean()), positive_score_reduction=float(positives.sum()),
                negative_score_reduction=float(np.minimum(score_rows, 0).sum()),
                top1pct_positive_share=float(positives[order[:count]].sum()/positives.sum()) if positives.sum() else 0.)


def load_inputs(spec):
    main = Path(spec['main_root'])
    verify_files(main, spec['inputs'])
    current = json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current[k] != v for k, v in spec['reference'].items()):
        raise ValueError('Current platform reference changed before execution')
    old_spec = json.loads((main/spec['original_diagnostic_spec']).read_text())
    frame, folds, base, time_members, ptarl, _ = load_saved(old_spec)
    from .component_regularization_run import collect
    _, iron_members = collect(main/spec['component_development'], frame, folds, {'tap_iron': ['EMA']})
    # Prior audits bind every persisted artifact; verify before using extra iron units.
    for seed in folds:
        for f in range(5):
            unit = main/spec['component_development']/f'tap_iron-EMA-s{seed}-f{f}'
            verify_files(unit, json.loads((unit/'complete.json').read_text())['hashes'])
    de3 = pd.read_csv(main/spec['de3_oof'], dtype={'sample_id': str})
    de3_audit = json.loads((main/spec['de3_development']/'audit.json').read_text())
    if (de3_audit['status'] != 'passed' or de3_audit['manifest_sha256'] != sha(main/spec['de3_development']/'manifest.json')
            or de3_audit['summary_sha256'] != sha(main/spec['de3_development']/'summary.json')):
        raise ValueError('DE3 original audit identity mismatch')
    bundles = {}
    y = frame[list(TARGETS)].to_numpy(float)
    for seed, fv in folds.items():
        b, v = base[seed]['tap_time_len'], base[seed]['v7_time']
        e, sam = time_members[seed, 'tap_time_len', 'EMA'], time_members[seed, 'tap_time_len', 'SAM']
        iron = base[seed]['tap_iron']
        ref = np.column_stack([iron, b+.75*(e-v)])
        paired_path = main/spec['previous_output']/f'paired-oof-{seed}.npz'
        with np.load(paired_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['query_ids'], frame.sample_id.to_numpy(str))
            np.testing.assert_array_equal(saved['folds'], fv)
            np.testing.assert_array_equal(saved['EMA'], ref[:, 1])
            np.testing.assert_array_equal(saved['PTARL_RAW'], ptarl[seed])
        d = de3.loc[(de3.target == 'tap_iron') & (de3.split_seed == seed)]
        if d.sample_id.duplicated().any() or set(d.sample_id) != set(frame.sample_id):
            raise ValueError('DE3 ID/seed coverage mismatch')
        d = d.set_index('sample_id').loc[frame.sample_id]
        np.testing.assert_array_equal(d.fold.to_numpy(), fv)
        np.testing.assert_array_equal(d.actual.to_numpy(), y[:, 0])
        np.testing.assert_allclose(d.current_prediction.to_numpy(), iron, rtol=0, atol=1e-10)
        np.testing.assert_allclose(d.seed_42_prediction.to_numpy(), iron, rtol=0, atol=1e-10)
        de3_iron = d[['seed_42_prediction', 'seed_104729_prediction', 'seed_130363_prediction']].to_numpy().mean(axis=1)
        c = .8*b+.2*ptarl[seed]
        candidates = {'V32': np.column_stack([iron, b]),
            'EMA_TIME_Q50': np.column_stack([iron, b+.5*(e-v)]),
            'EMA_TIME_Q100': np.column_stack([iron, b+(e-v)]),
            'SAM_TIME_Q50': np.column_stack([iron, b+.5*(sam-v)]),
            'PTARL_TIME_Q20': np.column_stack([iron, c]),
            'EMA_IRON_EMA_TIME': np.column_stack([iron+.5*(iron_members[seed, 'tap_iron', 'EMA']-base[seed]['v12_iron']), ref[:, 1]]),
            'DE3_IRON_Q75_RESERVE': np.column_stack([de3_iron, ref[:, 1]]),
            'EMA_PTARL_ENDPOINT_HALF': np.column_stack([iron, .5*ref[:, 1]+.5*c]),
            'EMA_PLUS_PTARL_INCREMENT_Q20': np.column_stack([iron, ref[:, 1]+.2*(ptarl[seed]-b)])}
        check_vectors(y, ref)
        for p in candidates.values():
            check_vectors(y, p)
        bundles[seed] = dict(y=y, ref=ref, candidates=candidates, folds=fv)
    return frame, bundles


def run(spec_path, workspace):
    import os
    import yaml
    from .v49_run import check_runtime
    spec_path, workspace = Path(spec_path).resolve(), Path(workspace).resolve()
    spec = json.loads(spec_path.read_text())
    if tuple(spec['candidates']) != CANDIDATES or spec['seeds'] != [42, 3407] or spec['new_fits'] != 0:
        raise ValueError('Frozen diagnostic scope changed')
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python3.12 required')
    runtime_spec = yaml.safe_load((Path(spec['main_root'])/'configs/strong_component_regularization/SPEC.yaml').read_text())
    versions = check_runtime(runtime_spec)
    import torch
    torch.set_num_threads(1)
    out = Path(spec['main_root'])/spec['output']
    if not out.resolve().is_relative_to(Path(spec['main_root'])/'local/runs'):
        raise ValueError('Private output required')
    out.mkdir(parents=True, exist_ok=False)
    files = {str(p): sha(p) for p in [spec_path, workspace/'docs/q75_error_relocation/PREREGISTRATION.md',
                                     Path(__file__), workspace/'scripts/q75_error_relocation.py',
                                     workspace/'scripts/audit_q75_error_relocation.py']}
    write_new(out/'manifest.json', dict(spec=spec, execution_files=files, versions=versions,
              threads={k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    try:
        frame, bundles = load_inputs(spec)
        report = dict(reference=spec['reference'], seeds={}, new_fits=0, new_split_seeds=0, packages=0,
                      desktop_writes=0, agent_uploads=0, G0='audit_pending', G1='descriptive_two_seeds_no_promotion',
                      error_sign='prediction-actual', cross_seed_prediction_averaging=False)
        top_sets = {}
        for seed, a in bundles.items():
            y, ref, fv = a['y'], a['ref'], a['folds']
            den = y.sum(axis=0)
            loss = 50*np.abs(y-ref)/den
            regions, thresholds = input_regions(frame, fv, spec['conditional_features'])
            rows = pd.DataFrame({'sample_id': frame.sample_id, 'split_seed': seed, 'fold': fv, 'spout_no': frame.spout_no})
            for k, label in enumerate(('iron', 'time')):
                rows[f'true_{label}'] = y[:, k]
                rows[f'prediction_{label}'] = ref[:, k]
                rows[f'error_{label}'] = ref[:, k]-y[:, k]
                rows[f'residual_{label}'] = y[:, k]-ref[:, k]
                rows[f'relative_error_{label}'] = np.abs(y[:, k]-ref[:, k])/y[:, k]
                rows[f'score_loss_{label}'] = loss[:, k]
            rows['score_loss_combined'] = loss.sum(axis=1)
            record = dict(rows=len(frame), folds={str(f): int((fv == f).sum()) for f in range(5)},
                denominators=den.tolist(), score=float(100-loss.sum()), input_thresholds=thresholds,
                regions={name: residual_summary(y, ref, m, den) for name, m in regions.items()},
                concentration={}, prediction_bins={}, candidates={})
            signed, absolute = y-ref, np.abs(y-ref)
            record['target_residual_correlation'] = dict(signed=float(np.corrcoef(signed.T)[0, 1]),
                                                        absolute=float(np.corrcoef(absolute.T)[0, 1]))
            for k, label in enumerate(('iron', 'time', 'combined')):
                values = loss[:, k] if k < 2 else loss.sum(axis=1)
                record['concentration'][label] = concentration(values)
                order = np.argsort(-values, kind='stable')
                top_sets[seed, label] = set(frame.sample_id.iloc[order[:20]])
                rows.iloc[order[:20]].to_csv(out/f'top20-{seed}-{label}.csv', index=False, mode='x')
                pd.DataFrame({'rank': np.arange(1, len(frame)+1), 'cumulative_share': np.cumsum(values[order])/values.sum()}).to_csv(
                    out/f'concentration-{seed}-{label}.csv', index=False, mode='x')
            for k, label in enumerate(('iron', 'time')):
                cuts = np.quantile(ref[:, k], [.25, .5, .75])
                q = np.searchsorted(cuts, ref[:, k], side='right')
                record['prediction_bins'][label] = dict(cuts=cuts.tolist(), interpretation='posthoc_not_calibration_training',
                    groups={f'Q{i+1}': residual_summary(y, ref, q == i, den) for i in range(4)})
            for name, candidate in a['candidates'].items():
                reduction = np.abs(y-ref)-np.abs(y-candidate)
                contributions = 50*(reduction/den).sum(axis=1)
                for k, label in enumerate(('iron', 'time')):
                    rows[f'{name}_prediction_{label}'] = candidate[:, k]
                    rows[f'{name}_error_reduction_{label}'] = reduction[:, k]
                rows[f'{name}_score_gain'] = contributions
                detail = movement_summary(y, ref, candidate, np.ones(len(frame), bool), den)
                detail['fold_gains'] = [movement_summary(y, ref, candidate, fv == f, den)['gain_region_denominator'] for f in range(5)]
                detail['regions'] = {n: movement_summary(y, ref, candidate, m, den) for n, m in regions.items()}
                record['candidates'][name] = detail
                positive = np.argsort(-contributions, kind='stable')[:20]
                negative = np.argsort(contributions, kind='stable')[:20]
                for direction, idx in [('positive', positive), ('negative', negative)]:
                    pd.DataFrame({'sample_id': frame.sample_id.iloc[idx].to_numpy(), 'score_gain': contributions[idx]}).to_csv(
                        out/f'candidate-top20-{seed}-{name}-{direction}.csv', index=False, mode='x')
            rows.to_csv(out/f'error-map-{seed}.csv', index=False, mode='x')
            np.savez_compressed(out/f'error-map-{seed}.npz', query_ids=frame.sample_id.to_numpy(str), y=y,
                                reference=ref, folds=fv, spout=frame.spout_no.to_numpy(), **a['candidates'])
            report['seeds'][str(seed)] = record
        report['top20_overlap_count'] = {label: len(top_sets[42, label] & top_sets[3407, label]) for label in ('iron', 'time', 'combined')}
        verify_files(Path(spec['main_root']), spec['inputs'])
        for path, expected in files.items():
            if sha(path) != expected:
                raise ValueError('Execution source changed')
        report['artifact_hashes'] = {p.name: sha(p) for p in out.iterdir() if p.name != 'manifest.json'}
        write_new(out/'report.json', report)
        return report
    except BaseException as error:
        write_new(out/'failure.json', dict(error=repr(error)))
        raise
