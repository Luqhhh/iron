"""OPT-27: exact error budgets and paired component diagnostics; zero new fits."""
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json, verify_file_identities, stable_digest
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import META, PRED, read_json
from bf_tap.optimization.dual_ratio_common import frame, verify_manifest
from bf_tap.optimization.structural import select_oof
from bf_tap.optimization.recency_run import coefficient_certificate, archive
from bf_tap.optimization.rate_model import schema
from bf_tap.optimization.trajectory_run import load_fold, builder_for
from bf_tap.optimization.v13_common import freeze, verify, verify_receipt, zero_fit

TARGETS = ('tap_iron', 'tap_time_len')


def statistic(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or y.sum() <= 0:
        raise ContractError('aligned finite targets with positive denominator required')
    e = p-y
    return {'N': len(y), 'absolute_error_sum': float(np.abs(e).sum()), 'target_sum': float(y.sum()),
            'WMAPE': float(np.abs(e).sum()/y.sum()), 'signed_mean_residual': float(e.mean()),
            'median_residual': float(np.median(e)), 'signed_error_sum': float(e.sum())}


def budgets(errors, origins, candidates, expected_summary, tolerance=1e-12):
    required = {*META, *TARGETS, *PRED, 'candidate', 'origin', 'unit', 'horizon'}
    if required-set(errors) or errors.empty or errors[list(required-{'horizon'})].isna().any().any():
        raise ContractError('complete archived errors required')
    if errors.duplicated(['candidate', 'unit', 'sample_id']).any():
        raise ContractError('duplicate ID inside a prediction unit')
    if set(errors.candidate) != set(candidates):
        raise ContractError('candidate coverage differs')
    canonical = errors.groupby('sample_id')
    if (canonical[[*META[1:], *TARGETS]].nunique(dropna=False) > 1).any().any():
        raise ContractError('repeated sample labels or metadata differ')
    expected_units = {f'O2024{m:02d}_H{h}' for m, n in origins.items() for h in range(1, n+1)} | {'DEV_LONG', 'DEV_SHORT'}
    pieces, units_out, top, unit_ids = [], {}, [], {}
    for candidate in candidates:
        rows = errors.loc[errors.candidate == candidate]
        if set(rows.unit) != expected_units:
            raise ContractError('incomplete evaluation grid')
        units_out[candidate] = {}
        for unit, g in rows.groupby('unit', sort=True):
            sorted_ids = sorted(g.sample_id.astype(str))
            if unit in unit_ids and unit_ids[unit] != sorted_ids:
                raise ContractError('candidate sample identity differs')
            unit_ids[unit] = sorted_ids
            is_grid = unit.startswith('O')
            h = int(unit[-1]) if is_grid else None
            if is_grid and (not (g.horizon == h).all() or not (g.origin == unit[:7]).all()):
                raise ContractError('unit horizon/origin certificate differs')
            n_h = sum(n >= h for n in origins.values()) if is_grid else None
            m = {}
            for target in TARGETS:
                stat = statistic(g[target], g['pred_'+target]); m[target] = stat
                z = g[[*META, 'candidate', 'origin', 'unit', 'horizon', target, 'pred_'+target]].copy()
                z = z.rename(columns={target: 'target_value', 'pred_'+target: 'prediction'})
                z['target'] = target; z['residual'] = z.prediction-z.target_value
                z['absolute_error'] = z.residual.abs(); z['unit_target_denominator'] = stat['target_sum']
                z['J_contribution'] = z.absolute_error/(8*n_h*stat['target_sum']) if is_grid else 0.0
                z['included_in_J'] = is_grid
                pieces.append(z)
                order = z.sort_values(['absolute_error', 'sample_id'], ascending=[False, True], kind='mergesort')
                for fraction in (.01, .05, .10):
                    selected = order.head(max(1, math.ceil(len(z)*fraction)))
                    top.append({'candidate': candidate, 'unit': unit, 'target': target, 'fraction': fraction,
                                'rows': len(selected), 'unique_sample_ids': selected.sample_id.nunique(),
                                'absolute_error_share': float(selected.absolute_error.sum()/z.absolute_error.sum()) if z.absolute_error.sum() else 0.0})
            m['E'] = .5*(m[TARGETS[0]]['WMAPE']+m[TARGETS[1]]['WMAPE']); units_out[candidate][unit] = m
    contributions = pd.concat(pieces, ignore_index=True)
    j, checks = {}, {}
    for candidate in candidates:
        grouped = units_out[candidate]
        horizons = {str(h): float(np.mean([v['E'] for k, v in grouped.items() if k.startswith('O') and k.endswith(f'_H{h}')])) for h in range(1, 5)}
        actual = float(np.mean(list(horizons.values())))
        total = float(contributions.loc[(contributions.candidate == candidate) & contributions.included_in_J, 'J_contribution'].sum())
        expected = expected_summary[candidate]['J']
        if abs(actual-expected) > tolerance or abs(total-actual) > tolerance:
            raise ContractError('E/J contribution reconstruction differs')
        j[candidate] = {'J': actual, 'J_from_contributions': total, 'horizon_mean_E': horizons,
                        'Delta_J_V1': actual-expected_summary['V1']['J']}
        checks[candidate] = {'J_error': abs(actual-expected), 'contribution_sum_error': abs(total-actual)}
    grid = contributions.loc[contributions.included_in_J]
    for candidate in candidates:
        left = grid.loc[grid.candidate == candidate].set_index(['unit', 'sample_id', 'target']).sort_index()
        right = grid.loc[grid.candidate == 'V1'].set_index(['unit', 'sample_id', 'target']).sort_index()
        if not left.index.equals(right.index) or not np.array_equal(left.target_value, right.target_value):
            raise ContractError('paired contribution identity differs')
        delta = float((left.J_contribution-right.J_contribution).sum())
        if abs(delta-j[candidate]['Delta_J_V1']) > tolerance:
            raise ContractError('paired Delta J contributions differ')
        checks[candidate]['Delta_J_contribution_error'] = abs(delta-j[candidate]['Delta_J_V1'])
    return contributions, {'units': units_out, 'candidates': j, 'checks': checks,
                            'top_error_by_unit': top, 'unique_grid_samples': grid.sample_id.nunique(),
                            'grid_prediction_exposures_per_candidate': int(grid.loc[grid.candidate == 'V1'].sample_id.count()/2),
                            'scope': 'CONSUMED_RETROSPECTIVE_SHARED_SAMPLES_NOT_INDEPENDENT'}


def binned(values, boundaries):
    if not boundaries or boundaries != sorted(set(boundaries)):
        raise ContractError('predeclared increasing bins required')
    return pd.cut(values, [-np.inf, *boundaries, np.inf], right=False).astype(str).where(values.notna(), 'MISSING')


def grouped_statistics(contributions, columns):
    records = []
    for keys, g in contributions.groupby(columns, dropna=False, sort=True):
        record = dict(zip(columns, keys if isinstance(keys, tuple) else (keys,)))
        record.update(statistic(g.target_value, g.prediction));record['unique_sample_ids'] = g.sample_id.nunique()
        record['J_contribution_sum'] = float(g.J_contribution.sum()); records.append(record)
    return pd.DataFrame(records)


def direct_diagnostic(errors, v12, component_path):
    components = frame(component_path); output = {}
    for unit, rows in errors.loc[errors.candidate == 'V1'].groupby('unit'):
        month = int(rows.origin.iloc[0][-2:]);old_base = frame(Path(v12['registration']['source_v8'])/'predictions'/f'{month}_inputs.csv')
        new_parts = frame(Path(v12['registration']['source_v12'])/'predictions'/f'{month}_parts.csv') if 'source_v12' in v12['registration'] else None
        if new_parts is None:
            raise ContractError('explicit new parts source required')
        old_direct = components.loc[(components.unit == unit) & (components.role == 'OR')]
        new_corrected = errors.loc[(errors.unit == unit) & (errors.candidate == 'V6B_RECENCY60_BOTH')]
        aligned = [x.set_index('sample_id').loc[rows.sample_id] for x in [old_base, new_parts, new_corrected]]
        out = {}
        for short, target in [('iron', 'tap_iron'), ('time', 'tap_time_len')]:
            y = rows[target].to_numpy(); ob, nparts, nc = aligned
            old = {'base': statistic(y, ob['pred_'+target]), 'corrected': statistic(y, rows['pred_'+target])}
            if len(old_direct) == len(rows) and set(old_direct.sample_id) == set(rows.sample_id):
                old['direct'] = statistic(y, old_direct.set_index('sample_id').loc[rows.sample_id, 'pred_'+target])
            else:
                old['direct'] = {'status': 'MISSING_SOURCE_NO_INFERENCE_FROM_FINAL_SCORES'}
            new = {'direct': statistic(y, nparts['direct_'+short]), 'base': statistic(y, nparts['base_'+short]),
                   'corrected': statistic(y, nc['pred_'+target])}
            deltas = {stage: new[stage]['WMAPE']-old[stage]['WMAPE'] for stage in ('direct', 'base', 'corrected') if 'WMAPE' in old[stage]}
            out[target] = {'old': old, 'new_recency60_closed': new, 'Delta_WMAPE': deltas,
                           'structural_effect_on_delta': deltas['corrected']-deltas['base']}
        output[unit] = out
    return {'scope': 'PAIRED_DIAGNOSTICS_ONLY_NOT_CANDIDATES', 'units': output}


def feature_diagnostic(errors, v12, root, config):
    builder = builder_for(v12); features = []; identities = []
    for month in config['origins']:
        fold, _ = load_fold(v12, month); samples = errors.loc[(errors.candidate == 'V1') & (errors.origin == f'O2024{month:02d}'), META].drop_duplicates('sample_id')
        for role, (model, history) in fold[1].items():
            x = builder.X(samples, fold[0][role], history)
            if schema(x) != model.feature_schema_:
                raise ContractError('as-of diagnostic schema differs')
            row = samples[['sample_id']].copy();row['origin'] = f'O2024{month:02d}';row['role'] = role
            numeric = x.select_dtypes(include='number')
            row['missing_fraction'] = numeric.isna().mean(axis=1).to_numpy()
            selected = [c for c in x if c.endswith(('__missing', '__stale', '__event_age_hours', '__latest_available_age_minutes'))]
            for c in selected: row[c] = x[c].to_numpy()
            features.append(row)
            identities.append({'origin': f'O2024{month:02d}', 'role': role, 'schema': model.feature_schema_,
                               'schema_sha256': stable_digest(schema(x)), 'history_available_max': str(history.available_at.max())})
        builder.cache.clear()
        print(f'OPT27 as-of source diagnostic origin {month}: fits=0', flush=True)
    feat = pd.concat(features, ignore_index=True)
    if feat.duplicated(['origin', 'sample_id', 'role']).any():
        raise ContractError('duplicate diagnostic feature identity')
    feat.to_csv(root/'asof_feature_states.csv', index=False)
    # Fixed bins only; views keep distinct origin predictions and report unique IDs.
    joined = errors.loc[errors.candidate == 'V1'].merge(feat, on=['origin', 'sample_id'], validate='many_to_many')
    rows = []
    fields = {'missing_fraction': config['feature_bins']['missing_fraction']}
    fields.update({c: config['feature_bins']['history_age_hours'] for c in feat if c.endswith('__latest_available_age_minutes')})
    fields.update({c: config['feature_bins']['flags'] for c in feat if c.endswith('__stale')})
    for c, bins in fields.items():
        values = joined[c]/60 if c.endswith('__latest_available_age_minutes') else joined[c]
        z = joined.assign(feature_bin=binned(values, bins))
        for (role, unit, bin_name), g in z.groupby(['role', 'unit', 'feature_bin']):
            for target in TARGETS:
                rows.append({'role': role, 'unit': unit, 'feature': c, 'bin': bin_name, 'target': target,
                             'unique_sample_ids': g.sample_id.nunique(), **statistic(g[target], g['pred_'+target])})
    pd.DataFrame(rows).to_csv(root/'feature_state_errors.csv', index=False)
    atomic_write_json(root/'feature_provenance.json', {'identities': identities, 'bins_frozen_before_read': True})



def failure_evidence(root, budget, paired):
    lines = ['# OPT-27 failure mode evidence', '',
             'Consumed retrospective diagnostics; shared samples are not independent evidence. No labels were removed and no predictions changed.', '',
             '| Observation | Numerical evidence | Supported interpretation | Missing evidence / unsupported conclusion |',
             '| --- | --- | --- | --- |']
    for item in budget['top_unique_sample_J_contributors']:
        if item['candidate'] == 'V1' and item['fraction'] == .10:
            lines.append(f"| {item['target']} top 10% unique IDs by accumulated J contribution | {item['unique_sample_ids']} unique IDs, {item['prediction_exposures']} prediction exposures; {item['J_target_error_share']:.6%} of target J error | Error concentration deserves analysis | Does not establish label error or justify removal/routing |")
    for target in TARGETS:
        cells = [v[target] for k, v in paired['units'].items() if k.endswith('_H1')]
        deltas = {stage: float(np.mean([v['Delta_WMAPE'][stage] for v in cells]))
                  for stage in ('direct', 'base', 'corrected') if all(stage in v['Delta_WMAPE'] for v in cells)}
        effect = float(np.mean([v['structural_effect_on_delta'] for v in cells]))
        lines.append(f"| {target} H1 paired recency60 layer changes | mean Delta WMAPE {deltas}; structural effect {effect:+.10f} | Identifies where paired error changes are retained or offset | Does not authorize raw/base release or a new LAD/calibration variant |")
        unit_stats = [v[target] for k, v in budget['units']['V1'].items() if k.endswith('_H1')]
        positive_means = sum(v['signed_mean_residual'] > 0 for v in unit_stats)
        positive_medians = sum(v['median_residual'] > 0 for v in unit_stats)
        lines.append(f"| {target} H1 residual signs | positive signed means {positive_means}/6; positive medians {positive_medians}/6 | Bias symptoms differ from tail sensitivity | No constant correction or optimal MAE calibration is inferred |")
    state = frame(root/'asof_feature_states.csv')
    ages = [c for c in state if c.endswith('__latest_available_age_minutes')]
    max_age = max((float(state[c].max()/60) for c in ages if state[c].notna().any()), default=None)
    stale = [c for c in state if c.endswith('__stale')]
    flags = {c: int((state[c] > 0).sum()) for c in stale}
    lines.append(f"| Frozen-history age and public-source status | max saved history age {max_age} hours; stale role-exposure counts {flags} | Existing as-of sources and frozen histories have measurable ages | Cannot distinguish availability uncertainty, conditional bias and irreducible extremes causally |")
    lines += ['', 'Detailed target_month_spout.csv and feature_state_errors.csv retain denominators, mean/median residuals and source bins. Source time semantics remain competition-timestamp-contract-v1 / ASSUMED. Test data were not used to select features or methods.', '',
              'A future intervention is not registered. Official second-round data identity and an explicit one-change experiment protocol are required before any training.']
    (root/'failure_mode_evidence.md').write_text('\n'.join(lines)+'\n')


def run(root):
    config = load_yaml('configs/optimization_v0_13/audit.yaml');src = Path(config['source_v12'])
    old = read_json(src/'manifest.json');verify_manifest(old)
    receipt = verify_receipt(config['v12_completion'])
    evidence = [src/'manifest.json', Path(config['v12_completion']), Path(config['component_export'])]
    evidence += list(map(Path, receipt['evidence_sha256']))
    source = Path(config['source_v8']);evidence += list((source/'predictions').glob('*.csv'))
    manifest = freeze(root, 'OPT27_EXACT_CONSUMED_ERROR_DIAGNOSTICS', evidence,
                      {k: v['path'] for k, v in old['inputs'].items()})
    root = Path(root)
    try:
        with zero_fit() as counts:
            errors = frame(src/'all_errors.csv'); errors['reference_time'] = pd.to_datetime(errors.reference_time).dt.tz_convert('Asia/Shanghai')
            start, end = [pd.Timestamp(manifest['scope'][key]) for key in ('reference_start', 'reference_end_exclusive')]
            if not ((errors.reference_time >= start) & (errors.reference_time < end)).all():
                raise ContractError('diagnostic label scope exceeded')
            contributions, budget = budgets(errors, config['origins'], config['candidates'], read_json(src/'summary.json'), config['metric_absolute_tolerance'])
            oldmetrics = read_json(src/'metrics.json')
            for candidate, units in budget['units'].items():
                for unit, result in units.items():
                    prior = oldmetrics[unit]['candidates'][candidate]['overall']
                    for target, short in zip(TARGETS, ('iron', 'time')):
                        if result[target]['target_sum'] != prior[short]['actual_sum'] or abs(result[target]['WMAPE']-prior[short]['wmape']) > config['metric_absolute_tolerance']:
                            raise ContractError('original target denominator/WMAPE differs')
                    if abs(result['E']-prior['loss']) > config['metric_absolute_tolerance']:
                        raise ContractError('original unit E differs')
            for month in config['origins']:
                selected = archive(src/'corrections'/f'{month}_OOF.csv')
                if not select_oof(selected, pd.Timestamp(f'2024-{month:02d}-01', tz='Asia/Shanghai')).sample_id.tolist() == selected.sample_id.tolist():
                    raise ContractError('coefficient OOF causal source differs')
                coefficient_certificate(selected, read_json(src/'corrections'/f'{month}.json')['beta'])
            contributions['calendar_month'] = contributions.reference_time.dt.strftime('%Y-%m')
            contributions.to_csv(root/'error_contributions.csv', index=False)
            groups = ['candidate', 'target', 'calendar_month', 'spout_no', 'horizon', 'included_in_J']
            grouped_statistics(contributions, groups).to_csv(root/'target_month_spout.csv', index=False)
            unique = contributions.loc[contributions.included_in_J].groupby(['candidate', 'target', 'sample_id']).agg(J_contribution=('J_contribution', 'sum'), exposures=('unit', 'size'), absolute_error=('absolute_error', 'sum')).reset_index()
            unique.to_csv(root/'unique_sample_contributions.csv', index=False)
            unique_top = []
            for (candidate, target), g in unique.groupby(['candidate', 'target']):
                ordered = g.sort_values(['J_contribution', 'sample_id'], ascending=[False, True])
                for fraction in config['top_error_fractions']:
                    selected = ordered.head(max(1, math.ceil(len(g)*fraction)))
                    unique_top.append({'candidate': candidate, 'target': target, 'fraction': fraction,
                                       'unique_sample_ids': len(selected), 'prediction_exposures': int(selected.exposures.sum()),
                                       'J_target_error_share': float(selected.J_contribution.sum()/g.J_contribution.sum())})
            budget['top_unique_sample_J_contributors'] = unique_top
            config_copy = {**old, 'registration': {**old['registration'], 'source_v12': str(src)}}
            paired = direct_diagnostic(errors, config_copy, config['component_export'])
            feature_diagnostic(errors, old, root, config)
        verify(manifest);verify_manifest(old)
        atomic_write_json(root/'error_budget.json', budget)
        atomic_write_json(root/'direct_structural_diagnostic.json', paired)
        failure_evidence(root, budget, paired)
        atomic_write_json(root/'validation.json', {'status': 'PASS_OPT27_ZERO_FIT', 'engineering_valid': True,
                          'quality_evaluated': False, 'new_candidate_quality_evaluated': False,
                          'consumed_historical_metrics_reconstructed': True, **counts})
        print('OPT27 full precision E/J, denominators and contributions PASS; fits=0', flush=True)
    except Exception as exc:
        atomic_write_json(root/'failure.json', {'exception': repr(exc), 'status': 'FAIL_PRESERVE_EVIDENCE'})
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output', required=True, type=Path);run(p.parse_args().output)
