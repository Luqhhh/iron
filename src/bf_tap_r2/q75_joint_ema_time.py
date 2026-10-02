"""Prospective zero-fit review of retained joint EMA time predictions."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import yaml

from .candidate_tiers import classify_candidates
from .component_regularization import ComponentRegressor
from .component_regularization_audit import verify_saved
from .data import TARGETS
from .ema_average_span import old_cache, task_frames
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .v3_4_bags import group_safe_inner_folds
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import paired_summary
from .v5_spec import load_v5_spec
from .v7_periodic import digest
from .v49_run import metric_detail, unit_id, verified_unit

SPEC = 'configs/q75_joint_ema_time/SPEC.json'
PROTOCOL = 'docs/q75_joint_ema_time/PREREGISTRATION.md'
ARMS = ('JOINT_BASE', 'JOINT_EMA')


def compose(q75, single, joint):
    q75, single, joint = [np.asarray(a, dtype=float) for a in (q75, single, joint)]
    if q75.ndim != 1 or q75.shape != single.shape or q75.shape != joint.shape:
        raise ValueError('Aligned single-split vectors required')
    if not all(np.isfinite(a).all() for a in (q75, single, joint)):
        raise ValueError('Nonfinite component')
    prediction = q75 + .2 * (joint - single)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid affine prediction; no clipping')
    return prediction


def choose(gains, paired):
    if set(gains) != {'42', '3407'} or set(paired) != set(gains):
        raise ValueError('Two complete declared splits required')
    if not all(math.isfinite(v) for v in [*gains.values(), *paired.values()]):
        raise ValueError('Finite paired gains required')
    return all(v > 0 for v in [*gains.values(), *paired.values()])


def freeze(work, out):
    spec = json.loads((work / SPEC).read_text())
    main = Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main / 'local/runs'):
        raise ValueError('Fresh private run required')
    versions = runtime(spec)
    current = json.loads((main / 'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current[k] != v for k, v in spec['reference'].items()):
        raise ValueError('Current reference changed')
    old = main / spec['old_development']
    manifest = json.loads((old / 'manifest.json').read_text())
    audit = json.loads((old / 'audit.json').read_text())
    if (audit['status'] != 'passed' or audit['manifest_sha256'] != sha(old / 'manifest.json')
            or audit['summary_sha256'] != sha(old / 'summary.json')):
        raise ValueError('Original audited cache required')
    for name in ('component_regularization.py', 'v12_joint.py', 'v7_periodic.py', 'v3_6_networks.py'):
        relative = 'src/bf_tap_r2/' + name
        if sha(work / relative) != manifest['source_hashes'][relative]:
            raise ValueError('Original scientific source changed: ' + name)
    paths = list((work / 'src').rglob('*.py'))
    paths += [work / p for p in [SPEC, PROTOCOL, 'uv.lock', 'pyproject.toml',
        'tests/test_q75_joint_ema_time.py', 'scripts/observe_q75_joint_ema_time.py',
        'scripts/observe_ema_fusion_selection.py']]
    paths += [main / p for p in ['configs/protection.yaml', 'configs/data.local.yaml',
        'configs/round2_v5/SPEC.yaml', 'configs/strong_component_regularization/SPEC.yaml',
        'configs/candidate_tiers.yaml', 'local/authorizations/optimization-standing-20261002-r1.json']]
    paths += [old / p for p in ['manifest.json', 'audit.json', 'summary.json',
                               'audit-recovery-provenance.json']]
    inputs = {str(main / p): h for p, h in manifest['data_hashes'].items()}
    for seed in spec['split_seeds']:
        paths.append(main / f'local/runs/round2-v2/comparison-r1/folds-{seed}.csv')
        for fold in range(5):
            for key in (f'reference-s{seed}-f{fold}', f'tap_time_len-EMA-s{seed}-f{fold}',
                        f'tap_iron-BASE-s{seed}-f{fold}', f'tap_iron-EMA-s{seed}-f{fold}'):
                if not verified_unit(old / key, unit_id(manifest, key)):
                    raise ValueError('Incomplete original unit: ' + key)
                paths.extend(p for p in (old / key).iterdir() if p.is_file())
    inputs.update({str(p): sha(p) for p in paths})
    verify_files(inputs)
    out.mkdir(parents=True, exist_ok=False)
    frozen = dict(spec=spec, workspace=str(work), files=inputs, versions=versions,
        python=sys.version, frozen_ns=time.time_ns(), old_manifest_sha256=sha(old / 'manifest.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=work, text=True).strip())
    write_new(out / 'manifest.json', frozen)
    return frozen


def context(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    verify_files(manifest['files'])
    spec = manifest['spec']
    runtime(spec)
    import torch
    torch.set_num_interop_threads(1)
    if spec['transfer_weight'] != .2 or spec['arms'] != list(ARMS) or spec['split_seeds'] != [42, 3407]:
        raise ValueError('Frozen scientific scope changed')
    return manifest, spec


def no_fit(*args, **kwargs):
    raise RuntimeError('This review permits no scientific fitting')


def cold_unit(directory, training, query, target, arm, spec, original):
    metadata = json.loads((directory / 'metadata.json').read_text())
    settings = original['training'][target]
    if (metadata['training_seed'] != 42 or metadata['fit_ids_digest'] != digest(training.sample_id.tolist())
            or metadata['target'] != target or metadata['arm'] != arm):
        raise ValueError('Cached model identity mismatch')
    inner = np.asarray(group_safe_inner_folds(training, seed=settings['inner_seed'])['fold'])
    cols = list(TARGETS) if target == 'tap_iron' else ['tap_time_len']
    # Preserve the native NumPy slice used by JointRegressor.fit. Rebuilding y
    # through a DataFrame changes reduction order in the historical joint model.
    y = training[cols].to_numpy(dtype=float)
    fitting = training.loc[inner != 0].reset_index(drop=True)
    validation = training.loc[inner == 0]
    selector = verify_saved(directory / 'selection.pt', fitting, y[inner != 0], arm,
                            settings, original['mechanisms'], validation)
    model = verify_saved(directory / 'refit.pt', training, y, arm, settings,
        original['mechanisms'], expected_epoch=selector.saved['trace']['selected_epoch'])
    for phase, state in [('selection', selector), ('refit', model)]:
        if state.saved['trace'] != metadata['model']['traces'][phase]:
            raise ValueError('Native trace identity mismatch')
    p = model.predict(query)
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['query_ids'], query.sample_id.to_numpy(str))
        np.testing.assert_array_equal(saved['prediction'], p)
    variants = [model.predict(query.iloc[::-1])[::-1],
                np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0, len(query), 37)])]
    delta = max(float(np.max(np.abs(v - p))) for v in variants)
    if delta > spec['cold_predict_atol']:
        raise ValueError('Cold order/chunk difference exceeded frozen tolerance')
    return p, dict(directory=str(directory), selected_epoch=selector.saved['trace']['selected_epoch'],
                   states=2, full_batch_difference=0., order_chunk_difference=delta)


def evaluate(out):
    manifest, spec = context(out)
    ComponentRegressor.fit = no_fit
    ComponentRegressor._train = no_fit
    main = Path(spec['main_root'])
    old = main / spec['old_development']
    original = yaml.safe_load((main / 'configs/strong_component_regularization/SPEC.yaml').read_text())
    with (out / 'access.jsonl').open('x') as stream:
        stream.write(json.dumps(dict(time_ns=time.time_ns(), scope='authorized_round2_v2_only',
            manifest_sha256=sha(out / 'manifest.json'), new_fits=0, protected_prelim_targets_read=False)) + '\n')
    frame = load_v5_training_frame(main)
    if len(frame) != 2754 or frame.sample_id.duplicated().any():
        raise ValueError('Expected complete unique round2 training frame')
    metrics = {'tap_time_len': {name: {} for name in ['Q75', *ARMS]}}
    records = {arm: {} for arm in ARMS}
    cold = []
    for seed in spec['split_seeds']:
        fv = fold_vector(main, frame, seed, load_v5_spec(main))
        if set(fv) != set(range(5)):
            raise ValueError('Incomplete fold inventory')
        columns = {name: np.full(len(frame), np.nan) for name in ['Q75', 'single', 'joint_base', 'joint_ema', 'iron']}
        for fold in range(5):
            mask = fv == fold
            training, query = task_frames(frame, fv, fold)
            base = old_cache(spec, seed, fold, training, query)
            columns['Q75'][mask] = base['q75']
            columns['iron'][mask] = base['iron']
            for target, arm, name in [('tap_iron', 'BASE', 'joint_base'),
                                      ('tap_iron', 'EMA', 'joint_ema'),
                                      ('tap_time_len', 'EMA', 'single')]:
                directory = old / f'{target}-{arm}-s{seed}-f{fold}'
                meta = json.loads((directory / 'metadata.json').read_text())
                if meta['seed'] != seed or meta['fold'] != fold:
                    raise ValueError('Cached split identity mismatch')
                p, receipt = cold_unit(directory, training, query, target, arm, spec, original)
                columns[name][mask] = p[:, 1 if target == 'tap_iron' else 0]
                cold.append(receipt)
            np.testing.assert_array_equal(columns['single'][mask], base['ema'])
        if not all(np.isfinite(v).all() for v in columns.values()):
            raise ValueError('Incomplete same-seed OOF coverage')
        y = frame.tap_time_len.to_numpy(float)
        spout = frame.spout_no.to_numpy()
        metrics['tap_time_len']['Q75'][str(seed)] = metric_detail(y, columns['Q75'], fv, spout)
        for arm, endpoint in zip(ARMS, ['joint_base', 'joint_ema']):
            p = compose(columns['Q75'], columns['single'], columns[endpoint])
            columns[arm] = p
            md = metric_detail(y, p, fv, spout)
            metrics['tap_time_len'][arm][str(seed)] = md
            contribution = np.abs(y - columns['Q75']) - np.abs(y - p)
            denominator = float(np.abs(y).sum())
            records[arm][str(seed)] = dict(gain=float(50 * contribution.sum() / denominator),
                positive_contribution=float(50 * contribution[contribution > 0].sum() / denominator),
                negative_contribution=float(50 * contribution[contribution < 0].sum() / denominator),
                improved_rows=int((contribution > 0).sum()), rows=len(y),
                standalone_wmape=float(np.abs(y - columns[endpoint]).sum() / denominator))
        with (out / f'oof-{seed}.npz').open('xb') as stream:
            np.savez_compressed(stream, **columns, y=y, folds=fv, spout=spout,
                ids=frame.sample_id.to_numpy(str))
    gain = {s: r['gain'] for s, r in records['JOINT_EMA'].items()}
    paired = {s: gain[s] - records['JOINT_BASE'][s]['gain'] for s in gain}
    policy = yaml.safe_load((main / 'configs/candidate_tiers.yaml').read_text())
    report = dict(records=records, paired_control_gains=paired,
        seed_summary=paired_summary(list(gain.values())), metrics=metrics,
        tiers=classify_candidates(metrics, spec, policy),
        confirmation_eligible=choose(gain, paired), formal_promotion=False,
        cold_states=sum(v['states'] for v in cold), cold=cold,
        maximum_cold_difference=max(v['order_chunk_difference'] for v in cold),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        manifest_sha256=sha(out / 'manifest.json'), new_fits=0, packages=0,
        G0='native_cold_passed_independent_scalar_and_actual_exit_pending', G1='two_development_splits_only')
    if report['cold_states'] != 60 or report['peak_rss_mib'] > spec['max_worker_rss_mib']:
        raise ValueError('Frozen state inventory or memory gate failed')
    verify_files(manifest['files'])
    write_new(out / 'report.json', report)


def audit(out):
    manifest, spec = context(out)
    report = json.loads((out / 'report.json').read_text())
    deltas, gains, pairs = [], {}, {}
    for seed in spec['split_seeds']:
        with np.load(out / f'oof-{seed}.npz', allow_pickle=False) as d:
            if len(d['ids']) != 2754 or len(set(d['ids'])) != 2754:
                raise ValueError('Independent OOF row coverage failed')
            if set(d['folds']) != set(range(5)):
                raise ValueError('Independent fold coverage failed')
            y = d['y'].tolist()
            denom = math.fsum(abs(v) for v in y)
            base_error = math.fsum(abs(a-b) for a,b in zip(y,d['Q75']))
            computed = {}
            for arm, endpoint in zip(ARMS, ['joint_base', 'joint_ema']):
                p = [q + (j-s)/5 for q,j,s in zip(d['Q75'], d[endpoint], d['single'])]
                deltas.extend(abs(a-b) for a,b in zip(p,d[arm]))
                e = math.fsum(abs(a-b) for a,b in zip(y,p))
                g = 50 * (base_error-e) / denom
                computed[arm] = g
                deltas.append(abs(g-report['records'][arm][str(seed)]['gain']))
                for name, pred in [('Q75', d['Q75']), (arm, p)]:
                    row = report['metrics']['tap_time_len'][name][str(seed)]
                    masks = [('wmape', np.ones(len(y), bool))]
                    masks += [(('by_fold', str(f)), d['folds'] == f) for f in range(5)]
                    masks += [(('by_spout', str(s)), d['spout'] == s) for s in sorted(set(d['spout']))]
                    for key, mask in masks:
                        inds = np.flatnonzero(mask)
                        actual = math.fsum(abs(y[i]-pred[i]) for i in inds) / math.fsum(abs(y[i]) for i in inds)
                        expected = row[key] if isinstance(key, str) else row[key[0]][key[1]]
                        deltas.append(abs(actual-expected))
            gains[str(seed)] = computed['JOINT_EMA']
            pairs[str(seed)] = computed['JOINT_EMA']-computed['JOINT_BASE']
    eligible = all(v > 0 for v in [*gains.values(), *pairs.values()])
    if eligible != report['confirmation_eligible'] or max(deltas) > spec['scalar_atol']:
        raise ValueError('Independent score/decision audit failed')
    verify_files(manifest['files'])
    write_new(out / 'independent-score.json', dict(status='passed', maximum_difference=max(deltas),
        scalar_checks=len(deltas), report_sha256=sha(out / 'report.json'),
        manifest_sha256=sha(out / 'manifest.json'), confirmation_eligible=eligible, new_fits=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'evaluate', 'audit'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.phase == 'freeze':
        freeze(Path(__file__).resolve().parents[2], out)
    else:
        globals()[args.phase](out)


if __name__ == '__main__':
    main()
