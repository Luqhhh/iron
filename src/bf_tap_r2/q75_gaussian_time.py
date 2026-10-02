"""Fixed-current-reference review of the retained V33 Gaussian control."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .candidate_tiers import classify_candidates
from .data import FEATURES, TARGETS
from .ema_average_span import old_cache
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v7_periodic import digest
from .v33_mixture import MixtureRegressor
from .v33_run import metric_detail, partitions, unit_id, verified_unit

SPEC = 'configs/q75_gaussian_time/SPEC.json'
PROTOCOL = 'docs/q75_gaussian_time/PREREGISTRATION.md'
CANDIDATE = 'GAUSS1_A20'


def compose(base, member):
    base, member = np.asarray(base, float), np.asarray(member, float)
    if base.ndim != 1 or member.shape != base.shape:
        raise ValueError('One aligned split required')
    if not np.isfinite([base, member]).all():
        raise ValueError('Nonfinite component')
    result = .8 * base + .2 * member
    if (result < 0).any():
        raise ValueError('Negative endpoint; no clipping')
    return result


def selected_epoch(trace, settings):
    best, chosen, stale = math.inf, 0, 0
    for index, row in enumerate(trace, 1):
        if row['epoch'] != index or stale >= settings['patience'] or index > settings['max_epochs']:
            raise ValueError('Invalid selector epoch history')
        value = row['calibration_standardized_mae']
        if not math.isfinite(value):
            raise ValueError('Nonfinite selector metric')
        if value < best - settings['min_delta_standardized_mae']:
            best, chosen, stale = value, index, 0
        else:
            stale += 1
    if not chosen or (len(trace) != settings['max_epochs'] and stale != settings['patience']):
        raise ValueError('Selector stopped outside frozen rule')
    return chosen, best


def reference_order(ids, saved_ids):
    ids, saved_ids = list(map(str, ids)), list(map(str, saved_ids))
    if len(ids) != len(set(ids)) or len(saved_ids) != len(set(saved_ids)) or set(ids) != set(saved_ids):
        raise ValueError('Reference ID coverage mismatch')
    lookup = {value: index for index, value in enumerate(saved_ids)}
    return np.array([lookup[value] for value in ids])


def freeze(work, out):
    spec = json.loads((work / SPEC).read_text())
    main = Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main / 'local/runs'):
        raise ValueError('Fresh private run required')
    runtime(spec)
    current = json.loads((main / 'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current[k] != v for k, v in spec['reference'].items()):
        raise ValueError('Current platform reference changed')
    old, ref = main / spec['old_development'], main / spec['reference_run']
    manifest = json.loads((old / 'manifest.json').read_text())
    native_audit = json.loads((old / 'audit.json').read_text())
    if (native_audit['status'] != 'passed' or native_audit['manifest_sha256'] != sha(old / 'manifest.json')
            or native_audit['summary_sha256'] != sha(old / 'summary.json')):
        raise ValueError('Original audit identity mismatch')
    supplemental = json.loads((old / 'supplemental-audit.json').read_text())
    if supplemental['status'] != 'passed' or supplemental['summary_sha256'] != sha(old / 'summary.json'):
        raise ValueError('Original supplemental audit mismatch')
    reference_manifest = json.loads((ref / 'manifest.json').read_text())
    verify_files(reference_manifest['files'])
    terminal = json.loads((ref / 'terminal.json').read_text())
    if terminal['status'] != 'passed' or terminal['actual_exit_codes'] != [0, 0]:
        raise ValueError('Current Q75 cold audit missing')
    for name, field in [('manifest.json', 'manifest_sha256'), ('report.json', 'report_sha256'),
                        ('independent-score.json', 'independent_score_sha256')]:
        if sha(ref / name) != terminal[field]:
            raise ValueError('Q75 terminal binding mismatch')
    for rel in ['src/bf_tap_r2/' + name for name in ('v33_mixture.py', 'v33_run.py',
            'v3_6_networks.py', 'v3_4_bags.py', 'v7_periodic.py')] + ['configs/round2_v33/SPEC.yaml']:
        if sha(work / rel) != manifest['source_hashes'][rel]:
            raise ValueError('Original scientific source changed: ' + rel)
    paths = list((work / 'src').rglob('*.py'))
    paths += [work / name for name in (SPEC, PROTOCOL, 'uv.lock', 'pyproject.toml',
              'configs/round2_v33/SPEC.yaml', 'tests/test_q75_gaussian_time.py',
              'tests/test_round2_v33_mixture.py', 'scripts/review_q75_gaussian_time.py',
              'scripts/observe_ema_fusion_selection.py')]
    paths += [main / name for name in ('configs/protection.yaml', 'configs/data.local.yaml',
              'configs/round2_v5/SPEC.yaml', 'configs/candidate_tiers.yaml',
              'local/authorizations/optimization-standing-20261002-r1.json')]
    paths += [old / name for name in ('manifest.json', 'audit.json', 'supplemental-audit.json', 'summary.json')]
    paths += [ref / name for name in ('manifest.json', 'report.json', 'terminal.json', 'independent-score.json')]
    for seed in spec['split_seeds']:
        paths += [ref / f'oof-{seed}.npz', main / f'local/runs/round2-v2/comparison-r1/folds-{seed}.csv']
        for fold in range(5):
            key = f'tap_time_len-GAUSS1-s{seed}-f{fold}'
            if not verified_unit(old / key, unit_id(manifest, key)):
                raise ValueError('Missing original cached unit')
            paths.extend(path for path in (old / key).iterdir() if path.is_file())
    files = {**reference_manifest['files'], **{str(main / p): h for p, h in manifest['data_hashes'].items()},
             **{str(path): sha(path) for path in paths}}
    verify_files(files)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'manifest.json', dict(spec=spec, workspace=str(work), files=files,
        frozen_ns=time.time_ns(), python=sys.version,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=work, text=True).strip()))


def context(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    verify_files(manifest['files'])
    spec = manifest['spec']
    runtime(spec)
    if spec['split_seeds'] != [42, 3407] or spec['weight'] != .2:
        raise ValueError('Scientific scope changed')
    import torch
    torch.set_num_interop_threads(1)
    return manifest, spec


def forbidden(*args, **kwargs):
    raise RuntimeError('Zero-fit review forbids training or optimization')


def cold_state(path, training, query, target, meta, settings, expected):
    model = MixtureRegressor.load(path)
    if model.recipe != 'GAUSS1' or model.settings != settings:
        raise ValueError('Model settings mismatch')
    if (meta['fit_ids_digest'] != digest(training.sample_id.astype(str).tolist())
            or meta['fit_rows'] != len(training) or model.preprocessor_.metadata() != meta['preprocessing']):
        raise ValueError('Training/preprocessing identity mismatch')
    x, y = training[list(FEATURES)].to_numpy(float), training[target].to_numpy(float)
    for actual, expected_stat in [(model.preprocessor_.means_, x.mean(0)), (model.preprocessor_.stds_, x.std(0)),
                                 (model.mean_, y.mean()), (model.std_, y.std())]:
        np.testing.assert_allclose(actual, expected_stat, rtol=1e-12, atol=1e-10)
    vocabulary = {int(value): i + 1 for i, value in enumerate(sorted(training.spout_no.unique()))}
    if model.preprocessor_.spout_to_index_ != vocabulary or model.preprocessor_.n_spout_categories_ != len(vocabulary) + 1:
        raise ValueError('Training vocabulary mismatch')
    if model.mean_ != meta['target_mean'] or model.std_ != meta['target_std']:
        raise ValueError('Saved target statistics mismatch')
    if any(not np.isfinite(value.detach().cpu().numpy()).all() for value in model.model_.state_dict().values()):
        raise ValueError('Nonfinite checkpoint')
    prediction = model.predict(query)
    alternatives = [prediction, model.predict(query.iloc[::-1])[::-1],
                    np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0, len(query), 37)])]
    difference = max(float(np.max(np.abs(value - expected))) for value in alternatives)
    return prediction, difference


def evaluate(out):
    manifest, spec = context(out)
    import torch
    MixtureRegressor.initialize = MixtureRegressor.train = NumericPreprocessor.fit = forbidden
    torch.optim.AdamW = torch.optim.Adam = forbidden
    main = Path(spec['main_root'])
    old, ref = main / spec['old_development'], main / spec['reference_run']
    native = yaml.safe_load((Path(manifest['workspace']) / 'configs/round2_v33/SPEC.yaml').read_text())
    original = json.loads((old / 'manifest.json').read_text())
    reference_spec = json.loads((ref / 'manifest.json').read_text())['spec']
    identity = dict(original)
    identity.pop('identity')
    if digest(identity) != original['identity']:
        raise ValueError('Original manifest identity mismatch')
    with (out / 'access.jsonl').open('x') as stream:
        stream.write(json.dumps(dict(time_ns=time.time_ns(), scope='authorized_round2_only',
            manifest_sha256=sha(out / 'manifest.json'), protected_prelim_targets_read=False, new_fits=0)) + '\n')
    frame = load_v5_training_frame(main)
    if len(frame) != 2754 or frame.sample_id.duplicated().any():
        raise ValueError('Complete unique round2 frame required')
    metrics, records, cold = {'tap_time_len': {'Q75': {}, CANDIDATE: {}}}, {}, []
    for seed in spec['split_seeds']:
        folds = fold_vector(main, frame, seed, load_v5_spec(main))
        if digest(folds.tolist()) != original['fold_hashes'][str(seed)]:
            raise ValueError('Original fold identity differs')
        with np.load(ref / f'oof-{seed}.npz', allow_pickle=False) as cached:
            order = reference_order(frame.sample_id, cached['ids'])
            np.testing.assert_array_equal(cached['folds'][order], folds)
            np.testing.assert_array_equal(cached['y'][order], frame.tap_time_len.to_numpy(float))
            q75 = cached['Q75'][order].copy()
        member = np.full(len(frame), np.nan)
        for fold in range(5):
            training, query, fitting, calibration = partitions(frame, folds, fold, native)
            parent = old_cache(reference_spec, seed, fold, training, query)
            np.testing.assert_array_equal(q75[folds == fold], parent['q75'])
            group = lambda d: set(pd.util.hash_pandas_object(d[list(FEATURES)], index=False))
            if group(training) & group(query) or group(fitting) & group(calibration):
                raise ValueError('Group overlap')
            key = f'tap_time_len-GAUSS1-s{seed}-f{fold}'
            directory = old / key
            verified_unit(directory, unit_id(original, key))
            meta = json.loads((directory / 'metadata.json').read_text())
            if (meta['seed'], meta['fold'], meta['recipe'], meta['target']) != (seed, fold, 'GAUSS1', 'tap_time_len'):
                raise ValueError('Cached model source/split/trial mismatch')
            chosen, best = selected_epoch(meta['calibration']['history'], native['training'])
            if (chosen != meta['calibration']['selected_epoch'] or chosen != meta['refit']['selected_epoch']
                    or meta['calibration']['stopped_epoch'] != len(meta['calibration']['history'])
                    or meta['refit']['stopped_epoch'] != chosen
                    or [r['epoch'] for r in meta['refit']['history']] != list(range(1, chosen + 1))):
                raise ValueError('Selection/refit epoch mismatch')
            with np.load(directory / 'predictions.npz', allow_pickle=False) as predictions:
                for phase, train, held, field, id_field, md in [
                    ('calibration_model.pt', fitting, calibration, 'calibration_prediction', 'calibration_ids', meta['calibration']),
                    ('model.pt', training, query, 'prediction', 'query_ids', meta['refit'])]:
                    np.testing.assert_array_equal(predictions[id_field], held.sample_id.to_numpy(str))
                    p, difference = cold_state(directory / phase, train, held.drop(columns=list(TARGETS), errors='ignore'),
                                                'tap_time_len', md, native['training'], predictions[field])
                    if difference > spec['cold_predict_atol']:
                        raise ValueError('Original cold tolerance failed')
                    if phase == 'calibration_model.pt':
                        actual = np.abs(p - held.tap_time_len.to_numpy()).mean() / md['target_std']
                        if abs(actual - best) > 1e-10:
                            raise ValueError('Selected checkpoint calibration metric mismatch')
                    else:
                        member[folds == fold] = p
                    cold.append(dict(directory=str(directory), state=phase, difference=difference, selected_epoch=chosen))
        y, spout = frame.tap_time_len.to_numpy(float), frame.spout_no.to_numpy()
        endpoint = compose(q75, member)
        gain = float(50 * (np.abs(y-q75)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        for name, prediction in [('Q75', q75), (CANDIDATE, endpoint)]:
            metrics['tap_time_len'][name][str(seed)] = metric_detail(y, prediction, folds, spout)
        records[str(seed)] = dict(gain=gain, rows=len(y), member_wmape=float(np.abs(y-member).sum()/np.abs(y).sum()))
        with (out / f'oof-{seed}.npz').open('xb') as stream:
            np.savez_compressed(stream, ids=frame.sample_id.to_numpy(str), y=y, folds=folds, spout=spout,
                                Q75=q75, member=member, endpoint=endpoint)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss > spec['max_worker_rss_mib'] or len(cold) != spec['cold_states']:
        raise ValueError('Resource/state count gate failed')
    policy = yaml.safe_load((main / 'configs/candidate_tiers.yaml').read_text())
    write_new(out / 'report.json', dict(records=records, metrics=metrics, cold=cold, peak_rss_mib=rss,
        max_cold_difference=max(row['difference'] for row in cold),
        confirmation_eligible=all(row['gain'] > 0 for row in records.values()), formal_promoted=False,
        tiers=classify_candidates(metrics, spec, policy), new_fits=0, new_optimizers=0,
        manifest_sha256=sha(out / 'manifest.json'), independent_audit='pending',
        oof_sha256={str(seed): sha(out / f'oof-{seed}.npz') for seed in spec['split_seeds']}))


def audit(out):
    manifest, spec = context(out)
    report = json.loads((out / 'report.json').read_text())
    differences, gains = [], {}
    for seed in spec['split_seeds']:
        path = out / f'oof-{seed}.npz'
        if sha(path) != report['oof_sha256'][str(seed)]:
            raise ValueError('Scored OOF identity changed')
        with np.load(path, allow_pickle=False) as data:
            if len(data['ids']) != 2754 or len(set(data['ids'])) != 2754 or set(data['folds']) != set(range(5)):
                raise ValueError('Incomplete independent coverage')
            y, base, member = data['y'].tolist(), data['Q75'].tolist(), data['member'].tolist()
            endpoint = [(.8*b)+(.2*m) for b, m in zip(base, member)]
            differences.extend(abs(a-b) for a, b in zip(endpoint, data['endpoint']))
            gain = 50 * math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,base,endpoint)) / math.fsum(map(abs, y))
            gains[str(seed)] = gain
            differences.append(abs(gain - report['records'][str(seed)]['gain']))
            for name, prediction in [('Q75', base), (CANDIDATE, endpoint)]:
                record = report['metrics']['tap_time_len'][name][str(seed)]
                groups = [('wmape', np.ones(len(y), bool))]
                groups += [(('by_fold', str(f)), data['folds'] == f) for f in range(5)]
                groups += [(('by_spout', str(s)), data['spout'] == s) for s in sorted(set(data['spout']))]
                for key, mask in groups:
                    indices = np.flatnonzero(mask)
                    value = math.fsum(abs(y[i]-prediction[i]) for i in indices) / math.fsum(abs(y[i]) for i in indices)
                    wanted = record[key] if isinstance(key, str) else record[key[0]][key[1]]
                    differences.append(abs(value-wanted))
    eligible = all(g > 0 for g in gains.values())
    if max(differences) > spec['scalar_atol'] or eligible != report['confirmation_eligible']:
        raise ValueError('Independent arithmetic/decision mismatch')
    verify_files(manifest['files'])
    write_new(out / 'independent-score.json', dict(status='passed', scalar_checks=len(differences),
        max_difference=max(differences), gains=gains, confirmation_eligible=eligible, new_fits=0,
        report_sha256=sha(out / 'report.json'), manifest_sha256=sha(out / 'manifest.json')))


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
