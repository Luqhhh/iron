"""Fixed Gaussian time confirmation using native training and existing Q75 OOF."""
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
import pandas as pd
import torch
import yaml
from scipy.stats import t

from .data import FEATURES, TARGETS
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .ema_reference_ledger import Binding, KINDS, NativeLedger, binding_sources, native_hooks
from .q75_gaussian_time import cold_state, compose, forbidden, reference_order, selected_epoch
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import paired_summary
from .v5_spec import load_v5_spec
from .v7_periodic import digest
from .v33_mixture import MixtureRegressor
from .v33_run import metric_detail, partitions

SPEC = 'configs/q75_gaussian_confirmation/SPEC.json'
PROTOCOL = 'docs/q75_gaussian_confirmation/PREREGISTRATION.md'
WORK = Path(__file__).resolve().parents[2]
SEEDS = (42, 3407, 271828, 314159)


def read(path):
    return json.loads(Path(path).read_text())


def hooks():
    return [Binding(torch.optim.Optimizer, '__init__', 'torch_optimizer')]


def validate_spec(spec):
    required = dict(candidate='GAUSS1_A20', weight=.2, development_seeds=[42, 3407],
        confirmation_seeds=[271828, 314159], folds=5, outer_estimators=10, optimizer_runs=20,
        new_states=20, engineering_optimizer_runs=2, reference_fits=0, repeated_development_fits=0,
        workers=1, numerical_threads=1, torch_interop_threads=1, maximum_runtime_seconds=None,
        automatic_retries=False, full_data_fits=0, packages=0, desktop_writes=0, agent_uploads=0)
    if any(spec.get(key) != value for key, value in required.items()):
        raise ValueError('Unregistered scientific scope or budget')


def gate(gains):
    if set(gains) != set(map(str, SEEDS)) or not all(math.isfinite(v) for v in gains.values()):
        raise ValueError('Four complete finite paired split gains required')
    summary = paired_summary([gains[str(seed)] for seed in SEEDS])
    failed = []
    if summary['positive'] != 4:
        failed.append('nonpositive_seed_gain')
    if summary['lcb95'] <= 0:
        failed.append('nonpositive_seed_lcb95')
    return dict(paired=summary, failed_conditions=failed, formal_promoted=not failed)


def access(out, stage):
    with (out / 'access.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage, time_ns=time.time_ns(),
            frozen_inputs_sha256=sha(out / 'frozen-files.json'), scope='synthetic_or_authorized_round2_only',
            protected_prelim_targets_read=False)) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def freeze(out, phase, engineering=None):
    spec = read(WORK / SPEC)
    validate_spec(spec)
    if phase not in ('engineering', 'confirmation'):
        raise ValueError('Explicit engineering or confirmation phase required')
    main = Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main / 'local/runs'):
        raise ValueError('New private output required')
    versions = runtime(spec)
    torch.set_num_interop_threads(1)
    current = read(main / 'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(current[k] != value for k, value in spec['reference'].items()):
        raise ValueError('Dynamic Q75 reference changed')
    dev = main / spec['development']
    dm, dr, dt = [read(dev / name) for name in ('manifest.json', 'report.json', 'terminal.json')]
    for rel in ('src/bf_tap_r2/v33_mixture.py', 'src/bf_tap_r2/v3_6_networks.py', spec['original_specification']):
        if sha(WORK / rel) != dm['files'][str(Path(dm['workspace']) / rel)]:
            raise ValueError('Native scientific implementation differs from validated development')
    if dt['status'] != 'passed' or dt['actual_exit_codes'] != [0, 0] or not dr['confirmation_eligible']:
        raise ValueError('Successful two-positive-seed development required')
    for name, field in [('manifest.json', 'manifest_sha256'), ('report.json', 'report_sha256'),
                        ('independent-score.json', 'independent_score_sha256')]:
        if sha(dev / name) != dt[field]:
            raise ValueError('Development terminal binding mismatch')
    if any(dr['records'][str(seed)]['gain'] <= 0 for seed in (42, 3407)):
        raise ValueError('Both development gains must be positive')
    paths = list((WORK / 'src').rglob('*.py'))
    paths += [WORK / p for p in (SPEC, PROTOCOL, 'uv.lock', 'pyproject.toml', spec['original_specification'],
              'scripts/q75_gaussian_confirmation.py', 'scripts/observe_ema_fusion_selection.py',
              'tests/test_q75_gaussian_confirmation.py')]
    paths += [dev / name for name in ('manifest.json', 'report.json', 'independent-score.json', 'terminal.json',
                                     'oof-42.npz', 'oof-3407.npz')]
    native_sources = binding_sources(hooks())
    native_sources.update({str(WORK / p): sha(WORK / p) for p in (
        'src/bf_tap_r2/v33_mixture.py', 'src/bf_tap_r2/v3_6_networks.py',
        'src/bf_tap_r2/q75_gaussian_confirmation.py')})
    binding_path = main / spec['reference_binding']
    binding = read(binding_path)
    if (binding['status'] != 'passed_zero_fit_existing_Q75_export' or binding['verified_split_seeds'] != list(SEEDS)
            or binding['candidate'] != spec['reference']['candidate']
            or binding['platform_package_sha256'] != spec['reference']['zip_sha256']):
        raise ValueError('Complete current Q75 binding required')
    files = {**dm['files'], **binding['frozen_original_evidence'], **native_sources}
    files[binding['ids']['path']] = binding['ids']['sha256']
    for seed in SEEDS:
        for record in binding['columns'][str(seed)].values():
            files[record['path']] = record['sha256']
    reference = main / spec['reference_confirmation']
    rt = read(reference / 'terminal-verification-r1.json')
    if rt['status'] != 'passed' or rt['actual_controller_exit_code'] != 0:
        raise ValueError('Original Q75 confirmation actual terminal missing')
    for name, key in [('manifest.json', 'manifest_sha256'), ('audit.json', 'audit_sha256'),
                      ('summary.json', 'summary_sha256'), ('process-terminal.json', 'process_terminal_sha256')]:
        if sha(reference / name) != rt[key]:
            raise ValueError('Original Q75 terminal identity changed')
        paths.append(reference / name)
    paths += [binding_path, reference / 'terminal-verification-r1.json']
    if phase == 'confirmation':
        if engineering is None:
            raise ValueError('Engineering terminal required')
        terminal = read(engineering / 'terminal.json')
        if terminal['status'] != 'passed' or terminal['actual_exit_codes'] != [0, 0]:
            raise ValueError('Actual successful engineering cold terminal required')
        if sha(engineering / 'manifest.json') != terminal['manifest_sha256']:
            raise ValueError('Engineering manifest binding mismatch')
        for name, value in read(engineering / 'manifest.json')['sources'].items():
            if sha(name) != value:
                raise ValueError('Engineering tested source changed')
        paths.extend(p for p in engineering.rglob('*') if p.is_file())
    files.update({str(path): sha(path) for path in paths})
    verify_files(files)
    out.mkdir(parents=True)
    write_new(out / 'frozen-files.json', files)
    access(out, 'freeze_' + phase)
    native = yaml.safe_load((WORK / spec['original_specification']).read_text())
    if phase == 'engineering':
        rng = np.random.default_rng(96104533)
        x = rng.normal(size=(200, len(FEATURES)))
        frame = pd.DataFrame(x, columns=FEATURES)
        frame['sample_id'] = [f'gaussian-synthetic-{i}' for i in range(200)]
        frame['spout_no'] = 1 + np.arange(200) % 2
        frame['tap_time_len'] = 120 + 20*x[:, 0] + 10*x[:, 1] + .3*rng.normal(size=200)
        folds = {-1: np.r_[np.ones(160, dtype=int), np.zeros(40, dtype=int)]}
        refs = {-1: np.full(200, np.nan)}
        units = [dict(seed=-1, fold=0)]
    else:
        frame = load_v5_training_frame(main)
        if len(frame) != 2754 or frame.sample_id.duplicated().any():
            raise ValueError('Complete native round2 data required')
        folds, refs = {}, {}
        order = reference_order(frame.sample_id, np.load(binding['ids']['path'], allow_pickle=False))
        for seed in SEEDS:
            fv = fold_vector(main, frame, seed, load_v5_spec(main))
            np.testing.assert_array_equal(fv, np.load(binding['columns'][str(seed)]['folds']['path'], allow_pickle=False)[order])
            folds[seed] = fv
            refs[seed] = np.load(binding['columns'][str(seed)]['time']['path'], allow_pickle=False)[order]
            if seed in (42, 3407):
                with np.load(dev / f'oof-{seed}.npz', allow_pickle=False) as saved:
                    index = reference_order(frame.sample_id, saved['ids'])
                    np.testing.assert_array_equal(refs[seed], saved['Q75'][index])
                    np.testing.assert_array_equal(frame.tap_time_len.to_numpy(), saved['y'][index])
            else:
                for fold in range(5):
                    old = reference / f's{seed}-f{fold}'
                    warm, cold = read(old / 'warm-complete.json'), read(old / 'cold-complete.json')
                    if (cold['status'] != 'passed' or cold['warm_receipt_sha256'] != sha(old / 'warm-complete.json')
                            or warm['predictions_sha256'] != sha(old / 'predictions.npz')):
                        raise ValueError('Native Q75 same-fold warm/cold identity differs')
                    with np.load(old / 'predictions.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['query_ids'], frame.loc[fv == fold, 'sample_id'].to_numpy(str))
                        q75 = saved['tap_time_len'] + .75*(saved['old_ema'] - saved['v7_time'])
                        np.testing.assert_array_equal(q75, saved['q75'])
                        np.testing.assert_array_equal(q75, refs[seed][fv == fold])
        units = [dict(seed=seed, fold=fold) for seed in spec['confirmation_seeds'] for fold in range(5)]
    arrays = dict(x=frame[list(FEATURES)].to_numpy(float), ids=frame.sample_id.to_numpy(str),
                  spouts=frame.spout_no.to_numpy(), y=frame.tap_time_len.to_numpy(float))
    arrays.update({f'folds_{seed}': value for seed, value in folds.items()})
    arrays.update({f'q75_{seed}': value for seed, value in refs.items()})
    with (out / 'inputs.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    sources = {str(p): sha(p) for p in paths if p.is_relative_to(WORK)}
    write_new(out / 'manifest.json', dict(phase=phase, spec=spec, native=native, files=files, sources=sources,
        native_sources=native_sources, versions=versions, python=sys.version, units=units,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip(),
        frozen_files_sha256=sha(out / 'frozen-files.json'), inputs_sha256=sha(out / 'inputs.npz')))


def context(out, *, no_fit=False):
    manifest = read(out / 'manifest.json')
    validate_spec(manifest['spec'])
    verify_files(manifest['files'])
    if sha(out / 'inputs.npz') != manifest['inputs_sha256'] or sha(out / 'frozen-files.json') != manifest['frozen_files_sha256']:
        raise ValueError('Frozen input identity changed')
    runtime(manifest['spec'])
    torch.set_num_interop_threads(1)
    if no_fit:
        MixtureRegressor.initialize = MixtureRegressor.train = forbidden
        torch.optim.Adam = torch.optim.AdamW = forbidden
    access(out, 'read_' + manifest['phase'])
    data = np.load(out / 'inputs.npz', allow_pickle=False)
    frame = pd.DataFrame(data['x'], columns=FEATURES)
    frame['sample_id'], frame['spout_no'], frame['tap_time_len'] = data['ids'], data['spouts'], data['y']
    return manifest, frame, data


def unit_parts(manifest, frame, data, seed, fold):
    training, query, fitting, calibration = partitions(frame, data[f'folds_{seed}'], fold, manifest['native'])
    group = lambda part: set(pd.util.hash_pandas_object(part[list(FEATURES)], index=False))
    if group(training) & group(query) or group(fitting) & group(calibration):
        raise ValueError('Group partition overlap')
    return training, query, fitting, calibration


def train(out):
    manifest, frame, data = context(out)
    for item in manifest['units']:
        seed, fold = item['seed'], item['fold']
        unit = out / f's{seed}-f{fold}'
        unit.mkdir(exist_ok=False)
        write_new(unit / 'start.json', dict(seed=seed, fold=fold, pid=os.getpid(), manifest_sha256=sha(out / 'manifest.json')))
        training, query, fitting, calibration = unit_parts(manifest, frame, data, seed, fold)
        expected = {key: 2 if key == 'torch_optimizer' else 0 for key in KINDS}
        ledger = NativeLedger(unit / 'native', identity=dict(source_directory=str(out), split_seed=seed,
            fold=fold, trial_id='GAUSS1'), expected=expected, training_ids=training.sample_id.tolist(),
            query_ids=query.sample_id.tolist(), source_hashes=manifest['native_sources'])
        try:
            with native_hooks(hooks()):
                with ledger.partition(fitting.sample_id.tolist(), calibration.sample_id.tolist()):
                    selector = MixtureRegressor('GAUSS1', manifest['native']['training']).initialize(fitting, fitting.tap_time_len.to_numpy())
                    epoch = selector.train(manifest['native']['training']['max_epochs'], (calibration, calibration.tap_time_len.to_numpy()))
                cp = selector.predict(calibration)
                selector.save(unit / 'calibration_model.pt')
                with ledger.partition(training.sample_id.tolist()):
                    model = MixtureRegressor('GAUSS1', manifest['native']['training']).initialize(training, training.tap_time_len.to_numpy())
                    model.train(epoch)
                prediction = model.predict(query)
                model.save(unit / 'model.pt')
            native_receipt = ledger.close()
            metadata = dict(seed=seed, fold=fold, target='tap_time_len', recipe='GAUSS1',
                            calibration=selector.metadata(), refit=model.metadata())
            write_new(unit / 'metadata.json', metadata)
            with (unit / 'predictions.npz').open('xb') as stream:
                np.savez_compressed(stream, prediction=prediction, calibration_prediction=cp,
                    query_ids=query.sample_id.to_numpy(str), calibration_ids=calibration.sample_id.to_numpy(str))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if rss > manifest['spec']['max_worker_rss_mib']:
                raise ValueError('Training RSS exceeds frozen gate')
            hashes = {str(p.relative_to(unit)): sha(p) for p in unit.rglob('*') if p.is_file()}
            write_new(unit / 'warm-complete.json', dict(hashes=hashes, counts=native_receipt['counts'], peak_rss_mib=rss,
                manifest_sha256=sha(out / 'manifest.json'), selected_epoch=epoch, pid=os.getpid()))
        except BaseException as error:
            write_new(unit / 'failure.json', dict(error=repr(error), automatic_retry=False))
            raise
    write_new(out / 'training-complete.json', dict(units=len(manifest['units']), optimizer_runs=2*len(manifest['units']),
        warm_hashes={f"s{i['seed']}-f{i['fold']}": sha(out / f"s{i['seed']}-f{i['fold']}" / 'warm-complete.json') for i in manifest['units']}))


def cold(out):
    manifest, frame, data = context(out, no_fit=True)
    completion = read(out / 'training-complete.json')
    differences, count, learnability = [], 0, None
    for item in manifest['units']:
        seed, fold = item['seed'], item['fold']
        unit = out / f's{seed}-f{fold}'
        warm = read(unit / 'warm-complete.json')
        if completion['warm_hashes'][unit.name] != sha(unit / 'warm-complete.json') or warm['pid'] == os.getpid():
            raise ValueError('Independent cold process and warm identity required')
        verify_files({str(unit / p): h for p, h in warm['hashes'].items()})
        training, query, fitting, calibration = unit_parts(manifest, frame, data, seed, fold)
        meta = read(unit / 'metadata.json')
        chosen, best = selected_epoch(meta['calibration']['history'], manifest['native']['training'])
        if (chosen != warm['selected_epoch'] or chosen != meta['calibration']['selected_epoch']
                or chosen != meta['refit']['selected_epoch'] or chosen != meta['refit']['stopped_epoch']
                or meta['calibration']['stopped_epoch'] != len(meta['calibration']['history'])
                or [r['epoch'] for r in meta['refit']['history']] != list(range(1, chosen+1))):
            raise ValueError('Cold selector/refit epochs mismatch')
        native = read(unit / 'native/scope-complete.json')
        if native['counts'] != warm['counts'] or native['counts']['torch_optimizer'] != 2:
            raise ValueError('Native budget mismatch')
        for index, fit, cal in [(1, fitting, calibration), (2, training, None)]:
            call = read(unit / f'native/call-{index:04d}/start.json')
            if call['partition']['training_ids'] != fit.sample_id.tolist() or call['partition']['calibration_ids'] != ([] if cal is None else cal.sample_id.tolist()):
                raise ValueError('Native optimizer partition mismatch')
        with np.load(unit / 'predictions.npz', allow_pickle=False) as saved:
            for file, fit, held, field, ids, metadata in [
                ('calibration_model.pt', fitting, calibration, 'calibration_prediction', 'calibration_ids', meta['calibration']),
                ('model.pt', training, query, 'prediction', 'query_ids', meta['refit'])]:
                np.testing.assert_array_equal(saved[ids], held.sample_id.to_numpy(str))
                prediction, difference = cold_state(unit / file, fit, held.drop(columns=list(TARGETS), errors='ignore'),
                    'tap_time_len', metadata, manifest['native']['training'], saved[field])
                if difference > manifest['spec']['cold_predict_atol']:
                    raise ValueError('Cold/order/chunk tolerance exceeded')
                if file == 'calibration_model.pt':
                    value = np.abs(prediction-calibration.tap_time_len.to_numpy()).mean()/metadata['target_std']
                    if abs(value-best) > 1e-10:
                        raise ValueError('Selected state is not the traced best checkpoint')
                elif manifest['phase'] == 'engineering':
                    y = frame.tap_time_len.to_numpy()[data[f'folds_{seed}'] == fold]
                    learnability = dict(model_mae=float(np.abs(prediction-y).mean()),
                        median_constant_mae=float(np.abs(np.median(training.tap_time_len)-y).mean()))
                    if learnability['model_mae'] >= learnability['median_constant_mae']:
                        raise ValueError('Synthetic learnability failed')
                differences.append(difference)
                count += 1
        write_new(unit / 'cold-complete.json', dict(status='passed', warm_sha256=sha(unit / 'warm-complete.json'),
                  cold_states=2, native_counts=native['counts']))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss > manifest['spec']['max_worker_rss_mib'] or count != 2*len(manifest['units']):
        raise ValueError('Cold RSS/count gate failed')
    write_new(out / 'cold-audit.json', dict(status='passed', cold_states=count, max_difference=max(differences),
        learnability=learnability, peak_rss_mib=rss, new_fits=0, manifest_sha256=sha(out / 'manifest.json')))


def evaluate(out):
    manifest, frame, data = context(out, no_fit=True)
    if manifest['phase'] != 'confirmation' or read(out / 'cold-audit.json')['cold_states'] != 20:
        raise ValueError('Complete independent cold confirmation required')
    spec = manifest['spec']
    dev = Path(spec['main_root']) / spec['development']
    metrics, gains, hashes = {}, {}, {}
    for seed in SEEDS:
        fv, base = data[f'folds_{seed}'], data[f'q75_{seed}']
        if seed in (42, 3407):
            with np.load(dev / f'oof-{seed}.npz', allow_pickle=False) as previous:
                order = reference_order(frame.sample_id, previous['ids'])
                member = previous['member'][order]
        else:
            member = np.full(len(frame), np.nan)
            for fold in range(5):
                unit = out / f's{seed}-f{fold}'
                warm, cold_receipt = read(unit / 'warm-complete.json'), read(unit / 'cold-complete.json')
                if cold_receipt['warm_sha256'] != sha(unit / 'warm-complete.json'):
                    raise ValueError('Cold candidate binding changed')
                verify_files({str(unit / p): h for p, h in warm['hashes'].items()})
                with np.load(unit / 'predictions.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['query_ids'], frame.loc[fv == fold, 'sample_id'].to_numpy(str))
                    member[fv == fold] = saved['prediction']
        endpoint = compose(base, member)
        y = data['y']
        gains[str(seed)] = float(50*(np.abs(y-base)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        metrics[str(seed)] = {name: metric_detail(y, prediction, fv, data['spouts']) for name, prediction in [('Q75', base), ('GAUSS1_A20', endpoint)]}
        path = out / f'oof-{seed}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, ids=data['ids'], y=y, folds=fv, spouts=data['spouts'], Q75=base, member=member, endpoint=endpoint)
        hashes[str(seed)] = sha(path)
    write_new(out / 'report.json', dict(gains=gains, metrics=metrics, **gate(gains),
        oof_sha256=hashes, candidate=spec['candidate'], new_estimators=10, new_states=20, optimizer_runs=20,
        reference_fits=0, repeated_development_fits=0, full_fits=0, packages=0, desktop_writes=0, uploads=0,
        independent_audit='pending', manifest_sha256=sha(out / 'manifest.json')))


def audit(out):
    manifest, _, _ = context(out, no_fit=True)
    report = read(out / 'report.json')
    differences, gains = [], {}
    for seed in SEEDS:
        path = out / f'oof-{seed}.npz'
        if sha(path) != report['oof_sha256'][str(seed)]:
            raise ValueError('OOF file identity changed')
        with np.load(path, allow_pickle=False) as values:
            if len(values['ids']) != 2754 or len(set(values['ids'])) != 2754 or set(values['folds']) != set(range(5)):
                raise ValueError('Incomplete four-seed OOF')
            y, base, member = [values[k].tolist() for k in ('y', 'Q75', 'member')]
            endpoint = [(.8*b)+(.2*m) for b, m in zip(base, member)]
            differences.extend(abs(a-b) for a,b in zip(endpoint, values['endpoint']))
            gain = 50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,base,endpoint))/math.fsum(map(abs,y))
            gains[str(seed)] = gain
            differences.append(abs(gain-report['gains'][str(seed)]))
            for name, prediction in [('Q75', base), ('GAUSS1_A20', endpoint)]:
                record = report['metrics'][str(seed)][name]
                groups = [('wmape', np.ones(len(y), bool))]
                groups += [(('by_fold',str(f)),values['folds']==f) for f in range(5)]
                groups += [(('by_spout',str(s)),values['spouts']==s) for s in sorted(set(values['spouts']))]
                for key, mask in groups:
                    ids = np.flatnonzero(mask)
                    actual = math.fsum(abs(y[i]-prediction[i]) for i in ids)/math.fsum(abs(y[i]) for i in ids)
                    expected = record[key] if isinstance(key,str) else record[key[0]][key[1]]
                    differences.append(abs(actual-expected))
    mean = math.fsum(gains.values())/4
    se = math.sqrt(math.fsum((gain-mean)**2 for gain in gains.values())/3)/2
    lcb = mean-float(t.ppf(.95,3))*se
    differences.extend([abs(mean-report['paired']['mean']),abs(lcb-report['paired']['lcb95'])])
    passed = all(gain>0 for gain in gains.values()) and lcb>0
    if passed != report['formal_promoted'] or max(differences)>manifest['spec']['scalar_atol']:
        raise ValueError('Independent score or formal gate mismatch')
    write_new(out / 'independent-score.json', dict(status='passed', gains=gains, mean_gain=mean, seed_lcb95=lcb,
        formal_promoted=passed, scalar_checks=len(differences), max_difference=max(differences),
        report_sha256=sha(out / 'report.json'), manifest_sha256=sha(out / 'manifest.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['freeze', 'train', 'cold', 'evaluate', 'audit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['engineering', 'confirmation'])
    parser.add_argument('--engineering', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.operation == 'freeze':
        freeze(out, args.phase, args.engineering.resolve() if args.engineering else None)
    else:
        globals()[args.operation](out)


if __name__ == '__main__':
    main()
