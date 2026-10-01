"""Fixed EMA decay development, preserving the original scientific trainer."""
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
import yaml

from .data import FEATURES, TARGETS
from .ema_evaluation_diagnostics import sha, verify_files, write_new
from .ema_training_scale import assert_isolated, event, require_memory
from .v3_4_bags import group_safe_inner_folds
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v7_periodic import digest

SPEC = 'configs/ema_average_span/SPEC.json'
ORDER = ['SHORT_SPAN', 'LONG_SPAN']


def bind_original_sources(workspace, main, hashes):
    """Original loaders use main-root private configs; never copy them into Git."""
    private = {}
    for name, expected in hashes.items():
        if name in ('configs/data.local.yaml', 'configs/predict.local.yaml'):
            verify_files(main, {name: expected})
            private[name] = expected
        else:
            verify_files(workspace, {name: expected})
    return private


def validate_scope(spec, original):
    expected = dict(target='tap_time_len', split_seeds=[42, 3407], folds=5,
                    candidate_order=ORDER, control_beta=.99, replacement_weight=.75,
                    workers=1, numerical_threads=1, monitor_seconds=600,
                    max_worker_rss_mib=1536, cold_predict_atol=.0005,
                    new_estimators=20, new_optimizer_runs=40,
                    maximum_runtime_seconds=None, automatic_scientific_retries=False)
    expected.update({k: 0 for k in ('new_confirmation_seeds', 'full_data_fits',
                                   'packages', 'desktop_writes', 'agent_uploads')})
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Unregistered scientific or execution scope')
    if (spec['candidate_beta'] != {'SHORT_SPAN': .99**2, 'LONG_SPAN': math.sqrt(.99)}
            or spec['training'] != original['training']['tap_time_len']
            or spec['mechanisms'] != original['mechanisms']):
        raise ValueError('Frozen trainer or averaging spans changed')


def candidate_column(reference, old_ema, new_ema):
    reference, old_ema, new_ema = [np.asarray(v, float) for v in (reference, old_ema, new_ema)]
    if reference.ndim != 1 or reference.shape != old_ema.shape or reference.shape != new_ema.shape:
        raise ValueError('Aligned one-dimensional predictions required')
    if not all(np.isfinite(v).all() for v in (reference, old_ema, new_ema)):
        raise ValueError('Nonfinite prediction')
    result = reference + .75*(new_ema-old_ema)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid affine prediction; no clipping permitted')
    return result


def choose_confirmation(gains):
    if list(gains) != ORDER or any(set(v) != {'42', '3407'} for v in gains.values()):
        raise ValueError('Complete declared split coverage required')
    if any(not np.isfinite(v) for row in gains.values() for v in row.values()):
        raise ValueError('Finite development gains required')
    eligible = [name for name in ORDER if min(gains[name].values()) > 0]
    if not eligible:
        return None
    means = {name: float(np.mean(list(gains[name].values()))) for name in eligible}
    best = max(means.values())
    return next(name for name in ORDER if name in means and means[name] >= best-1e-12)


def sources(workspace):
    workspace = Path(workspace)
    paths = list((workspace/'src').rglob('*.py')) + list((workspace/'tests').glob('*.py'))
    for extension in ('*.yaml', '*.json'):
        paths += list((workspace/'configs').rglob(extension))
    paths += [workspace/name for name in ('uv.lock', 'pyproject.toml',
        'docs/ema_average_span/PREREGISTRATION.md', 'scripts/ema_average_span.py',
        'scripts/audit_ema_average_span.py', 'scripts/observe_ema_average_span.py')]
    return {str(p.relative_to(workspace)): sha(p) for p in sorted(set(paths))}


def context(workspace, manifest):
    from .v49_run import check_runtime
    import torch
    workspace = Path(workspace)
    verify_files(workspace, manifest['sources'])
    spec = manifest['spec']; main = Path(spec['main_root'])
    verify_files(main, spec['inputs'])
    verify_files(main, manifest['original_private_dependencies'])
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python3.12 required')
    check_runtime(spec)
    torch.set_num_threads(1)
    original = yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text())
    validate_scope(spec, original)
    frame = load_v5_training_frame(main)
    folds = {s: fold_vector(main, frame, s, load_v5_spec(main)) for s in spec['split_seeds']}
    if {str(s): digest(f.tolist()) for s, f in folds.items()} != manifest['fold_digests']:
        raise ValueError('Frozen outer split identity changed')
    return spec, frame, folds


def task_frames(frame, fv, fold):
    training = frame.loc[fv != fold].reset_index(drop=True)
    query = frame.loc[fv == fold, ['sample_id', 'spout_no', *FEATURES]].reset_index(drop=True)
    assert_isolated(training, query)
    return training, query


def old_cache(spec, seed, fold, training, query):
    from .v49_run import verified_unit, unit_id
    root = Path(spec['main_root'])/spec['old_development']
    manifest = json.loads((root/'manifest.json').read_text())
    refkey, emakey = f'reference-s{seed}-f{fold}', f'tap_time_len-EMA-s{seed}-f{fold}'
    for key in (refkey, emakey):
        if not verified_unit(root/key, unit_id(manifest, key)):
            raise ValueError('Closed original cache unit required')
    base_meta = json.loads((root/refkey/'metadata.json').read_text())
    meta = json.loads((root/emakey/'metadata.json').read_text())
    fit_ids, query_ids = digest(training.sample_id.tolist()), digest(query.sample_id.tolist())
    if (base_meta['fit_ids_digest'] != fit_ids or base_meta['query_ids_digest'] != query_ids
            or meta['fit_ids_digest'] != fit_ids or meta['model']['fit_ids_digest'] != fit_ids
            or meta['seed'] != seed or meta['fold'] != fold or meta['training_seed'] != 42
            or base_meta['query_labels_received'] is not False):
        raise ValueError('Original cache partition identity mismatch')
    with np.load(root/refkey/'predictions.npz', allow_pickle=False) as saved:
        base = {k: saved[k].copy() for k in saved.files}
    with np.load(root/emakey/'predictions.npz', allow_pickle=False) as saved:
        if saved['prediction'].shape != (len(query), 1):
            raise ValueError('Original EMA prediction shape')
        ema = saved['prediction'][:, 0].copy()
        ids = saved['query_ids'].tolist()
    if ids != query.sample_id.tolist() or base['query_ids'].tolist() != ids:
        raise ValueError('Original cache query order mismatch')
    np.testing.assert_array_equal(base['tap_time_len'], .2*base['v36_time']+.3*base['n_time']+.5*base['v7_time'])
    np.testing.assert_array_equal(base['tap_iron'], .5*base['v36_iron']+.5*base['v12_iron'])
    q75 = base['tap_time_len']+.75*(ema-base['v7_time'])
    if any(v.shape != (len(query),) or not np.isfinite(v).all() or (v < 0).any()
           for v in (q75, ema, base['tap_iron'])):
        raise ValueError('Invalid original cache prediction')
    return dict(q75=q75, ema=ema, iron=base['tap_iron'], metadata=meta)


def require_previous_terminal(spec):
    import os
    import psutil
    root = Path(spec['main_root'])
    receipt = root/'local/runs/q75-error-relocation-20261001/calibration-development-r1/terminal-verification-r1.json'
    checked = json.loads(receipt.read_text()); out = receipt.parent
    terminal = json.loads((out/'process-terminal-r2.json').read_text())
    if (checked['status'] != 'passed' or checked['controller_exit_code'] != 0
            or checked['terminal_sha256'] != sha(out/'process-terminal-r2.json')
            or checked['manifest_sha256'] != sha(out/'manifest.json') or terminal['exit_code'] != 0):
        raise ValueError('Successful bound actual previous terminal required')
    names = ('q75_pressure_calibration.py', 'observe_q75_pressure_calibration_direct.py',
             'ema_training_scale.py', 'ema_average_span.py', 'observe_ema_average_span.py')
    active = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.info['pid'] == os.getpid():
            continue
        command = process.info['cmdline'] or []
        if any(Path(arg).name in names for arg in command):
            active.append(process.info)
    if active:
        raise ValueError('Serial scientific dependency still active: '+repr(active))
    return dict(receipt_sha256=sha(receipt), controller_exit_code=0)


def prepare(workspace, receipt):
    from .v49_run import check_runtime
    workspace = Path(workspace).resolve()
    spec = json.loads((workspace/SPEC).read_text())
    main = Path(spec['main_root'])
    validate_scope(spec, yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text()))
    source_hashes = sources(workspace)
    tests = json.loads(Path(receipt).read_text())
    if tests.get('status') != 'passed' or tests.get('sources') != source_hashes or tests.get('full_suite') is not True:
        raise ValueError('Passed exact-source full locked tests required')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *source_hashes], cwd=workspace, text=True).strip():
        raise ValueError('Commit tested scientific source before launch')
    verify_files(main, spec['inputs'])
    old = json.loads((main/spec['old_development']/'manifest.json').read_text())
    private_dependencies = bind_original_sources(workspace, main, old['source_hashes'])
    audit = json.loads((main/spec['old_development']/'audit.json').read_text())
    if audit['status'] != 'passed' or audit['manifest_sha256'] != sha(main/spec['old_development']/'manifest.json'):
        raise ValueError('Audited original model cache required')
    current = json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current[k] != v for k, v in spec['reference'].items()):
        raise ValueError('Current platform reference changed')
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python3.12 required')
    versions = check_runtime(spec); serial = require_previous_terminal(spec); available = require_memory(spec)
    out = Path(spec['output']).resolve()
    if not out.is_relative_to(main/'local/runs') or out.exists():
        raise ValueError('Fresh private output required')
    frame = load_v5_training_frame(main)
    folds = {s: fold_vector(main, frame, s, load_v5_spec(main)) for s in spec['split_seeds']}
    plan = {}
    for seed, fv in folds.items():
        for fold in range(5):
            training, query = task_frames(frame, fv, fold)
            old_cache(spec, seed, fold, training, query)
            inner = np.asarray(group_safe_inner_folds(training, seed=spec['training']['inner_seed'])['fold'])
            plan[f's{seed}-f{fold}'] = dict(training=digest(training.sample_id.tolist()),
                query=digest(query.sample_id.tolist()),
                inner_fit=digest(training.loc[inner != 0, 'sample_id'].tolist()),
                inner_validation=digest(training.loc[inner == 0, 'sample_id'].tolist()),
                training_rows=len(training), query_rows=len(query))
    out.mkdir(parents=True)
    manifest = dict(spec=spec, sources=source_hashes, versions=versions, plan=plan,
        fold_digests={str(s): digest(v.tolist()) for s, v in folds.items()}, workspace=str(workspace),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip(),
        available_memory_mib=available, previous_terminal=serial, receipt_sha256=sha(receipt),
        original_private_dependencies=private_dependencies)
    write_new(out/'manifest.json', manifest)
    return out


def worker(workspace, out, seed, fold, candidate):
    from .component_regularization import ComponentRegressor
    from .component_regularization_run import RECIPE
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text())
    spec, frame, folds = context(workspace, manifest)
    if seed not in spec['split_seeds'] or fold not in range(5) or candidate not in ORDER:
        raise ValueError('Unknown task')
    unit = out/f'{candidate}-s{seed}-f{fold}'; unit.mkdir(exist_ok=False)
    identifiers = dict(seed=seed, fold=fold, candidate=candidate)
    event(out/'events.jsonl', dict(event='unit_started', **identifiers))
    started = time.monotonic()
    try:
        require_memory(spec)
        training, query = task_frames(frame, folds[seed], fold)
        cache = old_cache(spec, seed, fold, training, query)
        mechanisms = dict(spec['mechanisms'], ema_beta=spec['candidate_beta'][candidate])

        class RecordedRegressor(ComponentRegressor):
            def _train(self, frame, y, epochs, validation=None):
                phase = 'selection' if validation is not None else 'refit'
                event(out/'events.jsonl', dict(event='optimizer_started', phase=phase, **identifiers))
                try:
                    result = super()._train(frame, y, epochs, validation)
                except BaseException as error:
                    event(out/'events.jsonl', dict(event='optimizer_failed', phase=phase, error=repr(error), **identifiers))
                    raise
                event(out/'events.jsonl', dict(event='optimizer_completed', phase=phase, **identifiers))
                return result

        model = RecordedRegressor(RECIPE, spec['training'], 'EMA', mechanisms, unit)
        model.fit(training, training[['tap_time_len']].to_numpy())
        ema = model.predict(query)[:, 0]
        prediction = candidate_column(cache['q75'], cache['ema'], ema)
        train_pred = model.predict(training.drop(columns=list(TARGETS)))[:, 0]
        with (unit/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, query_ids=query.sample_id.to_numpy(str), ema=ema,
                q75=cache['q75'], old_ema=cache['ema'], iron_reference=cache['iron'], candidate=prediction)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > spec['max_worker_rss_mib']:
            raise ValueError('Original worker memory gate failed')
        half_life = math.log(.5)/math.log(mechanisms['ema_beta'])
        phases = {}
        for phase, trace in model.traces.items():
            batches = math.ceil(trace['fit_rows']/spec['training']['batch_size'])
            phases[phase] = dict(half_life_updates=half_life, half_life_epochs=half_life/batches,
                updates_per_epoch=batches, selected_epoch=trace['selected_epoch'], stopped_epoch=trace['stopped_epoch'],
                total_updates=trace['updates'], cap_hit=trace['stopped_epoch'] == spec['training']['max_epochs'])
        write_new(unit/'metadata.json', dict(**identifiers, beta=mechanisms['ema_beta'], model=model.metadata_,
            partitions=manifest['plan'][f's{seed}-f{fold}'], averaging=phases,
            refit_training_mae=float(np.abs(training.tap_time_len.to_numpy()-train_pred).mean()),
            seconds=time.monotonic()-started, peak_rss_mib=peak))
        context(workspace, manifest)
        write_new(unit/'complete.json', dict(manifest_sha256=sha(out/'manifest.json'),
            hashes={p.name: sha(p) for p in unit.iterdir()}, new_optimizer_runs=2))
        event(out/'events.jsonl', dict(event='unit_completed', **identifiers))
    except BaseException as error:
        event(out/'events.jsonl', dict(event='unit_failed', error=repr(error), **identifiers))
        write_new(unit/'failure.json', dict(error=repr(error)))
        raise


def summarize(workspace, out):
    from .candidate_tiers import classify_candidates
    from .v49_run import metric_detail
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text())
    spec, frame, folds = context(workspace, manifest)
    metrics = {'tap_time_len': {name: {} for name in ['Q75', *ORDER]}}
    gains = {name: {} for name in ORDER}; scores = {}; diagnostics = {}
    for seed, fv in folds.items():
        vectors = {name: np.full(len(frame), np.nan) for name in ['Q75', *ORDER]}
        iron = np.full(len(frame), np.nan); diagnostics[str(seed)] = {}
        for fold in range(5):
            query_ids = frame.loc[fv == fold, 'sample_id'].tolist()
            for candidate in ORDER:
                unit = out/f'{candidate}-s{seed}-f{fold}'
                verify_files(unit, json.loads((unit/'complete.json').read_text())['hashes'])
                with np.load(unit/'predictions.npz', allow_pickle=False) as a:
                    if a['query_ids'].tolist() != query_ids:
                        raise ValueError('Candidate OOF order changed')
                    if candidate != ORDER[0]:
                        np.testing.assert_array_equal(vectors['Q75'][fv == fold], a['q75'])
                        np.testing.assert_array_equal(iron[fv == fold], a['iron_reference'])
                    vectors['Q75'][fv == fold] = a['q75']; iron[fv == fold] = a['iron_reference']
                    vectors[candidate][fv == fold] = a['candidate']
                    raw_ema = a['ema'].copy()
                meta = json.loads((unit/'metadata.json').read_text())
                query_mae = float(np.abs(frame.loc[fv == fold, 'tap_time_len'].to_numpy()-vectors[candidate][fv == fold]).mean())
                raw_mae = float(np.abs(frame.loc[fv == fold, 'tap_time_len'].to_numpy()-raw_ema).mean())
                diagnostics[str(seed)][f'{candidate}-f{fold}'] = dict(averaging=meta['averaging'],
                    refit_training_mae=meta['refit_training_mae'], outer_component_mae=raw_mae,
                    refit_train_outer_gap=raw_mae-meta['refit_training_mae'], candidate_mae=query_mae,
                    inner_selected_gap=(meta['model']['traces']['selection']['history'][meta['model']['selected_epoch']-1]['validation_mae']
                        -meta['model']['traces']['selection']['history'][meta['model']['selected_epoch']-1]['training_eval_mae']))
        y = frame.tap_time_len.to_numpy(); scores[str(seed)] = {}
        if not np.isfinite(iron).all():
            raise ValueError('Incomplete iron background')
        for name, p in vectors.items():
            if not np.isfinite(p).all():
                raise ValueError('Incomplete full-seed OOF')
            metrics['tap_time_len'][name][str(seed)] = metric_detail(y, p, fv, frame.spout_no.to_numpy())
            iron_wmape = np.abs(frame.tap_iron.to_numpy()-iron).sum()/np.abs(frame.tap_iron.to_numpy()).sum()
            scores[str(seed)][name] = float(100-50*(metrics['tap_time_len'][name][str(seed)]['wmape']+iron_wmape))
        for name in ORDER:
            gains[name][str(seed)] = float(50*(np.abs(y-vectors['Q75']).sum()-np.abs(y-vectors[name]).sum())/np.abs(y).sum())
    tier_spec = dict(split_seeds=spec['split_seeds'], folds=5, candidates={'tap_time_len': ORDER},
        tie_preference_by_target={'tap_time_len': ORDER}, reference_by_target={'tap_time_len': 'Q75'})
    tiers = classify_candidates(metrics, tier_spec, yaml.safe_load((Path(spec['main_root'])/'configs/candidate_tiers.yaml').read_text()))
    summary = dict(metrics=metrics, gains=gains, package_scores=scores, diagnostics=diagnostics, tiers=tiers,
        selected_for_confirmation=choose_confirmation(gains), G0='saved_state_audit_pending',
        G1='two_complete_development_seeds_not_formal_promotion', new_estimators=20, new_optimizer_runs=40,
        new_confirmation_seeds=0, full_data_fits=0, packages=0, desktop_writes=0, agent_uploads=0)
    write_new(out/'summary.json', summary)
    return summary


def execute(workspace, out):
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text())
    spec, _, _ = context(workspace, manifest)
    write_new(out/'activation.json', dict(started_ns=time.time_ns(), manifest_sha256=sha(out/'manifest.json')))
    try:
        for seed in spec['split_seeds']:
            for fold in range(5):
                for candidate in ORDER:
                    subprocess.run([sys.executable, str(Path(workspace)/'scripts/ema_average_span.py'),
                        '--workspace', str(workspace), '--output', str(out), 'worker',
                        '--seed', str(seed), '--fold', str(fold), '--candidate', candidate], check=True)
        summarize(workspace, out)
        subprocess.run([sys.executable, str(Path(workspace)/'scripts/audit_ema_average_span.py'), '--output', str(out)], check=True)
        write_new(out/'completion-event.json', dict(status='completed', completed_ns=time.time_ns(),
            summary_sha256=sha(out/'summary.json'), audit_sha256=sha(out/'audit.json')))
        print(json.dumps({'status': 'completed', 'output': str(out)}), flush=True)
    except BaseException as error:
        write_new(out/'failure.json', dict(error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare'); prepare_parser.add_argument('--receipt', type=Path, required=True)
    sub.add_parser('execute')
    worker_parser = sub.add_parser('worker')
    worker_parser.add_argument('--seed', type=int, required=True); worker_parser.add_argument('--fold', type=int, required=True)
    worker_parser.add_argument('--candidate', choices=ORDER, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare(args.workspace, args.receipt))
    elif args.command == 'execute': execute(args.workspace, args.output)
    else: worker(args.workspace, args.output, args.seed, args.fold, args.candidate)
