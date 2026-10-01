"""Paired EMA epoch selectors with honest, cached inner parent predictions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import yaml

from .component_regularization import ComponentRegressor, clone_state, inference_state, update_ema
from .component_regularization_run import RECIPE
from .data import TARGETS
from .ema_average_span import old_cache, task_frames
from .ema_evaluation_diagnostics import sha, write_new
from .ema_reference_artifacts import save_witness, audit_witness
from .ema_reference_ledger import KINDS, NativeLedger, native_hooks, reference_bindings, binding_sources
from .ema_training_scale import assert_isolated, event, require_memory
from .q75_combination_review import score
from .v3_4_bags import group_safe_inner_folds
from .v5_library import load_v5_training_frame, fold_vector
from .v5_spec import load_v5_spec
from .v5_resolution import paired_summary
from .v7_periodic import digest
from .v49_run import check_runtime, verify_reference_cache, read_reference, metric_detail

ARMS = ('COMPONENT_MAE', 'FUSION_MAE')
SPEC = 'configs/ema_fusion_selection/SPEC.json'


def validate_partitions(training, calibration, query):
    assert_isolated(training, calibration)
    assert_isolated(training, query)
    assert_isolated(calibration, query)


def select_step(state, value, epoch, patience, min_delta):
    """Independent original patience/min_delta policy, earliest strict best."""
    if not np.isfinite(value):
        raise ValueError('Nonfinite selection metric')
    if state['stopped_epoch'] is not None:
        return False
    improved = value < state['best']-min_delta
    if improved:
        state.update(best=value, selected_epoch=epoch, stale=0)
    else:
        state['stale'] += 1
    if state['stale'] >= patience:
        state['stopped_epoch'] = epoch
    return improved


def paired_selector(model, fitting, y, calibration, actual, remainder):
    """One identical training trajectory, two independently frozen stop policies.

    A stopped policy is never reconsidered. Continuing the common trajectory
    for the other policy cannot alter that policy's selected checkpoint.
    """
    import torch
    remainder = np.asarray(remainder, float)
    actual = np.asarray(actual, float)
    if (remainder.shape != (len(calibration),) or actual.shape != remainder.shape
            or not np.isfinite(remainder).all() or not np.isfinite(actual).all()):
        raise ValueError('Honest aligned inner parents required')
    x, cat = model._inputs(fitting)
    vx, vc = model._inputs(calibration)
    target = torch.as_tensor((y-model.mean_)/model.std_, dtype=torch.float32)
    vy = torch.as_tensor((actual[:, None]-model.mean_)/model.std_, dtype=torch.float32)
    rng = np.random.default_rng(model.settings['random_seed'])
    ema = clone_state(model.model_)
    states = {arm: dict(best=float('inf'), selected_epoch=0, stale=0, stopped_epoch=None) for arm in ARMS}
    checkpoints, history = {}, []
    for epoch in range(1, model.settings['max_epochs']+1):
        model.model_.train()
        order = rng.permutation(len(fitting))
        updates = 0
        for start in range(0, len(order), model.settings['batch_size']):
            idx = order[start:start+model.settings['batch_size']]
            model.optimizer_.zero_grad(set_to_none=True)
            from .v12_joint import joint_loss
            loss = joint_loss(model.model_(x[idx], cat[idx]), target[idx])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite EMA training loss')
            loss.backward()
            model.optimizer_.step()
            update_ema(model.model_, ema, model.mechanisms['ema_beta'])
            updates += 1
        model.model_.eval()
        with inference_state(model.model_, ema), torch.no_grad():
            standardized = model.model_(vx, vc).mean(1)
            component_mae = float((standardized-vy).abs().mean())
            prediction = standardized.numpy().astype(float)[:, 0]*model.std_[0]+model.mean_[0]
        fusion = remainder+.75*prediction
        if not np.isfinite(fusion).all() or (fusion < 0).any():
            raise ValueError('Invalid inner affine prediction; no clipping')
        fusion_mae = float(np.abs(actual-fusion).mean()/model.std_[0])
        values = dict(COMPONENT_MAE=component_mae, FUSION_MAE=fusion_mae)
        for arm in ARMS:
            if select_step(states[arm], values[arm], epoch, model.settings['patience'], model.settings['min_delta']):
                checkpoints[arm] = {k: v.clone() for k, v in ema.items()}
        history.append(dict(epoch=epoch, updates=updates, **values))
        if all(s['stopped_epoch'] is not None for s in states.values()):
            break
    for state in states.values():
        if state['stopped_epoch'] is None:
            state['stopped_epoch'] = epoch
        if state['selected_epoch'] < 1:
            raise ValueError('No selected EMA checkpoint')
    return states, checkpoints, history


def verify_files(files):
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Frozen source/data/cache changed: '+p)


def runtime(spec):
    import torch
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python 3.12 required')
    versions = check_runtime(spec)
    torch.set_num_threads(1)
    return versions


def freeze(workspace, main, out, tests_receipt):
    workspace, main, out = map(lambda p: Path(p).resolve(), (workspace, main, out))
    spec = json.loads((workspace/SPEC).read_text())
    original = yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text())
    if (spec['arms'] != list(ARMS) or spec['split_seeds'] != [42, 3407] or spec['folds'] != 5
            or spec['calibration_seed'] != 27001 or spec['replacement_weight'] != .75
            or spec['training'] != original['training']['tap_time_len']
            or spec['mechanisms'] != original['mechanisms'] or spec['optimizer_runs'] != 30
            or spec['monitor_seconds'] != 600 or spec['workers'] != 1
            or spec['maximum_runtime_seconds'] is not None):
        raise ValueError('Unregistered paired scientific scope')
    if out.exists() or not out.is_relative_to(main/'local/runs'):
        raise ValueError('Fresh private output required')
    current = json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current.get(k) != v for k, v in spec['reference'].items()):
        raise ValueError('Current platform reference changed before freeze')
    runtime(spec)
    require_memory(spec)
    verify_reference_cache(main, original)
    # Inherit verified historical data and source identities without changing them.
    inherited = json.loads((main/'configs/q75_error_relocation/CALIBRATION.json').read_text())['inputs']
    files = {str(main/p): h for p, h in inherited.items()}
    paths = list((workspace/'src').rglob('*.py'))
    paths += [workspace/SPEC, workspace/'docs/ema_fusion_selection/PREREGISTRATION.md',
              workspace/'scripts/ema_fusion_selection.py', workspace/'scripts/observe_ema_fusion_selection.py',
              workspace/'scripts/audit_q75_strategy_vectors.py', workspace/'uv.lock', workspace/'pyproject.toml',
              workspace/'tests/test_ema_fusion_selection.py']
    files.update({str(p): sha(p) for p in paths})
    frame = load_v5_training_frame(main)
    partitions = {}
    for seed in spec['split_seeds']:
        fv = fold_vector(main, frame, seed, load_v5_spec(main))
        for fold in range(5):
            training, query = task_frames(frame, fv, fold)
            inner = np.asarray(group_safe_inner_folds(training, seed=27001)['fold'])
            fitting = training.loc[inner != 0].reset_index(drop=True)
            calibration = training.loc[inner == 0].reset_index(drop=True)
            validate_partitions(fitting, calibration.drop(columns=list(TARGETS)), query)
            cache = main/spec['calibration_cache']/f'reference-calibration-s{seed}-f{fold}'
            read_reference(cache, fitting, calibration.drop(columns=list(TARGETS)), original)
            files.update({str(cache/n): sha(cache/n) for n in ('metadata.json', 'predictions.npz', 'complete.json')})
            for key in (f'reference-s{seed}-f{fold}', f'tap_time_len-EMA-s{seed}-f{fold}'):
                directory = main/spec['old_development']/key
                files.update({str(p): sha(p) for p in directory.iterdir() if p.suffix in ('.json', '.npz', '.pt')})
            old_cache(dict(main_root=str(main), old_development=spec['old_development']), seed, fold, training, query)
            partitions[f's{seed}-f{fold}'] = {name: digest(data.sample_id.tolist()) for name, data in
                [('fitting', fitting), ('calibration', calibration), ('training', training), ('query', query)]}
    files.update(binding_sources(reference_bindings()))
    model_sources = {p: h for p, h in files.items() if Path(p).suffix in ('.py', '.yaml', '.toml', '.lock')}
    verify_files(files)
    receipt = json.loads(Path(tests_receipt).read_text())
    if receipt.get('status') != 'passed' or receipt.get('python') != '3.12.12':
        raise ValueError('Passed locked Python 3.12 test receipt required')
    if any(receipt.get('tested_files', {}).get(str(p)) != sha(p) for p in paths):
        raise ValueError('Test receipt does not bind current scientific source')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(spec=spec, main_root=str(main), workspace=str(workspace), output=str(out), files=files,
        model_sources=model_sources,
        partitions=partitions, versions=runtime(spec), test_receipt_sha256=sha(tests_receipt),
        created_ns=time.time_ns(), authorization='user_requested_implementation_and_standing_optimization_grant',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip())
    manifest['identity'] = digest(manifest)
    write_new(out/'manifest.json', manifest)
    return manifest


def context(out, *, data=True):
    manifest = json.loads((Path(out)/'manifest.json').read_text())
    files = manifest['files'] if data else {p: h for p, h in manifest['files'].items() if Path(p).suffix not in ('.csv', '.xlsx')}
    verify_files(files)
    runtime(manifest['spec'])
    main = Path(manifest['main_root'])
    return manifest, load_v5_training_frame(main) if data else None


def worker(out, seed, fold):
    manifest, frame = context(out)
    spec, main, out = manifest['spec'], Path(manifest['main_root']), Path(out)
    if seed not in spec['split_seeds'] or fold not in range(5):
        raise ValueError('Undeclared paired task')
    require_memory(spec)
    unit = out/f's{seed}-f{fold}'; unit.mkdir(exist_ok=False)
    write_new(unit/'start.json', dict(pid=os.getpid(), manifest_sha256=sha(out/'manifest.json')))
    event(out/'events.jsonl', dict(event='unit_started', seed=seed, fold=fold))
    try:
        fv = fold_vector(main, frame, seed, load_v5_spec(main))
        training, query = task_frames(frame, fv, fold)
        inner = np.asarray(group_safe_inner_folds(training, seed=spec['calibration_seed'])['fold'])
        fitting = training.loc[inner != 0].reset_index(drop=True)
        calibration = training.loc[inner == 0].reset_index(drop=True)
        cal_query = calibration.drop(columns=list(TARGETS))
        validate_partitions(fitting, cal_query, query)
        part = {n: digest(d.sample_id.tolist()) for n, d in [('fitting', fitting),
            ('calibration', calibration), ('training', training), ('query', query)]}
        if part != manifest['partitions'][unit.name]:
            raise ValueError('Frozen training/calibration/query partition changed')
        original = yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text())
        values, metadata = read_reference(main/spec['calibration_cache']/f'reference-calibration-s{seed}-f{fold}',
            fitting, cal_query, original)
        remainder = .2*values['v36_time']+.3*values['n_time']-.25*values['v7_time']
        old = old_cache(dict(main_root=str(main), old_development=spec['old_development']), seed, fold, training, query)
        identity = dict(source_directory=str(out.resolve()), split_seed=seed, fold=fold, trial_id='PAIRED_SELECTOR')
        ledger = NativeLedger(unit/'native', identity=identity, expected={k: 3*int(k == 'torch_optimizer') for k in KINDS},
            training_ids=training.sample_id.tolist(), query_ids=query.sample_id.tolist(), source_hashes=manifest['model_sources'])
        witnesses, predictions = {}, {}
        with native_hooks(reference_bindings()):
            model = ComponentRegressor(RECIPE, spec['training'], 'EMA', spec['mechanisms'])
            with ledger.partition(fitting.sample_id.tolist(), calibration.sample_id.tolist()):
                model._initialize(fitting, fitting[['tap_time_len']].to_numpy())
            states, checkpoints, history = paired_selector(model, fitting, fitting[['tap_time_len']].to_numpy(),
                cal_query, calibration.tap_time_len.to_numpy(), remainder)
            for arm in ARMS:
                directory = unit/arm; directory.mkdir()
                model.model_.load_state_dict(checkpoints[arm]); model.model_.eval()
                trace = dict(**states[arm], history=history, fit_rows=len(fitting),
                    fit_ids_digest=part['fitting'], criterion=arm)
                model.save(directory/'selection.pt', trace)
                observed = model.predict(cal_query)
                witness_identity = dict(identity, trial_id=arm, fit_call_id='selection')
                witnesses[arm+'_selection'] = save_witness(model, cal_query, observed, directory/'selection-witness',
                    identity=witness_identity, training_ids=fitting.sample_id.tolist(), source_hashes=manifest['model_sources'],
                    full_batch_atol=0, row_atol=spec['cold_predict_atol'], fit_metadata=trace)
                refit = ComponentRegressor(RECIPE, spec['training'], 'EMA', spec['mechanisms'], directory)
                with ledger.partition(training.sample_id.tolist()):
                    refit._initialize(training, training[['tap_time_len']].to_numpy())
                refit._train(training, training[['tap_time_len']].to_numpy(), states[arm]['selected_epoch'])
                observed = refit.predict(query)
                predictions[arm] = observed[:, 0]
                witnesses[arm+'_refit'] = save_witness(refit, query, observed, directory/'refit-witness',
                    identity=dict(identity, trial_id=arm, fit_call_id='refit'), training_ids=training.sample_id.tolist(),
                    source_hashes=manifest['model_sources'], full_batch_atol=0, row_atol=spec['cold_predict_atol'],
                    fit_metadata=refit.traces['refit'])
        native = ledger.close()
        columns = {arm: old['q75']+.75*(predictions[arm]-old['ema']) for arm in ARMS}
        if any(not np.isfinite(p).all() or (p < 0).any() for p in columns.values()):
            raise ValueError('Invalid outer affine prediction; no clipping')
        with (unit/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, query_ids=query.sample_id.to_numpy(str), calibration_ids=cal_query.sample_id.to_numpy(str),
                calibration_actual=calibration.tap_time_len.to_numpy(), calibration_remainder=remainder,
                q75=old['q75'], iron=old['iron'], old_ema=old['ema'],
                **{arm+'_ema': predictions[arm] for arm in ARMS}, **columns)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > spec['max_worker_rss_mib']:
            raise ValueError('Worker memory gate failed')
        write_new(unit/'warm-complete.json', dict(status='passed', pid=os.getpid(), partition=part, states=states,
            parent_cache_metadata=metadata, predictions_sha256=sha(unit/'predictions.npz'),
            checkpoints={str(p.relative_to(unit)): sha(p) for p in unit.glob('*/*.pt')},
            witnesses=witnesses, native_counts=native['counts'], native_receipt_sha256=sha(unit/'native/scope-complete.json'),
            peak_rss_mib=peak, manifest_sha256=sha(out/'manifest.json')))
        event(out/'events.jsonl', dict(event='unit_warm_completed', seed=seed, fold=fold, optimizer_runs=3))
    except BaseException as error:
        write_new(unit/'failure.json', dict(error=repr(error)))
        event(out/'events.jsonl', dict(event='unit_failed', seed=seed, fold=fold, error=repr(error)))
        raise


def cold(out, seed, fold):
    manifest, _ = context(out, data=False)
    unit = Path(out)/f's{seed}-f{fold}'
    warm = json.loads((unit/'warm-complete.json').read_text())
    if os.getpid() == warm['pid'] or warm['manifest_sha256'] != sha(Path(out)/'manifest.json') or (unit/'failure.json').exists():
        raise ValueError('Independent cold process with successful warm receipt required')
    if sha(unit/'predictions.npz') != warm['predictions_sha256']:
        raise ValueError('Warm prediction identity changed')
    verify_files({str(unit/p): h for p, h in warm['checkpoints'].items()})
    for name, receipt in warm['witnesses'].items():
        arm, phase = name.rsplit('_', 1)
        audit_witness(unit/arm/(phase+'-witness'), receipt)
    native = json.loads((unit/'native/scope-complete.json').read_text())
    if (sha(unit/'native/scope-complete.json') != warm['native_receipt_sha256']
            or native['counts'] != {k: 3*int(k == 'torch_optimizer') for k in KINDS}):
        raise ValueError('Actual native fit budget did not close')
    verify_files({str(unit/'native'/p): h for p, h in native['call_hashes'].items()})
    with np.load(unit/'predictions.npz', allow_pickle=False) as saved:
        for arm in ARMS:
            observed = np.load(unit/arm/'refit-witness/observed.npy', allow_pickle=False)
            np.testing.assert_array_equal(saved[arm+'_ema'], observed[:, 0])
            np.testing.assert_array_equal(saved[arm], saved['q75']+.75*(saved[arm+'_ema']-saved['old_ema']))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak > manifest['spec']['max_worker_rss_mib']:
        raise ValueError('Cold worker memory gate failed')
    write_new(unit/'cold-complete.json', dict(status='passed', pid=os.getpid(), warm_receipt_sha256=sha(unit/'warm-complete.json'),
        retained_states=4, new_fits=0, native_counts=native['counts'], peak_rss_mib=peak))


def evaluate(out):
    manifest, frame = context(out)
    spec, main = manifest['spec'], Path(manifest['main_root'])
    metrics = {'tap_time_len': {name: {} for name in ('Q75', *ARMS)}}
    gains, paired_gains, epochs = {}, {}, {}
    for seed in spec['split_seeds']:
        fv = fold_vector(main, frame, seed, load_v5_spec(main))
        vectors = {k: np.full(len(frame), np.nan) for k in ('q75', 'iron', *ARMS)}
        for fold in range(5):
            unit = Path(out)/f's{seed}-f{fold}'
            warm = json.loads((unit/'warm-complete.json').read_text())
            checked = json.loads((unit/'cold-complete.json').read_text())
            if checked['status'] != 'passed' or checked['warm_receipt_sha256'] != sha(unit/'warm-complete.json'):
                raise ValueError('Complete independent cold coverage required')
            epochs[unit.name] = warm['states']
            with np.load(unit/'predictions.npz', allow_pickle=False) as saved:
                if saved['query_ids'].tolist() != frame.loc[fv == fold, 'sample_id'].tolist():
                    raise ValueError('Complete same-seed OOF order required')
                for k in vectors:
                    vectors[k][fv == fold] = saved[k]
        if any(not np.isfinite(v).all() for v in vectors.values()):
            raise ValueError('Incomplete full seed')
        for name, key in [('Q75', 'q75'), *[(arm, arm) for arm in ARMS]]:
            metrics['tap_time_len'][name][str(seed)] = metric_detail(frame.tap_time_len.to_numpy(), vectors[key], fv, frame.spout_no.to_numpy())
        scores = {k: score(frame[list(TARGETS)].to_numpy(), vectors['iron'], vectors[k]) for k in ('q75', *ARMS)}
        gains[str(seed)] = {arm: scores[arm]-scores['q75'] for arm in ARMS}
        paired_gains[str(seed)] = scores['FUSION_MAE']-scores['COMPONENT_MAE']
        with (Path(out)/f'oof-s{seed}.npz').open('xb') as stream:
            np.savez_compressed(stream, actual=frame[list(TARGETS)].to_numpy(), query_ids=frame.sample_id.to_numpy(str), folds=fv, **vectors)
    from .candidate_tiers import classify_candidates
    policy = yaml.safe_load((main/'configs/candidate_tiers.yaml').read_text())
    tier_spec = dict(split_seeds=spec['split_seeds'], folds=5, candidates={'tap_time_len': ['FUSION_MAE']},
        tie_preference_by_target={'tap_time_len': ['FUSION_MAE']}, reference_by_target={'tap_time_len': 'Q75'})
    eligible = all(g['FUSION_MAE'] > 0 for g in gains.values()) and all(g > 0 for g in paired_gains.values())
    report = dict(status='completed_paired_development', G0='passed_native_budget_and_independent_cold',
        gains_vs_original_q75=gains, fusion_minus_matched_control=paired_gains, epochs=epochs,
        paired_vs_control=paired_summary(list(paired_gains.values())), metrics=metrics,
        confirmation_eligible=eligible, formal_promotion=False, optimizer_runs=30,
        tiers=classify_candidates(metrics, tier_spec, policy), packages=0, uploads=0,
        interpretation='control uses common calibration_seed27001; isolate criterion using paired control, and report split change separately')
    write_new(Path(out)/'report.json', report)
    return report


def execute(out):
    manifest, _ = context(out, data=False)
    for seed in manifest['spec']['split_seeds']:
        for fold in range(5):
            for phase in ('worker', 'cold'):
                subprocess.run([sys.executable, '-m', 'bf_tap_r2.ema_fusion_selection', '--'+phase, '--output', str(out),
                    '--seed', str(seed), '--fold', str(fold)], check=True)
    evaluate(out)
    write_new(Path(out)/'completion-event.json', dict(status='completed', new_optimizer_runs=30, packages=0, uploads=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--workspace', default=Path(__file__).resolve().parents[2])
    parser.add_argument('--main-root', default='/home/lux1/iron')
    parser.add_argument('--tests-receipt')
    parser.add_argument('--seed', type=int); parser.add_argument('--fold', type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'worker', 'cold'):
        mode.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if args.prepare:
        freeze(args.workspace, args.main_root, args.output, args.tests_receipt)
    elif args.worker:
        worker(args.output, args.seed, args.fold)
    elif args.cold:
        cold(args.output, args.seed, args.fold)
    else:
        execute(args.output)


if __name__ == '__main__':
    main()
