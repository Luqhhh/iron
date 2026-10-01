"""Separate paired initialization and exact-update EMA refit experiments.

Original scientific trainers and cached runs remain immutable. Each phase has
its own frozen manifest, budget, source identity and terminal event.
"""
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

from .component_regularization import ComponentRegressor, clone_state, update_ema
from .component_regularization_audit import verify_saved
from .component_regularization_run import RECIPE
from .data import FEATURES, TARGETS
from .ema_average_span import candidate_column, old_cache, task_frames
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .ema_reference_artifacts import audit_witness, save_witness
from .ema_reference_ledger import KINDS, NativeLedger, binding_sources, native_hooks, reference_bindings
from .ema_training_scale import assert_isolated, event, require_memory
from .q75_combination_review import score
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import paired_summary
from .v5_spec import load_v5_spec
from .v7_periodic import digest
from .v49_run import metric_detail, unit_id, verified_unit

PHASES = ('initialization', 'steps')
INITIALIZATIONS = (42, 1042, 2042)
SPLITS = (42, 3407)
SPEC = 'configs/ema_retraining_validation/{phase}.json'
PROTOCOL = 'docs/ema_retraining_validation/{phase}/PREREGISTRATION.md'


def validate_scope(spec, original):
    phase = spec['phase']
    if phase not in PHASES:
        raise ValueError('Unregistered phase')
    expected = dict(split_seeds=list(SPLITS), folds=5, replacement_weight=.75,
        workers=1, numerical_threads=1, monitor_seconds=600,
        maximum_runtime_seconds=None, automatic_retries=False,
        max_worker_rss_mib=1536, cold_predict_atol=.0005,
        training=original['training']['tap_time_len'], mechanisms=original['mechanisms'])
    expected.update({k: 0 for k in ('new_confirmation_seeds', 'full_data_fits', 'packages', 'desktop_writes', 'uploads')})
    expected.update(dict(training_seeds=list(INITIALIZATIONS), new_estimators=40,
        optimizer_runs=80, reused_estimators=20, retained_states=120, arms=['BASE', 'EMA']) if phase == 'initialization'
        else dict(training_seeds=[42], new_estimators=10, optimizer_runs=10,
        reused_estimators=10, retained_states=30, arms=['EMA_EPOCH', 'EMA_UPDATES']))
    if any(spec.get(k) != value for k, value in expected.items()):
        raise ValueError('Frozen scope/trainer/weight/budget changed')


def source_files(workspace, phase):
    workspace = Path(workspace)
    paths = list((workspace/'src').rglob('*.py'))
    paths += [workspace/SPEC.format(phase=phase), workspace/PROTOCOL.format(phase=phase),
        workspace/'scripts/ema_retraining_validation.py', workspace/'scripts/observe_ema_retraining_validation.py',
        workspace/'scripts/observe_ema_fusion_selection.py',
        workspace/'scripts/audit_ema_retraining_vectors.py', workspace/'tests/test_ema_retraining_validation.py',
        workspace/'pyproject.toml', workspace/'uv.lock']
    return {str(p.resolve()): sha(p) for p in paths}


def partitions(training, query, settings):
    inner = np.asarray(group_safe_inner_folds(training, seed=settings['inner_seed'])['fold'])
    fitting = training.loc[inner != 0].reset_index(drop=True)
    calibration = training.loc[inner == 0].reset_index(drop=True)
    assert_isolated(fitting, calibration.drop(columns=list(TARGETS)))
    assert_isolated(fitting, query)
    assert_isolated(calibration, query)
    hashes = {k: digest(v.sample_id.tolist()) for k, v in
        [('fitting', fitting), ('calibration', calibration), ('training', training), ('query', query)]}
    return fitting, calibration, hashes


def require_previous(spec):
    path = Path(spec['previous_terminal'])
    previous = json.loads(path.read_text())
    if previous['status'] != 'passed' or previous['actual_exit_codes'] != [0, 0]:
        raise ValueError('Prior phase actual successful terminal/audit required')
    if 'report_sha256' in previous and sha(path.parent/'report.json') != previous['report_sha256']:
        raise ValueError('Prior terminal report identity mismatch')
    return sha(path)


def freeze(workspace, main, out, phase, tests_receipt):
    workspace, main, out = map(lambda p: Path(p).resolve(), (workspace, main, out))
    spec = json.loads((workspace/SPEC.format(phase=phase)).read_text())
    original = yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text())
    validate_scope(spec, original)
    if out.exists() or not out.is_relative_to(main/'local/runs'):
        raise ValueError('Fresh private run required')
    current = json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current.get(k) != v for k, v in spec['reference'].items()):
        raise ValueError('Latest reference changed before freeze')
    runtime(spec); require_memory(spec)
    previous_sha = require_previous(spec)
    sources = source_files(workspace, phase)
    receipt = json.loads(Path(tests_receipt).read_text())
    if (receipt.get('status') != 'passed' or receipt.get('python') != sys.version.split()[0]
            or receipt.get('sources') != sources or receipt.get('full_suite') is not True):
        raise ValueError('Exact-source locked Python3.12 full tests required')
    old_root = main/spec['old_development']
    old_manifest = json.loads((old_root/'manifest.json').read_text())
    old_audit = json.loads((old_root/'audit.json').read_text())
    if old_audit['status'] != 'passed' or old_audit['manifest_sha256'] != sha(old_root/'manifest.json'):
        raise ValueError('Original audited cache identity required')
    # Pin the actual trainer/recipe/preprocessor sources, not current documentation.
    for name in ('component_regularization.py', 'v12_joint.py', 'v7_periodic.py', 'v3_6_networks.py'):
        relative = 'src/bf_tap_r2/'+name
        if sha(workspace/relative) != old_manifest['source_hashes'][relative]:
            raise ValueError('Original scientific dependency changed: '+relative)
    files = dict(sources)
    files.update({str(main/p): h for p, h in old_manifest['data_hashes'].items()})
    for p in (main/'configs/protection.yaml', main/'configs/strong_component_regularization/SPEC.yaml',
              main/'configs/data.local.yaml',
              main/'configs/round2_v5/SPEC.yaml', old_root/'manifest.json', old_root/'audit.json',
              old_root/'summary.json', main/'local/runs/strong-component-regularization/preflight-r1/report.json'):
        files[str(p)] = sha(p)
    preflight = json.loads((main/'local/runs/strong-component-regularization/preflight-r1/report.json').read_text())
    if preflight['status'] != 'passed':
        raise ValueError('Original full-size engineering admission required')
    frame = load_v5_training_frame(main)
    plan, fold_hashes = {}, {}
    for seed in SPLITS:
        fv = fold_vector(main, frame, seed, load_v5_spec(main)); fold_hashes[str(seed)] = digest(fv.tolist())
        for fold in range(5):
            training, query = task_frames(frame, fv, fold)
            _, _, hashes = partitions(training, query, spec['training'])
            plan[f's{seed}-f{fold}'] = hashes
            for arm in (('BASE', 'EMA') if phase == 'initialization' else ('EMA',)):
                directory = old_root/f'tap_time_len-{arm}-s{seed}-f{fold}'
                if not verified_unit(directory, unit_id(old_manifest, directory.name)):
                    raise ValueError('Original cache unit incomplete')
                for p in directory.iterdir():
                    if p.is_file(): files[str(p)] = sha(p)
            directory = old_root/f'reference-s{seed}-f{fold}'
            for p in directory.iterdir():
                if p.is_file(): files[str(p)] = sha(p)
            old_cache(dict(main_root=str(main), old_development=spec['old_development']), seed, fold, training, query)
        for name in (f'folds-{seed}.csv',):
            p = main/'local/runs/round2-v2/comparison-r1'/name; files[str(p)] = sha(p)
    files.update(binding_sources(reference_bindings()))
    verify_files(files)
    model_sources = {p: h for p, h in files.items() if Path(p).suffix in ('.py', '.yaml', '.toml', '.lock')}
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(spec=spec, workspace=str(workspace), main_root=str(main), output=str(out),
        files=files, model_sources=model_sources, partitions=plan, fold_hashes=fold_hashes,
        previous_terminal_sha256=previous_sha, test_receipt_sha256=sha(tests_receipt),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip(),
        created_ns=time.time_ns(), authorization='user_20261002_execute_priority_and_standing_optimization_grant')
    manifest['identity'] = digest(manifest)
    write_new(out/'manifest.json', manifest)
    return manifest


def context(out, data=True):
    manifest = json.loads((Path(out)/'manifest.json').read_text())
    core = dict(manifest); identity = core.pop('identity')
    if digest(core) != identity:
        raise ValueError('Frozen manifest identity changed')
    files = manifest['files'] if data else {p: h for p, h in manifest['files'].items() if Path(p).suffix not in ('.csv', '.xlsx')}
    verify_files(files); runtime(manifest['spec'])
    main = Path(manifest['main_root'])
    return manifest, load_v5_training_frame(main) if data else None


def unit_name(seed, fold, init, arm):
    return f's{seed}-f{fold}-init{init}-{arm}'


def save_prediction_witness(model, query, observed, directory, manifest, seed, fold, trial, phase, ids, trace):
    return save_witness(model, query, observed, directory/(phase+'-witness'),
        identity=dict(source_directory=manifest['output'], split_seed=seed, fold=fold, trial_id=trial, fit_call_id=phase),
        training_ids=ids, source_hashes=manifest['model_sources'], full_batch_atol=0,
        row_atol=manifest['spec']['cold_predict_atol'], fit_metadata=trace)


def original_states(manifest, directory, training, calibration, fitting, query, seed, fold, arm):
    spec = manifest['spec']; root = Path(manifest['main_root'])/spec['old_development']
    original = root/f'tap_time_len-{arm}-s{seed}-f{fold}'
    old_manifest = json.loads((root/'manifest.json').read_text())
    if not verified_unit(original, unit_id(old_manifest, original.name)):
        raise ValueError('Original cached fit identity invalid')
    metadata = json.loads((original/'metadata.json').read_text())
    if metadata['training_seed'] != 42 or metadata['model']['fit_ids_digest'] != digest(training.sample_id.tolist()):
        raise ValueError('Original training seed/IDs mismatch')
    selector = verify_saved(original/'selection.pt', fitting, fitting[['tap_time_len']].to_numpy(),
        arm, spec['training'], spec['mechanisms'], calibration)
    epoch = selector.saved['trace']['selected_epoch']
    refit = verify_saved(original/'refit.pt', training, training[['tap_time_len']].to_numpy(),
        arm, spec['training'], spec['mechanisms'], expected_epoch=epoch)
    for phase, model in (('selection', selector), ('refit', refit)):
        if model.saved['trace'] != metadata['model']['traces'][phase]:
            raise ValueError('Original trace identity mismatch')
    observed = refit.predict(query)
    with np.load(original/'predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['query_ids'], query.sample_id.to_numpy(str))
        np.testing.assert_array_equal(saved['prediction'], observed)
    witnesses = {}
    for phase, model, q, ids in (('selection', selector, calibration.drop(columns=list(TARGETS)), fitting.sample_id.tolist()),
                                ('refit', refit, query, training.sample_id.tolist())):
        trace = model.saved['trace']
        witnesses[phase] = save_prediction_witness(model, q, model.predict(q), directory, manifest,
            seed, fold, arm+'_42', phase, ids, trace)
    return selector, refit, observed[:, 0], witnesses, original


def selected_updates(trace, batch_size):
    selected = trace['selected_epoch']
    expected = math.ceil(trace['fit_rows']/batch_size)
    rows = trace['history'][:selected]
    if len(rows) != selected or [r['epoch'] for r in rows] != list(range(1, selected+1)):
        raise ValueError('Complete selected prefix required')
    if any(r['updates'] != expected for r in rows):
        raise ValueError('Unexpected selector update count')
    return sum(r['updates'] for r in rows)


class StepMatchedRegressor(ComponentRegressor):
    """Fresh refit at the exact selector update budget, including partial epoch."""

    def train_updates(self, frame, y, updates):
        import torch
        from .v12_joint import joint_loss
        if self.arm != 'EMA' or type(updates) is not int or updates < 1:
            raise ValueError('Positive integral EMA update budget required')
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings['random_seed'])
        ema = clone_state(self.model_); done = 0; history = []
        batches = math.ceil(len(frame)/self.settings['batch_size'])
        for epoch in range(1, math.ceil(updates/batches)+1):
            self.model_.train(); order = rng.permutation(len(frame)); losses = []
            examples = 0
            for start in range(0, len(order), self.settings['batch_size']):
                idx = order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = joint_loss(self.model_(x[idx], cat[idx]), target[idx])
                if not torch.isfinite(loss): raise ValueError('Nonfinite update-matched loss')
                loss.backward(); self.optimizer_.step()
                update_ema(self.model_, ema, self.mechanisms['ema_beta'])
                losses.append(float(loss.detach())); done += 1; examples += len(idx)
                if done == updates: break
            self.model_.eval()
            from .component_regularization import inference_state
            with inference_state(self.model_, ema), torch.no_grad():
                prediction = self.model_(x, cat).mean(1)
                train_mae = float((prediction-target).abs().mean())
                train_mse = float((prediction-target).square().mean())
            history.append(dict(epoch=epoch, updates=len(losses), gradient_evaluations=len(losses),
                examples=examples, training_eval_mae=train_mae, training_eval_mse=train_mse,
                mean_batch_training_loss=float(np.mean(losses))))
        self.model_.load_state_dict(ema); self.model_.eval()
        trace = dict(history=history, selected_epoch=epoch, stopped_epoch=epoch,
            fit_rows=len(frame), fit_ids_digest=digest(frame.sample_id.tolist()), updates=done,
            gradient_evaluations=done, target_updates=updates, full_epoch_updates=batches,
            stopping_rule='exact_selector_selected_prefix_updates',
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        validate_update_trace(trace, updates, len(frame), self.settings['batch_size'])
        self.traces['refit'] = trace
        self.save(self.directory/'refit.pt', trace)
        return trace


def validate_update_trace(trace, updates, rows, batch_size):
    batches = math.ceil(rows/batch_size)
    count = math.ceil(updates/batches)
    expected = [batches]*(count-1)+[updates-batches*(count-1)]
    if (trace['updates'] != updates or trace['target_updates'] != updates or trace['fit_rows'] != rows
            or trace['selected_epoch'] != count or trace['stopped_epoch'] != count
            or [r['epoch'] for r in trace['history']] != list(range(1, count+1))
            or [r['updates'] for r in trace['history']] != expected
            or [r['gradient_evaluations'] for r in trace['history']] != expected
            or trace['gradient_evaluations'] != updates):
        raise ValueError('Exact selected-update stopping failed')


def worker(out, seed, fold, init, arm):
    manifest, frame = context(out); spec = manifest['spec']; main = Path(manifest['main_root']); out = Path(out)
    if seed not in spec['split_seeds'] or fold not in range(5) or init not in spec['training_seeds']:
        raise ValueError('Undeclared split/training seed task')
    phase = spec['phase']
    if arm not in (('BASE', 'EMA') if phase == 'initialization' else ('EMA_UPDATES',)):
        raise ValueError('Undeclared arm')
    directory = out/unit_name(seed, fold, init, arm); directory.mkdir(exist_ok=False)
    write_new(directory/'start.json', dict(pid=os.getpid(), manifest_sha256=sha(out/'manifest.json')))
    event(out/'events.jsonl', dict(event='unit_started', unit=directory.name))
    try:
        require_memory(spec)
        fv = fold_vector(main, frame, seed, load_v5_spec(main))
        if digest(fv.tolist()) != manifest['fold_hashes'][str(seed)]: raise ValueError('Outer folds changed')
        training, query = task_frames(frame, fv, fold)
        fitting, calibration, hashes = partitions(training, query, spec['training'])
        if hashes != manifest['partitions'][f's{seed}-f{fold}']: raise ValueError('Inner partitions changed')
        old = old_cache(dict(main_root=str(main), old_development=spec['old_development']), seed, fold, training, query)
        settings = dict(spec['training'], random_seed=init); witnesses = {}; paths = {}
        reused = phase == 'initialization' and init == 42
        native = None
        if reused or phase == 'steps':
            cached_arm = arm if reused else 'EMA'
            selector, refit, prediction, witnesses, original = original_states(manifest, directory,
                training, calibration, fitting, query, seed, fold, cached_arm)
            paths = {p: str(original/(p+'.pt')) for p in ('selection', 'refit')}
            if cached_arm == 'EMA': np.testing.assert_array_equal(prediction, old['ema'])
            else:
                with np.load(main/spec['old_development']/f'reference-s{seed}-f{fold}/predictions.npz') as saved:
                    np.testing.assert_array_equal(prediction, saved['v7_time'])
        if not reused:
            budget = 2 if phase == 'initialization' else 1
            ledger = NativeLedger(directory/'native',
                identity=dict(source_directory=manifest['output'], split_seed=seed, fold=fold, trial_id=f'{arm}_INIT{init}'),
                expected={k: budget*int(k == 'torch_optimizer') for k in KINDS},
                training_ids=training.sample_id.tolist(), query_ids=query.sample_id.tolist(), source_hashes=manifest['model_sources'])
            with native_hooks(reference_bindings()):
                if phase == 'initialization':
                    model = ComponentRegressor(RECIPE, settings, arm, spec['mechanisms'], directory)
                    with ledger.partition(fitting.sample_id.tolist(), calibration.sample_id.tolist()):
                        model._initialize(fitting, fitting[['tap_time_len']].to_numpy())
                    epoch = model._train(fitting, fitting[['tap_time_len']].to_numpy(), settings['max_epochs'],
                        (calibration, calibration[['tap_time_len']].to_numpy()))
                    witnesses['selection'] = save_prediction_witness(model, calibration.drop(columns=list(TARGETS)),
                        model.predict(calibration.drop(columns=list(TARGETS))), directory, manifest, seed, fold,
                        f'{arm}_INIT{init}', 'selection', fitting.sample_id.tolist(), model.traces['selection'])
                    with ledger.partition(training.sample_id.tolist()):
                        model._initialize(training, training[['tap_time_len']].to_numpy())
                    model._train(training, training[['tap_time_len']].to_numpy(), epoch)
                    paths = {p: str(directory/(p+'.pt')) for p in ('selection', 'refit')}
                    refit = model
                else:
                    updates = selected_updates(selector.saved['trace'], settings['batch_size'])
                    model = StepMatchedRegressor(RECIPE, settings, 'EMA', spec['mechanisms'], directory)
                    with ledger.partition(training.sample_id.tolist()):
                        model._initialize(training, training[['tap_time_len']].to_numpy())
                    model.train_updates(training, training[['tap_time_len']].to_numpy(), updates)
                    paths['matched'] = str(directory/'refit.pt')
                    refit = model
                prediction = refit.predict(query)[:, 0]
                key = 'refit' if phase == 'initialization' else 'matched'
                witnesses[key] = save_prediction_witness(refit, query, prediction[:, None], directory, manifest,
                    seed, fold, f'{arm}_INIT{init}', key, training.sample_id.tolist(), refit.traces['refit'])
            native = ledger.close()
        column = candidate_column(old['q75'], old['ema'], prediction)
        with (directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, query_ids=query.sample_id.to_numpy(str), prediction=prediction,
                q75=old['q75'], old_ema=old['ema'], iron=old['iron'], candidate=column)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > spec['max_worker_rss_mib']: raise ValueError('Worker RSS gate failed')
        metadata = dict(status='passed', pid=os.getpid(), seed=seed, fold=fold, training_seed=init, arm=arm,
            paths=paths, checkpoint_hashes={p: sha(p) for p in paths.values()}, witnesses=witnesses,
            predictions_sha256=sha(directory/'predictions.npz'), manifest_sha256=sha(out/'manifest.json'),
            partitions=hashes, reused=reused, optimizer_runs=0 if reused else (2 if phase == 'initialization' else 1),
            native_counts=None if native is None else native['counts'],
            native_receipt_sha256=None if native is None else sha(directory/'native/scope-complete.json'), peak_rss_mib=peak)
        write_new(directory/'warm-complete.json', metadata)
        event(out/'events.jsonl', dict(event='unit_warm_completed', unit=directory.name, optimizer_runs=metadata['optimizer_runs']))
    except BaseException as error:
        write_new(directory/'failure.json', dict(error=repr(error)))
        event(out/'events.jsonl', dict(event='unit_failed', unit=directory.name, error=repr(error)))
        raise


def cold(out, seed, fold, init, arm):
    manifest, _ = context(out, data=False); out = Path(out)
    directory = out/unit_name(seed, fold, init, arm)
    warm = json.loads((directory/'warm-complete.json').read_text())
    if warm['pid'] == os.getpid() or warm['manifest_sha256'] != sha(out/'manifest.json') or (directory/'failure.json').exists():
        raise ValueError('Independent successful warm identity required')
    verify_files(warm['checkpoint_hashes'])
    if sha(directory/'predictions.npz') != warm['predictions_sha256']: raise ValueError('Warm prediction changed')
    def no_training(*args, **kwargs):
        raise ValueError('Cold predictor attempted initialization/update training')
    initialize = ComponentRegressor._initialize
    train_updates = StepMatchedRegressor.train_updates
    ComponentRegressor._initialize = no_training
    StepMatchedRegressor.train_updates = no_training
    try:
        for phase, expected in warm['witnesses'].items():
            audit_witness(directory/(phase+'-witness'), expected)
    finally:
        ComponentRegressor._initialize = initialize
        StepMatchedRegressor.train_updates = train_updates
    if warm['native_counts'] is not None:
        native = json.loads((directory/'native/scope-complete.json').read_text())
        if (sha(directory/'native/scope-complete.json') != warm['native_receipt_sha256']
                or native['counts'] != {k: warm['optimizer_runs']*int(k == 'torch_optimizer') for k in KINDS}):
            raise ValueError('Actual native optimizer budget failed')
        verify_files({str(directory/'native'/p): h for p, h in native['call_hashes'].items()})
    with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
        phase = 'refit' if manifest['spec']['phase'] == 'initialization' else 'matched'
        observed = np.load(directory/(phase+'-witness')/'observed.npy', allow_pickle=False)
        np.testing.assert_array_equal(saved['prediction'], observed[:, 0])
        np.testing.assert_array_equal(saved['candidate'], candidate_column(saved['q75'], saved['old_ema'], saved['prediction']))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak > manifest['spec']['max_worker_rss_mib']: raise ValueError('Cold worker RSS gate failed')
    write_new(directory/'cold-complete.json', dict(status='passed', pid=os.getpid(), warm_sha256=sha(directory/'warm-complete.json'),
        cold_states=len(warm['witnesses']), optimizer_runs=warm['optimizer_runs'], peak_rss_mib=peak, new_fits=0))


def validate_checkpoints(manifest, directory, training, query, seed, fold, init, arm):
    spec = manifest['spec']; fitting, calibration, _ = partitions(training, query, spec['training'])
    warm = json.loads((directory/'warm-complete.json').read_text())
    verify_files(warm['checkpoint_hashes'])
    settings = dict(spec['training'], random_seed=init)
    actual_arm = arm if spec['phase'] == 'initialization' else 'EMA'
    selector = verify_saved(warm['paths']['selection'], fitting, fitting[['tap_time_len']].to_numpy(),
        actual_arm, settings, spec['mechanisms'], calibration)
    trace = selector.saved['trace']; epoch = trace['selected_epoch']
    refit = verify_saved(warm['paths']['refit'], training, training[['tap_time_len']].to_numpy(),
        actual_arm, settings, spec['mechanisms'], expected_epoch=epoch)
    result = dict(selected_epoch=epoch, stopped_epoch=trace['stopped_epoch'],
        selector_selected_updates=selected_updates(trace, settings['batch_size']),
        selector_total_updates=trace['updates'], epoch_refit_updates=refit.saved['trace']['updates'])
    if spec['phase'] == 'steps':
        saved = ComponentRegressor.load(warm['paths']['matched']).saved
        if saved['settings'] != settings or saved['mechanisms'] != spec['mechanisms'] or saved['recipe'] != RECIPE or saved['arm'] != 'EMA':
            raise ValueError('Matched model recipe mismatch')
        if saved['preprocessing'] != NumericPreprocessor(structure='raw_tabm').fit(training).metadata():
            raise ValueError('Matched train-only preprocessing mismatch')
        y = training[['tap_time_len']].to_numpy()
        np.testing.assert_array_equal(saved['mean'], y.mean(axis=0)); np.testing.assert_array_equal(saved['std'], y.std(axis=0))
        if saved['trace']['fit_ids_digest'] != digest(training.sample_id.tolist()): raise ValueError('Matched fit IDs mismatch')
        validate_update_trace(saved['trace'], result['selector_selected_updates'], len(training), settings['batch_size'])
        result['matched_refit_updates'] = saved['trace']['updates']
        result['matched_last_epoch_updates'] = saved['trace']['history'][-1]['updates']
    return result


def summarize_initialization(actual, components, q75, old_ema, iron, folds, spouts):
    """All averaging stays inside one outer split; no best seed selection."""
    if set(components) != {f'{arm}_{init}' for arm in ('BASE', 'EMA') for init in INITIALIZATIONS}:
        raise ValueError('Complete prespecified paired initialization pool required')
    components = dict(components)
    for arm in ('BASE', 'EMA'):
        components[arm+'_MEAN3'] = np.mean([components[f'{arm}_{init}'] for init in INITIALIZATIONS], axis=0)
    if any(np.asarray(v).shape != q75.shape for v in components.values()): raise ValueError('Component alignment mismatch')
    columns = {k: candidate_column(q75, old_ema, v) for k, v in components.items()}
    metrics = {k: metric_detail(actual[:, 1], v, folds, spouts) for k, v in columns.items()}
    scores = {k: score(actual, iron, v) for k, v in columns.items()}
    baseline = score(actual, iron, q75)
    paired = {str(init): dict(component_gain=50*(np.abs(actual[:, 1]-components[f'BASE_{init}']).sum()-
        np.abs(actual[:, 1]-components[f'EMA_{init}']).sum())/np.abs(actual[:, 1]).sum(),
        fusion_gain=scores[f'EMA_{init}']-scores[f'BASE_{init}']) for init in INITIALIZATIONS}
    return columns, dict(gains_vs_q75={k: s-baseline for k, s in scores.items()}, scores=scores,
        ema_minus_paired_base=paired, equal_mean_ema_minus_equal_mean_base=scores['EMA_MEAN3']-scores['BASE_MEAN3'],
        mean_ema_minus_each_single={str(i): scores['EMA_MEAN3']-scores[f'EMA_{i}'] for i in INITIALIZATIONS},
        component_mae={k: float(np.abs(actual[:, 1]-v).mean()) for k, v in components.items()}, metrics=metrics)


def evaluate(out):
    manifest, frame = context(out); spec = manifest['spec']; main = Path(manifest['main_root']); out = Path(out)
    records, traces, vectors_hashes = {}, {}, {}
    total_native = 0; total_states = 0; seen = 0
    for seed in spec['split_seeds']:
        fv = fold_vector(main, frame, seed, load_v5_spec(main))
        vectors = {k: np.full(len(frame), np.nan) for k in ('q75', 'old_ema', 'iron')}
        names = [f'{arm}_{i}' for i in INITIALIZATIONS for arm in ('BASE', 'EMA')] if spec['phase'] == 'initialization' else ['EMA_UPDATES']
        components = {k: np.full(len(frame), np.nan) for k in names}
        for fold in range(5):
            training, query = task_frames(frame, fv, fold)
            tasks = [(i, a) for i in INITIALIZATIONS for a in ('BASE', 'EMA')] if spec['phase'] == 'initialization' else [(42, 'EMA_UPDATES')]
            for init, arm in tasks:
                directory = out/unit_name(seed, fold, init, arm)
                warm = json.loads((directory/'warm-complete.json').read_text())
                checked = json.loads((directory/'cold-complete.json').read_text())
                if checked['status'] != 'passed' or checked['warm_sha256'] != sha(directory/'warm-complete.json'):
                    raise ValueError('Complete independent cold coverage required')
                traces[directory.name] = validate_checkpoints(manifest, directory, training, query, seed, fold, init, arm)
                total_native += checked['optimizer_runs']; total_states += checked['cold_states']; seen += 1
                with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['query_ids'], query.sample_id.to_numpy(str))
                    key = f'{arm}_{init}' if spec['phase'] == 'initialization' else arm
                    components[key][fv == fold] = saved['prediction']
                    for k in vectors:
                        if np.isfinite(vectors[k][fv == fold]).all(): np.testing.assert_array_equal(vectors[k][fv == fold], saved[k])
                        vectors[k][fv == fold] = saved[k]
        if any(not np.isfinite(v).all() for v in [*vectors.values(), *components.values()]):
            raise ValueError('Incomplete within-seed OOF')
        y = frame[list(TARGETS)].to_numpy()
        if spec['phase'] == 'initialization':
            columns, record = summarize_initialization(y, components, vectors['q75'], vectors['old_ema'], vectors['iron'], fv, frame.spout_no.to_numpy())
        else:
            columns = {'EMA_UPDATES': candidate_column(vectors['q75'], vectors['old_ema'], components['EMA_UPDATES'])}
            gain = score(y, vectors['iron'], columns['EMA_UPDATES'])-score(y, vectors['iron'], vectors['q75'])
            record = dict(gains_vs_q75=dict(EMA_UPDATES=gain),
                component_gain=50*(np.abs(y[:, 1]-vectors['old_ema']).sum()-np.abs(y[:, 1]-components['EMA_UPDATES']).sum())/np.abs(y[:, 1]).sum(),
                metrics=dict(EMA_UPDATES=metric_detail(y[:, 1], columns['EMA_UPDATES'], fv, frame.spout_no.to_numpy())))
        records[str(seed)] = record
        path = out/f'oof-s{seed}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, actual=y, query_ids=frame.sample_id.to_numpy(str), folds=fv,
                spouts=frame.spout_no.to_numpy(), **vectors, **columns,
                **{k+'_component': v for k, v in components.items()})
        vectors_hashes[str(seed)] = sha(path)
    if total_native != spec['optimizer_runs'] or total_states != spec['retained_states'] or seen != (60 if spec['phase'] == 'initialization' else 10):
        raise ValueError('Full native optimizer/state budget did not close')
    candidate = 'EMA_MEAN3' if spec['phase'] == 'initialization' else 'EMA_UPDATES'
    gains = [r['gains_vs_q75'][candidate] for r in records.values()]
    eligible = all(v > 0 for v in gains)
    if spec['phase'] == 'initialization': eligible = eligible and all(r['equal_mean_ema_minus_equal_mean_base'] > 0 for r in records.values())
    from .candidate_tiers import classify_candidates
    policy = yaml.safe_load((main/'configs/candidate_tiers.yaml').read_text())
    metrics = {'tap_time_len': {'Q75': {}, candidate: {}}}
    for seed, record in records.items():
        with np.load(out/f'oof-s{seed}.npz') as saved:
            metrics['tap_time_len']['Q75'][seed] = metric_detail(saved['actual'][:, 1], saved['q75'], saved['folds'], saved['spouts'])
        metrics['tap_time_len'][candidate][seed] = record['metrics'][candidate]
    tier_spec = dict(split_seeds=list(SPLITS), folds=5, candidates={'tap_time_len': [candidate]},
        tie_preference_by_target={'tap_time_len': [candidate]}, reference_by_target={'tap_time_len': 'Q75'})
    report = dict(status='completed_paired_development', phase=spec['phase'],
        G0='passed_native_budget_saved_partitions_epoch_trace_and_independent_cold', G1='two_complete_development_splits_not_formal_promotion',
        records=records, traces=traces, candidate=candidate, gains_vs_q75=paired_summary(gains),
        confirmation_eligible=eligible, formal_promotion=False, optimizer_runs=total_native, cold_states=total_states,
        vector_sha256=vectors_hashes, tiers=classify_candidates(metrics, tier_spec, policy),
        new_confirmation_seeds=0, full_data_fits=0, packages=0, uploads=0)
    write_new(out/'report.json', report)
    return report


def execute(out):
    manifest, _ = context(out, data=False); spec = manifest['spec']
    for seed in spec['split_seeds']:
        for fold in range(5):
            tasks = [(i, a) for i in INITIALIZATIONS for a in ('BASE', 'EMA')] if spec['phase'] == 'initialization' else [(42, 'EMA_UPDATES')]
            for init, arm in tasks:
                for mode in ('worker', 'cold'):
                    subprocess.run([sys.executable, '-m', 'bf_tap_r2.ema_retraining_validation', '--'+mode,
                        '--output', str(out), '--seed', str(seed), '--fold', str(fold), '--init', str(init), '--arm', arm], check=True)
    evaluate(out)
    write_new(Path(out)/'completion-event.json', dict(status='completed', optimizer_runs=spec['optimizer_runs'], packages=0, uploads=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--workspace', default=Path(__file__).resolve().parents[2])
    parser.add_argument('--main-root', default='/home/lux1/iron')
    parser.add_argument('--phase', choices=PHASES); parser.add_argument('--tests-receipt')
    parser.add_argument('--seed', type=int); parser.add_argument('--fold', type=int)
    parser.add_argument('--init', type=int); parser.add_argument('--arm')
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'worker', 'cold'): modes.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if args.prepare: freeze(args.workspace, args.main_root, args.output, args.phase, args.tests_receipt)
    elif args.worker: worker(args.output, args.seed, args.fold, args.init, args.arm)
    elif args.cold: cold(args.output, args.seed, args.fold, args.init, args.arm)
    else: execute(args.output)


if __name__ == '__main__':
    # Persisted subclasses must resolve to an importable module in cold workers.
    from bf_tap_r2.ema_retraining_validation import main as importable_main
    importable_main()
