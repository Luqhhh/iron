"""Serial, frozen SAM-EMA time development with isolated source identities."""
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

from .data import TARGETS
from .ema_average_span import candidate_column, old_cache as original_cache, task_frames
from .sam_ema import SAMEMARegressor
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import verify_files
from .ema_training_scale import event, require_memory
from .v3_4_bags import group_safe_inner_folds
from .v3_run import load_fold_vector, load_training_frame
from .v7_periodic import digest
from .v49_run import check_runtime, metric_detail

SPEC = 'configs/sam_ema_time/SPEC.json'
ARMS = ('SAM_EMA_TIME',)


def old_cache(spec, seed, fold, training, query):
    from .v49_run import verified_unit, unit_id
    old = original_cache(spec, seed, fold, training, query)
    root = Path(spec['main_root'])/spec['old_development']
    manifest = json.loads((root/'manifest.json').read_text())
    key = f'tap_time_len-SAM-s{seed}-f{fold}'
    if not verified_unit(root/key, unit_id(manifest,key)): raise ValueError('Closed SAM cache required')
    metadata = json.loads((root/key/'metadata.json').read_text())
    if (metadata['fit_ids_digest'] != digest(training.sample_id.tolist()) or
            metadata['model']['fit_ids_digest'] != digest(training.sample_id.tolist()) or
            metadata['seed'] != seed or metadata['fold'] != fold or metadata['training_seed'] != 42):
        raise ValueError('SAM cache training identity differs')
    with np.load(root/key/'predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['query_ids'],query.sample_id.to_numpy(str))
        if saved['prediction'].shape != (len(query),1): raise ValueError('SAM prediction shape differs')
        old['sam'] = saved['prediction'][:,0].copy()
    if not np.isfinite(old['sam']).all() or (old['sam']<0).any(): raise ValueError('Invalid SAM baseline')
    return old


def runtime(spec):
    import torch
    if sys.version_info[:2] != (3, 12): raise ValueError('Locked Python3.12 required')
    versions = check_runtime(spec)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    return versions


def validate_scope(spec, original):
    expected = dict(target='tap_time_len', split_seeds=[42, 3407], folds=5, training_seeds=[42],
        candidate='SAM_EMA_TIME', reused_original_EMA_states=20, reused_original_SAM_states=20,
        replacement_weight=.75, new_estimators=10, new_optimizer_runs=20, new_saved_states=20,
        workers=1, numerical_threads=1, torch_interop_threads=1, monitor_seconds=600,
        max_worker_rss_mib=1536, cold_predict_atol=.0005, maximum_runtime_seconds=None,
        automatic_scientific_retries=False, new_confirmation_seeds=0, full_data_fits=0,
        packages=0, desktop_writes=0, agent_uploads=0,
        confirmation_gate=dict(both_complete_development_seeds_positive_vs_Q75=True,
            formal_complete_split_seeds_at_least=4,
            formal_all_seeds_positive=True, formal_paired_seed_LCB95_positive=True, both_complete_development_seeds_positive_vs_SAM=True))
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError('Unregistered scientific or execution scope')
    if spec['training'] != original['training']['tap_time_len'] or spec['mechanisms'] != original['mechanisms']:
        raise ValueError('Native training protocol changed')


def source_files(work):
    work = Path(work)
    paths = list((work/'src').rglob('*.py')) + [work/name for name in (
        SPEC, 'docs/sam_ema_time/PREREGISTRATION.md', 'uv.lock', 'pyproject.toml',
        'tests/test_component_regularization.py', 'tests/test_sam_ema_development.py', 'tests/test_sam_ema.py',
        'scripts/audit_sam_ema_development.py', 'scripts/observe_sam_ema_development.py',
        'scripts/observe_ema_dropout_development.py')]
    return {str(path.resolve()): sha(path) for path in paths}


def freeze(work, output, tests_receipt):
    work, out = Path(work).resolve(), Path(output).resolve()
    spec = json.loads((work/SPEC).read_text()); main = Path(spec['main_root'])
    validate_scope(spec, yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text()))
    if out.exists() or not out.is_relative_to(main/'local/runs'): raise ValueError('Fresh private output required')
    files = source_files(work); tests = json.loads(Path(tests_receipt).read_text())
    if tests.get('status') != 'passed' or tests.get('sources') != files or tests.get('skipped') != 0:
        raise ValueError('Passed exact-source locked tests required')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *files], cwd=work, text=True).strip():
        raise ValueError('Commit tested source before scientific freeze')
    best = json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(best.get(key) != value for key, value in spec['reference'].items()):
        raise ValueError('Current platform reference changed')
    grant = main/spec['standing_grant']
    if sha(grant) != spec['standing_grant_sha256']: raise ValueError('Standing authorization changed')
    admission = main/'local/runs/sam-ema-time-20261002/admission-r1'
    terminal = json.loads((admission/'terminal-verification.json').read_text())
    observed = json.loads((admission/'observer-terminal-verification.json').read_text())
    actual = json.loads((admission/'actual-main-exit.json').read_text())
    if (terminal['status'] != 'passed' or terminal['actual_cold_exit_codes'] != [0]
            or terminal['engineering_optimizer_runs'] != 1 or observed['status'] != 'passed'
            or observed['actual_main']['exit_code'] != 0 or actual['actual_exit_code'] != 0):
        raise ValueError('Successful actual composed SAM-EMA engineering terminal required')
    if (actual['observer_terminal_sha256'] != sha(admission/'observer-terminal-verification.json')
            or observed['terminal_sha256'] != sha(admission/'terminal-verification.json')
            or observed['manifest_sha256'] != sha(admission/'manifest.json')):
        raise ValueError('Actual engineering terminal binding differs')
    for filename,field in [('warm-receipt.json','warm_receipt_sha256'),('cold-receipt.json','cold_receipt_sha256')]:
        if sha(admission/filename) != terminal[field] or json.loads((admission/filename).read_text())['status'] != 'passed':
            raise ValueError('Independent engineering receipt binding differs')
    admission_manifest = json.loads((admission/'manifest.json').read_text())
    if admission_manifest['spec'] != spec: raise ValueError('Engineering recipe differs')
    verify_files(admission_manifest['files'])
    for name,expected in admission_manifest['files'].items():
        path = Path(name)
        if '/src/' in name:
            relative = name.split('/src/',1)[1]
            if sha(work/'src'/relative) != expected: raise ValueError('Native engineering source differs')
    versions = runtime(spec); available = require_memory(spec)
    previous = main/spec['previous_terminal']
    previous_actual = previous.parent/'actual-main-exit.json'
    previous_exit = json.loads(previous_actual.read_text())
    if (json.loads(previous.read_text())['status'] != 'passed' or previous_exit['actual_supervisor_tool_exit_code'] != 0
            or previous_exit['terminal_sha256'] != sha(previous)):
        raise ValueError('Previous phase actual terminal binding differs')
    paths = [grant, previous, previous_actual, admission/'manifest.json', admission/'terminal-verification.json',
        admission/'warm-receipt.json', admission/'cold-receipt.json', admission/'actual-main-exit.json',
        admission/'observer-terminal-verification.json',
        main/'configs/protection.yaml', main/'configs/candidate_tiers.yaml',
        main/'configs/strong_component_regularization/SPEC.yaml', main/'uv.lock', main/'pyproject.toml',
        *sorted((main/'复赛_train').glob('*.csv')), Path(tests_receipt).resolve()]
    warm = json.loads((admission/'warm-receipt.json').read_text()); cold = json.loads((admission/'cold-receipt.json').read_text())
    for arm in ARMS:
        for filename,field in [('refit.pt','model_sha256'),('predictions.npz','prediction_sha256')]:
            path = admission/arm/filename
            if sha(path) != warm['records'][arm][field] or sha(path) != cold['records'][arm][field]:
                raise ValueError('Engineering model/prediction identity differs')
            paths.append(path)
    old = main/spec['old_development']
    paths += [old/'manifest.json', old/'audit.json']
    old_audit = json.loads((old/'audit.json').read_text())
    if old_audit['status'] != 'passed' or old_audit['manifest_sha256'] != sha(old/'manifest.json'):
        raise ValueError('Original cache audit is not closed')
    frame = load_training_frame(main); folds = {s: load_fold_vector(main, frame, s) for s in spec['split_seeds']}
    plan = {}
    for seed, fv in folds.items():
        paths.append(main/f'local/runs/round2-v2/comparison-r1/folds-{seed}.csv')
        for fold in range(5):
            training, query = task_frames(frame, fv, fold); old_cache(spec, seed, fold, training, query)
            inner = np.asarray(group_safe_inner_folds(training, seed=42)['fold'])
            plan[f's{seed}-f{fold}'] = dict(training=digest(training.sample_id.tolist()),
                query=digest(query.sample_id.tolist()), inner_fit=digest(training.loc[inner != 0, 'sample_id'].tolist()),
                inner_validation=digest(training.loc[inner == 0, 'sample_id'].tolist()))
            for key in (f'reference-s{seed}-f{fold}', f'tap_time_len-EMA-s{seed}-f{fold}', f'tap_time_len-SAM-s{seed}-f{fold}'):
                paths += [p for p in (old/key).iterdir() if p.suffix in ('.json', '.npz', '.pt')]
    files.update({str(path.resolve()): sha(path) for path in paths})
    verify_files(files)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'manifest.json', dict(spec=spec, workspace=str(work), files=files, plan=plan,
        versions=versions, available_memory_mib=available,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=work, text=True).strip(),
        fold_digests={str(s): digest(v.tolist()) for s, v in folds.items()}))


def context(output):
    out = Path(output); manifest = json.loads((out/'manifest.json').read_text())
    verify_files(manifest['files']); spec = manifest['spec']; main = Path(spec['main_root'])
    validate_scope(spec, yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text()))
    runtime(spec); frame = load_training_frame(main)
    folds = {s: load_fold_vector(main, frame, s) for s in spec['split_seeds']}
    if {str(s): digest(v.tolist()) for s, v in folds.items()} != manifest['fold_digests']:
        raise ValueError('Frozen split identity changed')
    return manifest, spec, frame, folds


def worker(output, seed, fold, arm):
    from .component_regularization_run import RECIPE
    out = Path(output); manifest, spec, frame, folds = context(out)
    if seed not in spec['split_seeds'] or fold not in range(5) or arm not in ARMS:
        raise ValueError('Unknown frozen task')
    directory = out/f'{arm}-s{seed}-f{fold}'; directory.mkdir(exist_ok=False)
    identity = dict(source_directory=str(out.resolve()), seed=seed, fold=fold, trial=arm)
    event(out/'events.jsonl', dict(event='unit_started', **identity))
    try:
        require_memory(spec); training, query = task_frames(frame, folds[seed], fold)
        old = old_cache(spec, seed, fold, training, query)
        mechanisms = dict(spec['mechanisms'])

        class Recorded(SAMEMARegressor):
            def _train(self, frame, y, epochs, validation=None):
                phase = 'selection' if validation is not None else 'refit'
                event(out/'events.jsonl', dict(event='optimizer_started', phase=phase, **identity))
                result = super()._train(frame, y, epochs, validation)
                event(out/'events.jsonl', dict(event='optimizer_completed', phase=phase, **identity))
                return result

        model = Recorded(RECIPE, spec['training'], 'SAM_EMA', mechanisms, directory)
        model.fit(training, training[['tap_time_len']].to_numpy())
        prediction = model.predict(query)[:, 0]
        candidate = candidate_column(old['q75'], old['ema'], prediction)
        with (directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, query_ids=query.sample_id.to_numpy(str), ema=prediction,
                old_ema=old['ema'], old_sam=old['sam'], q75=old['q75'], iron=old['iron'], candidate=candidate)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > spec['max_worker_rss_mib']: raise ValueError('Worker memory gate failed')
        write_new(directory/'metadata.json', dict(**identity, model=model.metadata_, mechanisms=mechanisms,
            partitions=manifest['plan'][f's{seed}-f{fold}'], peak_rss_mib=peak, optimizer_runs=2))
        verify_files(manifest['files'])
        write_new(directory/'complete.json', dict(manifest_sha256=sha(out/'manifest.json'), optimizer_runs=2,
            hashes={p.name: sha(p) for p in directory.iterdir()}))
        event(out/'events.jsonl', dict(event='unit_completed', **identity))
    except BaseException as error:
        write_new(directory/'failure.json', dict(error=repr(error)))
        event(out/'events.jsonl', dict(event='unit_failed', error=repr(error), **identity))
        raise


def report_arrays(arrays, policy):
    from .candidate_tiers import classify_candidates
    from .q75_combination_review import score
    if set(arrays) != {'42', '3407'}: raise ValueError('Complete frozen split pool required')
    gains, records = {}, {}
    metrics = {'tap_time_len': {name: {} for name in ('Q75', *ARMS)}}
    for seed, a in arrays.items():
        if any(a[name].shape != (len(a['actual']),) or not np.isfinite(a[name]).all()
               for name in ('iron', 'Q75', 'SAM_MATCHED', *ARMS)):
            raise ValueError('Complete finite same-split predictions required')
        scores = {name: score(a['actual'], a['iron'], a[name]) for name in ('Q75', 'SAM_MATCHED', *ARMS)}
        gains[seed] = scores['SAM_EMA_TIME']-scores['Q75']
        records[seed] = dict(scores=scores, gains={name:scores[name]-scores['Q75'] for name in ARMS},
            matched_sam_gain=scores['SAM_EMA_TIME']-scores['SAM_MATCHED'])
        for name in ('Q75', *ARMS):
            metrics['tap_time_len'][name][seed] = metric_detail(a['actual'][:, 1], a[name], a['folds'], a['spouts'])
    tiers_spec = dict(split_seeds=[42, 3407], folds=5, candidates={'tap_time_len':['SAM_EMA_TIME']},
        tie_preference_by_target={'tap_time_len':['SAM_EMA_TIME']}, reference_by_target={'tap_time_len':'Q75'})
    eligible = all(value > 0 for value in gains.values()) and all(r['matched_sam_gain']>0 for r in records.values())
    return dict(records=records, metrics=metrics, tiers=classify_candidates(metrics, tiers_spec, policy),
        selected_for_confirmation='SAM_EMA_TIME' if eligible else None, confirmation_eligible=eligible,
        formal_promotion=False, G0='independent_saved_state_audit_pending', G1='two_development_splits',
        new_estimators=10, optimizer_runs=20, new_saved_states=20, new_confirmation_seeds=0,
        full_data_fits=0, packages=0, desktop_writes=0, uploads=0)


def summarize(output):
    out = Path(output); manifest, spec, frame, folds = context(out); arrays = {}
    for seed, fv in folds.items():
        a = dict(actual=frame[list(TARGETS)].to_numpy(), folds=fv, spouts=frame.spout_no.to_numpy(),
            query_ids=frame.sample_id.to_numpy(str), **{name:np.full(len(frame),np.nan) for name in ('iron','Q75','SAM_MATCHED',*ARMS)})
        for fold in range(5):
            held = fv == fold
            for arm in ARMS:
                directory = out/f'{arm}-s{seed}-f{fold}'; receipt = json.loads((directory/'complete.json').read_text())
                if receipt['manifest_sha256'] != sha(out/'manifest.json') or receipt['optimizer_runs'] != 2:
                    raise ValueError('Unit binding or budget changed')
                verify_files({str(directory/name): value for name,value in receipt['hashes'].items()})
                with np.load(directory/'predictions.npz', allow_pickle=False) as values:
                    np.testing.assert_array_equal(values['query_ids'], a['query_ids'][held])
                    a['Q75'][held] = values['q75']; a['iron'][held] = values['iron']; a[arm][held] = values['candidate']
                    a['SAM_MATCHED'][held] = values['q75']+.75*(values['old_sam']-values['old_ema'])
        with (out/f'oof-s{seed}.npz').open('xb') as stream: np.savez_compressed(stream, **a)
        arrays[str(seed)] = a
    report = report_arrays(arrays, yaml.safe_load((Path(spec['main_root'])/'configs/candidate_tiers.yaml').read_text()))
    report.update(manifest_sha256=sha(out/'manifest.json'), vector_sha256={str(s):sha(out/f'oof-s{s}.npz') for s in folds})
    write_new(out/'report.json', report)


def execute(output):
    out = Path(output); manifest, spec, _, _ = context(out); work = Path(manifest['workspace'])
    write_new(out/'activation.json', dict(pid=os.getpid(), started_ns=time.time_ns(), manifest_sha256=sha(out/'manifest.json')))
    for seed in spec['split_seeds']:
        for fold in range(5):
            for arm in ARMS:
                subprocess.run([sys.executable, '-m', 'bf_tap_r2.sam_ema_development', '--output', str(out),
                    'worker', '--seed', str(seed), '--fold', str(fold), '--arm', arm], cwd=work, check=True)
    summarize(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--workspace', required=True); f.add_argument('--tests-receipt', required=True)
    sub.add_parser('execute')
    w = sub.add_parser('worker'); w.add_argument('--seed', type=int, required=True); w.add_argument('--fold', type=int, required=True); w.add_argument('--arm', choices=ARMS, required=True)
    args = p.parse_args()
    if args.command == 'freeze': freeze(args.workspace, args.output, args.tests_receipt)
    elif args.command == 'execute': execute(args.output)
    else: worker(args.output, args.seed, args.fold, args.arm)


if __name__ == '__main__': main()
