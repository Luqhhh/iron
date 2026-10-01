"""Honest inner-holdout calibration of Q75; all learning stays in outer training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, TARGETS
from .ema_evaluation_diagnostics import sha, verify_files, write_new
from .ema_training_scale import assert_isolated, event, require_memory
from .v3_4_bags import group_safe_inner_folds
from .v5_library import load_v5_training_frame, fold_vector
from .v5_spec import load_v5_spec
from .v7_periodic import digest

SPEC = 'configs/q75_error_relocation/CALIBRATION.json'


def fit_offsets(residual, bins, family, minimum=30):
    residual, bins = np.asarray(residual, float), np.asarray(bins)
    if residual.shape != bins.shape or not len(residual) or not np.isfinite(residual).all():
        raise ValueError('Aligned finite calibration residuals required')
    global_value = float(np.median(residual))
    if family == 'GLOBAL':
        return np.full(4, global_value)
    if family != 'PRESSURE' or not np.isin(bins, np.arange(4)).all():
        raise ValueError('Unknown calibration family/bin')
    return np.array([np.median(residual[bins == q]) if (bins == q).sum() >= minimum else global_value for q in range(4)])


def learn_calibration(calibration_inputs, actual, prediction, cuts, family, settings):
    """Receives only calibration rows; no outer query or labels enter selection."""
    if any(t in calibration_inputs for t in TARGETS):
        raise ValueError('Label-free calibration inputs required')
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    if actual.shape != (len(calibration_inputs),) or prediction.shape != actual.shape or not np.isfinite(actual-prediction).all():
        raise ValueError('Calibration label/prediction alignment')
    bins = np.searchsorted(cuts, calibration_inputs.total_press_diff.to_numpy(float), side='right')
    folds = np.asarray(group_safe_inner_folds(calibration_inputs, seed=settings['cv_seed'], n_splits=3)['fold'])
    oof = np.full(len(actual), np.nan)
    residual = actual-prediction
    training_digests = {}
    for f in range(3):
        held = folds == f
        offsets = fit_offsets(residual[~held], bins[~held], family, settings['minimum_bin_rows'])
        oof[held] = offsets[bins[held]]
        training_digests[str(f)] = digest(calibration_inputs.loc[~held, 'sample_id'].tolist())
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete correction inner OOF')
    losses = [float(np.abs(actual-(prediction+g*oof)).sum()) for g in settings['gamma_grid']]
    best = min(losses)
    selected = next(i for i, value in enumerate(losses) if value <= best+1e-12)
    offsets = fit_offsets(residual, bins, family, settings['minimum_bin_rows'])
    return dict(family=family, gamma=settings['gamma_grid'][selected], offsets=offsets.tolist(), cuts=list(cuts),
                calibration_rows=len(actual), bin_counts=[int((bins == q).sum()) for q in range(4)],
                correction_cv_folds=folds.tolist(), correction_cv_fit_ids=training_digests,
                cv_absolute_error=losses, fitted_calibration_ids_digest=digest(calibration_inputs.sample_id.tolist()))


def apply_calibration(query, prediction, fitted):
    if any(t in query for t in TARGETS):
        raise ValueError('Outer prediction query contains labels')
    bins = np.searchsorted(fitted['cuts'], query.total_press_diff.to_numpy(float), side='right')
    p = np.asarray(prediction, float)+fitted['gamma']*np.asarray(fitted['offsets'])[bins]
    if p.shape != (len(query),) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid corrected prediction; no clipping permitted')
    return p


def sources(workspace):
    workspace = Path(workspace)
    paths = list((workspace/'src').rglob('*.py'))+list((workspace/'tests').glob('*.py'))
    for extension in ('*.yaml','*.json'):
        paths += list((workspace/'configs').rglob(extension))
    paths += [workspace/n for n in (SPEC,'uv.lock','pyproject.toml','docs/q75_error_relocation/CALIBRATION_PREREGISTRATION.md',
        'scripts/q75_pressure_calibration.py','scripts/audit_q75_pressure_calibration.py')]
    return {str(p.relative_to(workspace)): sha(p) for p in sorted(set(paths))}


def context(workspace, manifest):
    from .v49_run import check_runtime
    import torch
    workspace = Path(workspace)
    verify_files(workspace, manifest['sources'])
    spec = manifest['spec']; main = Path(spec['main_root'])
    verify_files(main, spec['inputs'])
    if sys.version_info[:2] != (3,12):
        raise ValueError('Locked Python3.12 required')
    check_runtime(spec)
    torch.set_num_threads(1)
    frame = load_v5_training_frame(main)
    return spec, frame


def partition(frame, seed, fold, spec):
    fv = fold_vector(Path(spec['main_root']), frame, seed, load_v5_spec(Path(spec['main_root'])))
    training = frame.loc[fv != fold].reset_index(drop=True)
    query = frame.loc[fv == fold, ['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
    inner = np.asarray(group_safe_inner_folds(training, seed=27001, n_splits=5)['fold'])
    fitting = training.loc[inner != 0].reset_index(drop=True)
    calibration = training.loc[inner == 0].reset_index(drop=True)
    assert_isolated(fitting, calibration.drop(columns=list(TARGETS)))
    assert_isolated(training, query)
    return fitting, calibration, training, query


def prepare(workspace, receipt):
    workspace = Path(workspace).resolve()
    spec = json.loads((workspace/SPEC).read_text())
    source_hashes = sources(workspace)
    checked = json.loads(Path(receipt).read_text())
    if checked.get('status') != 'passed' or checked.get('sources') != source_hashes or checked.get('full_suite') is not True:
        raise ValueError('Passed exact-source full locked tests required')
    if subprocess.check_output(['git','status','--porcelain','--',*source_hashes],cwd=workspace,text=True).strip():
        raise ValueError('Commit tested scientific source before launch')
    if (spec['split_seeds'] != [42,3407] or spec['families'] != ['GLOBAL','PRESSURE'] or spec['new_estimators'] != 10
            or spec['new_optimizer_runs'] != 20 or spec['correction']['gamma_grid'] != [0,.25,.5,1]
            or any(spec[k] != 0 for k in ['full_data_fits','packages','desktop_writes','agent_uploads'])):
        raise ValueError('Unregistered scientific scope')
    main = Path(spec['main_root'])
    current = json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current[k] != v for k,v in spec['reference'].items()):
        raise ValueError('Current reference changed')
    out = Path(spec['output'])
    if not out.is_relative_to(main/'local/runs') or out.exists():
        raise ValueError('Fresh private run required')
    verify_files(main, spec['inputs'])
    from .v49_run import check_runtime
    versions = check_runtime(spec)
    available = require_memory(spec)
    original = yaml.safe_load((main/'configs/strong_component_regularization/SPEC.yaml').read_text())
    if spec['training'] != original['training']['tap_time_len'] or spec['mechanisms'] != original['mechanisms']:
        raise ValueError('Original trainer/settings changed')
    audit = json.loads((main/spec['reference_cache']/'audit.json').read_text())
    if audit['status'] != 'passed' or audit['manifest_sha256'] != sha(main/spec['reference_cache']/'manifest.json'):
        raise ValueError('Audited calibration reference cache required')
    frame = load_v5_training_frame(main)
    plan = {}
    for seed in spec['split_seeds']:
        for fold in range(5):
            fitting, calibration, training, query = partition(frame, seed, fold, spec)
            cache = main/spec['reference_cache']/f'reference-calibration-s{seed}-f{fold}'
            meta = json.loads((cache/'metadata.json').read_text())
            if meta['fit_ids_digest'] != digest(fitting.sample_id.tolist()) or meta['query_ids_digest'] != digest(calibration.sample_id.tolist()):
                raise ValueError('Calibration base fitted on the wrong partition')
            verify_files(cache, json.loads((cache/'complete.json').read_text())['hashes'])
            key = f's{seed}-f{fold}'
            plan[key] = {label: digest(data.sample_id.tolist()) for label,data in
                [('fitting',fitting),('calibration',calibration),('outer_training',training),('query',query)]}
    out.mkdir(parents=True)
    manifest = dict(spec=spec, sources=source_hashes, versions=versions, plan=plan, workspace=str(workspace),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),
        available_memory_mib=available, receipt_sha256=sha(receipt))
    write_new(out/'manifest.json', manifest)
    return out


def worker(workspace, out, seed, fold):
    from .component_regularization import ComponentRegressor
    from .component_regularization_run import RECIPE
    from .v49_run import read_reference
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text())
    spec, frame = context(workspace, manifest)
    if seed not in spec['split_seeds'] or fold not in range(5):
        raise ValueError('Unknown task')
    unit = out/f's{seed}-f{fold}'; unit.mkdir(exist_ok=False)
    event(out/'events.jsonl',dict(event='unit_started',seed=seed,fold=fold))
    start = time.monotonic()
    try:
        require_memory(spec)
        fitting, calibration, training, query = partition(frame, seed, fold, spec)
        original_spec = yaml.safe_load((Path(spec['main_root'])/'configs/strong_component_regularization/SPEC.yaml').read_text())
        cal_query = calibration.drop(columns=list(TARGETS))
        cached, _ = read_reference(Path(spec['main_root'])/spec['reference_cache']/f'reference-calibration-s{seed}-f{fold}', fitting, cal_query, original_spec)
        class RecordedRegressor(ComponentRegressor):
            def _train(self, *args, **kwargs):
                phase = 'selection' if kwargs.get('validation') is not None or (len(args)>3 and args[3] is not None) else 'refit'
                event(out/'events.jsonl', dict(event='optimizer_started',seed=seed,fold=fold,phase=phase))
                result = super()._train(*args, **kwargs)
                event(out/'events.jsonl', dict(event='optimizer_completed',seed=seed,fold=fold,phase=phase))
                return result
        model = RecordedRegressor(RECIPE,spec['training'],'EMA',spec['mechanisms'],unit)
        model.fit(fitting, fitting[['tap_time_len']].to_numpy())
        ema = model.predict(cal_query)[:,0]
        cal_prediction = cached['tap_time_len']+.75*(ema-cached['v7_time'])
        # The saved full outer Q75 sees only training; no new baseline fit is needed.
        with np.load(Path(spec['main_root'])/spec['diagnostic_output']/f'error-map-{seed}.npz',allow_pickle=False) as saved:
            held = saved['folds'] == fold
            if saved['query_ids'][held].tolist() != query.sample_id.tolist():
                raise ValueError('Outer Q75 prediction identity changed')
            reference = saved['reference'][held,1].copy()
        cuts = np.quantile(fitting.total_press_diff.to_numpy(float), [.25,.5,.75])
        fitted, predictions = {}, {}
        for family in spec['families']:
            fitted[family] = learn_calibration(cal_query, calibration.tap_time_len.to_numpy(), cal_prediction, cuts, family, spec['correction'])
            predictions[family] = apply_calibration(query, reference, fitted[family])
        with (unit/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, query_ids=query.sample_id.to_numpy(str), calibration_ids=cal_query.sample_id.to_numpy(str),
                calibration_prediction=cal_prediction, ema_calibration=ema, reference=reference, **predictions)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > spec['max_worker_rss_mib']:
            raise ValueError('Original memory gate failed')
        write_new(unit/'metadata.json',dict(seed=seed,fold=fold,model=model.metadata_,fitted=fitted,
            partitions=manifest['plan'][f's{seed}-f{fold}'],seconds=time.monotonic()-start,peak_rss_mib=peak))
        context(workspace, manifest)
        write_new(unit/'complete.json',dict(manifest_sha256=sha(out/'manifest.json'),
            hashes={p.name:sha(p) for p in unit.iterdir()},new_optimizer_runs=2))
        event(out/'events.jsonl',dict(event='unit_completed',seed=seed,fold=fold))
    except BaseException as error:
        event(out/'events.jsonl',dict(event='unit_failed',seed=seed,fold=fold,error=repr(error)))
        write_new(unit/'failure.json',dict(error=repr(error)))
        raise


def summarize(workspace, out):
    from .candidate_tiers import classify_candidates
    from .v49_run import metric_detail
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text())
    spec, frame = context(workspace, manifest)
    metrics = {'tap_time_len': {name:{} for name in ['Q75',*spec['families']]}}
    gains = {name:{} for name in spec['families']}
    for seed in spec['split_seeds']:
        fv = fold_vector(Path(spec['main_root']),frame,seed,load_v5_spec(Path(spec['main_root'])))
        vectors = {name:np.full(len(frame),np.nan) for name in ['Q75',*spec['families']]}
        for fold in range(5):
            unit = out/f's{seed}-f{fold}'
            verify_files(unit,json.loads((unit/'complete.json').read_text())['hashes'])
            with np.load(unit/'predictions.npz',allow_pickle=False) as a:
                for name in vectors:
                    vectors[name][fv == fold] = a['reference' if name == 'Q75' else name]
        y = frame.tap_time_len.to_numpy()
        for name,p in vectors.items():
            if not np.isfinite(p).all():
                raise ValueError('Incomplete OOF')
            metrics['tap_time_len'][name][str(seed)] = metric_detail(y,p,fv,frame.spout_no.to_numpy())
        for name in spec['families']:
            gains[name][str(seed)] = float(50*(np.abs(y-vectors['Q75']).sum()-np.abs(y-vectors[name]).sum())/y.sum())
    tier_spec = dict(split_seeds=spec['split_seeds'],folds=5,candidates={'tap_time_len':spec['families']},
        tie_preference_by_target={'tap_time_len':spec['families']},reference_by_target={'tap_time_len':'Q75'})
    tiers = classify_candidates(metrics,tier_spec,yaml.safe_load((Path(spec['main_root'])/'configs/candidate_tiers.yaml').read_text()))
    eligible = [n for n in spec['families'] if min(gains[n].values()) > 0]
    selected = min(eligible,key=lambda n:(-np.mean(list(gains[n].values())),spec['families'].index(n))) if eligible else None
    summary = dict(metrics=metrics,gains=gains,tiers=tiers,selected_for_confirmation=selected,
        G0='saved_state_audit_pending',G1='two_complete_development_seeds_no_formal_promotion',
        new_estimators=10,new_optimizer_runs=20,full_data_fits=0,packages=0,desktop_writes=0,agent_uploads=0)
    write_new(out/'summary.json',summary)
    return summary


def execute(workspace, out):
    out = Path(out)
    manifest = json.loads((out/'manifest.json').read_text()); spec = manifest['spec']
    try:
        for seed in spec['split_seeds']:
            for fold in range(5):
                subprocess.run([sys.executable,str(Path(workspace)/'scripts/q75_pressure_calibration.py'),
                    '--workspace',str(workspace),'--output',str(out),'worker','--seed',str(seed),'--fold',str(fold)],check=True)
        summarize(workspace,out)
        subprocess.run([sys.executable,str(Path(workspace)/'scripts/audit_q75_pressure_calibration.py'),'--output',str(out)],check=True)
        write_new(out/'completion-event.json',dict(status='completed',completed_ns=time.time_ns(),
            summary_sha256=sha(out/'summary.json'),audit_sha256=sha(out/'audit.json')))
        print(json.dumps({'status':'completed','output':str(out)}),flush=True)
    except BaseException as error:
        write_new(out/'failure.json',dict(error=repr(error)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path)
    sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare'); a.add_argument('--receipt',type=Path,required=True)
    sub.add_parser('execute')
    a = sub.add_parser('worker');a.add_argument('--seed',type=int,required=True);a.add_argument('--fold',type=int,required=True)
    a = p.parse_args()
    if a.command == 'prepare': print(prepare(a.workspace,a.receipt))
    elif a.command == 'execute': execute(a.workspace,a.output)
    else: worker(a.workspace,a.output,a.seed,a.fold)
