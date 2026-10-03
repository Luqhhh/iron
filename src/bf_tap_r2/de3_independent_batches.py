"""Frozen two-split joint BASE independent-batch development for DE3 iron."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np
from .data import TARGETS
from .ema_nested_residual import (forbid_training, memory, read, save_arrays, setup, sha, verify, write)
from .ema_independent_batches import (merge_legacy_files, optimizer_ledger, verify_counts,
    validate_frames, reference_arrays, scalar_score)
from .v7_periodic import digest

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path('/home/lux1/iron')
RUN = MAIN/'local/runs/de3-independent-batches-20261004/development-r1'
PREVIOUS = MAIN/'local/runs/ema-independent-batches-20261004/development-r1'
NESTED = MAIN/'local/runs/ema-nested-residual-20261003/development-r1'
DE3 = MAIN/'local/runs/independent-ensemble-checkpoints-20260929/development-DE3'
DE3_SOURCE = MAIN/'local/worktrees/independent-ensemble-checkpoints'
PROTOCOL = ROOT/'docs/de3_independent_batches/PREREGISTRATION.md'
SPEC = ROOT/'configs/de3_independent_batches/SPEC.json'
MODEL = ROOT/'src/bf_tap_r2/de3_independent_batches_model.py'
SEEDS = (42, 3407)
INITS = (42, 104729, 130363)
CANDIDATES = {'JOINT_IBATCH_IRON_A100':1., 'JOINT_IBATCH_IRON_A20':.2}
REFERENCE = 'DE3_EMA_MEAN3_Q100'


def columns(reference, old_members, new_members):
    ref, old, new = [np.asarray(x, float) for x in (reference, old_members, new_members)]
    if (ref.ndim != 1 or old.shape != (3, len(ref)) or new.shape != old.shape
            or any(not np.isfinite(x).all() for x in (ref,old,new))):
        raise ValueError('Exactly three aligned finite old/new joint iron members required')
    result = {k:ref+.5*weight*(new.mean(0)-old.mean(0)) for k,weight in CANDIDATES.items()}
    if any(not np.isfinite(x).all() or (x<0).any() for x in (ref,*result.values())):
        raise ValueError('Invalid final iron column; no clipping')
    return result


def choose(gains):
    if list(gains) != list(CANDIDATES) or any(set(v) != {'42','3407'} for v in gains.values()):
        raise ValueError('Complete fixed pool and both full splits required')
    if any(not math.isfinite(v) for values in gains.values() for v in values.values()):
        raise ValueError('Nonfinite gain')
    passing = [k for k in gains if min(gains[k].values()) > 0]
    return max(passing, key=lambda k:sum(gains[k].values())) if passing else None


def require_previous():
    e = PREVIOUS/'execution'; t = read(e/'terminal.json')
    a, f = read(e/'independent-terminal-audit.json'), read(e/'final-reconciliation.json')
    if (any(x['status'] != 'passed' for x in (t,a,f)) or len(t['events']) != 72
            or any(x['exit_code'] != 0 for x in t['events']) or not f['owned_processes_absent']
            or any(f[k] != 0 for k in ('actual_supervisor_exec_exit_code',
                'actual_followup_exec_exit_code','actual_independent_audit_exit_code'))
            or f['independent_terminal_audit_sha256'] != sha(e/'independent-terminal-audit.json')
            or a['original_terminal_sha256'] != sha(e/'terminal.json')
            or a['report_sha256'] != sha(PREVIOUS/'report.json')):
        raise ValueError('Previous scientific batch must be independently closed')


def prepare(checks):
    import yaml
    if RUN.exists():
        raise FileExistsError('Private scientific run already consumed')
    require_previous()
    c, spec = read(checks), read(SPEC)
    if (c['status'] != 'passed' or c['python'] != sys.version.split()[0]
            or c['source_sha256'] != sha(__file__) or c['model_sha256'] != sha(MODEL)
            or c['test_sha256'] != sha(ROOT/'tests/test_de3_independent_batches.py')
            or c['junit_sha256'] != sha(c['junit'])
            or c['synthetic_cold_sha256'] != sha(c['synthetic_cold'])
            or read(c['synthetic_cold'])['status'] != 'passed'):
        raise ValueError('Matching locked engineering and fresh synthetic cold checks required')
    verify(c['files'])
    best = read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best[k] != v for k,v in spec['reference'].items()):
        raise ValueError('Latest reference changed before input freeze')
    old = read(DE3/'manifest.json'); previous = read(PREVIOUS/'manifest.json')
    old_spec_path = DE3_SOURCE/'configs/independent_ensemble_checkpoints/SPEC.yaml'
    old_spec = yaml.safe_load(old_spec_path.read_text())
    if (sha(old_spec_path) != old['spec_sha256'] or spec['training'] != old_spec['training']['tap_iron']
            or spec['training_seeds'] != old_spec['training_seeds'] or spec['arm'] != 'BASE'
            or spec['mechanisms'] or spec['joint_training_outputs'] != list(TARGETS)
            or spec['candidate_weights'] != CANDIDATES or spec['split_seeds'] != list(SEEDS)):
        raise ValueError('Frozen joint scientific contract differs')
    original_audit = read(DE3/'audit.json')
    if (original_audit['status'] != 'passed' or original_audit['manifest_sha256'] != sha(DE3/'manifest.json')
            or original_audit['summary_sha256'] != sha(DE3/'summary.json')):
        raise ValueError('Original DE3 source not independently closed')
    files = {}; merge_legacy_files(files, previous['files'], MAIN)
    merge_legacy_files(files, c['files'], MAIN)
    merge_legacy_files(files, old['source_hashes'], DE3_SOURCE)
    merge_legacy_files(files, old['data_hashes'], MAIN)
    paths = list((ROOT/'src').rglob('*.py')) + [PROTOCOL, SPEC, old_spec_path,
        ROOT/'tests/test_de3_independent_batches.py', ROOT/'tests/test_de3_independent_terminal_audit.py',
        ROOT/'scripts/audit_de3_independent_batches_terminal.py', ROOT/'scripts/watch_ema_independent_terminal.py',
        ROOT/'uv.lock', ROOT/'pyproject.toml',
        Path(checks), Path(c['junit']), Path(c['synthetic_cold']), MAIN/best['package'], MAIN/best['platform_feedback_record']]
    paths += [PREVIOUS/n for n in ('manifest.json','report.json','independent-audit.json',
        'execution/terminal.json','execution/independent-terminal-audit.json','execution/final-reconciliation.json')]
    paths += [DE3/n for n in ('manifest.json','summary.json','audit.json','arithmetic-audit.json')]
    identities = {}
    for seed in SEEDS:
        for fold in range(5):
            unit = DE3/f'tap_iron-DE3-s{seed}-f{fold}'
            for init in INITS:
                d = unit/f'training-seed-{init}'; completion = read(d/'complete.json')
                paths.append(d/'complete.json')
                merge_legacy_files(files, completion['hashes'], d)
                identity = dict(state_directory=str(d), source_directory=str(d), split_seed=seed,
                    fold=fold, trial_id=completion['key'], original_identity=completion['identity'])
                if (d/'reuse.json').exists():
                    reuse = read(d/'reuse.json'); source = Path(reuse['source'])
                    if sha(source/'complete.json') != reuse['source_complete_sha256']:
                        raise ValueError('Original reused DE3 model identity differs')
                    identity.update(source_directory=str(source), original_identity=reuse['source_identity'],
                        trial_id=read(source/'complete.json')['key'], copied_trial_id=completion['key'],
                        copied_identity=completion['identity'])
                    paths.append(source/'complete.json')
                    merge_legacy_files(files, read(source/'complete.json')['hashes'], source)
                identities[f's{seed}-f{fold}-init{init}'] = identity
    for p in paths:
        merge_legacy_files(files, {str(p.resolve()):sha(p)}, MAIN)
    verify(files); RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'manifest.json', dict(files=files, plans=previous['plans'], training=spec['training'],
        recipe=spec['recipe'], mechanisms={}, reference=best, old_identities=identities,
        seeds=list(SEEDS), training_seeds=list(INITS), candidates=CANDIDATES,
        budget=spec['scientific_budget'], monitor_seconds=600, time_budget_seconds=None,
        automatic_retries=False, created_ns=time.time_ns(), source_directory=str(ROOT),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    write(RUN/'round2-access-before-read.json', dict(scope='authorized_round2_cached_outer_training_only',
        protected_preliminary_targets_read=False, manifest_sha256=sha(RUN/'manifest.json')))
    print(json.dumps(dict(status='frozen', files=len(files), new_estimators=30, new_optimizers=60)),flush=True)


def unit_frames(seed, fold):
    if seed not in SEEDS or fold not in range(5):
        raise ValueError('Undeclared outer unit')
    m = read(RUN/'manifest.json'); verify(m['files'])
    u = NESTED/f's{seed}-f{fold}'
    with (u/'training.pkl').open('rb') as f: training = pickle.load(f)
    with (u/'query.pkl').open('rb') as f: query = pickle.load(f)
    validate_frames(training, query, m['plans'][u.name])
    return m, training, query


def audit_models(directory, training, query, expected, settings, mechanisms):
    import torch
    from .component_regularization_audit import verify_saved
    from .de3_independent_batches_model import JointIndependentBatchRegressor
    from .v3_4_bags import group_safe_inner_folds
    mask = group_safe_inner_folds(training, seed=settings['inner_seed'])['fold'] != 0
    fitting, calibration = training.loc[mask].reset_index(drop=True), training.loc[~mask]
    # Preserve the exact NumPy layout/reduction order used by JointRegressor.fit.
    targets = training[list(TARGETS)].to_numpy()
    def forbidden(*args, **kwargs):
        raise ValueError('Cold process attempted joint training')
    with forbid_training(), patch.object(JointIndependentBatchRegressor, '_train', forbidden):
        selector = verify_saved(directory/'selection.pt', fitting, targets[mask],
            'BASE', settings, mechanisms, calibration)
        model = verify_saved(directory/'refit.pt', training, targets,
            'BASE', settings, mechanisms, expected_epoch=selector.saved['trace']['selected_epoch'])
        if len(model.saved['mean']) != 2:
            raise ValueError('Two-output native model required')
        prediction = model.predict(query)
        np.testing.assert_array_equal(prediction, expected)
        if prediction.shape != (len(query),2):
            raise ValueError('Both joint outputs must be cold checked')
        maximum = 0.
        for p in (model.predict(query.iloc[::-1])[::-1],
                np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])):
            maximum = max(maximum, float(np.max(np.abs(p-prediction))))
    if maximum > .0005:
        raise ValueError('Joint cold order/chunk gate failed')
    return dict(status='passed', maximum_difference=maximum, states=2,
        selected_epoch=selector.saved['trace']['selected_epoch'], peak_rss_mib=memory())


def worker(seed, fold, init):
    from .de3_independent_batches_model import JointIndependentBatchRegressor
    if init not in INITS:
        raise ValueError('Undeclared training seed')
    m, training, query = unit_frames(seed, fold)
    d = RUN/f's{seed}-f{fold}-init{init}'; d.mkdir(exist_ok=False)
    identity = dict(source_directory=str(d),split_seed=seed,fold=fold,trial_id=f'JOINT_IBATCH_INIT{init}')
    write(d/'estimator-start.json',dict(identity=identity,pid=os.getpid(),time_ns=time.time_ns(),
        training_ids=training.sample_id.tolist(),query_ids=query.sample_id.tolist()))
    try:
        settings = dict(m['training'],random_seed=init,batch_order='independent_without_replacement')
        with optimizer_ledger(d,identity) as counts:
            model = JointIndependentBatchRegressor(m['recipe'],settings,d)
            model.fit(training.drop(columns=list(TARGETS)),training[list(TARGETS)].to_numpy())
        verify_counts(counts,model.traces); prediction = model.predict(query)
        if prediction.shape != (len(query),2) or not np.isfinite(prediction).all():
            raise ValueError('Invalid joint predictions')
        save_arrays(d/'predictions.npz',ids=query.sample_id.to_numpy(str),prediction=prediction)
        write(d/'complete.json',dict(identity=identity,pid=os.getpid(),settings=settings,
            manifest_sha256=sha(RUN/'manifest.json'),state_hashes={r:sha(d/(r+'.pt')) for r in ('selection','refit')},
            training_ids=training.sample_id.tolist(),query_ids=query.sample_id.tolist(),
            prediction_sha256=sha(d/'predictions.npz'),**counts,peak_rss_mib=memory()))
    except BaseException as error:
        write(d/'failure.json',dict(error=repr(error),automatic_retry=False));raise


def cold(seed, fold, init):
    import torch
    from .ema_independent_batches_model import member_orders
    if init not in INITS:
        raise ValueError('Undeclared training seed')
    m,training,query = unit_frames(seed,fold);d = RUN/f's{seed}-f{fold}-init{init}'
    c = read(d/'complete.json'); start = read(d/'estimator-start.json')
    identity = dict(source_directory=str(d),split_seed=seed,fold=fold,trial_id=f'JOINT_IBATCH_INIT{init}')
    settings = dict(m['training'],random_seed=init,batch_order='independent_without_replacement')
    if (c['identity'] != identity or c['settings'] != settings or c['pid'] == os.getpid()
            or c['manifest_sha256'] != sha(RUN/'manifest.json') or (d/'failure.json').exists()
            or c['training_ids'] != training.sample_id.tolist() or c['query_ids'] != query.sample_id.tolist()
            or c['prediction_sha256'] != sha(d/'predictions.npz')):
        raise ValueError('Native joint model identity mismatch')
    traces = {}
    for role,h in c['state_hashes'].items():
        if sha(d/(role+'.pt')) != h: raise ValueError('Native joint state changed')
        receipt = read(d/(role+'-optimizer-start.json'))
        if receipt['identity'] != identity or receipt['role'] != role or receipt['time_ns'] <= start['time_ns']:
            raise ValueError('Optimizer pre-construction identity differs')
        trace = torch.load(d/(role+'.pt'),map_location='cpu',weights_only=True)['trace'];traces[role]=trace
        rng = np.random.default_rng(init)
        orders = [digest(member_orders(rng,trace['fit_rows'],settings['tabm_k']).tolist())
                  for _ in range(trace['stopped_epoch'])]
        if (trace['member_order_digests'] != orders or trace['samples_per_head_per_epoch'] != trace['fit_rows']
                or trace['batch_order'] != 'independent_without_replacement'):
            raise ValueError('Joint member coverage/order identity differs')
    verify_counts(c,traces)
    with np.load(d/'predictions.npz',allow_pickle=False) as p:
        np.testing.assert_array_equal(p['ids'],query.sample_id.to_numpy(str))
        receipt = audit_models(d,training,query,p['prediction'],settings,{})
    write(d/'cold.json',dict(**receipt,pid=os.getpid(),complete_sha256=sha(d/'complete.json'),
        native_optimizer_constructors=2,native_steps=sum(c['steps'].values())))


def reuse(seed, fold):
    import torch
    m,training,query = unit_frames(seed,fold);unit = DE3/f'tap_iron-DE3-s{seed}-f{fold}'
    receipts, members = {}, []
    with np.load(unit/'predictions.npz',allow_pickle=False) as combined:
        np.testing.assert_array_equal(combined['query_ids'],query.sample_id.to_numpy(str))
        for init in INITS:
            d = unit/f'training-seed-{init}'; c = read(d/'complete.json')
            identity = m['old_identities'][f's{seed}-f{fold}-init{init}']
            if c['identity'] != identity.get('copied_identity',identity['original_identity']):
                raise ValueError('Original DE3 physical fit identity differs')
            for n,h in c['hashes'].items():
                if sha(d/n) != h: raise ValueError('Original DE3 artifact changed')
            state = torch.load(d/'refit.pt',map_location='cpu',weights_only=True)
            settings = dict(m['training'],random_seed=init)
            with np.load(d/'predictions.npz',allow_pickle=False) as p:
                np.testing.assert_array_equal(p['query_ids'],query.sample_id.to_numpy(str))
                np.testing.assert_array_equal(p['prediction'][:,0],combined[f'seed_{init}'])
                receipts[str(init)] = audit_models(d,training,query,p['prediction'],settings,state['mechanisms'])
                members.append(p['prediction'][:,0].copy())
        np.testing.assert_array_equal(np.mean(members,axis=0),combined['DE3'])
    write(RUN/f'reuse-s{seed}-f{fold}.json',dict(status='passed',receipts=receipts,cold_states=6,
        old_identities={str(i):m['old_identities'][f's{seed}-f{fold}-init{i}'] for i in INITS},
        manifest_sha256=sha(RUN/'manifest.json'),new_optimizers=0,peak_rss_mib=memory()))


def tasks():
    result = []
    for seed in SEEDS:
        for fold in range(5):
            result.append(('reuse',seed,fold,-1))
            for init in INITS:
                result.extend((stage,seed,fold,init) for stage in ('worker','cold'))
    return result+[('report',42,0,-1),('audit',42,0,-1)]


def verify_model_events():
    for index,(stage,seed,fold,init) in enumerate(tasks()[:-2]):
        key = f'{index:03d}-{stage}-s{seed}-f{fold}-i{init}'
        e = read(RUN/'execution'/(key+'-terminal.json'))
        if e['task'] != key or e['exit_code'] != 0 or e['peak_rss_mib'] > 1536:
            raise ValueError('Complete model exits required before quality reads')


def report():
    import yaml
    from .candidate_tiers import classify_candidates
    from .component_regularization_run import metric_detail
    verify_model_events();m = read(RUN/'manifest.json');verify(m['files'])
    gains = {k:{} for k in CANDIDATES};hashes={};states=optimizers=steps=0
    metrics = {'tap_iron':{k:{} for k in [REFERENCE,*CANDIDATES]}}
    for seed in SEEDS:
        a = reference_arrays(seed);ids,fv,y = (a[k] for k in ('ids','folds','actual'))
        if len(ids)!=2754 or len(set(ids))!=2754 or set(fv)!=set(range(5)):
            raise ValueError('Complete within-seed OOF required')
        old,new = np.full((3,len(ids)),np.nan),np.full((3,len(ids)),np.nan)
        for fold in range(5):
            reuse_receipt = read(RUN/f'reuse-s{seed}-f{fold}.json');mask=fv==fold
            if reuse_receipt['status']!='passed' or reuse_receipt['manifest_sha256']!=sha(RUN/'manifest.json'):
                raise ValueError('Original joint cold coverage incomplete')
            states += reuse_receipt['cold_states']
            with np.load(DE3/f'tap_iron-DE3-s{seed}-f{fold}/predictions.npz',allow_pickle=False) as p:
                np.testing.assert_array_equal(p['query_ids'],ids[mask])
                for j,init in enumerate(INITS):old[j,mask]=p[f'seed_{init}']
            for j,init in enumerate(INITS):
                d=RUN/f's{seed}-f{fold}-init{init}';c=read(d/'cold.json');w=read(d/'complete.json')
                if (c['status']!='passed' or c['complete_sha256']!=sha(d/'complete.json')
                        or w['prediction_sha256']!=sha(d/'predictions.npz')):
                    raise ValueError('New joint cold coverage incomplete')
                states+=c['states'];optimizers+=c['native_optimizer_constructors'];steps+=c['native_steps']
                with np.load(d/'predictions.npz',allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['ids'],ids[mask]);new[j,mask]=p['prediction'][:,0]
        candidates=columns(a['iron'],old,new)
        metrics['tap_iron'][REFERENCE][str(seed)]=metric_detail(y[:,0],a['iron'],fv,a['spouts'])
        for name,p in candidates.items():
            gains[name][str(seed)]=float(50*np.sum(np.abs(y[:,0]-a['iron'])-np.abs(y[:,0]-p))/np.abs(y[:,0]).sum())
            metrics['tap_iron'][name][str(seed)]=metric_detail(y[:,0],p,fv,a['spouts'])
        path=RUN/f'oof-s{seed}.npz'
        save_arrays(path,ids=ids,folds=fv,actual=y,reference_iron=a['iron'],time=a['current'],
            old_members=old,new_members=new,spouts=a['spouts'],**candidates);hashes[str(seed)]=sha(path)
    if (states,optimizers)!=(120,60):raise ValueError('Joint cold/optimizer inventory differs')
    spec=dict(split_seeds=list(SEEDS),folds=5,candidates={'tap_iron':list(CANDIDATES)},
        reference_by_target={'tap_iron':REFERENCE},tie_preference_by_target={'tap_iron':list(CANDIDATES)})
    value=dict(status='completed_two_split_joint_iron_development',gains=gains,selected_for_confirmation=choose(gains),
        formal_promoted=False,metrics=metrics,candidate_tiers=classify_candidates(metrics,spec,
        yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text())),new_optimizer_constructors=optimizers,
        native_steps=steps,cold_base_states=states,manifest_sha256=sha(RUN/'manifest.json'),oof_sha256=hashes,
        new_confirmation_seeds=0,full_fits=0,packages=0)
    verify(m['files']);write(RUN/'report.json',value)
    print(json.dumps({k:value[k] for k in ('status','gains','selected_for_confirmation')}),flush=True)


def audit():
    verify_model_events();m,r=read(RUN/'manifest.json'),read(RUN/'report.json');verify(m['files'])
    if r['manifest_sha256']!=sha(RUN/'manifest.json'):raise ValueError('Report identity differs')
    truth={};maximum=0.;gains={k:{} for k in CANDIDATES};steps=0
    for seed in SEEDS:
        for fold in range(5):
            _,training,_=unit_frames(seed,fold)
            for row in training[['sample_id',*TARGETS]].itertuples(index=False,name=None):
                if row[0] in truth and truth[row[0]]!=row[1:]:raise ValueError('Target binding differs')
                truth[row[0]]=row[1:]
        with np.load(RUN/f'oof-s{seed}.npz',allow_pickle=False) as a:
            if sha(RUN/f'oof-s{seed}.npz')!=r['oof_sha256'][str(seed)]:raise ValueError('OOF identity differs')
            np.testing.assert_array_equal(a['actual'],np.asarray([truth[k] for k in a['ids']]))
            original=reference_arrays(seed)
            for key,target in [('ids','ids'),('folds','folds'),('reference_iron','iron'),('time','current')]:
                np.testing.assert_array_equal(a[key],original[target])
            for fold in range(5):
                mask=a['folds']==fold
                with np.load(DE3/f'tap_iron-DE3-s{seed}-f{fold}/predictions.npz',allow_pickle=False) as old:
                    np.testing.assert_array_equal(old['query_ids'],a['ids'][mask])
                    for j,init in enumerate(INITS):np.testing.assert_array_equal(old[f'seed_{init}'],a['old_members'][j,mask])
                for j,init in enumerate(INITS):
                    d=RUN/f's{seed}-f{fold}-init{init}';w=read(d/'complete.json');c=read(d/'cold.json')
                    if c['complete_sha256']!=sha(d/'complete.json') or w['prediction_sha256']!=sha(d/'predictions.npz'):
                        raise ValueError('Joint model prediction binding differs')
                    for role,h in w['state_hashes'].items():
                        if sha(d/(role+'.pt'))!=h:raise ValueError('Native state changed')
                    steps+=sum(w['steps'].values())
                    with np.load(d/'predictions.npz',allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['ids'],a['ids'][mask])
                        np.testing.assert_array_equal(p['prediction'][:,0],a['new_members'][j,mask])
            base=scalar_score(a['actual'],a['reference_iron'],a['time'])
            for name,weight in CANDIDATES.items():
                expected=[float(b)+.5*weight*(math.fsum(map(float,n))/3-math.fsum(map(float,o))/3)
                          for b,n,o in zip(a['reference_iron'],a['new_members'].T,a['old_members'].T)]
                maximum=max(maximum,max(abs(x-float(y)) for x,y in zip(expected,a[name])))
                gain=scalar_score(a['actual'],expected,a['time'])-base;gains[name][str(seed)]=gain
                maximum=max(maximum,abs(gain-r['gains'][name][str(seed)]))
    if maximum>1e-9 or choose(gains)!=r['selected_for_confirmation'] or r['formal_promoted'] or steps!=r['native_steps']:
        raise ValueError('Independent joint arithmetic/decision/inventory differs')
    write(RUN/'independent-audit.json',dict(status='passed',gains=gains,maximum_difference=maximum,
        manifest_sha256=sha(RUN/'manifest.json'),report_sha256=sha(RUN/'report.json'),new_fits=0,packages=0))
    print(json.dumps(dict(status='passed',gains=gains,maximum_difference=maximum)),flush=True)


def synthetic_cold(directory):
    """One fresh-process audit of the already consumed synthetic fit budget."""
    import torch
    from .ema_independent_batches_model import member_orders
    d=Path(directory).resolve();w=read(d/'warm.json');verify(w['files'])
    if (w['scope']!='synthetic_only' or w['pid']==os.getpid()
            or w['source_sha256']!=sha(__file__) or w['model_sha256']!=sha(MODEL)):
        raise ValueError('Synthetic warm/source/process identity differs')
    with (d/'training.pkl').open('rb') as f:training=pickle.load(f)
    with (d/'query.pkl').open('rb') as f:query=pickle.load(f)
    if not all(k.startswith('synthetic-joint-') for k in training.sample_id):
        raise ValueError('Only synthetic engineering rows allowed')
    traces={}
    for role in ('selection','refit'):
        trace=torch.load(d/(role+'.pt'),map_location='cpu',weights_only=True)['trace'];traces[role]=trace
        rng=np.random.default_rng(w['settings']['random_seed'])
        expected=[digest(member_orders(rng,trace['fit_rows'],w['settings']['tabm_k']).tolist())
                  for _ in range(trace['stopped_epoch'])]
        if trace['member_order_digests']!=expected:raise ValueError('Synthetic head orders differ')
    verify_counts(w['counts'],traces)
    with np.load(d/'predictions.npz',allow_pickle=False) as p:
        np.testing.assert_array_equal(p['ids'],query.sample_id.to_numpy(str))
        receipt=audit_models(d,training,query,p['prediction'],w['settings'],{})
    write(d/'cold.json',dict(**receipt,pid=os.getpid(),warm_sha256=sha(d/'warm.json'),new_fits=0,
        new_optimizer_constructors=0,original_synthetic_optimizer_constructors=2))
    print(json.dumps(receipt),flush=True)


def controller():
    m = read(RUN/'manifest.json'); verify(m['files']); launch = RUN/'execution'; launch.mkdir(exist_ok=False)
    write(launch/'started.json',dict(pid=os.getpid(),started_ns=time.time_ns(),manifest_sha256=sha(RUN/'manifest.json')))
    stop = threading.Event(); lock = threading.Lock(); owned = {'pid':None,'task':None}; events = []
    def monitor():
        import psutil
        number = 0
        while not stop.wait(600):
            number += 1
            with lock:
                row = dict(number=number,time_ns=time.time_ns(),**owned)
            if row['pid'] is not None:
                try:
                    p=psutil.Process(row['pid']);row.update(live=p.is_running(),rss_mib=p.memory_info().rss/1024**2)
                except psutil.NoSuchProcess:
                    row['live']=False
            write(launch/f'observation-{number:03d}.json',row);print(json.dumps(row),flush=True)
    observer = threading.Thread(target=monitor,daemon=True); observer.start()
    try:
        for index,(stage,seed,fold,init) in enumerate(tasks()):
            key=f'{index:03d}-{stage}-s{seed}-f{fold}-i{init}'
            with (launch/(key+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-m','bf_tap_r2.de3_independent_batches',stage,'--seed',str(seed),
                    '--fold',str(fold),'--init',str(init)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                with lock:
                    owned.update(pid=child.pid,task=key)
                write(launch/(key+'-start.json'),dict(pid=child.pid,time_ns=time.time_ns()))
                _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
            e=dict(task=key,exit_code=code,peak_rss_mib=usage.ru_maxrss/1024,completed_ns=time.time_ns())
            write(launch/(key+'-terminal.json'),e);events.append(e)
            with lock:
                owned.update(pid=None,task=None)
            print(json.dumps(e),flush=True)
            if code!=0 or e['peak_rss_mib']>1536:
                raise RuntimeError('Scientific child failed: '+key)
        verify(m['files'])
        write(launch/'terminal.json',dict(status='passed',events=events,report_sha256=sha(RUN/'report.json'),
            independent_audit_sha256=sha(RUN/'independent-audit.json'),dependencies_unchanged=True))
    except BaseException as error:
        write(launch/'terminal.json',dict(status='failed',events=events,error=repr(error),automatic_retry=False));raise
    finally:
        stop.set();observer.join()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','reuse','worker','cold','report','audit','synthetic-cold'])
    p.add_argument('--checks');p.add_argument('--seed',type=int,default=42);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--init',type=int,default=-1);p.add_argument('--directory');a=p.parse_args();setup()
    if a.stage=='synthetic-cold':synthetic_cold(a.directory)
    elif a.stage=='prepare':prepare(a.checks)
    elif a.stage=='run':controller()
    elif a.stage=='reuse':reuse(a.seed,a.fold)
    elif a.stage in ('worker','cold'):{'worker':worker,'cold':cold}[a.stage](a.seed,a.fold,a.init)
    else:{'report':report,'audit':audit}[a.stage]()
