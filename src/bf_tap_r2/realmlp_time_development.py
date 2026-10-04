"""Complete native RealMLP time development with fixed current-parent blends."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from . import realmlp_state_adapter as adapter
from .data import TARGETS
from .ema_nested_residual import read,write,sha,verify,save_arrays
from .realmlp_native_audit import observe_native,verify_trace,native_state_digest,cold_audit,forbid_native_fit

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_time_development/SPEC.json'
REVIEW=MAIN/'local/runs/realmlp-current-review-20261004/review-r1'
RECOVERY=MAIN/'local/runs/realmlp-native-admission-20261004/recovery-r1'
OLD=MAIN/'local/runs/round2-v9-realmlp/development-r1'
SEEDS=(42,3407)
INITS=(42,1042,2042)
CANDIDATES=('REALMLP_SINGLE_A20','REALMLP_MEAN3_A20')


def source_hashes():
    paths=list((WORK/'src').rglob('*.py'))+[WORK/p for p in [SPEC,
        'docs/realmlp_time_development/PREREGISTRATION.md','tests/test_realmlp_time_development.py',
        'configs/round2_v9/SPEC.yaml','uv.lock','pyproject.toml']]
    return {str(p.relative_to(WORK)):sha(p) for p in paths}


def setup():
    import psutil
    import torch
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError('Set serial numerical threads before import')
    if sys.version_info[:2]!=(3,12):raise ValueError('Locked Python3.12 required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original=yaml.safe_load((WORK/'configs/round2_v9/SPEC.yaml').read_text())
    if {p:importlib.metadata.version(p) for p in original['runtime_versions']}!=original['runtime_versions']:
        raise ValueError('Frozen native runtime changed')
    if psutil.virtual_memory().available/2**20<3072:raise ValueError('Insufficient entry memory')
    return original


def recipe_for_init(original,init):
    if init not in INITS:raise ValueError('Unregistered training initialization')
    recipe=deepcopy(original)
    recipe['constructor']['random_state']=init
    recipe['resolved']['random_state']=init
    return recipe


def combine(reference,members):
    reference,members=np.asarray(reference,float),np.asarray(members,float)
    if reference.ndim!=1 or members.shape!=(3,len(reference)) or not np.isfinite(reference).all() or not np.isfinite(members).all():
        raise ValueError('Three aligned finite members required')
    result={CANDIDATES[0]:.8*reference+.2*members[0],CANDIDATES[1]:.8*reference+.2*members.mean(0)}
    if any((p<0).any() or not np.isfinite(p).all() for p in result.values()):
        raise ValueError('Invalid frozen blend; no added clipping')
    return result


def select(gains):
    if tuple(gains)!=CANDIDATES or any(set(v)!={'42','3407'} or not all(math.isfinite(x) for x in v.values()) for v in gains.values()):
        raise ValueError('Both complete finite split results required for every candidate')
    eligible=[c for c,v in gains.items() if all(x>0 for x in v.values())]
    return max(eligible,key=lambda c:(min(gains[c].values()),sum(gains[c].values()),-CANDIDATES.index(c))) if eligible else None


def reference(seed):
    with np.load(REVIEW/f'oof-s{seed}.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def prepare(checks):
    import pytabkit
    from .v5_library import load_v5_training_frame,fold_vector
    from .v5_spec import load_v5_spec
    from .v2_release import load_v2
    original=setup();spec=read(WORK/SPEC);run=Path(spec['run_directory'])
    if run.exists():raise FileExistsError('Development directory already consumed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Clean committed source required')
    recovery=read(RECOVERY/'complete.json');terminal=read(RECOVERY/'terminal-reconciliation.json')
    if recovery['status']!='passed' or terminal['status']!='passed' or terminal['recovery_actual_exit_code']!=0:
        raise ValueError('Completed native admission required')
    if sha(adapter.__file__)!=read(RECOVERY/'manifest.json')['new_validator_sha256']:
        raise ValueError('Admitted state adapter changed')
    best=read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['reference'].items()):raise ValueError('Current reference changed before freeze')
    c=read(checks);sources=source_hashes()
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['source_hashes']!=sources or c['junit_sha256']!=sha(c['junit']):
        raise ValueError('Exact-source scientific runner checks required')
    files={}
    for directory in [REVIEW,RECOVERY]:
        for p,h in read(directory/'manifest.json')['files'].items():
            p=str((MAIN/p).resolve())
            if p in files and files[p]!=h:raise ValueError('Conflicting historical identities')
            files[p]=h
    paths=[WORK/p for p in sources]+[Path(checks),Path(c['junit'])]
    paths+=list(Path(pytabkit.__file__).parent.rglob('*.py'))
    paths += [RECOVERY/n for n in ['manifest.json','complete.json','terminal-reconciliation.json']]
    paths += [REVIEW/n for n in ['manifest.json','report.json','independent-audit.json','terminal-reconciliation.json']]
    paths += [MAIN/f'复赛_{stage}/{stage}_{kind}.csv' for stage in ['train','test'] for kind in ['samples','features']]
    paths += [MAIN/'复赛_test/result_template.csv',MAIN/best['package'],MAIN/best['platform_feedback_record']]
    for seed in SEEDS:
        p=REVIEW/f'oof-s{seed}.npz';paths.append(p)
        if sha(p)!=read(REVIEW/'report.json')['oof_sha256'][str(seed)]:raise ValueError('Current review OOF changed')
        paths += [OLD/f'tap_time_len-realmlp_td-s{seed}-f{fold}.npy' for fold in range(5)]
    for p in paths:
        p=p.resolve();h=sha(p)
        if str(p) in files and files[str(p)]!=h:raise ValueError('Changed frozen file: '+str(p))
        files[str(p)]=h
    verify(files)
    frame=load_v5_training_frame(MAIN)
    if hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()!=read(OLD/'manifest.json')['data_digest']:
        raise ValueError('Original V9 training data identity differs')
    test=load_v2(MAIN/'复赛_test','test',322)
    template=pd.read_csv(MAIN/'复赛_test/result_template.csv',dtype={'sample_id':str})
    if test.sample_id.tolist()!=template.sample_id.tolist() or any(t in test for t in TARGETS):raise ValueError('Official test input identity differs')
    folds={s:fold_vector(MAIN,frame,s,load_v5_spec(MAIN)) for s in SEEDS}
    for seed,fv in folds.items():
        a=reference(seed)
        for x,y in [(a['ids'],frame.sample_id.to_numpy(str)),(a['folds'],fv),
            (a['actual'],frame[list(TARGETS)].to_numpy(float)),(a['spouts'],frame.spout_no.to_numpy())]:np.testing.assert_array_equal(x,y)
    run.mkdir(parents=True);frame.to_pickle(run/'training.pkl');test.to_pickle(run/'test.pkl')
    save_arrays(run/'folds.npz',**{str(s):f for s,f in folds.items()})
    files.update({str(run/p):sha(run/p) for p in ['training.pkl','test.pkl','folds.npz']})
    write(run/'manifest.json',dict(spec=spec,files=files,sources=sources,recipe=original['recipes']['realmlp_td'],
        runtime=original['runtime_versions'],source_directory=str(WORK),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        engineering_admission_sha256=sha(RECOVERY/'terminal-reconciliation.json'),formal_promoted=False))
    print(json.dumps(dict(status='frozen',files=len(files),procedures=30,optimizers=60)),flush=True)


def context(seed=None,fold=None,init=None):
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);m=read(run/'manifest.json')
    verify(m['files'])
    if m['source_directory']!=str(WORK) or m['spec']!=spec:raise ValueError('Frozen source context differs')
    if seed is None:return run,m
    if seed not in SEEDS or fold not in range(5) or init not in INITS:raise ValueError('Unregistered scientific task')
    frame=pd.read_pickle(run/'training.pkl');test=pd.read_pickle(run/'test.pkl')
    with np.load(run/'folds.npz',allow_pickle=False) as a:fv=a[str(seed)]
    fitting=frame.loc[fv!=fold].reset_index(drop=True)
    query=frame.loc[fv==fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    return run,m,fitting,query,test


def worker(seed,fold,init):
    run,m,frame,query,test=context(seed,fold,init);d=run/f's{seed}-f{fold}-i{init}';d.mkdir(exist_ok=False)
    identity=dict(source_directory=str(d),split_seed=seed,fold=fold,trial_id=f'TD_TIME_INIT{init}')
    write(d/'estimator-start.json',dict(identity=identity,pid=os.getpid(),fitting_ids=frame.sample_id.tolist(),query_ids=query.sample_id.tolist()))
    recipe=recipe_for_init(m['recipe'],init);y=frame.tap_time_len.to_numpy(float)
    with observe_native(d) as trace:captured=adapter.capture_fit(recipe,frame,y,inner_seed=42)
    model=captured.regressor;epoch=model.metadata_['selected_epoch'];verify_trace(trace,frame,epoch,256)
    predictions={'refit':model.predict(query),
        'selection':captured.estimators[0].predict(captured.encoders[0].transform(query)),
        'test_refit':model.predict(test)}
    if init==42:
        np.testing.assert_array_equal(predictions['refit'],np.load(OLD/f'tap_time_len-realmlp_td-s{seed}-f{fold}.npy',allow_pickle=False))
    states={role:native_state_digest(captured.estimators[i]) for i,role in enumerate(('selection','refit'))}
    for role in ('selection','refit'):adapter.save_snapshot(captured,role,d/(role+'.pkl'),identity)
    save_arrays(d/'predictions.npz',ids=query.sample_id.to_numpy(str),test_ids=test.sample_id.to_numpy(str),**predictions)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>m['spec']['max_rss_mib']:raise ValueError('Scientific peak memory gate failed')
    verify(m['files'])
    write(d/'complete.json',dict(status='passed',identity=identity,recipe=recipe,native_trace=trace,metadata=model.metadata_,
        selected_epoch=epoch,parameter_states=states,predictions_sha256=sha(d/'predictions.npz'),
        native_optimizer_runs=len(trace['optimizers']),native_updates=sum(v['steps'] for v in trace['optimizers']),
        state_hashes={role:sha(d/(role+'.pkl')) for role in ['selection','refit']},
        peak_rss_mib=peak,pid=os.getpid(),manifest_sha256=sha(run/'manifest.json')))
    print(json.dumps(dict(status='passed',seed=seed,fold=fold,init=init,selected_epoch=epoch)),flush=True)


def cold(seed,fold,init):
    run,m,frame,query,test=context(seed,fold,init);d=run/f's{seed}-f{fold}-i{init}';c=read(d/'complete.json')
    if sha(d/'predictions.npz')!=c['predictions_sha256']:raise ValueError('Original warm predictions changed')
    with np.load(d/'predictions.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str));np.testing.assert_array_equal(a['test_ids'],test.sample_id.to_numpy(str))
        result=cold_audit(d,frame,query,frame.tap_time_len.to_numpy(float),c['recipe'],c['identity'],c,
            {k:a[k].copy() for k in ['selection','refit']})
        with forbid_native_fit():
            payload=adapter.load_snapshot(d/'refit.pkl',expected_identity=c['identity'],expected_role='refit',expected_recipe=c['recipe'])
            np.testing.assert_array_equal(adapter.predict_snapshot(payload,test),a['test_refit'])
    result.update(pid=os.getpid(),complete_sha256=sha(d/'complete.json'),official_test_cold_exact=True)
    verify(m['files']);write(d/'cold.json',result)
    print(json.dumps(dict(status='passed',seed=seed,fold=fold,init=init,states=2)),flush=True)


def report():
    from .candidate_tiers import classify_candidates
    from .component_regularization_run import metric_detail
    from .ema_evaluation_diagnostics import sample_indices,simulated_gains,distribution,reduction_detail
    run,m=context();spec=m['spec'];gains={k:{} for k in CANDIDATES};metrics={'tap_time_len':{k:{} for k in ['CURRENT',*CANDIDATES]}}
    diagnostics={};hashes={};optimizers=0;updates=0;states=0
    test=pd.read_pickle(run/'test.pkl');quotas={int(k):int(v) for k,v in test.spout_no.value_counts().items()}
    for seed in SEEDS:
        a=reference(seed);fv=a['folds'];members=np.full((3,len(fv)),np.nan);diagnostics[str(seed)]={}
        for fold in range(5):
            mask=fv==fold
            for j,init in enumerate(INITS):
                d=run/f's{seed}-f{fold}-i{init}';c=read(d/'complete.json');cold=read(d/'cold.json')
                if (cold['status']!='passed' or cold['complete_sha256']!=sha(d/'complete.json')
                        or sha(d/'predictions.npz')!=c['predictions_sha256']):raise ValueError('Incomplete audited model coverage')
                with np.load(d/'predictions.npz',allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['ids'],a['ids'][mask]);members[j,mask]=p['refit']
                optimizers+=c['native_optimizer_runs'];updates+=c['native_updates'];states+=cold['states']
        candidates=combine(a['time'],members);y=a['actual'][:,1]
        np.testing.assert_allclose(candidates[CANDIDATES[0]],a['REALMLP_TD_TIME_A20'],rtol=0,atol=1e-12)
        metrics['tap_time_len']['CURRENT'][str(seed)]=metric_detail(y,a['time'],fv,a['spouts'])
        draws={'uniform':sample_indices(len(y),322,spec['draws'],spec['draw_seed']+seed),
            'spout':sample_indices(len(y),322,spec['draws'],spec['draw_seed']+seed,groups=a['spouts'],quotas=quotas)}
        save_arrays(run/f'draws-s{seed}.npz',**draws)
        for name,p in candidates.items():
            gains[name][str(seed)]=float(50*np.sum(abs(y-a['time'])-abs(y-p))/np.sum(abs(y)))
            metrics['tap_time_len'][name][str(seed)]=metric_detail(y,p,fv,a['spouts'])
            diagnostics[str(seed)][name]=dict(reduction=reduction_detail(y,a['time'],p),
                draws={k:distribution(simulated_gains(y,a['time'],p,d)[0]) for k,d in draws.items()})
        save_arrays(run/f'oof-s{seed}.npz',ids=a['ids'],folds=fv,actual=a['actual'],spouts=a['spouts'],
            iron=a['iron'],reference=a['time'],members=members,**candidates)
        hashes[str(seed)]=sha(run/f'oof-s{seed}.npz')
    if (optimizers,states)!=(60,60):raise ValueError('Frozen native budget/cold inventory differs')
    tier_spec=dict(split_seeds=list(SEEDS),folds=5,candidates={'tap_time_len':list(CANDIDATES)},
        reference_by_target={'tap_time_len':'CURRENT'},tie_preference_by_target={'tap_time_len':list(CANDIDATES)})
    result=dict(status='completed_two_split_native_development',gains=gains,
        mean3_minus_single={s:gains[CANDIDATES[1]][s]-gains[CANDIDATES[0]][s] for s in ['42','3407']},
        selected_for_confirmation=select(gains),metrics=metrics,
        candidate_tiers=classify_candidates(metrics,tier_spec,yaml.safe_load((WORK/'configs/candidate_tiers.yaml').read_text())),
        diagnostics=diagnostics,spout_quotas=quotas,formal_promoted=False,
        native_optimizer_runs=optimizers,native_updates=updates,cold_states=states,oof_sha256=hashes,
        manifest_sha256=sha(run/'manifest.json'),new_confirmation_seeds=0,full_fits=0,packages=0)
    verify(m['files']);write(run/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','gains','selected_for_confirmation']}),flush=True)


def audit():
    run,m=context();r=read(run/'report.json');maximum=0.;gains={k:{} for k in CANDIDATES}
    for seed in SEEDS:
        with np.load(run/f'oof-s{seed}.npz',allow_pickle=False) as a:
            if sha(run/f'oof-s{seed}.npz')!=r['oof_sha256'][str(seed)] or len(set(a['ids']))!=2754:
                raise ValueError('OOF identity/coverage mismatch')
            original=reference(seed)
            for k,x in [('ids',original['ids']),('actual',original['actual']),('folds',original['folds']),('reference',original['time'])]:np.testing.assert_array_equal(a[k],x)
            for fold in range(5):
                mask=a['folds']==fold
                for j,init in enumerate(INITS):
                    d=run/f's{seed}-f{fold}-i{init}';c=read(d/'complete.json');cold=read(d/'cold.json')
                    if cold['complete_sha256']!=sha(d/'complete.json') or c['predictions_sha256']!=sha(d/'predictions.npz'):
                        raise ValueError('Unit evidence changed')
                    for role in ['selection','refit']:
                        if sha(d/(role+'.pkl'))!=c['state_hashes'][role]:raise ValueError('State changed')
                    with np.load(d/'predictions.npz',allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['ids'],a['ids'][mask]);np.testing.assert_array_equal(p['refit'],a['members'][j,mask])
            y=a['actual'][:,1]
            for name in CANDIDATES:
                scalar=np.array([.8*float(a['reference'][i])+.2*(float(a['members'][0,i]) if name==CANDIDATES[0]
                    else math.fsum(float(x) for x in a['members'][:,i])/3) for i in range(len(y))])
                maximum=max(maximum,float(np.max(abs(scalar-a[name]))))
                value=50*math.fsum(abs(float(t)-float(b))-abs(float(t)-float(p)) for t,b,p in zip(y,a['reference'],scalar))/math.fsum(float(t) for t in y)
                gains[name][str(seed)]=value;maximum=max(maximum,abs(value-r['gains'][name][str(seed)]))
            with np.load(run/f'draws-s{seed}.npz',allow_pickle=False) as draws:
                for method in draws.files:
                    d=draws[method]
                    if d.shape!=(10000,322) or (np.diff(np.sort(d,axis=1),axis=1)==0).any():raise ValueError('Sampling identity failed')
                    if method=='spout':
                        for group,count in r['spout_quotas'].items():
                            if not ((a['spouts'][d]==int(group)).sum(1)==count).all():raise ValueError('Sampling spout quota differs')
                    for name in CANDIDATES:
                        values=[]
                        for rows in d:
                            values.append(50*math.fsum(abs(float(y[i])-float(a['reference'][i]))-abs(float(y[i])-float(a[name][i])) for i in rows)/math.fsum(float(y[i]) for i in rows))
                        expected=r['diagnostics'][str(seed)][name]['draws'][method]
                        maximum=max(maximum,abs(float(np.mean(values))-expected['mean']),
                            abs(float(np.mean(np.asarray(values)<0))-expected['negative_fraction']))
                        for q,v in expected['quantiles'].items():maximum=max(maximum,abs(float(np.quantile(values,float(q)))-v))
    if maximum>m['spec']['scalar_atol'] or select(gains)!=r['selected_for_confirmation']:
        raise ValueError('Independent scalar or selection mismatch')
    verify(m['files']);write(run/'independent-audit.json',dict(status='passed',maximum_scalar_difference=maximum,
        report_sha256=sha(run/'report.json'),cold_states=60,new_fits=0,new_optimizers=0,pid=os.getpid()))
    print(json.dumps(dict(status='passed',maximum_scalar_difference=maximum)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','cold','report','audit']);p.add_argument('--checks')
    p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);p.add_argument('--init',type=int)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action in ['worker','cold']:(worker if a.action=='worker' else cold)(a.seed,a.fold,a.init)
    elif a.action=='report':report()
    else:audit()


if __name__=='__main__':main()
