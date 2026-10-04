"""Fixed native MAE loss development, reusing the admitted native state path."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from . import realmlp_time_confirmation as native_units
from . import realmlp_state_adapter as adapter
from .data import FEATURES,TARGETS
from .ema_nested_residual import read,write,sha,verify,save_arrays
from .realmlp_development_audit import arrays,scalar_gain,scalar_wmape
from .realmlp_mae_recipe import mae_recipe
from .realmlp_mae_observer import observe_mae_native
from .realmlp_native_audit import forbid_native_fit
from .realmlp_time_development import setup

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_mae_development/SPEC.json'
OLD=MAIN/'local/runs/realmlp-time-development-20261004/development-r1'
ADMISSION=MAIN/'local/runs/realmlp-mae-admission-20261004/engineering-r1'
SEEDS=(42,3407)
CANDIDATE='REALMLP_MAE_A20'


def sources():
    return {str(p.relative_to(WORK)):sha(p) for p in [*list((WORK/'src').rglob('*.py')),WORK/SPEC,
        WORK/'docs/realmlp_mae/DEVELOPMENT.md',WORK/'tests/test_realmlp_mae_development.py',WORK/'uv.lock',WORK/'pyproject.toml']}


def eligible(gains):
    if set(gains)!={'42','3407'} or not all(math.isfinite(x) for x in gains.values()):raise ValueError('Two complete finite splits required')
    return all(x>0 for x in gains.values())


def blend(parent,prediction):
    parent,prediction=np.asarray(parent,float),np.asarray(prediction,float)
    if parent.ndim!=1 or parent.shape!=prediction.shape or not np.isfinite([parent,prediction]).all():raise ValueError('Aligned finite OOF vectors required')
    result=.8*parent+.2*prediction
    if (result<0).any():raise ValueError('Invalid fixed blend; clipping prohibited')
    return result


def prepare(checks):
    original=setup();spec=read(WORK/SPEC);run=Path(spec['run_directory'])
    if run.exists():raise FileExistsError('Consumed native MAE development directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Committed source required')
    c=read(checks)
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or c['junit_sha256']!=sha(c['junit']):raise ValueError('Exact locked no-fit checks required')
    admission=read(ADMISSION/'report.json');terminal=read(ADMISSION/'terminal-reconciliation.json')
    if admission['status']!='passed' or admission['optimizer_runs']!=4 or terminal['status']!='passed' or terminal['actual_supervisor_exit_code']!=0 or terminal['report_sha256']!=sha(ADMISSION/'report.json'):
        raise ValueError('Completed native MAE training/capture/cold admission required')
    admitted=read(ADMISSION/'manifest.json');recipe=mae_recipe(original['recipes']['realmlp_td'])
    if recipe!=admitted['recipe']:raise ValueError('Admitted MAE recipe changed')
    for name in ['v9_realmlp.py','realmlp_state_adapter.py','realmlp_native_audit.py','realmlp_mae_recipe.py','realmlp_mae_observer.py']:
        if sha(WORK/'src/bf_tap_r2'/name)!=sha(Path(admitted['source_directory'])/'src/bf_tap_r2'/name):raise ValueError('Admitted native source changed')
    best=read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['reference'].items()):raise ValueError('Current incumbent changed before MAE freeze')
    recovery=OLD.parent/'development-audit-recovery-r1';closed=read(recovery/'terminal-reconciliation.json')
    if closed['status']!='passed' or closed['actual_recovery_exit_code']!=0 or closed['original_report_sha256']!=sha(OLD/'report.json'):raise ValueError('Original MSE evidence not closed')
    files=dict(read(recovery/'manifest.json')['files'])
    for p,h in admitted['files'].items():
        if p in files and files[p]!=h:raise ValueError('Conflicting admitted and historical identities')
        files[p]=h
    paths=[WORK/p for p in sources()]+[Path(checks),Path(c['junit']),ADMISSION/'manifest.json',ADMISSION/'report.json',ADMISSION/'terminal-reconciliation.json',
        recovery/'manifest.json',recovery/'independent-audit.json',recovery/'terminal-reconciliation.json',MAIN/best['package'],MAIN/best['platform_feedback_record']]
    for seed in SEEDS:
        paths += [OLD/f'oof-s{seed}.npz',OLD/f'draws-s{seed}.npz']
        for fold in range(5):paths += [p for p in (OLD/f's{seed}-f{fold}-i42').iterdir() if p.is_file()]
    for p in paths:
        p=p.resolve();h=sha(p)
        if str(p) in files and files[str(p)]!=h:raise ValueError('Frozen dependency changed')
        files[str(p)]=h
    verify(files);frame=pd.read_pickle(OLD/'training.pkl');test=pd.read_pickle(OLD/'test.pkl');folds=arrays(OLD/'folds.npz')
    for seed in SEEDS:
        a=arrays(OLD/f'oof-s{seed}.npz')
        for x,y in [(a['ids'],frame.sample_id.to_numpy(str)),(a['actual'],frame[list(TARGETS)].to_numpy(float)),(a['folds'],folds[str(seed)])]:np.testing.assert_array_equal(x,y)
    run.mkdir();frame.to_pickle(run/'training.pkl');test.to_pickle(run/'test.pkl');save_arrays(run/'folds.npz',**folds)
    for seed in SEEDS:
        for fold in range(5):(run/f's{seed}-f{fold}').mkdir()
    files.update({str(run/n):sha(run/n) for n in ['training.pkl','test.pkl','folds.npz']})
    write(run/'manifest.json',dict(spec=spec,files=files,realmlp_recipe=recipe,source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        admission_terminal_sha256=sha(ADMISSION/'terminal-reconciliation.json'),formal_promoted=False))
    print(json.dumps(dict(status='frozen',new_native_procedures=10,new_adam=20)),flush=True)


def context(seed=None,fold=None):
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);m=read(run/'manifest.json');verify(m['files'])
    if m['source_directory']!=str(WORK) or m['spec']!=spec:raise ValueError('Frozen MAE development context changed')
    if seed is None:return run,m
    if seed not in SEEDS or fold not in range(5):raise ValueError('Unregistered MAE development unit')
    frame=pd.read_pickle(run/'training.pkl');fv=arrays(run/'folds.npz')[str(seed)]
    return run,m,frame.loc[fv!=fold].reset_index(drop=True),frame.loc[fv==fold,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)


def worker(seed,fold,cold=False):
    ctx=context(seed,fold);run,m,training,query=ctx
    def require_admitted_mae(actual_run):
        terminal=read(ADMISSION/'terminal-reconciliation.json')
        if actual_run!=run or sha(ADMISSION/'terminal-reconciliation.json')!=m['admission_terminal_sha256'] or terminal['status']!='passed' or terminal['actual_supervisor_exit_code']!=0:
            raise ValueError('Closed native MAE admission required in this development scope')
    # Only the experiment context and already admitted MAE observation wrapper
    # differ. The original native fitting/state/cold functions stay unchanged.
    with patch.object(native_units,'context',lambda *args:ctx),patch.object(native_units,'require_bridges',require_admitted_mae),patch.object(native_units,'observe_native',observe_mae_native):
        native_units.realmlp(seed,fold,cold)
    if cold:
        d=run/f's{seed}-f{fold}'/'realmlp';c=read(d/'complete.json');test=pd.read_pickle(run/'test.pkl')
        with forbid_native_fit():
            saved=adapter.load_snapshot(d/'refit.pkl',expected_identity=c['identity'],expected_role='refit',expected_recipe=m['realmlp_recipe'])
            p=adapter.predict_snapshot(saved,test)
            variants=[adapter.predict_snapshot(saved,test.iloc[::-1])[::-1],np.concatenate([adapter.predict_snapshot(saved,test.iloc[i:i+37]) for i in range(0,len(test),37)]),adapter.predict_snapshot(saved,test.iloc[:1])]
            maximum=max(float(np.max(abs(v-p[:len(v)]))) for v in variants)
            if maximum>.0005:raise ValueError('Same official query row/chunk gate failed')
        save_arrays(d/'test-predictions.npz',ids=test.sample_id.to_numpy(str),prediction=p)
        write(d/'test-cold.json',dict(status='passed',maximum_order_chunk_difference=maximum,predictions_sha256=sha(d/'test-predictions.npz'),
            native_complete_sha256=sha(d/'complete.json'),native_cold_sha256=sha(d/'cold.json'),new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=native_units.peak()))
    verify(m['files'])


def collect(run,m,seed):
    a=arrays(OLD/f'oof-s{seed}.npz');prediction=np.full(len(a['ids']),np.nan);updates=0
    for fold in range(5):
        d=run/f's{seed}-f{fold}'/'realmlp';c=read(d/'complete.json');cold=read(d/'cold.json');test=read(d/'test-cold.json');p=arrays(d/'predictions.npz');mask=a['folds']==fold
        if cold['status']!=test['status'] or cold['status']!='passed' or cold['complete_sha256']!=sha(d/'complete.json') or c['predictions_sha256']!=sha(d/'predictions.npz'):
            raise ValueError('Complete MAE warm/cold coverage required')
        if c['recipe']!=m['realmlp_recipe'] or c['manifest_sha256']!=sha(run/'manifest.json') or c['native_optimizer_runs']!=2 or cold['states']!=2:
            raise ValueError('Native identity or per-unit budget changed')
        for role,h in c['state_hashes'].items():
            if sha(d/(role+'.pkl'))!=h:raise ValueError('MAE native state changed')
        if test['predictions_sha256']!=sha(d/'test-predictions.npz') or test['native_cold_sha256']!=sha(d/'cold.json') or test['native_complete_sha256']!=sha(d/'complete.json'):raise ValueError('Official query cold proof changed')
        np.testing.assert_array_equal(p['ids'],a['ids'][mask]);prediction[mask]=p['refit'];updates+=c['native_updates']
    a['native_mae']=prediction;a[CANDIDATE]=blend(a['reference'],prediction)
    return a,updates


def report():
    from .candidate_tiers import classify_candidates
    from .component_regularization_run import metric_detail
    from .ema_evaluation_diagnostics import distribution,simulated_gains,reduction_detail
    run,m=context();gains={};comparison={};metrics={'tap_time_len':{k:{} for k in ['CURRENT',CANDIDATE]}};diagnostics={};hashes={};updates=0
    for seed in SEEDS:
        a,n=collect(run,m,seed);updates+=n;y=a['actual'][:,1];p=a[CANDIDATE];base=a['reference'];mse=a['REALMLP_SINGLE_A20']
        gains[str(seed)]=float(50*np.sum(abs(y-base)-abs(y-p))/np.sum(abs(y)))
        comparison[str(seed)]=float(50*np.sum(abs(y-mse)-abs(y-p))/np.sum(abs(y)))
        for name,v in [('CURRENT',base),(CANDIDATE,p)]:metrics['tap_time_len'][name][str(seed)]=metric_detail(y,v,a['folds'],a['spouts'])
        diagnostics[str(seed)]=dict(reduction=reduction_detail(y,base,p),draws={method:distribution(simulated_gains(y,base,p,d)[0]) for method,d in arrays(OLD/f'draws-s{seed}.npz').items()})
        save_arrays(run/f'oof-s{seed}.npz',ids=a['ids'],folds=a['folds'],actual=a['actual'],spouts=a['spouts'],iron=a['iron'],reference=base,mse_control=mse,native_mae=a['native_mae'],candidate=p)
        hashes[str(seed)]=sha(run/f'oof-s{seed}.npz')
    tiers=dict(split_seeds=list(SEEDS),folds=5,candidates={'tap_time_len':[CANDIDATE]},reference_by_target={'tap_time_len':'CURRENT'},tie_preference_by_target={'tap_time_len':[CANDIDATE]})
    result=dict(status='completed_two_split_native_MAE_development',candidate=CANDIDATE,gains=gains,mae_minus_mse=comparison,metrics=metrics,
        diagnostics=diagnostics,candidate_tiers=classify_candidates(metrics,tiers,yaml.safe_load((WORK/'configs/candidate_tiers.yaml').read_text())),
        confirmation_qualified=eligible(gains),native_optimizer_runs=20,native_states=20,native_updates=updates,oof_sha256=hashes,
        manifest_sha256=sha(run/'manifest.json'),formal_promoted=False,new_confirmation_seeds=0,full_fits=0,packages=0,peak_rss_mib=native_units.peak())
    verify(m['files']);write(run/'report.json',result);print(json.dumps({k:result[k] for k in ['status','gains','mae_minus_mse','confirmation_qualified']}),flush=True)


def audit():
    run,m=context();r=read(run/'report.json');maximum=0.;gains={};updates=0;cells=0
    def check(x,y):
        nonlocal maximum,cells
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):raise ValueError('Nonfinite scalar evidence')
        maximum=max(maximum,abs(float(x)-float(y)));cells+=1
        if maximum>1e-10:raise ValueError('Independent scalar mismatch')
    for seed in SEEDS:
        source,n=collect(run,m,seed);updates+=n;a=arrays(run/f'oof-s{seed}.npz');y=a['actual'][:,1]
        if sha(run/f'oof-s{seed}.npz')!=r['oof_sha256'][str(seed)]:raise ValueError('Complete OOF changed')
        for k in ['ids','folds','actual','spouts','iron','reference','native_mae']:np.testing.assert_array_equal(a[k],source[k])
        np.testing.assert_array_equal(a['mse_control'],source['REALMLP_SINGLE_A20'])
        expected=np.array([.8*float(b)+.2*float(p) for b,p in zip(a['reference'],a['native_mae'])]);np.testing.assert_array_equal(expected,a['candidate'])
        gains[str(seed)]=scalar_gain(y,a['reference'],expected);check(gains[str(seed)],r['gains'][str(seed)]);check(scalar_gain(y,a['mse_control'],expected),r['mae_minus_mse'][str(seed)])
        for name,p in [('CURRENT',a['reference']),(CANDIDATE,expected)]:
            metric=r['metrics']['tap_time_len'][name][str(seed)];check(scalar_wmape(y,p,range(len(y))),metric['wmape'])
            for fold in range(5):check(scalar_wmape(y,p,np.flatnonzero(a['folds']==fold)),metric['by_fold'][str(fold)])
            for spout in set(a['spouts']):check(scalar_wmape(y,p,np.flatnonzero(a['spouts']==spout)),metric['by_spout'][str(spout)])
        for method,d in arrays(OLD/f'draws-s{seed}.npz').items():
            values=np.array([scalar_gain(y,a['reference'],expected,rows) for rows in d]);record=r['diagnostics'][str(seed)]['draws'][method]
            check(values.mean(),record['mean']);check((values<0).mean(),record['negative_fraction'])
            for q,v in record['quantiles'].items():check(np.quantile(values,float(q)),v)
    if eligible(gains)!=r['confirmation_qualified'] or updates!=r['native_updates']:raise ValueError('Independent qualification or updates changed')
    verify(m['files']);result=dict(status='passed',maximum_scalar_difference=maximum,scalar_cells=cells,gains=gains,confirmation_qualified=eligible(gains),
        native_optimizer_runs=20,native_states=20,native_updates=updates,report_sha256=sha(run/'report.json'),manifest_sha256=sha(run/'manifest.json'),new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=native_units.peak())
    write(run/'independent-audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','cold','report','audit']);p.add_argument('--checks');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action in ['worker','cold']:worker(a.seed,a.fold,a.action=='cold')
    elif a.action=='report':report()
    else:audit()
