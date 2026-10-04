"""Conditional MAE confirmation using closed, same-split Q100 references."""
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
from scipy.stats import t

from . import realmlp_mae_development as development
from .data import FEATURES
from .ema_nested_residual import read,write,sha,verify,save_arrays
from .realmlp_development_audit import arrays,scalar_gain,scalar_wmape
from .realmlp_time_confirmation import gate,peak
from .realmlp_time_development import setup

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_mae_confirmation/SPEC.json'
DEV=MAIN/'local/runs/realmlp-mae-development-20261004/development-r1'
BASE=MAIN/'local/runs/realmlp-time-confirmation-20261004/confirmation-r1'
SEEDS=(271828,314159)
ALL_SEEDS=(42,3407,*SEEDS)
CANDIDATE='REALMLP_MAE_A20'


def sources():
    return {str(p.relative_to(WORK)):sha(p) for p in [*list((WORK/'src').rglob('*.py')),WORK/SPEC,
        WORK/'docs/realmlp_mae/CONFIRMATION.md',WORK/'tests/test_realmlp_mae_confirmation.py',WORK/'uv.lock',WORK/'pyproject.toml']}


def require_development(report,audit,terminal):
    if (report['candidate']!=CANDIDATE or not development.eligible(report['gains']) or not report['confirmation_qualified']
            or audit['status']!='passed' or not audit['confirmation_qualified'] or terminal['status']!='passed'
            or terminal['actual_supervisor_exit_code']!=0 or not terminal['confirmation_qualified']):
        raise ValueError('Closed two-complete-positive-split MAE development required')


def prepare(checks):
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);c=read(checks)
    if run.exists():raise FileExistsError('Consumed MAE confirmation directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Committed source required')
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or c['junit_sha256']!=sha(c['junit']):raise ValueError('Exact source checks required')
    r=read(DEV/'report.json');audit=read(DEV/'independent-audit.json');terminal=read(DEV/'terminal-reconciliation.json')
    require_development(r,audit,terminal)
    if terminal['report_sha256']!=audit['report_sha256'] or audit['report_sha256']!=sha(DEV/'report.json') or terminal['audit_sha256']!=sha(DEV/'independent-audit.json'):
        raise ValueError('Development terminal chain changed')
    old=read(DEV/'manifest.json');base=read(BASE/'manifest.json');bc=read(BASE/'terminal-reconciliation.json');br=read(BASE/'report.json')
    if bc['status']!='passed' or bc['actual_supervisor_exit_code']!=0 or bc['report_sha256']!=sha(BASE/'report.json'):raise ValueError('Actual same-split reference completion required')
    best=read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['reference'].items()):raise ValueError('Current reference changed')
    for name in ['v9_realmlp.py','realmlp_state_adapter.py','realmlp_native_audit.py','realmlp_mae_recipe.py','realmlp_mae_observer.py','realmlp_mae_development.py']:
        if sha(WORK/'src/bf_tap_r2'/name)!=sha(Path(old['source_directory'])/'src/bf_tap_r2'/name):raise ValueError('Qualified MAE training source changed')
    files={}
    for block in [old['files'],base['files']]:
        for path,h in block.items():
            if path in files and files[path]!=h:raise ValueError('Frozen input identity conflict')
            files[path]=h
    paths=[WORK/p for p in sources()]+[Path(checks),Path(c['junit']),MAIN/best['package'],MAIN/best['platform_feedback_record']]
    paths += [p for root in [DEV,BASE] for p in root.rglob('*') if p.is_file()]
    for p in paths:
        p=p.resolve();h=sha(p)
        if str(p) in files and files[str(p)]!=h:raise ValueError('Frozen dependency changed')
        files[str(p)]=h
    verify(files);frame=pd.read_pickle(BASE/'training.pkl');folds=arrays(BASE/'folds.npz');query=pd.read_pickle(DEV/'test.pkl')
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str),pd.read_pickle(DEV/'training.pkl').sample_id.to_numpy(str))
    for seed in ALL_SEEDS:
        a=arrays(BASE/f'oof-s{seed}.npz')
        if sha(BASE/f'oof-s{seed}.npz')!=br['oof_sha256'][str(seed)]:raise ValueError('Closed reference OOF changed')
        for x,y in [(a['ids'],frame.sample_id.to_numpy(str)),(a['actual'],frame.tap_time_len.to_numpy(float)),(a['folds'],folds[str(seed)])]:np.testing.assert_array_equal(x,y)
        if seed not in SEEDS:
            b=arrays(DEV/f'oof-s{seed}.npz');np.testing.assert_array_equal(a['reference'],b['reference']);np.testing.assert_array_equal(a['candidate'],b['mse_control'])
    run.mkdir(parents=True);frame.to_pickle(run/'training.pkl');query.to_pickle(run/'test.pkl');save_arrays(run/'folds.npz',**folds)
    for seed in SEEDS:
        for fold in range(5):(run/f's{seed}-f{fold}').mkdir()
    files.update({str(run/n):sha(run/n) for n in ['training.pkl','test.pkl','folds.npz']})
    write(run/'manifest.json',dict(spec=spec,files=files,realmlp_recipe=old['realmlp_recipe'],admission_terminal_sha256=old['admission_terminal_sha256'],
        development_gains=r['gains'],source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),formal_promoted=False))
    print(json.dumps(dict(status='frozen',new_native_procedures=10,new_adam=20,new_reference_fits=0)),flush=True)


def context(seed=None,fold=None):
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);m=read(run/'manifest.json');verify(m['files'])
    if m['source_directory']!=str(WORK) or m['spec']!=spec:raise ValueError('Frozen MAE confirmation changed')
    if seed is None:return run,m
    if seed not in SEEDS or fold not in range(5):raise ValueError('Unregistered MAE confirmation unit')
    frame=pd.read_pickle(run/'training.pkl');fv=arrays(run/'folds.npz')[str(seed)]
    return run,m,frame.loc[fv!=fold].reset_index(drop=True),frame.loc[fv==fold,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)


def worker(seed,fold,cold=False):
    with patch.object(development,'context',context):development.worker(seed,fold,cold)


def collect(run,m,seed):
    base=arrays(BASE/f'oof-s{seed}.npz');a=dict(ids=base['ids'],folds=base['folds'],actual=base['actual'],spouts=base['spouts'],reference=base['reference'],mse_control=base['candidate'])
    if seed not in SEEDS:
        d=arrays(DEV/f'oof-s{seed}.npz');a.update(native_mae=d['native_mae'],candidate=d['candidate']);return a,0
    prediction=np.full(len(a['ids']),np.nan);updates=0
    for fold in range(5):
        d=run/f's{seed}-f{fold}/realmlp';c=read(d/'complete.json');cold=read(d/'cold.json');q=read(d/'test-cold.json');p=arrays(d/'predictions.npz');mask=a['folds']==fold
        if c['status']!=cold['status'] or cold['status']!=q['status'] or q['status']!='passed' or c['recipe']!=m['realmlp_recipe']:raise ValueError('Complete admitted MAE state coverage required')
        if c['manifest_sha256']!=sha(run/'manifest.json') or c['native_optimizer_runs']!=2 or cold['states']!=2:raise ValueError('Native identity or count differs')
        if cold['complete_sha256']!=q['native_complete_sha256'] or q['native_complete_sha256']!=sha(d/'complete.json') or q['native_cold_sha256']!=sha(d/'cold.json'):raise ValueError('Native receipt chain differs')
        if c['predictions_sha256']!=sha(d/'predictions.npz') or q['predictions_sha256']!=sha(d/'test-predictions.npz'):raise ValueError('Cold predictions changed')
        for role,h in c['state_hashes'].items():
            if sha(d/(role+'.pkl'))!=h:raise ValueError('Native state changed')
        np.testing.assert_array_equal(p['ids'],a['ids'][mask]);prediction[mask]=p['refit'];updates+=c['native_updates']
    a.update(native_mae=prediction,candidate=development.blend(a['reference'],prediction));return a,updates


def report():
    from .component_regularization_run import metric_detail
    run,m=context();gains={};comparison={};metrics={};hashes={};updates=0
    for seed in ALL_SEEDS:
        a,n=collect(run,m,seed);updates+=n;y=a['actual'];p=a['candidate']
        gains[str(seed)]=float(50*np.sum(abs(y-a['reference'])-abs(y-p))/np.sum(abs(y)))
        comparison[str(seed)]=float(50*np.sum(abs(y-a['mse_control'])-abs(y-p))/np.sum(abs(y)))
        metrics[str(seed)]={k:metric_detail(y,a[k],a['folds'],a['spouts']) for k in ['reference','candidate']}
        save_arrays(run/f'oof-s{seed}.npz',**a);hashes[str(seed)]=sha(run/f'oof-s{seed}.npz')
    for s,value in m['development_gains'].items():
        if abs(gains[s]-value)>1e-10:raise ValueError('Closed development gain changed')
    result=dict(status='complete_four_split_native_MAE_confirmation',candidate=CANDIDATE,gains=gains,mae_minus_mse=comparison,paired=gate(gains),metrics=metrics,
        native_optimizer_runs=20,native_states=20,native_updates=updates,new_reference_fits=0,oof_sha256=hashes,manifest_sha256=sha(run/'manifest.json'),
        formal_promoted=False,full_fits=0,packages=0,peak_rss_mib=peak())
    verify(m['files']);write(run/'report.json',result);print(json.dumps({k:result[k] for k in ['status','gains','mae_minus_mse','paired']}),flush=True)


def audit():
    run,m=context();r=read(run/'report.json');gains={};maximum=0.;cells=0;updates=0
    def check(x,y):
        nonlocal maximum,cells
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):raise ValueError('Nonfinite scalar metric')
        maximum=max(maximum,abs(float(x)-float(y)));cells+=1
        if maximum>1e-10:raise ValueError('Independent scalar metric differs')
    for seed in ALL_SEEDS:
        expected,n=collect(run,m,seed);updates+=n;a=arrays(run/f'oof-s{seed}.npz')
        if sha(run/f'oof-s{seed}.npz')!=r['oof_sha256'][str(seed)]:raise ValueError('Complete OOF changed')
        for k,v in expected.items():np.testing.assert_array_equal(a[k],v)
        scalar=np.array([.8*float(b)+.2*float(p) for b,p in zip(a['reference'],a['native_mae'])]);np.testing.assert_array_equal(scalar,a['candidate'])
        y=a['actual'];gains[str(seed)]=scalar_gain(y,a['reference'],scalar);check(gains[str(seed)],r['gains'][str(seed)]);check(scalar_gain(y,a['mse_control'],scalar),r['mae_minus_mse'][str(seed)])
        for name in ['reference','candidate']:
            metric=r['metrics'][str(seed)][name];check(scalar_wmape(y,a[name],range(len(y))),metric['wmape'])
            for fold in range(5):check(scalar_wmape(y,a[name],np.flatnonzero(a['folds']==fold)),metric['by_fold'][str(fold)])
            for spout in set(a['spouts']):check(scalar_wmape(y,a[name],np.flatnonzero(a['spouts']==spout)),metric['by_spout'][str(spout)])
    mean=math.fsum(gains.values())/4;sd=math.sqrt(math.fsum((x-mean)**2 for x in gains.values())/3);lcb=mean-float(t.ppf(.95,3))*sd/2
    for k,v in dict(mean=mean,sd=sd,se=sd/2,lcb95=lcb).items():check(v,r['paired'][k])
    passed=all(x>0 for x in gains.values()) and lcb>0
    if passed!=r['paired']['passed'] or updates!=r['native_updates']:raise ValueError('Independent gate or native count differs')
    verify(m['files']);result=dict(status='passed',four_seed_gate_passed=passed,gains=gains,seed_paired_lcb95=lcb,maximum_scalar_difference=maximum,scalar_cells=cells,
        native_optimizer_runs=20,native_states=20,native_updates=updates,report_sha256=sha(run/'report.json'),manifest_sha256=sha(run/'manifest.json'),new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=peak())
    write(run/'independent-audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','cold','report','audit']);p.add_argument('--checks');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action in ['worker','cold']:worker(a.seed,a.fold,a.action=='cold')
    elif a.action=='report':report()
    else:audit()
