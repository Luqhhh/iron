"""Zero-fit scalar audit recovery using arrays materialized once from each NPZ."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np

from .ema_nested_residual import read,write,sha,verify
from .realmlp_time_development import CANDIDATES,INITS,SEEDS,select

WORK=Path(__file__).resolve().parents[2]
RUN=Path('/home/lux1/iron/local/runs/realmlp-time-development-20261004/development-r1')
OUT=RUN.parent/'development-audit-recovery-r1'
EXPECTED_MANIFEST='b7ff4e88d7b176ca668e2dab1edc916dfc3440601ea212aecd509c596c92b7fc'
EXPECTED_REPORT='6e0364afa8d878d402256904097032e2960f4275d19c2b635185976360ce4de1'


def arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


def scalar_gain(y,reference,candidate,rows=None):
    if rows is None:rows=range(len(y))
    total=math.fsum(abs(float(y[i])) for i in rows)
    if total<=0:raise ValueError('Positive WMAPE denominator required')
    return 50*math.fsum(abs(float(y[i])-float(reference[i]))-abs(float(y[i])-float(candidate[i])) for i in rows)/total


def scalar_wmape(y,p,rows):
    return math.fsum(abs(float(y[i])-float(p[i])) for i in rows)/math.fsum(abs(float(y[i])) for i in rows)


def sources():
    return {str(p.relative_to(WORK)):sha(p) for p in [*list((WORK/'src').rglob('*.py')),
        WORK/'tests/test_realmlp_development_audit.py',WORK/'docs/realmlp_time_development/AUDIT_RECOVERY.md']}


def run(checks):
    if sys.version_info[:2]!=(3,12):raise ValueError('Locked Python3.12 required')
    if any(os.environ.get(k)!='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']):
        raise ValueError('Frozen serial numerical threads required')
    if OUT.exists():raise FileExistsError('Recovery directory already consumed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Committed source required')
    c=read(checks)
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or sha(c['junit'])!=c['junit_sha256']:
        raise ValueError('Exact source checks required')
    if sha(RUN/'manifest.json')!=EXPECTED_MANIFEST or sha(RUN/'report.json')!=EXPECTED_REPORT:
        raise ValueError('Original science/report identity changed')
    failure=read(RUN/'failure.json');interruption=read(RUN/'audit-interruption.json')
    if failure['exits'].get('audit')!=-15 or any(v!=0 for k,v in failure['exits'].items() if k!='audit') or len(failure['exits'])!=62:
        raise ValueError('Original 60 warm/cold stages and report must have finished successfully')
    if any(Path('/proc',str(pid)).exists() for pid in failure['owned_pids']):raise ValueError('Previous owned child still active')
    if interruption['new_fits']!=0 or interruption['new_optimizers']!=0:raise ValueError('Wrong recovery scope')
    manifest=read(RUN/'manifest.json');report=read(RUN/'report.json');files=dict(manifest['files'])
    files.update({str(WORK/p):h for p,h in sources().items()})
    for p in [*RUN.rglob('*.json'),*RUN.rglob('*.pkl'),*RUN.rglob('*.npz'),RUN/'supervise.py',Path(checks),Path(c['junit'])]:files[str(p.resolve())]=sha(p)
    verify(files);OUT.mkdir()
    write(OUT/'manifest.json',dict(files=files,original_manifest_sha256=EXPECTED_MANIFEST,original_report_sha256=EXPECTED_REPORT,
        source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        original_audit_exit_code=-15,scientific_retry=False,new_fits=0,new_optimizers=0,original_tolerances_unchanged=True))
    maximum=0.;cells=0;gains={k:{} for k in CANDIDATES};optimizers=0;states=0;updates=0
    def check(actual,expected):
        nonlocal maximum,cells
        if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):raise ValueError('Nonfinite scalar result')
        maximum=max(maximum,abs(float(actual)-float(expected)));cells+=1
        if maximum>manifest['spec']['scalar_atol']:raise ValueError('Frozen independent scalar tolerance failed')
    review=Path('/home/lux1/iron/local/runs/realmlp-current-review-20261004/review-r1')
    for seed in SEEDS:
        a=arrays(RUN/f'oof-s{seed}.npz');base=arrays(review/f'oof-s{seed}.npz')
        if sha(RUN/f'oof-s{seed}.npz')!=report['oof_sha256'][str(seed)] or len(set(a['ids']))!=2754:raise ValueError('Complete OOF identity required')
        for k,b in [('ids','ids'),('actual','actual'),('folds','folds'),('spouts','spouts'),('iron','iron'),('reference','time')]:np.testing.assert_array_equal(a[k],base[b])
        for fold in range(5):
            mask=a['folds']==fold
            if not mask.any():raise ValueError('Missing fold')
            for j,init in enumerate(INITS):
                d=RUN/f's{seed}-f{fold}-i{init}';unit=read(d/'complete.json');cold=read(d/'cold.json');p=arrays(d/'predictions.npz')
                if cold['status']!='passed' or cold['complete_sha256']!=sha(d/'complete.json') or unit['predictions_sha256']!=sha(d/'predictions.npz'):
                    raise ValueError('Warm/cold evidence changed')
                if unit['manifest_sha256']!=EXPECTED_MANIFEST or unit['identity']!=dict(source_directory=str(d),split_seed=seed,fold=fold,trial_id=f'TD_TIME_INIT{init}'):
                    raise ValueError('Physical model identity changed')
                for role,h in unit['state_hashes'].items():
                    if sha(d/(role+'.pkl'))!=h:raise ValueError('Native state changed')
                np.testing.assert_array_equal(p['ids'],a['ids'][mask]);np.testing.assert_array_equal(p['refit'],a['members'][j,mask])
                optimizers+=unit['native_optimizer_runs'];states+=cold['states'];updates+=unit['native_updates']
        y=a['actual'][:,1]
        for name in CANDIDATES:
            expected=np.array([.8*float(a['reference'][i])+.2*(float(a['members'][0,i]) if name==CANDIDATES[0] else math.fsum(float(x) for x in a['members'][:,i])/3) for i in range(len(y))])
            check(float(np.max(abs(expected-a[name]))),0)
            gains[name][str(seed)]=scalar_gain(y,a['reference'],expected)
            check(gains[name][str(seed)],report['gains'][name][str(seed)])
        for name,p in [('CURRENT',a['reference']),*[(n,a[n]) for n in CANDIDATES]]:
            metric=report['metrics']['tap_time_len'][name][str(seed)]
            check(scalar_wmape(y,p,range(len(y))),metric['wmape'])
            for fold in range(5):check(scalar_wmape(y,p,np.flatnonzero(a['folds']==fold)),metric['by_fold'][str(fold)])
            for spout in set(a['spouts']):check(scalar_wmape(y,p,np.flatnonzero(a['spouts']==spout)),metric['by_spout'][str(spout)])
        for method,d in arrays(RUN/f'draws-s{seed}.npz').items():
            if d.shape!=(10000,322) or (d<0).any() or (d>=len(y)).any() or (np.diff(np.sort(d,axis=1),axis=1)==0).any():raise ValueError('Sampling design changed')
            if method=='spout':
                for group,count in report['spout_quotas'].items():
                    if not ((a['spouts'][d]==int(group)).sum(1)==count).all():raise ValueError('Sampling spout quota differs')
            for name in CANDIDATES:
                values=np.array([scalar_gain(y,a['reference'],a[name],rows) for rows in d]);expected=report['diagnostics'][str(seed)][name]['draws'][method]
                check(values.mean(),expected['mean']);check((values<0).mean(),expected['negative_fraction'])
                for q,v in expected['quantiles'].items():check(np.quantile(values,float(q)),v)
    if optimizers!=60 or states!=60 or updates!=report['native_updates'] or select(gains)!=report['selected_for_confirmation']:
        raise ValueError('Native inventory or frozen candidate selection changed')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>manifest['spec']['max_rss_mib']:raise ValueError('Recovery memory gate failed')
    verify(files)
    result=dict(status='passed',maximum_scalar_difference=maximum,scalar_cells=cells,native_optimizer_runs=optimizers,
        cold_states=states,native_updates=updates,selected_for_confirmation=select(gains),report_sha256=EXPECTED_REPORT,
        manifest_sha256=sha(OUT/'manifest.json'),original_audit_exit_code=-15,original_failure_preserved=True,
        new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=peak)
    write(OUT/'independent-audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checks',required=True);run(p.parse_args().checks)
