"""Conditional native MAE full fit and unchanged-iron A20 release."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import realmlp_time_release as native
from .data import TARGETS
from .ema_nested_residual import read,write,sha,verify,save_arrays
from .realmlp_development_audit import arrays
from .realmlp_mae_observer import observe_mae_native
from .realmlp_time_development import setup
from .platform_transfer_diagnostics import movement_detail
from .ema_silu_release import package_rows,payload
from .submission import package,ZIP_NAME

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_mae_release/SPEC.json'
CONF=MAIN/'local/runs/realmlp-mae-confirmation-20261004/confirmation-r1'
DEV=MAIN/'local/runs/realmlp-mae-development-20261004/development-r1'
CANDIDATE='REALMLP_MAE_A20'


def sources():
    return {str(p.relative_to(WORK)):sha(p) for p in [*list((WORK/'src').rglob('*.py')),WORK/SPEC,
        WORK/'docs/realmlp_mae/RELEASE.md',WORK/'tests/test_realmlp_mae_release.py',WORK/'uv.lock',WORK/'pyproject.toml']}


def require_quality(report,audit,terminal):
    if (report['candidate']!=CANDIDATE or set(report['gains'])!={'42','3407','271828','314159'}
            or not all(math.isfinite(x) and x>0 for x in report['gains'].values()) or not report['paired']['passed'] or report['paired']['lcb95']<=0
            or audit['status']!='passed' or not audit['four_seed_gate_passed'] or terminal['status']!='passed'
            or terminal['actual_supervisor_exit_code']!=0 or not terminal['four_seed_gate_passed']):
        raise ValueError('Closed four-positive-split native MAE quality required')


def prepare(checks):
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);c=read(checks)
    if run.exists():raise FileExistsError('Consumed native MAE release directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Committed source required')
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or c['junit_sha256']!=sha(c['junit']):raise ValueError('Exact source no-fit checks required')
    r=read(CONF/'report.json');a=read(CONF/'independent-audit.json');t=read(CONF/'terminal-reconciliation.json');old=read(CONF/'manifest.json')
    require_quality(r,a,t)
    if t['report_sha256']!=a['report_sha256'] or a['report_sha256']!=sha(CONF/'report.json') or t['audit_sha256']!=sha(CONF/'independent-audit.json'):raise ValueError('Qualified terminal chain changed')
    best=read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['reference'].items()):raise ValueError('Latest incumbent changed')
    for name in ['v9_realmlp.py','realmlp_state_adapter.py','realmlp_native_audit.py','realmlp_mae_recipe.py','realmlp_mae_observer.py','realmlp_time_release.py']:
        if sha(WORK/'src/bf_tap_r2'/name)!=sha(Path(old['source_directory'])/'src/bf_tap_r2'/name):raise ValueError('Qualified native source changed')
    files=dict(old['files']);paths=[WORK/p for p in sources()]+[Path(checks),Path(c['junit']),MAIN/best['package'],MAIN/best['platform_feedback_record']]
    paths += [p for p in CONF.rglob('*') if p.is_file()]
    for p in paths:
        p=p.resolve();h=sha(p)
        if str(p) in files and files[str(p)]!=h:raise ValueError('Qualified dependency changed')
        files[str(p)]=h
    verify(files);frame=pd.read_pickle(CONF/'training.pkl');query=pd.read_pickle(CONF/'test.pkl')
    ids=pd.read_csv(MAIN/'复赛_test/result_template.csv',dtype={'sample_id':str}).sample_id.tolist()
    if len(frame)!=2754 or query.sample_id.tolist()!=ids or any(t in query for t in TARGETS):raise ValueError('Official full/query identity differs')
    package_rows(MAIN/best['package'],ids)
    run.mkdir(parents=True);frame.to_pickle(run/'training.pkl');query.to_pickle(run/'query.pkl')
    files.update({str(run/n):sha(run/n) for n in ['training.pkl','query.pkl']})
    write(run/'manifest.json',dict(spec=spec,files=files,source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        recipe=old['realmlp_recipe'],parent_zip=str(MAIN/best['package']),confirmation_terminal_sha256=sha(CONF/'terminal-reconciliation.json'),local_four_seed_gate_passed=True,platform_score=None))
    print(json.dumps(dict(status='frozen',full_procedures=1,native_adam=2,packages=1)),flush=True)


def context(inference=False):
    if inference:sys.addaudithook(native.block_training)
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);m=read(run/'manifest.json')
    if inference:
        paths=[*[str(WORK/p) for p in sources()],str(run/'query.pkl'),m['parent_zip']]
        verify({p:m['files'][p] for p in paths})
    else:verify(m['files'])
    if m['source_directory']!=str(WORK) or m['spec']!=spec:raise ValueError('Frozen MAE release changed')
    return run,m


def full(cold=False):
    with patch.object(native,'context',context),patch.object(native,'observe_native',observe_mae_native):native.full(cold)


def query_members(seed,ids):
    result=[]
    for fold in range(5):
        d=DEV/f's{seed}-f{fold}/realmlp';r=read(d/'test-cold.json');p=d/'test-predictions.npz'
        if r['status']!='passed' or sha(p)!=r['predictions_sha256']:raise ValueError('Closed official query control changed')
        a=arrays(p);np.testing.assert_array_equal(a['ids'],ids);result.append(a['prediction'])
    return result


def build():
    run,m=context(inference=True);query,parent,base,prediction=native.label_free(run,m);values=native.blend(base,prediction)
    movement={str(s):movement_detail(.2*(np.mean(query_members(s,query.sample_id.to_numpy(str)),axis=0)-base),.2*(prediction-base)) for s in [42,3407]}
    directory=run/'package';directory.mkdir(exist_ok=False);package(directory,payload(parent,values),query.sample_id.tolist())
    save_arrays(run/'release-predictions.npz',ids=query.sample_id.to_numpy(str),reference=base,realmlp=prediction,candidate=values)
    write(run/'release.json',dict(candidate=CANDIDATE,zip=str(directory/ZIP_NAME),zip_sha256=sha(directory/ZIP_NAME),csv_sha256=sha(directory/'result.csv'),
        prediction_sha256=sha(run/'release-predictions.npz'),native_complete_sha256=sha(run/'native/complete.json'),native_cold_sha256=sha(run/'native/cold.json'),
        movement=movement,training_reads_prohibited=True,pid=os.getpid(),platform_score=None,desktop_writes=0,agent_uploads=0,peak_rss_mib=native.memory()))
    print(json.dumps(dict(status='packaged',zip_sha256=sha(directory/ZIP_NAME),movement=movement)),flush=True)


def audit():
    run,m=context(inference=True);r=read(run/'release.json');query,parent,base,prediction=native.label_free(run,m);rows,data=package_rows(r['zip'],query.sample_id.tolist())
    if r['pid']==os.getpid() or r['candidate']!=CANDIDATE or sha(r['zip'])!=r['zip_sha256'] or sha(Path(r['zip']).with_name('result.csv'))!=r['csv_sha256'] or data!=Path(r['zip']).with_name('result.csv').read_bytes():raise ValueError('Independent package identity differs')
    if r['prediction_sha256']!=sha(run/'release-predictions.npz'):raise ValueError('Release prediction identity changed')
    if any(a['pred_tap_iron']!=b['pred_tap_iron'] for a,b in zip(parent,rows)):raise ValueError('Parent iron field strings changed')
    expected=np.array([.8*float(b)+.2*float(p) for b,p in zip(base,prediction)]);a=arrays(run/'release-predictions.npz')
    for x,y in [(a['candidate'],expected),(a['realmlp'],prediction),(a['reference'],base),(a['ids'],query.sample_id.to_numpy(str))]:np.testing.assert_array_equal(x,y)
    maximum=max(abs(float(row['pred_tap_time_len'])-float(v)) for row,v in zip(rows,expected))
    for seed in [42,3407]:
        members=query_members(seed,query.sample_id.to_numpy(str));cv=np.array([.2*(math.fsum(float(p[i]) for p in members)/5-float(base[i])) for i in range(322)]);full=.2*(prediction-base)
        norm=math.sqrt(math.fsum(float(x*x) for x in cv)*math.fsum(float(x*x) for x in full))
        fields=dict(cv_mean=math.fsum(cv)/322,full_mean=math.fsum(full)/322,cv_rms=math.sqrt(math.fsum(float(x*x) for x in cv)/322),full_rms=math.sqrt(math.fsum(float(x*x) for x in full)/322),
            difference_rms=math.sqrt(math.fsum(float((x-y)**2) for x,y in zip(cv,full))/322),cosine=math.fsum(float(x*y) for x,y in zip(cv,full))/norm if norm else None,
            sign_agreement=sum(np.sign(x)==np.sign(y) for x,y in zip(cv,full))/322)
        for name,value in fields.items():
            if value is None:
                if r['movement'][str(seed)][name] is not None:raise ValueError('Undefined movement changed')
            else:maximum=max(maximum,abs(value-r['movement'][str(seed)][name]))
    if maximum>1e-10:raise ValueError('Independent scalar release audit differs')
    result=dict(status='passed',rows=322,unique_ids=322,official_order=True,zip_only_result_csv=True,crc_passed=True,iron_string_mismatches=0,maximum_scalar_difference=maximum,
        release_sha256=sha(run/'release.json'),zip_sha256=sha(r['zip']),training_reads_prohibited=True,new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=native.memory())
    write(run/'package-audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','full','cold','build','audit']);p.add_argument('--checks');a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action in ['full','cold']:full(a.action=='cold')
    elif a.action=='build':build()
    else:audit()
