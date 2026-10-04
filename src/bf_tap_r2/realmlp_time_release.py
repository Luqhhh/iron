"""One qualified native RealMLP full fit and fixed time-only private package."""
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
import pandas as pd

from . import realmlp_state_adapter as adapter
from .data import TARGETS
from .ema_nested_residual import read,write,sha,verify,save_arrays
from .ema_silu_release import package_rows,payload
from .platform_transfer_diagnostics import movement_detail
from .realmlp_development_audit import arrays
from .realmlp_native_audit import observe_native,verify_trace,native_state_digest,cold_audit,forbid_native_fit
from .realmlp_time_development import setup
from .submission import ZIP_NAME,package,deny_training_reads

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_time_release/SPEC.json'
CONF=MAIN/'local/runs/realmlp-time-confirmation-20261004/confirmation-r1'
DEV=MAIN/'local/runs/realmlp-time-development-20261004/development-r1'


def sources():
    return {str(p.relative_to(WORK)):sha(p) for p in [*list((WORK/'src').rglob('*.py')),WORK/SPEC,
        WORK/'docs/realmlp_time_release/PREREGISTRATION.md',WORK/'tests/test_realmlp_time_release.py',WORK/'uv.lock',WORK/'pyproject.toml']}


def blend(parent,model):
    parent,model=np.asarray(parent,float),np.asarray(model,float)
    if parent.ndim!=1 or parent.shape!=model.shape or not np.isfinite([parent,model]).all():raise ValueError('Aligned finite single-model predictions required')
    result=.8*parent+.2*model
    if (result<0).any() or not np.isfinite(result).all():raise ValueError('Invalid fixed blend; clipping forbidden')
    return result


def require_quality(report,audit,terminal):
    if (report['candidate']!='REALMLP_SINGLE_A20' or set(report['gains'])!={'42','3407','271828','314159'}
            or not all(math.isfinite(x) and x>0 for x in report['gains'].values()) or not report['paired']['passed']
            or report['paired']['lcb95']<=0 or audit['status']!='passed' or not audit['four_seed_gate_passed']
            or terminal['status']!='passed' or terminal['actual_supervisor_exit_code']!=0):
        raise ValueError('Completed four-positive-split quality and independent terminal required')


def memory():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if value>1536:raise ValueError('Frozen memory gate failed')
    return value


def prepare(checks):
    from .v5_library import load_v5_training_frame
    from .v2_release import load_v2
    original=setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);checks=Path(checks)
    if run.exists():raise FileExistsError('Consumed release directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Committed source required')
    c=read(checks)
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or c['junit_sha256']!=sha(c['junit']):raise ValueError('Exact source locked no-fit checks required')
    report=read(CONF/'report.json');audit=read(CONF/'independent-audit.json');terminal=read(CONF/'terminal-reconciliation.json')
    require_quality(report,audit,terminal)
    if (terminal['independent_audit_sha256']!=sha(CONF/'independent-audit.json') or audit['report_sha256']!=sha(CONF/'report.json')
            or terminal['report_sha256']!=sha(CONF/'report.json') or audit['manifest_sha256']!=sha(CONF/'manifest.json')):raise ValueError('Confirmation receipt chain changed')
    best=read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['reference'].items()):raise ValueError('Current parent changed')
    previous=read(CONF/'manifest.json');files=dict(previous['files'])
    if previous['realmlp_recipe']!=original['recipes']['realmlp_td']:raise ValueError('Qualified native recipe changed')
    for name in ['v9_realmlp.py','realmlp_state_adapter.py','realmlp_native_audit.py']:
        if sha(WORK/'src/bf_tap_r2'/name)!=sha(Path(previous['source_directory'])/'src/bf_tap_r2'/name):raise ValueError('Qualified native source changed')
    paths=[WORK/p for p in sources()]+[checks,Path(c['junit']),MAIN/best['package'],MAIN/best['platform_feedback_record']]
    paths += [p for p in CONF.rglob('*.json')]+[p for p in CONF.glob('oof-*.npz')]
    for stage in ['train','test']:
        paths += [MAIN/f'复赛_{stage}/{stage}_{kind}.csv' for kind in ['samples','features']]
    paths += [MAIN/'复赛_test/result_template.csv']
    for seed in [42,3407]:
        for fold in range(5):paths.append(DEV/f's{seed}-f{fold}-i42/predictions.npz')
    for p in paths:
        p=p.resolve();h=sha(p)
        if str(p) in files and files[str(p)]!=h:raise ValueError('Frozen dependency changed')
        files[str(p)]=h
    verify(files);frame=load_v5_training_frame(MAIN);query=load_v2(MAIN/'复赛_test','test',322)
    ids=pd.read_csv(MAIN/'复赛_test/result_template.csv',dtype={'sample_id':str}).sample_id.tolist()
    if len(frame)!=2754 or query.sample_id.tolist()!=ids or any(t in query for t in TARGETS):raise ValueError('Official full/query identity differs')
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str),pd.read_pickle(CONF/'training.pkl').sample_id.to_numpy(str))
    parent,_=package_rows(MAIN/best['package'],ids)
    if sha(MAIN/best['package'])!=spec['reference']['zip_sha256']:raise ValueError('Parent package hash differs')
    run.mkdir();frame.to_pickle(run/'training.pkl');query.to_pickle(run/'query.pkl')
    files.update({str(run/p):sha(run/p) for p in ['training.pkl','query.pkl']})
    write(run/'manifest.json',dict(spec=spec,files=files,source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        recipe=original['recipes']['realmlp_td'],parent_zip=str(MAIN/best['package']),confirmation_terminal_sha256=sha(CONF/'terminal-reconciliation.json'),
        local_four_seed_gate_passed=True,platform_score=None))
    print(json.dumps(dict(status='frozen',full_procedures=1,native_adam=2,packages=1,desktop_writes=0)),flush=True)


def block_training(event,args):
    deny_training_reads(event,args)
    if event=='open' and isinstance(args[0],(str,bytes)) and 'training.pkl' in str(args[0]):raise RuntimeError('Release inference cannot read training frames')


def context(inference=False):
    if inference:sys.addaudithook(block_training)
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);m=read(run/'manifest.json')
    if inference:
        # Training identities are checked by warm/cold and terminal auditors.
        # This separate inference process may open only its source, query,
        # trained states, label-free predictions and fixed parent package.
        required=[*[str(WORK/p) for p in sources()],str(run/'query.pkl'),m['parent_zip']]
        verify({p:m['files'][p] for p in required})
    else:verify(m['files'])
    if m['source_directory']!=str(WORK) or m['spec']!=spec:raise ValueError('Frozen release context changed')
    return run,m


def full(cold=False):
    run,m=context();frame=pd.read_pickle(run/'training.pkl');query=pd.read_pickle(run/'query.pkl');d=run/'native';recipe=m['recipe']
    identity=dict(source_directory=str(d),split_seed=42,fold=-1,trial_id='TD_TIME_FULL_INIT42')
    if cold:
        c=read(d/'complete.json');a=arrays(d/'predictions.npz')
        if c['pid']==os.getpid() or c['identity']!=identity or c['predictions_sha256']!=sha(d/'predictions.npz'):raise ValueError('Independent full cold identity differs')
        np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str))
        result=cold_audit(d,frame,query,frame.tap_time_len.to_numpy(float),recipe,identity,c,{k:a[k] for k in ['selection','refit']})
        write(d/'cold.json',dict(**result,pid=os.getpid(),complete_sha256=sha(d/'complete.json'),peak_rss_mib=memory()))
    else:
        d.mkdir(exist_ok=False);write(d/'estimator-start.json',dict(identity=identity,pid=os.getpid(),fitting_ids=frame.sample_id.tolist(),query_ids=query.sample_id.tolist()))
        with observe_native(d) as trace:captured=adapter.capture_fit(recipe,frame,frame.tap_time_len.to_numpy(float),inner_seed=42)
        model=captured.regressor;epoch=model.metadata_['selected_epoch'];verify_trace(trace,frame,epoch,256)
        predictions={'refit':model.predict(query),'selection':captured.estimators[0].predict(captured.encoders[0].transform(query))}
        states={role:native_state_digest(captured.estimators[i]) for i,role in enumerate(('selection','refit'))}
        for role in states:adapter.save_snapshot(captured,role,d/(role+'.pkl'),identity)
        save_arrays(d/'predictions.npz',ids=query.sample_id.to_numpy(str),**predictions)
        write(d/'complete.json',dict(status='passed',identity=identity,recipe=recipe,native_trace=trace,metadata=model.metadata_,selected_epoch=epoch,parameter_states=states,
            predictions_sha256=sha(d/'predictions.npz'),native_optimizer_runs=len(trace['optimizers']),native_updates=sum(v['steps'] for v in trace['optimizers']),
            state_hashes={r:sha(d/(r+'.pkl')) for r in states},peak_rss_mib=memory(),pid=os.getpid(),manifest_sha256=sha(run/'manifest.json')))
    verify(m['files']);print(json.dumps(dict(status='passed',action='cold' if cold else 'full')),flush=True)


def label_free(run,m):
    query=pd.read_pickle(run/'query.pkl');c=read(run/'native/complete.json');cold=read(run/'native/cold.json')
    if cold['status']!='passed' or cold['complete_sha256']!=sha(run/'native/complete.json') or c['native_optimizer_runs']!=2:raise ValueError('Full native cold proof required')
    with forbid_native_fit():
        saved=adapter.load_snapshot(run/'native/refit.pkl',expected_identity=c['identity'],expected_role='refit',expected_recipe=m['recipe'])
        prediction=adapter.predict_snapshot(saved,query)
    a=arrays(run/'native/predictions.npz');np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str));np.testing.assert_array_equal(prediction,a['refit'])
    parent,_=package_rows(m['parent_zip'],query.sample_id.tolist());base=np.array([float(row['pred_tap_time_len']) for row in parent])
    return query,parent,base,prediction


def build():
    run,m=context(inference=True);query,parent,base,prediction=label_free(run,m);values=blend(base,prediction);movement={}
    for seed in [42,3407]:
        members=[]
        for fold in range(5):
            a=arrays(DEV/f's{seed}-f{fold}-i42/predictions.npz');np.testing.assert_array_equal(a['test_ids'],query.sample_id.to_numpy(str));members.append(a['test_refit'])
        foldmean=np.mean(members,axis=0)
        movement[str(seed)]=movement_detail(.2*(foldmean-base),.2*(prediction-base))
    directory=run/'package';directory.mkdir(exist_ok=False);package(directory,payload(parent,values),query.sample_id.tolist())
    save_arrays(run/'release-predictions.npz',ids=query.sample_id.to_numpy(str),reference=base,realmlp=prediction,candidate=values)
    write(run/'release.json',dict(candidate='REALMLP_SINGLE_A20',zip=str(directory/ZIP_NAME),zip_sha256=sha(directory/ZIP_NAME),csv_sha256=sha(directory/'result.csv'),
        prediction_sha256=sha(run/'release-predictions.npz'),native_complete_sha256=sha(run/'native/complete.json'),native_cold_sha256=sha(run/'native/cold.json'),
        movement=movement,training_reads_prohibited=True,pid=os.getpid(),platform_score=None,desktop_writes=0,agent_uploads=0))
    print(json.dumps(dict(status='packaged',zip_sha256=sha(directory/ZIP_NAME),movement=movement)),flush=True)


def audit():
    run,m=context(inference=True);r=read(run/'release.json');query,parent,base,prediction=label_free(run,m);rows,data=package_rows(r['zip'],query.sample_id.tolist())
    if (r['pid']==os.getpid() or sha(r['zip'])!=r['zip_sha256'] or sha(Path(r['zip']).with_name('result.csv'))!=r['csv_sha256']
            or data!=Path(r['zip']).with_name('result.csv').read_bytes() or r['prediction_sha256']!=sha(run/'release-predictions.npz')):raise ValueError('Independent release byte identity differs')
    expected=np.array([.8*float(b)+.2*float(p) for b,p in zip(base,prediction)])
    if any(a['pred_tap_iron']!=b['pred_tap_iron'] for a,b in zip(parent,rows)):raise ValueError('Unchanged iron field strings changed')
    maximum=max(abs(float(row['pred_tap_time_len'])-float(v)) for row,v in zip(rows,expected))
    a=arrays(run/'release-predictions.npz');np.testing.assert_array_equal(a['candidate'],expected);np.testing.assert_array_equal(a['realmlp'],prediction)
    for seed in [42,3407]:
        members=[arrays(DEV/f's{seed}-f{fold}-i42/predictions.npz')['test_refit'] for fold in range(5)]
        cv=np.array([.2*(math.fsum(float(p[i]) for p in members)/5-float(base[i])) for i in range(322)]);full=.2*(prediction-base);detail=r['movement'][str(seed)]
        norm=math.sqrt(math.fsum(float(x*x) for x in cv)*math.fsum(float(x*x) for x in full))
        expected_fields=dict(cv_mean=math.fsum(cv)/322,full_mean=math.fsum(full)/322,cv_rms=math.sqrt(math.fsum(float(x*x) for x in cv)/322),
            full_rms=math.sqrt(math.fsum(float(x*x) for x in full)/322),difference_rms=math.sqrt(math.fsum(float((x-y)**2) for x,y in zip(cv,full))/322),
            cosine=math.fsum(float(x*y) for x,y in zip(cv,full))/norm if norm else None,sign_agreement=sum(np.sign(x)==np.sign(y) for x,y in zip(cv,full))/322)
        for name,value in expected_fields.items():
            if value is None:
                if detail[name] is not None:raise ValueError('Undefined cosine changed')
            else:maximum=max(maximum,abs(value-detail[name]))
    if maximum>1e-10:raise ValueError('Independent package or movement scalar mismatch')
    result=dict(status='passed',rows=322,unique_ids=322,official_order=True,zip_only_result_csv=True,crc_passed=True,iron_string_mismatches=0,
        maximum_scalar_difference=maximum,release_sha256=sha(run/'release.json'),zip_sha256=sha(r['zip']),training_reads_prohibited=True,new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=memory())
    write(run/'package-audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','full','cold','build','audit']);p.add_argument('--checks');a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action in ['full','cold']:full(a.action=='cold')
    elif a.action=='build':build()
    else:audit()
