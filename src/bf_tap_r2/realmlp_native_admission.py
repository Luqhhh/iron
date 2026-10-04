"""Synthetic full-shape original/capture parity before RealMLP scientific fits."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from . import realmlp_state_adapter as adapter
from .data import FEATURES
from .ema_nested_residual import read,write,sha,verify,save_arrays
from .realmlp_native_audit import observe_native,verify_trace,native_state_digest,cold_audit

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_native_admission/SPEC.json'


def setup():
    import psutil
    import torch
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError('Set numerical threads before import')
    if sys.version_info[:2]!=(3,12):raise ValueError('Locked Python3.12 required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    spec=read(WORK/SPEC)
    original=yaml.safe_load((WORK/spec['recipe_source']).read_text())
    versions={p:importlib.metadata.version(p) for p in original['runtime_versions']}
    if versions!=original['runtime_versions']:raise ValueError('Native runtime changed')
    if sha(adapter.original.__file__)!=spec['original_source_sha256'] or sha(adapter.__file__)!=spec['adapter_source_sha256']:
        raise ValueError('Original fit or capture adapter changed')
    if psutil.virtual_memory().available/2**20<spec['minimum_available_mib']:
        raise ValueError('Insufficient entry memory')
    return spec,original['recipes'][spec['recipe']],versions


def synthetic(spec):
    rng=np.random.default_rng(spec['synthetic_seed'])
    frames=[]
    for stage,n in [('training',spec['synthetic_training_rows']),('query',spec['synthetic_query_rows'])]:
        a=rng.normal(size=(n,len(FEATURES)))
        x=pd.DataFrame(a,columns=FEATURES)
        x['spout_no']=np.arange(n)%2+1
        x['sample_id']=[f'synthetic-{stage}-{i:05d}' for i in range(n)]
        frames.append(x)
    x=frames[0]
    y=100+5*np.sin(x.air_volume.to_numpy())+3*x.hot_air_temp.to_numpy()+2*x.oxygen.to_numpy()**2+.3*rng.normal(size=len(x))
    return *frames,y


def source_hashes():
    paths=list((WORK/'src').rglob('*.py'))+[WORK/p for p in
        [SPEC,'configs/round2_v9/SPEC.yaml','docs/realmlp_native_admission/PREREGISTRATION.md',
         'tests/test_realmlp_native_audit.py','tests/test_realmlp_state_adapter.py','uv.lock','pyproject.toml']]
    return {str(p.relative_to(WORK)):sha(p) for p in paths}


def prepare(checks):
    import pytabkit
    spec,recipe,versions=setup();run=Path(spec['run_directory'])
    if run.exists():raise FileExistsError('Private engineering run already exists')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():
        raise ValueError('Committed isolated source snapshot required')
    c=read(checks);sources=source_hashes()
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['source_hashes']!=sources or c['junit_sha256']!=sha(c['junit']):
        raise ValueError('Exact-source no-fit checks required')
    files={str(WORK/p):h for p,h in sources.items()}
    files.update({str(p):sha(p) for p in Path(pytabkit.__file__).parent.rglob('*.py')})
    files[str(Path(checks))]=sha(checks);files[c['junit']]=sha(c['junit'])
    run.mkdir(parents=True)
    training,query,y=synthetic(spec)
    training.to_pickle(run/'training.pkl');query.to_pickle(run/'query.pkl')
    save_arrays(run/'targets.npz',y=y)
    files.update({str(run/n):sha(run/n) for n in ('training.pkl','query.pkl','targets.npz')})
    write(run/'manifest.json',dict(spec=spec,recipe=recipe,versions=versions,files=files,sources=sources,
        source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()))
    print(json.dumps(dict(status='frozen',files=len(files),synthetic_outer_procedures=2,optimizer_budget=4)),flush=True)


def context():
    spec,recipe,versions=setup();run=Path(spec['run_directory']);m=read(run/'manifest.json')
    verify(m['files'])
    if m['spec']!=spec or m['recipe']!=recipe or m['source_directory']!=str(WORK):raise ValueError('Frozen native context differs')
    training=pd.read_pickle(run/'training.pkl');query=pd.read_pickle(run/'query.pkl')
    with np.load(run/'targets.npz',allow_pickle=False) as a:y=a['y'].copy()
    return run,m,training,query,y


def worker(mode):
    run,m,training,query,y=context();d=run/mode;d.mkdir(exist_ok=False)
    write(d/'estimator-start.json',dict(mode=mode,pid=os.getpid(),synthetic=True))
    with observe_native(d) as trace:
        if mode=='control':
            model=adapter.original.RealMLPRegressor(m['recipe'],inner_seed=42).fit(training,y)
        else:
            captured=adapter.capture_fit(m['recipe'],training,y,inner_seed=42);model=captured.regressor
    epoch=model.metadata_['selected_epoch'];verify_trace(trace,training,epoch,256)
    identity=dict(source_directory=str(d),split_seed=-1,fold=-1,trial_id='synthetic-full-shape-TD')
    predictions=dict(refit=model.predict(query))
    states=dict(refit=native_state_digest(model.model_))
    if mode=='capture':
        predictions['selection']=captured.estimators[0].predict(captured.encoders[0].transform(query))
        states['selection']=native_state_digest(captured.estimators[0])
        for role in ('selection','refit'):
            adapter.save_snapshot(captured,role,d/(role+'.pkl'),identity)
    save_arrays(d/'predictions.npz',ids=query.sample_id.to_numpy(str),**predictions)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>m['spec']['max_rss_mib']:raise ValueError('Native peak memory gate failed')
    verify(m['files'])
    write(d/'complete.json',dict(status='passed',identity=identity,native_trace=trace,metadata=model.metadata_,
        selected_epoch=epoch,parameter_states=states,predictions_sha256=sha(d/'predictions.npz'),
        native_optimizer_runs=len(trace['optimizers']),native_updates=sum(v['steps'] for v in trace['optimizers']),
        peak_rss_mib=peak,pid=os.getpid(),manifest_sha256=sha(run/'manifest.json')))
    print(json.dumps(dict(status='passed',mode=mode,selected_epoch=epoch,peak_rss_mib=peak)),flush=True)


def cold():
    run,m,training,query,y=context();d=run/'capture';c=read(d/'complete.json')
    if sha(d/'predictions.npz')!=c['predictions_sha256']:raise ValueError('Warm predictions changed')
    with np.load(d/'predictions.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str))
        result=cold_audit(d,training,query,y,m['recipe'],c['identity'],c,
            {k:a[k].copy() for k in ('selection','refit')})
    result.update(pid=os.getpid(),complete_sha256=sha(d/'complete.json'))
    verify(m['files']);write(run/'cold.json',result)
    print(json.dumps(result),flush=True)


def parity():
    run,m,training,query,y=context();control=read(run/'control/complete.json');capture=read(run/'capture/complete.json')
    for key in ('selected_epoch','metadata','native_trace','native_optimizer_runs','native_updates'):
        if control[key]!=capture[key]:raise ValueError('Original/capture differs: '+key)
    if control['parameter_states']['refit']!=capture['parameter_states']['refit']:
        raise ValueError('Original/capture final native parameters differ')
    with np.load(run/'control/predictions.npz',allow_pickle=False) as a,np.load(run/'capture/predictions.npz',allow_pickle=False) as b:
        np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['refit'],b['refit'])
    cold=read(run/'cold.json')
    if cold['status']!='passed' or cold['complete_sha256']!=sha(run/'capture/complete.json'):
        raise ValueError('Native cold admission missing')
    verify(m['files'])
    result=dict(status='passed',G0='original_native_fit_capture_parity_and_two_state_cold_passed',G1='no_scientific_measurement',
        original_capture_prediction_difference=0,optimizer_runs=control['native_optimizer_runs']+capture['native_optimizer_runs'],
        native_updates=control['native_updates']+capture['native_updates'],
        selected_epoch=control['selected_epoch'],cold=cold,
        peak_rss_mib=max(control['peak_rss_mib'],capture['peak_rss_mib']),
        receipts={p:sha(run/p) for p in ('control/complete.json','capture/complete.json','cold.json')},
        manifest_sha256=sha(run/'manifest.json'),official_data_reads=0,scientific_fits=0,packages=0)
    write(run/'report.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','control','capture','cold','parity']);p.add_argument('--checks')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action in ('control','capture'):worker(a.action)
    elif a.action=='cold':cold()
    else:parity()


if __name__=='__main__':main()
