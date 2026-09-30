"""One-shot full-size synthetic PTaRL resource admission; zero official fits.

All six roles are exercised at frozen official architecture and row bounds.
Probe cap32 is scaled to the formal240 ceiling, without changing that ceiling.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from .data import FEATURES
from .ptarl_execution import execute_unit, audit_unit
from .ptarl_freeze import verify_manifest, private_path, load_anchored
from .ptarl_model import ARMS, clean
from .ptarl_protocol import ReservationLedger, file_hash, write_new

LIMITS=dict(pair_unit=1,optimizer=6,kmeans=2)
TASK=dict(target='tap_time_len',seed=42,fold=0)
ROLES=('teacher_inner','teacher_outer','CONTROL_selector','CONTROL_refit','PTARL_AUX_selector','PTARL_AUX_refit')
SYNTHETIC_SPEC=dict(seed=57001,rows=2755,train_rows=2204,query_rows=551,max_epochs=32,
    task=TASK,limits=LIMITS,formula='10+2*sin(x0)+x1*x2+0.5*x3+0.1*epsilon',
    each_arm_must_beat_training_median=True,cold_tolerance=1e-8)
RESOURCE_SPEC=dict(workers=4,numeric_threads=1,maximum_worker_mib=1536,
    free_memory_rule='4*maximum_measured_peak+1024',
    projection_seconds='20*(sum_role_training_seconds_per_step*9*240+kmeans_seconds)/4*1.5+300',
    projection_maximum_seconds=7200)


def synthetic_data():
    rng=np.random.default_rng(57001)
    x=rng.normal(size=(2755,len(FEATURES)))
    y=10+2*np.sin(x[:,0])+x[:,1]*x[:,2]+.5*x[:,3]+.1*rng.normal(size=2755)
    f=pd.DataFrame(x,columns=FEATURES)
    f['spout_no']=np.arange(len(x))%2+1
    f['sample_id']=[f'ptarl-resource-{i:04}' for i in range(len(x))]
    f['tap_time_len']=y;f['tap_iron']=y+20
    return f.iloc[:2204].reset_index(drop=True),clean(f.iloc[2204:]).reset_index(drop=True),y[2204:]


def synthetic_settings(manifest):
    settings=deepcopy(manifest['settings'])
    if settings['max_epochs']!=240 or settings['batch_size']!=256:
        raise ValueError('Formal epoch/batch ceiling mismatch')
    settings['max_epochs']=32
    return settings


def resource_decision(measured,available_mib):
    if (measured['train_rows']!=2204 or measured['query_rows']!=551 or measured['probe_max_epochs']!=32
            or measured['models_checked']!=6 or set(measured['costs'])!=set(ROLES)
            or set(measured['mae'])!=set(ARMS)):
        raise ValueError('Frozen probe shape/role coverage mismatch')
    values=[available_mib,measured['peak_mib'],measured['cold_difference'],measured['median_mae'],measured['kmeans_seconds']]
    rates=[]
    for role in ROLES:
        c=measured['costs'][role]
        if (type(c['training_steps']) is not int or not 1<=c['training_steps']<=32*9
                or not np.isfinite(c['training_seconds']) or c['training_seconds']<=0):
            raise ValueError('Invalid per-role training cost')
        rates.append(c['training_seconds']/c['training_steps'])
        values.append(c['training_seconds'])
    values.extend(measured['mae'].values())
    if not np.isfinite(values).all() or min(values)<0 or measured['peak_mib']<=0 or measured['kmeans_seconds']<=0:
        raise ValueError('Invalid resource/quality measurements')
    # Each inner role is conservatively projected as nine full batches per epoch.
    # Role-specific initialization/serialization and cold audit get the300s buffer.
    projection=20*(sum(rates)*9*240+measured['kmeans_seconds'])/4*1.5+300
    required=4*measured['peak_mib']+1024
    checks=dict(synthetic_quality=all(v<measured['median_mae'] for v in measured['mae'].values()),
        cold_inference=measured['cold_difference']<=1e-8,worker_peak=measured['peak_mib']<=1536,
        available_memory=available_mib>=required,development_cost=projection<=7200)
    return dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
        projected_development_seconds=projection,maximum_worker_mib=measured['peak_mib'],
        available_mib=available_mib,required_available_mib=required)


def _costs(root,policy_sha256):
    ledger=ReservationLedger.open(root/'ledger',policy_sha256)
    counts=ledger.inspect()
    if (ledger.limits!=LIMITS or counts['started']!=LIMITS or counts['completed']!=LIMITS
            or any(counts[s][k] for s in ('failed','incomplete') for k in LIMITS)):
        raise ValueError('Failed/incomplete/extra resource reservations')
    return counts


def measure(manifest_path,manifest_sha256,policy_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    from .rfm_run import initialize_worker
    initialize_worker()
    root=private_path(manifest['workspace'],Path(manifest_path).resolve().parent/'preflight')
    training,query,truth=synthetic_data();settings=synthetic_settings(manifest)
    start=time.perf_counter()
    anchor=execute_unit(TASK,training,query,settings,root/'unit',root/'ledger',policy_sha256)
    prediction,audit=audit_unit(root/'unit',TASK,training,query,settings,
        expected_sha256=anchor['complete_sha256'],ledger_root=root/'ledger')
    complete=load_anchored(root/'unit/complete.json',anchor['complete_sha256'])
    km_seconds=0.
    for path in (root/'ledger/events').glob('kmeans-*.started.json'):
        a=json.loads(path.read_text())
        b=json.loads(path.with_name(path.name.replace('started.json','complete.json')).read_text())
        km_seconds+=(b['ended_ns']-a['started_ns'])/1e9
    report=dict(train_rows=len(training),query_rows=len(query),probe_max_epochs=32,
        models_checked=audit['models_checked'],cold_difference=audit['maximum_difference'],
        mae={a:float(np.abs(prediction[a]-truth).mean()) for a in ARMS},
        median_mae=float(np.abs(truth-np.median(training.tap_time_len)).mean()),
        peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        elapsed_seconds=time.perf_counter()-start,kmeans_seconds=km_seconds,costs=complete['costs'],
        unit_complete_sha256=anchor['complete_sha256'],policy_sha256=policy_sha256,manifest_sha256=manifest_sha256)
    verify_manifest(manifest_path,manifest_sha256)
    write_new(root/'measurement.json',report)
    return report


def run_synthetic_admission(manifest_path,manifest_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    root=private_path(manifest['workspace'],Path(manifest_path).resolve().parent/'preflight')
    root.mkdir(exist_ok=False)
    write_new(root/'started.json',dict(manifest_sha256=manifest_sha256,limits=LIMITS))
    try:
        ledger=ReservationLedger.create(root/'ledger',LIMITS)
        command=[sys.executable,'-m','bf_tap_r2.ptarl_preflight','worker','--manifest',str(manifest_path),
            '--manifest-sha256',manifest_sha256,'--policy-sha256',ledger.policy_sha256]
        with (root/'worker.log').open('x') as log:
            subprocess.run(command,check=True,cwd=manifest['workspace'],stdout=log,stderr=subprocess.STDOUT)
        measured=json.loads((root/'measurement.json').read_text())
        if measured['manifest_sha256']!=manifest_sha256 or measured['policy_sha256']!=ledger.policy_sha256:
            raise ValueError('Wrong resource measurement identity')
        counts=_costs(root,ledger.policy_sha256)
        decision=resource_decision(measured,psutil.virtual_memory().available/1024**2)
        hashes={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}
        report=dict(**decision,measurement=measured,counts=counts,artifact_hashes=hashes,
            manifest_sha256=manifest_sha256,policy_sha256=ledger.policy_sha256,official_fits=0,release_authorized=False)
        verify_manifest(manifest_path,manifest_sha256)
        write_new(root/'admission.json',report)
        if decision['status']!='passed':raise ValueError('Frozen PTaRL synthetic resource admission failed')
    except BaseException as exc:
        write_new(root/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(admission_sha256=file_hash(root/'admission.json'),**decision)


def verify_admission(path,expected_sha256,manifest_path,manifest_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    path=private_path(manifest['workspace'],path);root=path.parent
    if root!=Path(manifest_path).resolve().parent/'preflight' or (root/'failed.json').exists():
        raise ValueError('Invalid/failed PTaRL admission directory')
    report=load_anchored(path,expected_sha256)
    if report['status']!='passed' or report['manifest_sha256']!=manifest_sha256:
        raise ValueError('Resource admission belongs to a failed/different manifest')
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=set(report['artifact_hashes'])|{'admission.json'}:
        raise ValueError('Unexpected/missing resource artifact')
    for name,sha in report['artifact_hashes'].items():
        p=root/name
        if not p.resolve().is_relative_to(root) or file_hash(p)!=sha:
            raise ValueError('Resource artifact changed')
    if _costs(root,report['policy_sha256'])!=report['counts']:
        raise ValueError('Resource ledger changed')
    measured=report['measurement']
    if load_anchored(root/'measurement.json',report['artifact_hashes']['measurement.json'])!=measured:
        raise ValueError('Resource measurement changed')
    train,query,_=synthetic_data()
    audit_unit(root/'unit',TASK,train,query,synthetic_settings(manifest),
        expected_sha256=measured['unit_complete_sha256'],ledger_root=root/'ledger')
    decision=resource_decision(measured,report['available_mib'])
    if any(report[k]!=v for k,v in decision.items()):raise ValueError('Resource gate arithmetic changed')
    if psutil.virtual_memory().available/1024**2<report['required_available_mib']:
        raise ValueError('Available memory no longer meets admitted worker bound')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','worker'])
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--policy-sha256');a=p.parse_args()
    result=run_synthetic_admission(a.manifest,a.manifest_sha256) if a.action=='run' else measure(a.manifest,a.manifest_sha256,a.policy_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
