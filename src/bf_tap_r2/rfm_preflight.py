"""Single-use, frozen-size synthetic RFM resource admission. No official fits."""
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
from .rfm_audit import audit_path
from .rfm_freeze import load_anchored, private_path, verify_manifest
from .rfm_model import RFMRegressor, SavedRFM
from .rfm_protocol import ARMS, ReservationLedger, canonical, file_hash, write_new

LIMITS = dict(procedure=4,solve=10,update=6)


def synthetic_data():
    rng=np.random.default_rng(56001)
    x=rng.normal(size=(2755,len(FEATURES)))
    y=10+2*np.sin(x[:,0])+x[:,1]*x[:,2]+.5*x[:,3]+.1*rng.normal(size=2755)
    frame=pd.DataFrame(x,columns=FEATURES)
    frame['spout_no']=np.arange(2755)%2+1
    frame['sample_id']=[f'rfm-resource-{i:04}' for i in range(2755)]
    return frame.iloc[:2204].copy(),y[:2204],frame.iloc[2204:].copy(),y[2204:]


def resource_decision(arms, available_mib):
    if set(arms)!=set(ARMS):
        raise ValueError('both complete synthetic arms required')
    values=[float(available_mib)]
    for arm, report in arms.items():
        if (report['arm']!=arm or report['procedures']!=2
                or report['solves']!=(8 if arm=='FULL_RFM' else 2)
                or report['updates']!=(6 if arm=='FULL_RFM' else 0)
                or report['refit_updates']!=(3 if arm=='FULL_RFM' else 0)
                or report['train_rows']!=2204 or report['query_rows']!=551):
            raise ValueError('synthetic shape/path/budget differs from frozen design')
        values.extend(report[k] for k in ('seconds','peak_mib','mae','median_mae','cold_difference'))
    if not np.isfinite(values).all() or min(values)<0:
        raise ValueError('invalid resource measurement')
    if any(r['seconds']<=0 or r['peak_mib']<=0 for r in arms.values()):
        raise ValueError('missing resource measurement')
    peak=max(r['peak_mib'] for r in arms.values())
    projection=20*sum(r['seconds'] for r in arms.values())/4*1.5+300
    checks=dict(synthetic_quality=all(r['mae']<r['median_mae'] for r in arms.values()),
        cold_inference=all(r['cold_difference']<=1e-8 for r in arms.values()),
        worker_peak=peak<=1536,available_memory=available_mib>=4*peak+1024,
        development_cost=projection<=7200)
    return dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
                projected_development_seconds=projection,maximum_worker_mib=peak,
                available_mib=available_mib,required_available_mib=4*peak+1024)


def measure_arm(manifest_path, manifest_sha256, arm, policy_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    from .rfm_run import initialize_worker
    initialize_worker()
    root=private_path(manifest['workspace'],Path(manifest_path).resolve().parent/'preflight')
    directory=root/arm;directory.mkdir(exist_ok=False)
    ledger=ReservationLedger.open(root/'ledger',policy_sha256)
    train,y,query,truth=synthetic_data()
    started=time.perf_counter()
    reports={};maximum=0.
    for stage in ('inner','refit'):
        regressor=RFMRegressor(arm)
        # Both paths deliberately use the full training bound; refit always
        # executes all three FULL updates irrespective of synthetic selection.
        regressor.fit_path(train,y,3 if arm=='FULL_RFM' else 0,
                           ledger.scoped((arm,stage)),directory=directory/stage)
        models,difference=audit_path(directory/stage,train,y,arm,regressor.last_artifacts_)
        errors=[float(np.mean(np.abs(m.predict(query)-truth))) for m in models]
        maximum=max(maximum,difference)
        final=models[-1];prediction=final.predict(query)
        reversed_prediction=final.predict(query.iloc[::-1])[::-1]
        chunks=np.concatenate([final.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])
        maximum=max(maximum,float(np.max(abs(prediction-reversed_prediction))),
                    float(np.max(abs(prediction-chunks))))
        reports[stage]=dict(artifacts=regressor.last_artifacts_,mae=errors,
                            selected_state=int(np.argmin(errors)),forced_final_state=len(models)-1)
    report=dict(arm=arm,train_rows=len(train),query_rows=len(query),paths=reports,
        refit_updates=3 if arm=='FULL_RFM' else 0,procedures=2,
        solves=8 if arm=='FULL_RFM' else 2,updates=6 if arm=='FULL_RFM' else 0,
        mae=reports['refit']['mae'][-1],median_mae=float(np.mean(abs(truth-np.median(y)))),
        cold_difference=maximum,seconds=time.perf_counter()-started,
        peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        manifest_sha256=manifest_sha256,policy_sha256=policy_sha256)
    verify_manifest(manifest_path,manifest_sha256)
    write_new(directory/'complete.json',report)
    return report


def _verify_costs(root, policy_sha256):
    ledger=ReservationLedger.open(root/'ledger',policy_sha256)
    if ledger.limits!=LIMITS:
        raise ValueError('wrong synthetic budget')
    counts=ledger.inspect()
    if (counts['started']!=LIMITS or counts['completed']!=LIMITS
            or any(counts[s][k] for s in ('failed','incomplete') for k in LIMITS)):
        raise ValueError('incomplete or failed synthetic procedures')
    expected=set()
    for arm in ARMS:
        states=4 if arm=='FULL_RFM' else 1
        for stage in ('inner','refit'):
            expected.add(('procedure',canonical([arm,stage,'procedure',None])))
            expected.update(('solve',canonical([arm,stage,'solve',s])) for s in range(states))
            expected.update(('update',canonical([arm,stage,'update',s])) for s in range(1,states))
    actual=set()
    for path in (ledger.root/'events').glob('*.started.json'):
        r=json.loads(path.read_text());actual.add((r['kind'],canonical(r['key'])))
    if actual!=expected:
        raise ValueError('synthetic reservation keys differ from frozen paths')
    return counts


def run_synthetic_admission(manifest_path, manifest_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    root=private_path(manifest['workspace'],Path(manifest_path).resolve().parent/'preflight')
    root.mkdir(exist_ok=False)
    write_new(root/'started.json',dict(manifest_sha256=manifest_sha256,limits=LIMITS))
    try:
        ledger=ReservationLedger.create(root/'ledger',LIMITS)
        arms={}
        for arm in ARMS:
            args=[sys.executable,'-m','bf_tap_r2.rfm_preflight','worker','--manifest',str(manifest_path),
                  '--manifest-sha256',manifest_sha256,'--arm',arm,'--policy-sha256',ledger.policy_sha256]
            with (root/f'{arm}.log').open('x') as log:
                subprocess.run(args,check=True,cwd=manifest['workspace'],stdout=log,stderr=subprocess.STDOUT)
            arms[arm]=json.loads((root/arm/'complete.json').read_text())
        counts=_verify_costs(root,ledger.policy_sha256)
        # The two arm definitions must begin at the identical fixed model.
        for stage in ('inner','refit'):
            models=[SavedRFM.load(root/a/stage/'state-0',
                expected_sha256=arms[a]['paths'][stage]['artifacts'][0]['complete_sha256']) for a in ARMS]
            if not np.array_equal(models[0].alpha,models[1].alpha):
                raise ValueError('synthetic FULL state zero differs from fixed control')
        available=psutil.virtual_memory().available/1024**2
        decision=resource_decision(arms,available)
        hashes={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}
        report=dict(**decision,arms=arms,counts=counts,artifact_hashes=hashes,
                    manifest_sha256=manifest_sha256,policy_sha256=ledger.policy_sha256,
                    official_fits=0,release_authorized=False)
        verify_manifest(manifest_path,manifest_sha256)
        write_new(root/'admission.json',report)
        if decision['status']!='passed':
            raise ValueError('frozen synthetic resource admission failed')
    except BaseException as exc:
        write_new(root/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(admission_sha256=file_hash(root/'admission.json'),**decision)


def verify_admission(path, expected_sha256, manifest_path, manifest_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    path=private_path(manifest['workspace'],path);root=path.parent
    if root!=Path(manifest_path).resolve().parent/'preflight' or (root/'failed.json').exists():
        raise ValueError('invalid or failed admission directory')
    report=load_anchored(path,expected_sha256)
    if report['manifest_sha256']!=manifest_sha256 or report['status']!='passed':
        raise ValueError('resource admission belongs to another or failed manifest')
    present={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if present!=set(report['artifact_hashes'])|{'admission.json'}:
        raise ValueError('unexpected or missing synthetic admission artifact')
    for relative,sha in report['artifact_hashes'].items():
        artifact=root/relative
        if not artifact.resolve().is_relative_to(root) or file_hash(artifact)!=sha:
            raise ValueError('synthetic admission artifact changed')
    if _verify_costs(root,report['policy_sha256'])!=report['counts']:
        raise ValueError('synthetic counts changed')
    recomputed=resource_decision(report['arms'],report['available_mib'])
    if any(report[k]!=v for k,v in recomputed.items()):
        raise ValueError('resource gate arithmetic mismatch')
    if psutil.virtual_memory().available/1024**2 < report['required_available_mib']:
        raise ValueError('available memory no longer meets admitted worker bound')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','worker'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--policy-sha256')
    a=p.parse_args()
    if a.action=='run':result=run_synthetic_admission(a.manifest,a.manifest_sha256)
    else:result=measure_arm(a.manifest,a.manifest_sha256,a.arm,a.policy_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
