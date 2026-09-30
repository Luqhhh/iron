"""Independent held-out-vector and gate arithmetic. No production scorer.

Source/reference binding is verified separately; never fits or clusters.
"""
import argparse
from pathlib import Path
import json

import numpy as np

from .data import TARGETS
from .ptarl_unbudgeted_freeze import verify_manifest, reload_references, load_anchored
from .ptarl_protocol import phase_tasks, file_hash, write_new
from .ptarl_execution import task_name
from .ptarl_unbudgeted_run import verify_audited_artifacts,development_records

ARMS=('CONTROL','PTARL_AUX')


def direct_decision(records,phase,eligible=None):
    if phase=='development':
        if eligible is not None:raise ValueError('Development selection must start unfitted')
        targets,seeds=TARGETS,(42,3407)
    elif phase=='confirmation':
        if eligible is None or len(set(eligible))!=len(eligible) or not set(eligible)<=set(TARGETS):
            raise ValueError('Explicit valid earned targets required')
        targets,seeds=tuple(t for t in TARGETS if t in eligible),(42,3407,7777,12011)
    else:raise ValueError('Invalid arithmetic phase')
    rows={(r['target'],r['arm'],r['seed']):r for r in records}
    expected={(t,a,s) for t in TARGETS for a in ARMS for s in (42,3407)}
    if phase=='confirmation':expected|={(t,a,s) for t in targets for a in ARMS for s in (7777,12011)}
    if len(rows)!=len(records) or set(rows)!=expected:
        raise ValueError('Incomplete/duplicate/extra arithmetic records')
    if any(not np.isfinite([r['gain'],r['candidate_score']]).all() for r in rows.values()):
        raise ValueError('Invalid arithmetic metrics')
    selected,details=[],{}
    for target in targets:
        full=[rows[target,'PTARL_AUX',s] for s in seeds]
        gains=np.array([r['gain'] for r in full])
        contrast=gains-np.array([rows[target,'CONTROL',s]['gain'] for s in seeds])
        dev=[rows[target,'PTARL_AUX',s] for s in (42,3407)]
        development_pass=(min(r['gain'] for r in dev)>0 and np.mean([r['gain'] for r in dev])>=.01
            and np.mean([r['candidate_score'] for r in dev])>=96.25
            and np.mean([rows[target,'PTARL_AUX',s]['gain']-rows[target,'CONTROL',s]['gain'] for s in (42,3407)])>0)
        if phase=='development':
            passed=bool(development_pass);details[target]=dict(passed=passed)
        else:
            if not development_pass:raise ValueError('Unqualified target reached confirmation arithmetic')
            lower=float(gains.mean()-2.3533634348018264*np.sqrt(np.sum((gains-gains.mean())**2)/3)/2)
            passed=bool(min(gains)>0 and lower>0 and contrast.mean()>0)
            details[target]=dict(passed=passed,seed_lcb95=lower)
        if passed:selected.append(target)
    return selected,details


def recover_records(output,complete,frame,folds,current,historical,expected_tasks):
    anchors=complete['unit_anchors']
    if set(anchors)!={task_name(t) for t in expected_tasks}:
        raise ValueError('Incomplete arithmetic paired-unit anchors')
    members={}
    for task in expected_tasks:
        unit=output/'units'/task_name(task)
        record=load_anchored(unit/'complete.json',anchors[task_name(task)])
        if record['task']!=task:raise ValueError('Arithmetic unit/task mismatch')
        if file_hash(unit/'predictions.npz')!=record['prediction_sha256']:
            raise ValueError('Arithmetic predictions changed')
        mask=folds[task['seed']]==task['fold']
        with np.load(unit/'predictions.npz',allow_pickle=False) as archive:
            if archive['query_ids'].tolist()!=frame.loc[mask,'sample_id'].tolist():
                raise ValueError('Arithmetic query identity mismatch')
            for arm in ARMS:
                raw=members.setdefault((task['target'],arm,task['seed']),np.full(len(frame),np.nan))
                if np.isfinite(raw[mask]).any():raise ValueError('Duplicate arithmetic coverage')
                p=archive[arm]
                if p.shape!=(int(mask.sum()),) or not np.isfinite(p).all():
                    raise ValueError('Invalid arithmetic held-out values')
                raw[mask]=p
    y={t:frame[t].to_numpy(float) for t in TARGETS}
    records=[]
    for (target,arm,seed),raw in sorted(members.items()):
        if not np.isfinite(raw).all():raise ValueError('Incomplete arithmetic OOF coverage')
        errors={t:float(np.abs(y[t]-current[seed][t]).sum()/np.abs(y[t]).sum()) for t in TARGETS}
        old={t:float(np.abs(y[t]-historical[seed][t]).sum()/np.abs(y[t]).sum()) for t in TARGETS}
        endpoint=.8*current[seed][target]+.2*raw
        changed=float(np.abs(y[target]-endpoint).sum()/np.abs(y[target]).sum())
        other=next(t for t in TARGETS if t!=target)
        score=100-50*(changed+errors[other])
        records.append(dict(target=target,arm=arm,seed=seed,gain=50*(errors[target]-changed),
            candidate_score=score,historical_target_gain=50*(old[target]-changed),
            candidate_minus_historical_package=score-(100-50*sum(old.values()))))
    return records


def check_phase(manifest_path,manifest_sha256,phase,audit_sha256,development_arithmetic_sha256=None):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    if phase not in ('development','confirmation'):raise ValueError('Invalid arithmetic phase')
    root=Path(manifest_path).resolve().parent;output=root/phase
    write_new(output/'arithmetic-started.json',dict(audit_sha256=audit_sha256))
    try:
        audit=load_anchored(output/'audit.json',audit_sha256)
        if audit['status']!='passed' or audit['manifest_sha256']!=manifest_sha256 or audit['phase']!=phase:
            raise ValueError('Arithmetic requires passed matching phase audit')
        verify_audited_artifacts(output,audit)
        prior=None;eligible=None
        if phase=='confirmation':
            if audit['development_arithmetic_sha256']!=development_arithmetic_sha256:
                raise ValueError('Confirmation development arithmetic anchor differs')
            development_records(root,manifest_sha256,audit['development_sha256'],development_arithmetic_sha256)
            prior=load_anchored(root/'development/arithmetic.json',development_arithmetic_sha256)
            if prior['audit_sha256']!=audit['development_sha256'] or prior['status']!='passed':
                raise ValueError('Wrong development arithmetic parent')
            eligible,_=direct_decision(prior['records'],'development')
            if eligible!=prior['selected_targets']:raise ValueError('Development selection changed')
        elif development_arithmetic_sha256 is not None:
            raise ValueError('Development cannot inherit arithmetic')
        complete=load_anchored(output/'complete.json',audit['phase_complete_sha256'])
        frame,folds,current,historical,_=reload_references(manifest)
        recovered=recover_records(output,complete,frame,folds,current,historical,phase_tasks(phase,eligible))
        original={(r['target'],r['arm'],r['seed']):r for r in audit['records']}
        if len(original)!=len(audit['records']) or set(original)!={(r['target'],r['arm'],r['seed']) for r in recovered}:
            raise ValueError('Arithmetic/production record coverage mismatch')
        for row in recovered:
            old=original[row['target'],row['arm'],row['seed']]
            if any(abs(row[k]-old[k])>1e-12 for k in ('gain','candidate_score','historical_target_gain','candidate_minus_historical_package')):
                raise ValueError('Independent arithmetic differs from production')
        combined=recovered if phase=='development' else prior['records']+recovered
        selected,details=direct_decision(combined,phase,eligible)
        expected=audit['decision']['eligible_targets' if phase=='development' else 'promoted_targets']
        if selected!=expected:raise ValueError('Independent gate decision differs')
        if phase=='confirmation' and any(abs(details[t]['seed_lcb95']-audit['decision']['decisions'][t]['seed_lcb95'])>1e-12 for t in eligible):
            raise ValueError('Independent paired confidence bound differs')
        verify_manifest(manifest_path,manifest_sha256)
        report=dict(status='passed',phase=phase,manifest_sha256=manifest_sha256,audit_sha256=audit_sha256,records=recovered,
            selected_targets=selected,details=details,new_fits=0,release_authorized=False)
        write_new(output/'arithmetic.json',report)
    except BaseException as exc:
        write_new(output/'arithmetic-failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(arithmetic_sha256=file_hash(output/'arithmetic.json'),selected_targets=selected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--phase',choices=['development','confirmation'],required=True)
    p.add_argument('--audit-sha256',required=True);p.add_argument('--development-arithmetic-sha256')
    a=p.parse_args();print(json.dumps(check_phase(a.manifest,a.manifest_sha256,a.phase,a.audit_sha256,
        a.development_arithmetic_sha256),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
