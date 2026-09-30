"""Independent OOF recovery, isolated-score arithmetic and paired seed gates."""
from pathlib import Path
import argparse

import numpy as np

from .data import TARGETS
from .danet_phase_protocol import DEV, CONFIRM, ELIGIBLE, CONTROLS, phase_tasks, required_arms, task_name
from .danet_phase_freeze import verify_manifest, reload_references
from .danet_phase_run import earned_context
from .dnnr_preflight import anchored
from .danet_ledger import write_new, file_hash


def direct_decision(records,phase,eligible=None):
    if phase not in ('development','confirmation'):raise ValueError('Unknown independent arithmetic phase')
    dev={(t,a,s) for t in TARGETS for a in ('DANET_FIXED','DANET_LEARNED') for s in DEV}
    expected=set(dev)
    if phase=='confirmation':
        if eligible is None:raise ValueError('Explicit earned methods required')
        targets={t for t,a in eligible}
        expected|={(t,a,s) for t in targets for a in required_arms(t,phase,eligible) for s in CONFIRM}
    elif eligible is not None:raise ValueError('Development cannot inherit selection')
    index={(r['target'],r['arm'],r['seed']):r for r in records}
    if len(index)!=len(records) or set(index)!=expected:raise ValueError('Independent gate coverage differs')
    if any(not np.isfinite([r['gain'],r['candidate_score']]).all() for r in records):
        raise ValueError('Independent gate requires finite scores')
    selected=[];details={}
    pairs=[(t,a) for t in TARGETS for a in ELIGIBLE] if phase=='development' else [tuple(p) for p in eligible]
    for t,a in pairs:
        d=[index[t,a,s] for s in DEV]
        dev_good=(min(r['gain'] for r in d)>0 and sum(r['gain'] for r in d)/2>=.01
            and sum(r['candidate_score'] for r in d)/2>=96.25
            and sum(index[t,a,s]['gain']-index[t,CONTROLS[a],s]['gain'] for s in DEV)/2>0)
        if phase=='development':
            passed=bool(dev_good);lower=None
        else:
            if not dev_good:raise ValueError('Unqualified method reached derived-seed arithmetic')
            values=[index[t,a,s]['gain'] for s in (*DEV,*CONFIRM)]
            mean=sum(values)/4;sd=(sum((v-mean)**2 for v in values)/3)**.5
            lower=mean-2.3533634348018264*sd/2
            contrast=sum(index[t,a,s]['gain']-index[t,CONTROLS[a],s]['gain'] for s in (*DEV,*CONFIRM))/4
            passed=bool(min(values)>0 and lower>0 and contrast>0)
        details.setdefault(t,{})[a]=dict(passed=passed,seed_lcb95=lower)
        if passed:selected.append([t,a])
    return selected,details


def recover_records(output,complete,frame,folds,current,parent,phase,eligible):
    tasks=phase_tasks(phase,eligible)
    if set(complete['unit_anchors'])!={task_name(t) for t in tasks}:raise ValueError('Independent unit coverage differs')
    vectors={}
    for task in tasks:
        unit=output/'units'/task_name(task)
        record=anchored(unit/'complete.json',complete['unit_anchors'][task_name(task)])
        mask=folds[task['seed']]==task['fold']
        if record['task']!=task or record['query_ids']!=frame.loc[mask,'sample_id'].astype(str).tolist():
            raise ValueError('Independent held-out task/ID identity differs')
        for arm in required_arms(task['target'],phase,eligible):
            name=f'prediction-{arm}.npy';path=unit/name
            if file_hash(path)!=record['prediction_hashes'][name]:raise ValueError('Independent prediction identity changed')
            p=np.load(path,allow_pickle=False)
            vector=vectors.setdefault((task['target'],arm,task['seed']),np.full(len(frame),np.nan))
            if p.shape!=(int(mask.sum()),) or not np.isfinite(p).all() or np.isfinite(vector[mask]).any():
                raise ValueError('Independent OOF duplicate/missing coverage')
            vector[mask]=p
    y={t:frame[t].to_numpy(float) for t in TARGETS};records=[]
    for (target,arm,seed),raw in sorted(vectors.items()):
        if not np.isfinite(raw).all():raise ValueError('Independent OOF vector incomplete')
        error={t:float(sum(abs(y[t]-current[seed][t]))/sum(abs(y[t]))) for t in TARGETS}
        old={t:float(sum(abs(y[t]-parent[seed][t]))/sum(abs(y[t]))) for t in TARGETS}
        endpoint=.8*current[seed][target]+.2*raw
        changed=float(sum(abs(y[target]-endpoint))/sum(abs(y[target])))
        other=next(t for t in TARGETS if t!=target)
        score=100-50*(changed+error[other])
        records.append(dict(target=target,arm=arm,seed=seed,gain=50*(error[target]-changed),
            candidate_score=score,parent_target_gain=50*(old[target]-changed),
            candidate_minus_parent_package=score-(100-50*sum(old.values()))))
    return records


def check_phase(manifest_path,manifest_sha256,phase,audit_sha256,development_audit_sha256=None,development_arithmetic_sha256=None):
    m=verify_manifest(manifest_path,manifest_sha256);root=Path(manifest_path).resolve().parent;output=root/phase
    eligible,prior=earned_context(root,phase,development_audit_sha256,development_arithmetic_sha256)
    audit=anchored(output/'audit.json',audit_sha256)
    if audit['status']!='passed' or audit['manifest_sha256']!=manifest_sha256 or audit['phase']!=phase:
        raise ValueError('Independent arithmetic requires passed matching cold audit')
    write_new(output/'arithmetic-started.json',dict(audit_sha256=audit_sha256,new_fits=0))
    try:
        for name,sha in audit['artifact_hashes'].items():
            if file_hash(output/name)!=sha:raise ValueError('Cold audited artifact changed before arithmetic')
        complete=anchored(output/'complete.json',audit['phase_complete_sha256'])
        frame,folds,current,parent,_=reload_references(m)
        recovered=recover_records(output,complete,frame,folds,current,parent,phase,eligible)
        original={(r['target'],r['arm'],r['seed']):r for r in audit['records']}
        if len(original)!=len(audit['records']) or set(original)!={(r['target'],r['arm'],r['seed']) for r in recovered}:
            raise ValueError('Independent/production record coverage differs')
        maximum=0.
        for row in recovered:
            other=original[row['target'],row['arm'],row['seed']]
            for key in ('gain','candidate_score','parent_target_gain','candidate_minus_parent_package'):
                maximum=max(maximum,abs(row[key]-other[key]))
        if maximum>1e-12:raise ValueError('Independent score arithmetic differs')
        combined=recovered if phase=='development' else prior['records']+recovered
        selected,details=direct_decision(combined,phase,eligible)
        expected=audit['decision']['eligible_pairs' if phase=='development' else 'promoted_pairs']
        if selected!=expected:raise ValueError('Independent gate decision differs')
        if phase=='confirmation':
            for t,a in eligible:
                if abs(details[t][a]['seed_lcb95']-audit['decision']['decisions'][t][a]['seed_lcb95'])>1e-12:
                    raise ValueError('Independent paired confidence bound differs')
        verify_manifest(manifest_path,manifest_sha256)
        report=dict(status='passed',phase=phase,manifest_sha256=manifest_sha256,audit_sha256=audit_sha256,
            records=recovered,selected_pairs=selected,details=details,maximum_score_difference=maximum,
            new_fits=0,release_authorized=False)
        write_new(output/'arithmetic.json',report)
    except BaseException as exc:
        write_new(output/'arithmetic-failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(arithmetic_sha256=file_hash(output/'arithmetic.json'),selected_pairs=selected)


def main():
    import json
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--phase',choices=['development','confirmation'],required=True);p.add_argument('--audit-sha256',required=True)
    p.add_argument('--development-audit-sha256');p.add_argument('--development-arithmetic-sha256');a=p.parse_args()
    print(json.dumps(check_phase(a.manifest,a.manifest_sha256,a.phase,a.audit_sha256,a.development_audit_sha256,
        a.development_arithmetic_sha256),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
