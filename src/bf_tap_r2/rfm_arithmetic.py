"""Independent direct arithmetic from anchored held-out prediction files.

Does not import the production scorer or decision helpers; never fits models.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .rfm_freeze import verify_manifest, reload_references, load_anchored
from .rfm_protocol import file_hash, write_new, TARGETS


def direct_decision(records, phase, eligible=None):
    rows={(r['target'],r['arm'],r['seed']):r for r in records}
    if len(rows)!=len(records):raise ValueError('duplicate arithmetic records')
    result=[];details={}
    for target in TARGETS if phase=='development' else eligible:
        seeds=(42,3407) if phase=='development' else (42,3407,7777,12011)
        full=[rows[target,'FULL_RFM',s] for s in seeds]
        gains=np.array([r['gain'] for r in full])
        contrast=gains-np.array([rows[target,'FIXED_KRR',s]['gain'] for s in seeds])
        if phase=='development':
            passed=bool(min(gains)>0 and gains.mean()>=.01 and contrast.mean()>0
                        and np.mean([r['candidate_score'] for r in full])>=96.25)
            details[target]={'passed':passed}
        else:
            lower=float(gains.mean()-2.3533634348018264*np.sqrt(np.sum((gains-gains.mean())**2)/3)/2)
            passed=bool(min(gains)>0 and lower>0 and contrast.mean()>0)
            details[target]={'passed':passed,'seed_lcb95':lower}
        if passed:result.append(target)
    return result,details


def check_phase(manifest_path,manifest_sha256,phase,audit_sha256,development_arithmetic_sha256=None):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    root=Path(manifest_path).resolve().parent
    if phase not in ('development','confirmation'):raise ValueError('invalid phase')
    output=root/phase
    write_new(output/'arithmetic-started.json',{'audit_sha256':audit_sha256})
    audit=load_anchored(output/'audit.json',audit_sha256)
    if audit['status']!='passed' or audit['manifest_sha256']!=manifest_sha256:
        raise ValueError('arithmetic requires passed matching audit')
    complete=load_anchored(output/'complete.json',audit['phase_complete_sha256'])
    frame,folds,current,historical,_=reload_references(manifest)
    predictions={}
    for name,anchor in complete['unit_anchors'].items():
        unit=output/'units'/name;record=load_anchored(unit/'complete.json',anchor)
        task=record['task'];target,arm,seed=task['target'],task['arm'],task['seed']
        mask=folds[seed]==task['fold']
        if file_hash(unit/'predictions.npz')!=record['prediction_sha256']:raise ValueError('changed predictions')
        with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
            if saved['query_ids'].tolist()!=frame.loc[mask,'sample_id'].tolist():raise ValueError('wrong query rows')
            vector=predictions.setdefault((target,arm,seed),np.full(len(frame),np.nan))
            if np.isfinite(vector[mask]).any():raise ValueError('duplicate arithmetic coverage')
            vector[mask]=saved['prediction']
    recovered=[]
    for original in audit['records']:
        target,arm,seed=original['target'],original['arm'],original['seed']
        raw=predictions[target,arm,seed]
        if not np.isfinite(raw).all():raise ValueError('incomplete arithmetic coverage')
        actual={t:frame[t].to_numpy(float) for t in TARGETS}
        errors={t:float(np.abs(actual[t]-current[seed][t]).sum()/np.abs(actual[t]).sum()) for t in TARGETS}
        endpoint=.8*current[seed][target]+.2*raw
        changed=float(np.abs(actual[target]-endpoint).sum()/np.abs(actual[target]).sum())
        old={t:float(np.abs(actual[t]-historical[seed][t]).sum()/np.abs(actual[t]).sum()) for t in TARGETS}
        other=next(t for t in TARGETS if t!=target)
        score=100-50*(changed+errors[other])
        row=dict(target=target,arm=arm,seed=seed,gain=50*(errors[target]-changed),candidate_score=score,
            historical_target_gain=50*(old[target]-changed),
            candidate_minus_historical_package=score-(100-50*sum(old.values())))
        for key in ('gain','candidate_score','historical_target_gain','candidate_minus_historical_package'):
            if abs(row[key]-original[key])>1e-12:raise ValueError('independent arithmetic mismatch: '+key)
        recovered.append(row)
    if phase=='development':
        selected,details=direct_decision(recovered,phase)
        expected=audit['decision']['eligible_targets']
    else:
        load_anchored(root/'development'/'audit.json',audit['development_sha256'])
        prior=load_anchored(root/'development'/'arithmetic.json',development_arithmetic_sha256)
        if prior['audit_sha256']!=audit['development_sha256']:raise ValueError('wrong development arithmetic')
        eligible,_=direct_decision(prior['records'],'development')
        selected,details=direct_decision(prior['records']+recovered,phase,eligible)
        expected=audit['decision']['promoted_targets']
        for t in eligible:
            if abs(details[t]['seed_lcb95']-audit['decision']['decisions'][t]['seed_lcb95'])>1e-12:
                raise ValueError('independent confidence bound mismatch')
    if selected!=expected:raise ValueError('independent gate mismatch')
    verify_manifest(manifest_path,manifest_sha256)
    report=dict(status='passed',phase=phase,audit_sha256=audit_sha256,records=recovered,
                selected_targets=selected,details=details,new_fits=0,release_authorized=False)
    write_new(output/'arithmetic.json',report)
    return dict(arithmetic_sha256=file_hash(output/'arithmetic.json'),selected_targets=selected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--phase',required=True)
    p.add_argument('--audit-sha256',required=True);p.add_argument('--development-arithmetic-sha256');a=p.parse_args()
    print(json.dumps(check_phase(a.manifest,a.manifest_sha256,a.phase,a.audit_sha256,a.development_arithmetic_sha256),sort_keys=True))


if __name__=='__main__':main()
