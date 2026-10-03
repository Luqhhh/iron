"""Frozen, zero-fit comparison of three EMA predictions' median versus mean."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'local/runs/ema-retraining-initialization-20261002/development-r1'
RUN=ROOT/'local/runs/ema-median3-20261003/development-r1'
PROTOCOL=ROOT/'docs/ema_median3/PREREGISTRATION.md'
SEEDS=(42,3407)
INITS=(42,1042,2042)
CANDIDATE='EMA_MEDIAN3'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def write(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Frozen artifact changed: '+p)

def columns(q75,members):
    import numpy as np
    q75=np.asarray(q75,float);members=np.asarray(members,float)
    if q75.ndim!=1 or members.shape!=(3,len(q75)) or not np.isfinite(members).all():
        raise ValueError('Exactly three finite aligned member vectors required')
    mean=q75+.75*(members.mean(axis=0)-members[0])
    median=q75+.75*(np.median(members,axis=0)-members[0])
    if any(not np.isfinite(v).all() or (v<0).any() for v in (mean,median)):
        raise ValueError('Invalid final prediction; no clipping')
    return mean,median

def score(y,iron,time_column):
    return 100-50*sum(math.fsum(abs(float(a)-float(b)) for a,b in zip(y[:,j],p))/
        math.fsum(abs(float(a)) for a in y[:,j]) for j,p in enumerate((iron,time_column)))

def validate_unit(warm,refit,seed,fold,init,training_ids,query_ids):
    if (warm['seed'],warm['fold'],warm['training_seed'],warm['arm'])!=(seed,fold,init,'EMA'):
        raise ValueError('Model split/fold/training seed identity differs')
    if (refit['training_ids']!=training_ids or refit['query_ids']!=query_ids
            or set(training_ids)&set(query_ids)
            or warm['partitions']['training']!=digest(training_ids)
            or warm['partitions']['query']!=digest(query_ids)):
        raise ValueError('Training/query partition identity differs')
    expected=dict(source_directory=str(SOURCE),split_seed=seed,fold=fold,
        trial_id='EMA_42' if init==42 else f'EMA_INIT{init}',fit_call_id='refit')
    if refit['identity']!=expected:raise ValueError('Witness source/trial identity differs')

def runtime():
    if sys.version_info[:2]!=(3,12) or any(os.environ.get(k)!='1' for k in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('Locked Python3.12 and pre-import thread pinning required')

def prepare(checks):
    runtime();checks=Path(checks);check=read(checks)
    if check['status']!='passed' or check['script_sha256']!=sha(__file__) or check['junit_sha256']!=sha(check['junit']):
        raise ValueError('Matching locked checks required')
    best=read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'],best['score'],best['zip_sha256'])!=('EMA_MEAN3_FULL_Q75',96.3954,
            '016e7e9cb3f750c74509dbb51961c61fea205d0296bb233056da66005f69a396'):
        raise ValueError('Latest incumbent changed before freeze')
    manifest=read(SOURCE/'manifest.json');report=read(SOURCE/'report.json');terminal=read(SOURCE/'terminal-verification.json')
    if terminal['status']!='passed' or terminal['actual_exit_codes']!=[0,0]:raise ValueError('Original actual terminal required')
    for key,name in [('manifest','manifest'),('report','report'),('independent_score','independent-score')]:
        if sha(SOURCE/(name+'.json'))!=terminal[key+'_sha256']:raise ValueError('Original terminal binding differs')
    paths=[Path(__file__),PROTOCOL,checks,Path(check['junit']),ROOT/'tests/test_ema_median3_review.py',
        ROOT/'uv.lock',ROOT/'pyproject.toml',ROOT/'configs/candidate_tiers.yaml',
        ROOT/'src/bf_tap_r2/candidate_tiers.py',ROOT/'src/bf_tap_r2/weak_models.py',
        ROOT/'src/bf_tap_r2/component_regularization_run.py',
        ROOT/best['package'],ROOT/best['platform_feedback_record'],
        SOURCE/'manifest.json',SOURCE/'report.json',SOURCE/'independent-score.json',SOURCE/'terminal-verification.json']
    files={str(p.resolve()):sha(p) for p in paths};files.update(manifest['model_sources'])
    states=set();units={}
    for seed in SEEDS:
        path=SOURCE/f'oof-s{seed}.npz';files[str(path)]=report['vector_sha256'][str(seed)]
        for fold in range(5):
            for init in INITS:
                d=SOURCE/f's{seed}-f{fold}-init{init}-EMA';w=read(d/'warm-complete.json');c=read(d/'cold-complete.json')
                if c['status']!='passed' or c['warm_sha256']!=sha(d/'warm-complete.json') or c['cold_states']!=2:
                    raise ValueError('Original independent model cold receipt required')
                if w['manifest_sha256']!=sha(SOURCE/'manifest.json'):raise ValueError('Wrong model manifest')
                if w['partitions']!=manifest['partitions'][f's{seed}-f{fold}']:raise ValueError('Partition metadata differs')
                for name in ['warm-complete.json','cold-complete.json']:files[str(d/name)]=sha(d/name)
                files[str(d/'predictions.npz')]=w['predictions_sha256'];files.update(w['checkpoint_hashes']);states.update(w['checkpoint_hashes'])
                for role,h in w['witnesses'].items():
                    wd=d/(role+'-witness');v=read(wd/'complete.json');cold=read(wd/'cold-audit.json')
                    if sha(wd/'complete.json')!=h or cold['status']!='passed' or cold['receipt_sha256']!=h:
                        raise ValueError('Witness/cold binding differs')
                    files[str(wd/'complete.json')]=h;files[str(wd/'cold-audit.json')]=sha(wd/'cold-audit.json')
                    files.update({str(wd/name):h for name,h in v['hashes'].items()})
                units[d.name]=dict(paths=w['paths'],reused_original=w['reused'],warm_sha256=sha(d/'warm-complete.json'))
    if len(states)!=60 or len(units)!=30:raise ValueError('Distinct physical saved state inventory differs')
    verify(files);RUN.mkdir(parents=True,exist_ok=False)
    write(RUN/'manifest.json',dict(files=files,source=str(SOURCE),units=units,reference=best['candidate'],
        reference_score_user_reported=best['score'],reference_zip_sha256=best['zip_sha256'],
        candidate_order=[CANDIDATE],seeds=list(SEEDS),training_seeds=list(INITS),folds=5,component_weight=.75,
        retained_states=60,new_fits=0,new_optimizer_constructors=0,new_model_predict_calls=0,
        confirmation_seeds=0,packages=0,desktop_writes=0,agent_uploads=0,created_ns=time.time_ns()))
    print(json.dumps(dict(status='frozen',files=len(files),saved_states=60,new_fits=0)))

def evaluate():
    runtime();m=read(RUN/'manifest.json');verify(m['files'])
    import numpy as np
    import yaml
    from bf_tap_r2.candidate_tiers import classify_candidates
    from bf_tap_r2.component_regularization_run import metric_detail
    metrics={'tap_time_len':{'EMA_MEAN3':{},CANDIDATE:{}}};gains={};outputs={};maximum=0.
    for seed in SEEDS:
        with np.load(SOURCE/f'oof-s{seed}.npz',allow_pickle=False) as a:
            ids=a['query_ids'];fv=a['folds']
            if len(ids)!=2754 or len(set(ids))!=2754 or set(fv)!=set(range(5)):raise ValueError('Incomplete OOF')
            members=np.stack([a[f'EMA_{i}_component'] for i in INITS]);q75=a['q75'];iron=a['iron']
            for fold in range(5):
                mask=fv==fold;fit_ids=ids[~mask].tolist();query_ids=ids[mask].tolist()
                for j,init in enumerate(INITS):
                    d=SOURCE/f's{seed}-f{fold}-init{init}-EMA';w=read(d/'warm-complete.json');v=read(d/'refit-witness/complete.json')
                    validate_unit(w,v,seed,fold,init,fit_ids,query_ids)
                    with np.load(d/'predictions.npz',allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['query_ids'],ids[mask]);np.testing.assert_array_equal(p['prediction'],members[j,mask])
                        for key in ['q75','old_ema','iron']:np.testing.assert_array_equal(p[key],a[key][mask])
            np.testing.assert_array_equal(members[0],a['old_ema'])
            mean,median=columns(q75,members);np.testing.assert_array_equal(mean,a['EMA_MEAN3'])
            scalar=np.array([float(q)+.75*(sorted(float(x) for x in v)[1]-float(v[0])) for q,v in zip(q75,members.T)])
            maximum=max(maximum,float(np.max(np.abs(scalar-median))))
            y=a['actual'];base=score(y,iron,mean);delta=score(y,iron,median)-base
            one=50*(math.fsum(abs(float(t)-float(p)) for t,p in zip(y[:,1],mean))-
                math.fsum(abs(float(t)-float(p)) for t,p in zip(y[:,1],median)))/math.fsum(abs(float(t)) for t in y[:,1])
            maximum=max(maximum,abs(delta-one))
            gains[str(seed)]=dict(vs_mean3=delta,vs_q75=score(y,iron,median)-score(y,iron,q75),
                independently_scored_vs_mean3=one,mean3_score=base,median3_score=score(y,iron,median))
            for name,p in [('EMA_MEAN3',mean),(CANDIDATE,median)]:
                metrics['tap_time_len'][name][str(seed)]=metric_detail(y[:,1],p,fv,a['spouts'])
            path=RUN/f'oof-s{seed}.npz'
            with path.open('xb') as stream:np.savez_compressed(stream,ids=ids,folds=fv,spouts=a['spouts'],actual=y,
                iron=iron,q75=q75,mean3=mean,median3=median,members=members)
            outputs[str(seed)]=sha(path)
    if maximum>1e-11:raise ValueError('Independent formula/score mismatch')
    spec=dict(split_seeds=list(SEEDS),folds=5,candidates={'tap_time_len':[CANDIDATE]},
        reference_by_target={'tap_time_len':'EMA_MEAN3'},tie_preference_by_target={'tap_time_len':[CANDIDATE]})
    report=dict(status='complete_two_seed_zero_fit_review',gains=gains,metrics=metrics,
        mean_gain=sum(x['vs_mean3'] for x in gains.values())/2,
        confirmation_eligible=all(x['vs_mean3']>0 for x in gains.values()),formal_promoted=False,
        candidate_tiers=classify_candidates(metrics,spec,yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text())),
        independent_maximum_difference=maximum,manifest_sha256=sha(RUN/'manifest.json'),oof_sha256=outputs,
        new_fits=0,new_model_predict_calls=0,confirmation_seeds=0,packages=0,desktop_writes=0,agent_uploads=0)
    verify(m['files']);write(RUN/'report.json',report)
    print(json.dumps({k:report[k] for k in ['status','gains','mean_gain','confirmation_eligible','formal_promoted']}))

def audit():
    runtime();m=read(RUN/'manifest.json');r=read(RUN/'report.json');verify(m['files'])
    import numpy as np
    if sha(RUN/'manifest.json')!=r['manifest_sha256']:raise ValueError('Report manifest differs')
    results={};maximum=0.
    for seed in SEEDS:
        path=RUN/f'oof-s{seed}.npz'
        if sha(path)!=r['oof_sha256'][str(seed)]:raise ValueError('Derived OOF differs')
        with np.load(path,allow_pickle=False) as a:
            with np.load(SOURCE/f'oof-s{seed}.npz',allow_pickle=False) as original:
                for saved,key in [('ids','query_ids'),('folds','folds'),('spouts','spouts'),('actual','actual'),('iron','iron'),('q75','q75')]:
                    np.testing.assert_array_equal(a[saved],original[key])
                for j,init in enumerate(INITS):np.testing.assert_array_equal(a['members'][j],original[f'EMA_{init}_component'])
            med=[];avg=[]
            for q,row in zip(a['q75'],a['members'].T):
                v=[float(x) for x in row];med.append(float(q)+.75*(sorted(v)[1]-v[0]));avg.append(float(q)+.75*(math.fsum(v)/3-v[0]))
            maximum=max(maximum,max(abs(x-float(y)) for x,y in zip(med,a['median3'])),max(abs(x-float(y)) for x,y in zip(avg,a['mean3'])))
            den=math.fsum(abs(float(v)) for v in a['actual'][:,1])
            gain=50*math.fsum(abs(float(y)-p)-abs(float(y)-c) for y,p,c in zip(a['actual'][:,1],avg,med))/den
            maximum=max(maximum,abs(gain-r['gains'][str(seed)]['vs_mean3']));results[str(seed)]=gain
    if maximum>1e-11 or r['formal_promoted']:raise ValueError('Independent arithmetic/scope gate failed')
    if r['confirmation_eligible']!=all(v>0 for v in results.values()):raise ValueError('Eligibility differs')
    write(RUN/'independent-audit.json',dict(status='passed',independent_gains=results,maximum_difference=maximum,
        manifest_sha256=sha(RUN/'manifest.json'),report_sha256=sha(RUN/'report.json'),new_fits=0,new_model_predict_calls=0))
    print(json.dumps(dict(status='passed',independent_gains=results,maximum_difference=maximum)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate','audit']);p.add_argument('--checks')
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.checks)
    elif args.stage=='evaluate':evaluate()
    else:audit()
