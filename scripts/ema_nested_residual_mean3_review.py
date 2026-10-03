"""Reweight the frozen seed42 residual correction in a three-EMA incumbent."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'local/runs/ema-nested-residual-20261003/development-r1'
MEAN=ROOT/'local/runs/ema-median3-20261003/development-r1'
RUN=ROOT/'local/runs/ema-nested-residual-mean3-20261003/review-r1'
ORDER=('RIDGE','GBM')
SEEDS=(42,3407)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Frozen dependency changed: '+p)
def runtime():
    if sys.version_info[:2]!=(3,12) or any(os.environ.get(k)!='1' for k in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('Locked Python3.12 and pinned threads required')

def corrected(mean3,head):
    import numpy as np
    mean3,head=np.asarray(mean3,float),np.asarray(head,float)
    if mean3.ndim!=1 or head.shape!=mean3.shape:raise ValueError('Aligned one-dimensional vectors required')
    value=mean3+.25*.25*head
    if not np.isfinite(value).all() or (value<0).any():raise ValueError('Invalid prediction; no clipping')
    return value

def choose(gains):
    if list(gains)!=list(ORDER) or any(set(v)!={'42','3407'} for v in gains.values()):
        raise ValueError('Complete fixed two-seed pool required')
    if any(not math.isfinite(x) for v in gains.values() for x in v.values()):raise ValueError('Nonfinite gain')
    eligible=[k for k in ORDER if min(gains[k].values())>0]
    if not eligible:return None
    best=max(math.fsum(gains[k].values())/2 for k in eligible)
    return next(k for k in ORDER if k in eligible and math.fsum(gains[k].values())/2>=best-1e-12)

def require_terminal(directory):
    directory=Path(directory);e=directory/'execution'
    t=read(e/'terminal.json');a=read(e/'independent-terminal-audit.json');r=read(e/'final-reconciliation.json')
    if (t.get('status')!='passed' or a.get('status')!='passed' or r.get('status')!='passed'
            or r.get('actual_supervisor_exec_exit_code')!=0 or r.get('actual_independent_audit_exit_code')!=0
            or r.get('owned_processes_absent') is not True):
        raise ValueError('Actual successful original terminal and independent audit required')
    if (a['original_terminal_sha256']!=sha(e/'terminal.json') or a['report_sha256']!=sha(directory/'report.json')
            or r['independent_audit_sha256']!=sha(e/'independent-terminal-audit.json')):
        raise ValueError('Original terminal/audit binding differs')
    if len(t['events'])!=121 or any(v['exit_code']!=0 for v in t['events']):raise ValueError('Original children incomplete')

def aligned(old,mean):
    import numpy as np
    if len(old['ids'])!=2754 or len(set(old['ids']))!=2754 or set(old['folds'])!=set(range(5)):
        raise ValueError('Incomplete same-seed OOF')
    for key in ['ids','folds','actual','q75','iron']:np.testing.assert_array_equal(old[key],mean[key])

def score(y,iron,time_column):
    return 100-50*math.fsum(math.fsum(abs(float(a)-float(b)) for a,b in zip(y[:,j],p))/
        math.fsum(abs(float(a)) for a in y[:,j]) for j,p in enumerate((iron,time_column)))

def prepare(checks):
    runtime();checks=Path(checks);c=read(checks)
    if c['status']!='passed' or c['script_sha256']!=sha(__file__) or sha(c['junit'])!=c['junit_sha256']:
        raise ValueError('Matching locked checks required')
    require_terminal(OLD)
    best=read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'],best['score'],best['zip_sha256'])!=('EMA_MEAN3_FULL_Q75',96.3954,
            '016e7e9cb3f750c74509dbb51961c61fea205d0296bb233056da66005f69a396'):
        raise ValueError('Latest reference changed before activation')
    old=read(OLD/'manifest.json');mean=read(MEAN/'manifest.json');mt=read(MEAN/'terminal-reconciliation.json')
    if (mt['status']!='passed' or mt['actual_evaluation_exit_code']!=0 or mt['actual_independent_audit_exit_code']!=0
            or mt['manifest_sha256']!=sha(MEAN/'manifest.json') or mt['report_sha256']!=sha(MEAN/'report.json')
            or mt['independent_audit_sha256']!=sha(MEAN/'independent-audit.json')):
        raise ValueError('Audited matching mean3 cache required')
    files=dict(old['files'])
    for path,h in mean['files'].items():
        if path in files and files[path]!=h:raise ValueError('Incompatible original cache dependencies')
        files[path]=h
    paths=[Path(__file__),checks,Path(c['junit']),ROOT/'tests/test_ema_nested_residual_mean3_review.py',
        ROOT/'docs/ema_nested_residual/MEAN3_FOLLOWUP_PREREGISTRATION.md',ROOT/best['package'],
        ROOT/best['platform_feedback_record'],OLD/'manifest.json',OLD/'report.json',
        OLD/'execution/terminal.json',OLD/'execution/independent-terminal-audit.json',OLD/'execution/final-reconciliation.json',
        MEAN/'manifest.json',MEAN/'report.json',MEAN/'independent-audit.json',MEAN/'terminal-reconciliation.json']
    for seed in SEEDS:
        paths.extend([OLD/f'oof-s{seed}.npz',MEAN/f'oof-s{seed}.npz'])
        for fold in range(5):
            u=OLD/f's{seed}-f{fold}';w=read(u/'heads-complete.json');cold=read(u/'heads-cold.json')
            if cold['status']!='passed' or cold['complete_sha256']!=sha(u/'heads-complete.json'):
                raise ValueError('Original head cold receipt differs')
            if sha(u/'predictions.npz')!=w['prediction_sha256']:raise ValueError('Head prediction hash differs')
            paths.extend([u/'heads-complete.json',u/'heads-cold.json',u/'predictions.npz'])
            for family,h in w['states'].items():
                p=u/family/('state.npz' if family=='RIDGE' else 'state.txt')
                if sha(p)!=h:raise ValueError('Original head state differs')
                paths.append(p)
    files.update({str(p):sha(p) for p in paths});verify(files);RUN.mkdir(parents=True,exist_ok=False)
    write(RUN/'manifest.json',dict(files=files,reference=best['candidate'],reference_score_user_reported=best['score'],
        candidate_order=list(ORDER),seeds=list(SEEDS),component_weight=.25,gamma=.25,
        new_fits=0,new_model_predict_calls=0,new_confirmation_seeds=0,packages=0,desktop_writes=0,agent_uploads=0))
    print(json.dumps(dict(status='frozen',files=len(files),new_fits=0)))

def evaluate():
    runtime();m=read(RUN/'manifest.json');verify(m['files']);require_terminal(OLD)
    import numpy as np
    import yaml
    from bf_tap_r2.candidate_tiers import classify_candidates
    from bf_tap_r2.component_regularization_run import metric_detail
    gains={k:{} for k in ORDER};old_comparison={k:{} for k in ORDER};hashes={};maximum=0.
    metrics={'tap_time_len':{k:{} for k in ('EMA_MEAN3',*ORDER)}}
    for seed in SEEDS:
        with np.load(OLD/f'oof-s{seed}.npz',allow_pickle=False) as old,np.load(MEAN/f'oof-s{seed}.npz',allow_pickle=False) as mean:
            aligned(old,mean);ids=old['ids'];folds=old['folds'];y=old['actual'];iron=old['iron'];base=mean['mean3']
            heads={k:np.full(len(ids),np.nan) for k in ORDER}
            for fold in range(5):
                mask=folds==fold
                with np.load(OLD/f's{seed}-f{fold}'/'predictions.npz',allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['ids'],ids[mask])
                    for family in ORDER:
                        np.testing.assert_array_equal(p[family],old[family][mask]);heads[family][mask]=p[family+'_head']
                        np.testing.assert_array_equal(p[family],old['q75'][mask]+.75*.25*p[family+'_head'])
            output={};baseline=score(y,iron,base)
            metrics['tap_time_len']['EMA_MEAN3'][str(seed)]=metric_detail(y[:,1],base,folds,mean['spouts'])
            for family in ORDER:
                prediction=corrected(base,heads[family]);output[family]=prediction;output[family+'_head']=heads[family]
                # Algebraic check against the old Q75 recipe, without selecting its scale.
                maximum=max(maximum,float(np.max(np.abs(prediction-(base+(old[family]-old['q75'])/3)))))
                gains[family][str(seed)]=score(y,iron,prediction)-baseline
                old_comparison[family][str(seed)]=score(y,iron,old[family])-baseline
                metrics['tap_time_len'][family][str(seed)]=metric_detail(y[:,1],prediction,folds,mean['spouts'])
            path=RUN/f'oof-s{seed}.npz'
            with path.open('xb') as stream:np.savez_compressed(stream,ids=ids,folds=folds,actual=y,iron=iron,
                q75=old['q75'],mean3=base,spouts=mean['spouts'],**output)
            hashes[str(seed)]=sha(path)
    if maximum>1e-11:raise ValueError('Fixed component arithmetic mismatch')
    spec=dict(split_seeds=list(SEEDS),folds=5,candidates={'tap_time_len':list(ORDER)},
        reference_by_target={'tap_time_len':'EMA_MEAN3'},tie_preference_by_target={'tap_time_len':list(ORDER)})
    result=dict(status='complete_rebased_review',gains=gains,original_Q75_recipes_vs_mean3_descriptive=old_comparison,
        selected_for_confirmation=choose(gains),formal_promoted=False,metrics=metrics,
        candidate_tiers=classify_candidates(metrics,spec,yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text())),
        arithmetic_maximum_difference=maximum,manifest_sha256=sha(RUN/'manifest.json'),oof_sha256=hashes,
        new_fits=0,new_model_predict_calls=0,new_confirmation_seeds=0,packages=0,desktop_writes=0,agent_uploads=0)
    verify(m['files']);write(RUN/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','gains','selected_for_confirmation','formal_promoted']}))

def audit():
    runtime();m=read(RUN/'manifest.json');r=read(RUN/'report.json');verify(m['files']);require_terminal(OLD)
    import numpy as np
    if r['manifest_sha256']!=sha(RUN/'manifest.json'):raise ValueError('Report binding differs')
    gains={k:{} for k in ORDER};maximum=0.
    for seed in SEEDS:
        path=RUN/f'oof-s{seed}.npz'
        if sha(path)!=r['oof_sha256'][str(seed)]:raise ValueError('Output hash differs')
        with np.load(path,allow_pickle=False) as a,np.load(MEAN/f'oof-s{seed}.npz',allow_pickle=False) as mean:
            for k in ['ids','folds','actual','iron','q75','mean3','spouts']:np.testing.assert_array_equal(a[k],mean[k])
            for family in ORDER:
                heads=np.full(2754,np.nan)
                for fold in range(5):
                    with np.load(OLD/f's{seed}-f{fold}'/'predictions.npz',allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['ids'],a['ids'][a['folds']==fold]);heads[a['folds']==fold]=p[family+'_head']
                np.testing.assert_array_equal(heads,a[family+'_head'])
                expected=[]
                for q,members,h in zip(mean['q75'],mean['members'].T,heads):
                    v=[float(x) for x in members];v[0]+=.25*float(h)
                    expected.append(float(q)+.75*(math.fsum(v)/3-float(members[0])))
                maximum=max(maximum,max(abs(x-float(y)) for x,y in zip(expected,a[family])))
                denominator=math.fsum(abs(float(v)) for v in a['actual'][:,1])
                gain=50*math.fsum(abs(float(y)-float(p))-abs(float(y)-c) for y,p,c in zip(a['actual'][:,1],a['mean3'],expected))/denominator
                gains[family][str(seed)]=gain;maximum=max(maximum,abs(gain-r['gains'][family][str(seed)]))
    if maximum>1e-11 or choose(gains)!=r['selected_for_confirmation'] or r['formal_promoted']:
        raise ValueError('Independent scalar/selection audit failed')
    write(RUN/'independent-audit.json',dict(status='passed',gains=gains,maximum_difference=maximum,
        manifest_sha256=sha(RUN/'manifest.json'),report_sha256=sha(RUN/'report.json'),new_fits=0,new_model_predict_calls=0))
    print(json.dumps(dict(status='passed',gains=gains,maximum_difference=maximum)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate','audit']);p.add_argument('--checks');a=p.parse_args()
    {'prepare':lambda:prepare(a.checks),'evaluate':evaluate,'audit':audit}[a.stage]()
