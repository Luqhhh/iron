"""Frozen four-split RealMLP confirmation with the actual EMA3 Q100 reference."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from . import realmlp_state_adapter as adapter
from .data import FEATURES,TARGETS
from .ema_nested_residual import read,write,sha,verify,save_arrays,audit_models
from .realmlp_development_audit import arrays,scalar_gain,scalar_wmape
from .realmlp_native_audit import observe_native,verify_trace,native_state_digest,cold_audit
from .realmlp_time_development import setup as native_setup
from .v7_periodic import digest

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_time_confirmation/SPEC.json'
DEV=MAIN/'local/runs/realmlp-time-development-20261004/development-r1'
RECOVERY=DEV.parent/'development-audit-recovery-r1'
OLD=MAIN/'local/runs/ema-span-confirmation-20261001/confirmation-r1'
SEEDS=(271828,314159)
ALL_SEEDS=(42,3407,*SEEDS)
CANDIDATE='REALMLP_SINGLE_A20'


def sources():
    paths=[*list((WORK/'src').rglob('*.py')),WORK/SPEC,WORK/'tests/test_realmlp_time_confirmation.py',
        WORK/'docs/realmlp_time_confirmation/PREREGISTRATION.md',WORK/'uv.lock',WORK/'pyproject.toml']
    return {str(p.relative_to(WORK)):sha(p) for p in paths}


def setup():
    original=native_setup()
    old=read(OLD/'manifest.json')
    if {k:importlib.metadata.version(k) for k in old['versions']}!=old['versions']:raise ValueError('Original reference runtime differs')
    return original


def columns(v32,v7,members,realmlp):
    v32,v7,members,realmlp=[np.asarray(x,float) for x in (v32,v7,members,realmlp)]
    if v32.ndim!=1 or v7.shape!=v32.shape or realmlp.shape!=v32.shape or members.shape!=(3,len(v32)):
        raise ValueError('Aligned current-parent and single candidate vectors required')
    current=v32+(members.mean(0)-v7)
    candidate=.8*current+.2*realmlp
    if any(not np.isfinite(x).all() or (x<0).any() for x in (current,candidate)):raise ValueError('Invalid extrapolation; clipping forbidden')
    return current,candidate


def gate(gains):
    if set(gains)!=set(map(str,ALL_SEEDS)) or not all(math.isfinite(x) for x in gains.values()):
        raise ValueError('Four complete finite split gains required')
    values=np.array([gains[str(s)] for s in ALL_SEEDS]);mean=float(values.mean());sd=float(values.std(ddof=1))
    lower=mean-float(t.ppf(.95,3))*sd/2
    return dict(n=4,mean=mean,sd=sd,se=sd/2,lcb95=lower,passed=bool((values>0).all() and lower>0))


def peak():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if value>1536:raise ValueError('Frozen memory gate failed')
    return value


def prepare(checks):
    from .v5_library import load_v5_training_frame,fold_vector
    from .v5_spec import load_v5_spec
    original=setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);checks=Path(checks)
    if run.exists():raise FileExistsError('Consumed confirmation directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Clean committed source required')
    c=read(checks)
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or c['junit_sha256']!=sha(c['junit']):raise ValueError('Exact source locked checks required')
    terminal=read(RECOVERY/'terminal-reconciliation.json');r=read(DEV/'report.json');audit=read(RECOVERY/'independent-audit.json')
    if (sha(RECOVERY/'terminal-reconciliation.json')!='8104a4695fefbff501f6bb96e966ce07a7f7e567b6a37a18c65850ae02a312f7'
            or terminal['status']!='passed' or terminal['actual_recovery_exit_code']!=0 or audit['status']!='passed'
            or audit['report_sha256']!=sha(DEV/'report.json') or r['selected_for_confirmation']!=CANDIDATE
            or min(r['gains'][CANDIDATE].values())<=0):raise ValueError('Closed two-positive-split development required')
    best=read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['reference'].items()):raise ValueError('Current incumbent changed')
    # Path-name inventory is a limited cache search, not a global absence proof.
    matches=[]
    roots=[MAIN/'local/runs',MAIN/'local/worktrees/ema-retraining-initialization/local/runs']
    for root in roots:
        if not root.exists():continue
        for p in root.rglob('*.pt'):
            name=str(p)
            if any(f's{s}-' in name for s in SEEDS) and any(f'init{i}' in name or f'seed{i}' in name for i in (1042,2042)) and 'EMA' in name:
                matches.append(str(p))
    if matches:raise ValueError('Potential reusable current-reference cache found; refreeze after identity intake: '+repr(matches))
    old=read(OLD/'manifest.json');old_terminal=read(OLD/'terminal-verification-r1.json')
    if old_terminal['status']!='passed':raise ValueError('Original reference batch terminal not passed')
    files=dict(read(RECOVERY/'manifest.json')['files'])
    files.update(old['model_sources'])
    for p,h in old['sources'].items():files[str(Path(old['workspace'])/p)]=h
    for p,h in old['spec']['inputs'].items():files[str(MAIN/p)]=h
    paths=[WORK/p for p in sources()]+[checks,Path(c['junit']),RECOVERY/'manifest.json',RECOVERY/'independent-audit.json',RECOVERY/'terminal-reconciliation.json',
        OLD/'manifest.json',OLD/'audit.json',OLD/'terminal-verification-r1.json',MAIN/best['package'],MAIN/best['platform_feedback_record']]
    for seed in SEEDS:
        for fold in range(5):
            d=OLD/f's{seed}-f{fold}'
            paths += [p for p in d.rglob('*') if p.is_file() and 'SHORT_SPAN' not in p.parts]
    for p in paths:
        p=p.resolve();h=sha(p)
        if str(p) in files and files[str(p)]!=h:raise ValueError('Conflicting frozen dependency: '+str(p))
        files[str(p)]=h
    verify(files)
    frame=load_v5_training_frame(MAIN);folds={s:fold_vector(MAIN,frame,s,load_v5_spec(MAIN)) for s in ALL_SEEDS}
    for s in ALL_SEEDS:
        if digest(folds[s].tolist())!=old['fold_digests'][str(s)]:raise ValueError('Confirmation fold identity changed')
    for s in [42,3407]:
        a=arrays(DEV/f'oof-s{s}.npz')
        for x,y in [(a['ids'],frame.sample_id.to_numpy(str)),(a['actual'],frame[list(TARGETS)].to_numpy(float)),(a['folds'],folds[s])]:np.testing.assert_array_equal(x,y)
    current=yaml.safe_load((WORK/'configs/strong_component_regularization/SPEC.yaml').read_text())
    if current['training']['tap_time_len']!=old['spec']['training'] or current['mechanisms']!=old['spec']['mechanisms']:raise ValueError('Original EMA recipe changed')
    run.mkdir();frame.to_pickle(run/'training.pkl');save_arrays(run/'folds.npz',**{str(s):a for s,a in folds.items()})
    files.update({str(run/n):sha(run/n) for n in ['training.pkl','folds.npz']})
    write(run/'manifest.json',dict(spec=spec,files=files,source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        realmlp_recipe=original['recipes']['realmlp_td'],ema_settings=current['training']['tap_time_len'],mechanisms=current['mechanisms'],
        reference_composition=old['reference_composition'],old_partitions=old['partitions'],cache_search_roots=list(map(str,roots)),cache_matches=matches,
        development_gains=r['gains'][CANDIDATE],formal_promoted=False))
    print(json.dumps(dict(status='frozen',files=len(files),new_procedures=30,new_optimizers=60)),flush=True)


def context(seed=None,fold=None):
    setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);m=read(run/'manifest.json');verify(m['files'])
    if m['source_directory']!=str(WORK) or m['spec']!=spec:raise ValueError('Frozen context changed')
    if seed is None:return run,m
    if seed not in SEEDS or fold not in range(5):raise ValueError('Unregistered confirmation unit')
    frame=pd.read_pickle(run/'training.pkl');fv=arrays(run/'folds.npz')[str(seed)]
    training=frame.loc[fv!=fold].reset_index(drop=True);query=frame.loc[fv==fold,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
    from .ema_span_confirmation import partition
    if partition(training,query,m['ema_settings'])!=m['old_partitions'][f's{seed}-f{fold}']:raise ValueError('Original outer/inner training partition changed')
    return run,m,training,query


def require_bridges(run):
    for seed in SEEDS:
        for fold in range(5):
            d=run/f's{seed}-f{fold}'/'reference-replay';r=read(d/'complete.json')
            if r['status']!='passed' or r['reference_states']!=40 or r['ema_native_states']!=2 or sha(d/'predictions.npz')!=r['predictions_sha256']:
                raise ValueError('All ten current-reference cold bridges required before fitting')


def reference_time_composition(source,graph,original):
    store={}
    for slot in graph['slots']:
        name=slot['name']
        if name not in store:store[name]=np.load(source/'reference'/name/'final/observed.npy',allow_pickle=False)
    matrix=np.column_stack([store[s['name']] if s['outputs']==1 else store[s['name']][:,s['column']] for s in graph['slots']])
    if graph['existing_b36_nonnegative_projection'] is not True:raise ValueError('Original reference projection changed')
    a=matrix@np.asarray(graph['a_coefficients']['tap_time_len']);weights=graph['b_weights']['tap_time_len'];v36=weights[0]*a
    for weight,name in zip(weights[1:],graph['b_experts']['tap_time_len']):v36=v36+weight*store[name]
    v36=np.maximum(v36,0)
    expected=dict(v36_time=v36,n_time=store['N0048'],v7_time=store['V7_periodic'])
    expected['tap_time_len']=.2*v36+.3*expected['n_time']+.5*expected['v7_time']
    maximum=max(float(np.max(abs(value-original[name]))) for name,value in expected.items())
    if maximum>1e-9:raise ValueError('Independent original time composition differs')
    return maximum


def reuse(seed,fold):
    from .ema_reference_artifacts import audit_witness
    from .v4_1_reference import V36FixedRecipeFactory
    run,m,training,query=context(seed,fold);source=OLD/f's{seed}-f{fold}';out=run/f's{seed}-f{fold}'/'reference-replay';out.mkdir(parents=True,exist_ok=False)
    V36FixedRecipeFactory(MAIN,workers=1)  # Register immutable legacy pickle classes; no fit.
    warm=read(source/'warm-complete.json');cold=read(source/'cold-complete.json');ref=read(source/'reference/complete.json')
    if (cold['status']!='passed' or cold['warm_receipt_sha256']!=sha(source/'warm-complete.json')
            or warm['reference_receipt_sha256']!=sha(source/'reference/complete.json')
            or warm['predictions_sha256']!=sha(source/'predictions.npz')):raise ValueError('Original reference receipts changed')
    receipts={};maxdiff=0.;state_count=0
    for name,record in ref['roles'].items():
        role=source/'reference'/name
        if sha(role/'complete.json')!=record['receipt_sha256']:raise ValueError('Reference role changed')
        for key,h in record['witnesses'].items():
            witness=role/key;w=read(witness/'complete.json');copy=out/'witnesses'/name/key;copy.mkdir(parents=True,exist_ok=False)
            expected=dict(source_directory=str(OLD),split_seed=seed,fold=fold,trial_id=name,fit_call_id='selector-terminal' if key=='selector-terminal/model-witness' else key)
            if sha(witness/'complete.json')!=h or w['identity']!=expected:raise ValueError('Original witness identity changed')
            if key=='final' and (w['training_ids']!=training.sample_id.tolist() or w['query_ids']!=query.sample_id.tolist()):raise ValueError('Original final partition differs')
            for file in ['complete.json',*w['hashes']]:
                shutil.copyfile(witness/file,copy/file)
                if sha(witness/file)!=sha(copy/file):raise ValueError('Read-only witness copy changed')
            receipt=audit_witness(copy,h);receipts[name+'/'+key]=dict(source=str(witness),receipt_sha256=h,cold=receipt)
            maxdiff=max(maxdiff,*receipt['differences'].values());state_count+=1;gc.collect()
    if state_count!=40:raise ValueError('Original reference cold state count differs')
    original=arrays(source/'predictions.npz');np.testing.assert_array_equal(original['query_ids'],query.sample_id.to_numpy(str))
    composition=reference_time_composition(source,m['reference_composition'],original)
    oldema=read(source/'OLD_EMA/start.json')
    if oldema['settings']!=m['ema_settings'] or oldema['mechanisms']!=m['mechanisms'] or oldema['training_ids']!=training.sample_id.tolist() or oldema['query_ids']!=query.sample_id.tolist():raise ValueError('Original EMA42 identity differs')
    ema=audit_models(source/'OLD_EMA',training,query,{'refit':original['old_ema']},m['ema_settings'],m['mechanisms'])
    save_arrays(out/'predictions.npz',ids=query.sample_id.to_numpy(str),v32=original['tap_time_len'],v7=original['v7_time'],ema42=original['old_ema'])
    verify(m['files']);write(out/'complete.json',dict(status='passed',source_directory=str(source),reference_states=state_count,ema_native_states=ema['states'],
        native_ema_audit=ema,witnesses=receipts,maximum_cold_difference=maxdiff,composition_difference=composition,predictions_sha256=sha(out/'predictions.npz'),
        original_warm_sha256=sha(source/'warm-complete.json'),new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=peak()))
    print(json.dumps(dict(status='passed',action='reuse',seed=seed,fold=fold,states=42)),flush=True)


def realmlp(seed,fold,cold=False):
    run,m,frame,query=context(seed,fold);require_bridges(run);d=run/f's{seed}-f{fold}'/'realmlp';recipe=m['realmlp_recipe']
    identity=dict(source_directory=str(d),split_seed=seed,fold=fold,trial_id='TD_TIME_INIT42')
    if cold:
        c=read(d/'complete.json');a=arrays(d/'predictions.npz')
        if c['identity']!=identity or c['predictions_sha256']!=sha(d/'predictions.npz') or c['pid']==os.getpid():raise ValueError('Independent native cold identity differs')
        np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str))
        result=cold_audit(d,frame,query,frame.tap_time_len.to_numpy(float),recipe,identity,c,{k:a[k] for k in ['selection','refit']})
        write(d/'cold.json',dict(**result,pid=os.getpid(),complete_sha256=sha(d/'complete.json'),peak_rss_mib=peak()))
    else:
        d.mkdir(exist_ok=False);write(d/'estimator-start.json',dict(identity=identity,pid=os.getpid(),fitting_ids=frame.sample_id.tolist(),query_ids=query.sample_id.tolist()))
        with observe_native(d) as trace:captured=adapter.capture_fit(recipe,frame,frame.tap_time_len.to_numpy(float),inner_seed=42)
        model=captured.regressor;epoch=model.metadata_['selected_epoch'];verify_trace(trace,frame,epoch,256)
        predictions={'refit':model.predict(query),'selection':captured.estimators[0].predict(captured.encoders[0].transform(query))}
        states={role:native_state_digest(captured.estimators[i]) for i,role in enumerate(('selection','refit'))}
        for role in states:adapter.save_snapshot(captured,role,d/(role+'.pkl'),identity)
        save_arrays(d/'predictions.npz',ids=query.sample_id.to_numpy(str),**predictions)
        write(d/'complete.json',dict(status='passed',identity=identity,recipe=recipe,native_trace=trace,metadata=model.metadata_,selected_epoch=epoch,
            parameter_states=states,predictions_sha256=sha(d/'predictions.npz'),native_optimizer_runs=len(trace['optimizers']),native_updates=sum(x['steps'] for x in trace['optimizers']),
            state_hashes={r:sha(d/(r+'.pkl')) for r in states},peak_rss_mib=peak(),pid=os.getpid(),manifest_sha256=sha(run/'manifest.json')))
    verify(m['files']);print(json.dumps(dict(status='passed',action='realmlp_cold' if cold else 'realmlp',seed=seed,fold=fold)),flush=True)


def ema(seed,fold,init,cold=False):
    from .component_regularization import ComponentRegressor
    from .component_regularization_run import RECIPE
    from .ema_mean5 import optimizer_ledger,verify_counts
    run,m,frame,query=context(seed,fold);require_bridges(run)
    if init not in (1042,2042):raise ValueError('Only missing current-reference members admitted')
    d=run/f's{seed}-f{fold}'/f'ema-init{init}';identity=dict(source_directory=str(d),split_seed=seed,fold=fold,trial_id=f'EMA_INIT{init}');settings=dict(m['ema_settings'],random_seed=init)
    if cold:
        import torch
        c=read(d/'complete.json');a=arrays(d/'predictions.npz')
        if c['identity']!=identity or c['pid']==os.getpid() or c['settings']!=settings or c['predictions_sha256']!=sha(d/'predictions.npz'):raise ValueError('EMA cold identity differs')
        traces={}
        for role,h in c['state_hashes'].items():
            if sha(d/(role+'.pt'))!=h:raise ValueError('EMA native state changed')
            traces[role]=torch.load(d/(role+'.pt'),map_location='cpu',weights_only=True)['trace']
        verify_counts(c,traces);np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str))
        result=audit_models(d,frame,query,{k:a[k] for k in ['selection','refit']},settings,m['mechanisms'])
        write(d/'cold.json',dict(**result,pid=os.getpid(),complete_sha256=sha(d/'complete.json')))
    else:
        d.mkdir(exist_ok=False);write(d/'estimator-start.json',dict(identity=identity,pid=os.getpid(),fitting_ids=frame.sample_id.tolist(),query_ids=query.sample_id.tolist()))
        with optimizer_ledger(d,identity) as counts:
            model=ComponentRegressor(RECIPE,settings,'EMA',m['mechanisms'],d);model.fit(frame.drop(columns=list(TARGETS)),frame[['tap_time_len']].to_numpy())
        verify_counts(counts,model.traces)
        save_arrays(d/'predictions.npz',ids=query.sample_id.to_numpy(str),refit=model.predict(query)[:,0],selection=ComponentRegressor.load(d/'selection.pt').predict(query)[:,0])
        write(d/'complete.json',dict(status='passed',identity=identity,settings=settings,**counts,state_hashes={r:sha(d/(r+'.pt')) for r in ['selection','refit']},
            predictions_sha256=sha(d/'predictions.npz'),native_optimizer_runs=2,native_updates=sum(counts['steps'].values()),peak_rss_mib=peak(),pid=os.getpid(),manifest_sha256=sha(run/'manifest.json')))
    verify(m['files']);print(json.dumps(dict(status='passed',action='ema_cold' if cold else 'ema',seed=seed,fold=fold,init=init)),flush=True)


def collect(run,m,frame,folds):
    vectors={};optimizers=0;states=0;updates=0
    for seed in [42,3407]:
        a=arrays(DEV/f'oof-s{seed}.npz');vectors[seed]=dict(reference=a['reference'],candidate=a[CANDIDATE])
    for seed in SEEDS:
        fv=folds[str(seed)];members=np.full((3,len(frame)),np.nan);v32=np.full(len(frame),np.nan);v7=v32.copy();r=v32.copy()
        for fold in range(5):
            mask=fv==fold;d=run/f's{seed}-f{fold}';base=arrays(d/'reference-replay/predictions.npz')
            np.testing.assert_array_equal(base['ids'],frame.loc[mask,'sample_id'].to_numpy(str));v32[mask]=base['v32'];v7[mask]=base['v7'];members[0,mask]=base['ema42']
            for name in ['realmlp','ema-init1042','ema-init2042']:
                unit=d/name;c=read(unit/'complete.json');cold=read(unit/'cold.json');a=arrays(unit/'predictions.npz')
                if cold['status']!='passed' or cold['complete_sha256']!=sha(unit/'complete.json') or c['predictions_sha256']!=sha(unit/'predictions.npz') or c['manifest_sha256']!=sha(run/'manifest.json'):raise ValueError('Incomplete scientific state coverage')
                for role,h in c['state_hashes'].items():
                    if sha(unit/(role+('.pkl' if name=='realmlp' else '.pt')))!=h:raise ValueError('Saved state changed')
                np.testing.assert_array_equal(a['ids'],base['ids'])
                if name=='realmlp':r[mask]=a['refit']
                else:members[1 if name.endswith('1042') else 2,mask]=a['refit']
                optimizers+=c['native_optimizer_runs'];states+=cold['states'];updates+=c['native_updates']
        current,candidate=columns(v32,v7,members,r);vectors[seed]=dict(reference=current,candidate=candidate,v32=v32,v7=v7,ema_members=members,realmlp=r)
    if optimizers!=60 or states!=60:raise ValueError('Frozen new native budget differs')
    return vectors,dict(new_optimizers=optimizers,new_states=states,new_updates=updates,reused_reference_states=400,reused_ema_states=20)


def report():
    from .component_regularization_run import metric_detail
    run,m=context();require_bridges(run);frame=pd.read_pickle(run/'training.pkl');folds=arrays(run/'folds.npz');vectors,inventory=collect(run,m,frame,folds)
    gains={};metrics={};hashes={};y=frame.tap_time_len.to_numpy(float)
    for seed,v in vectors.items():
        gains[str(seed)]=float(50*np.sum(abs(y-v['reference'])-abs(y-v['candidate']))/np.sum(abs(y)))
        metrics[str(seed)]={k:metric_detail(y,v[k],folds[str(seed)],frame.spout_no.to_numpy()) for k in ['reference','candidate']}
        save_arrays(run/f'oof-s{seed}.npz',ids=frame.sample_id.to_numpy(str),folds=folds[str(seed)],actual=y,spouts=frame.spout_no.to_numpy(),**v);hashes[str(seed)]=sha(run/f'oof-s{seed}.npz')
    for s in ['42','3407']:
        if abs(gains[s]-m['development_gains'][s])>1e-10:raise ValueError('Original development gain changed')
    result=dict(status='complete_four_split_confirmation',candidate=CANDIDATE,gains=gains,paired=gate(gains),metrics=metrics,inventory=inventory,oof_sha256=hashes,
        manifest_sha256=sha(run/'manifest.json'),formal_promoted=False,independent_audit_pending=True,absolute_package_scores_computed=False,full_fits=0,packages=0)
    verify(m['files']);write(run/'report.json',result);print(json.dumps({k:result[k] for k in ['status','gains','paired']}),flush=True)


def audit():
    run,m=context();require_bridges(run);report=read(run/'report.json');frame=pd.read_pickle(run/'training.pkl');folds=arrays(run/'folds.npz');vectors,inventory=collect(run,m,frame,folds);gains={};maximum=0.;cells=0
    def check(a,b):
        nonlocal maximum,cells
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):raise ValueError('Nonfinite independent metric')
        maximum=max(maximum,abs(float(a)-float(b)));cells+=1
        if maximum>1e-10:raise ValueError('Independent scalar mismatch')
    for seed in SEEDS:
        for fold in range(5):
            bridge=run/f's{seed}-f{fold}'/'reference-replay';receipt=read(bridge/'complete.json')
            for key,entry in receipt['witnesses'].items():
                copy=bridge/'witnesses'/key;old=read(Path(entry['source'])/'complete.json')
                if sha(copy/'complete.json')!=entry['receipt_sha256'] or read(copy/'cold-audit.json')!=entry['cold']:
                    raise ValueError('Read-only cold replay custody changed')
                for name,h in old['hashes'].items():
                    if sha(copy/name)!=h:raise ValueError('Read-only witness bytes changed')
    for seed in ALL_SEEDS:
        a=arrays(run/f'oof-s{seed}.npz')
        if sha(run/f'oof-s{seed}.npz')!=report['oof_sha256'][str(seed)]:raise ValueError('OOF changed')
        np.testing.assert_array_equal(a['ids'],frame.sample_id.to_numpy(str));np.testing.assert_array_equal(a['actual'],frame.tap_time_len.to_numpy(float));np.testing.assert_array_equal(a['folds'],folds[str(seed)])
        for k,v in vectors[seed].items():np.testing.assert_array_equal(a[k],v)
        if seed in SEEDS:
            current=np.array([float(b)+math.fsum(float(x) for x in p)/3-float(v) for b,p,v in zip(a['v32'],a['ema_members'].T,a['v7'])])
            candidate=np.array([.8*float(b)+.2*float(r) for b,r in zip(current,a['realmlp'])])
            check(np.max(abs(current-a['reference'])),0);check(np.max(abs(candidate-a['candidate'])),0)
        y=a['actual'];gains[str(seed)]=scalar_gain(y,a['reference'],a['candidate']);check(gains[str(seed)],report['gains'][str(seed)])
        for name in ['reference','candidate']:
            p=a[name];metric=report['metrics'][str(seed)][name];check(scalar_wmape(y,p,range(len(y))),metric['wmape'])
            for fold in range(5):check(scalar_wmape(y,p,np.flatnonzero(a['folds']==fold)),metric['by_fold'][str(fold)])
            for spout in set(a['spouts']):check(scalar_wmape(y,p,np.flatnonzero(a['spouts']==spout)),metric['by_spout'][str(spout)])
    mean=math.fsum(gains.values())/4;sd=math.sqrt(math.fsum((x-mean)**2 for x in gains.values())/3);lcb=mean-float(t.ppf(.95,3))*sd/2
    for k,v in dict(mean=mean,sd=sd,se=sd/2,lcb95=lcb).items():check(v,report['paired'][k])
    passed=all(x>0 for x in gains.values()) and lcb>0
    if passed!=report['paired']['passed'] or inventory!=report['inventory']:raise ValueError('Independent gate or budget mismatch')
    verify(m['files']);result=dict(status='passed',gains=gains,seed_paired_lcb95=lcb,four_seed_gate_passed=passed,maximum_scalar_difference=maximum,scalar_cells=cells,
        inventory=inventory,report_sha256=sha(run/'report.json'),manifest_sha256=sha(run/'manifest.json'),new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=peak())
    write(run/'independent-audit.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','reuse','realmlp','realmlp_cold','ema','ema_cold','report','audit']);p.add_argument('--checks');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);p.add_argument('--init',type=int);a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action=='reuse':reuse(a.seed,a.fold)
    elif a.action.startswith('realmlp'):realmlp(a.seed,a.fold,a.action.endswith('_cold'))
    elif a.action.startswith('ema'):ema(a.seed,a.fold,a.init,a.action.endswith('_cold'))
    elif a.action=='report':report()
    else:audit()


if __name__=='__main__':main()
