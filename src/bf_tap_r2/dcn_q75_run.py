"""Independent fixed DCN development; preserve the original scientific trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import dcn_cross as native
from .dcn_cross import ARMS, CrossRegressor, clean, CrossNetwork, state_digest
from .dcn_cross_verify import independent_prediction, verify_model
from .data import FEATURES,TARGETS
from . import local_ridge_run as common
from .local_ridge_run import read,write,sha,verify,event,load_labels,reference
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest

SPEC='configs/dcn_q75_development/SPEC.json'


def runtime():
    result=common.runtime()
    if torch.__version__!='2.14.0+cpu':raise ValueError('Verified CPU torch changed')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    return dict(**result,torch=torch.__version__,torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads())


def sources(work):
    names=[SPEC,'docs/dcn_q75_development/PREREGISTRATION.md','tests/test_dcn_q75.py',
        'tests/test_dcn_cross.py','uv.lock','pyproject.toml','configs/protection.yaml',
        'configs/candidate_tiers.yaml','configs/dcn_cross_preparation/SPEC.yaml','scripts/observe_dcn_q75.py']
    paths=list((work/'src').rglob('*.py'))+[work/n for n in names]
    return {str(p.resolve()):sha(p) for p in paths}


def partitions(training):
    folds=np.asarray(group_safe_inner_folds(training,seed=42)['fold'])
    return training.loc[folds!=0],training.loc[folds==0]


def freeze(work,out,tests):
    work,out=Path(work).resolve(),Path(out).resolve();spec=read(work/SPEC);main=Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main/'local/runs'):raise ValueError('Fresh private directory required')
    source=sources(work)
    if read(tests)['status']!='passed' or read(tests)['sources']!=source:raise ValueError('Exact-source test receipt required')
    if subprocess.check_output(['git','status','--porcelain','--',*source],cwd=work,text=True).strip():raise ValueError('Commit tested source')
    for n,h in spec['original_source_hashes'].items():
        if sha(work/n)!=h or sha(Path(spec['original_source_worktree'])/n)!=h:raise ValueError('Original source differs')
    best=read(main/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best[k]!=v for k,v in spec['reference'].items()):raise ValueError('Current reference changed')
    bp=main/spec['reference_binding']
    if sha(bp)!=spec['reference_binding_sha256']:raise ValueError('Reference binding changed')
    binding=read(bp);files=dict(source);files.update(binding['frozen_original_evidence']);files[str(bp)]=sha(bp)
    for item in [binding['ids'],*[v for a in binding['columns'].values() for v in a.values()]]:files[item['path']]=item['sha256']
    units={}
    for seed in spec['development_seeds']+spec['confirmation_seeds']:
        for fold in range(5):
            dev=seed in spec['development_seeds'];p=Path(binding['original_development_source'] if dev else binding['original_confirmation_source'])
            p/=f'SHORT_SPAN-s{seed}-f{fold}' if dev else f's{seed}-f{fold}'
            units[f's{seed}-f{fold}']=dict(path=str(p/'predictions.npz'),iron_field='iron_reference' if dev else 'iron')
    for path in [main/'configs/protection.yaml',main/'configs/candidate_tiers.yaml',Path(tests),*sorted((main/'复赛_train').glob('*.csv'))]:files[str(path.resolve())]=sha(path)
    for seed in spec['development_seeds']:
        path=main/f'local/runs/round2-v2/comparison-r1/folds-{seed}.csv';files[str(path)]=sha(path)
    probe=out.parent/'engineering-r1';admission=read(probe/'cold.json');actual=read(probe/'actual-exit.json')
    if admission['status']!='passed' or actual['exit_code']!=0 or read(probe/'manifest.json')['sources']!=source:raise ValueError('Full-size actual cold admission required')
    if read(probe/'manifest.json')['runtime']!=runtime():raise ValueError('Engineering runtime differs')
    files.update({str(p):sha(p) for p in probe.rglob('*') if p.is_file()});verify(files)
    out.mkdir(parents=True,exist_ok=False)
    m=dict(spec=spec,sources=source,files=files,workspace=str(work),binding_path=str(bp),reference_units=units,
           runtime=runtime(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip())
    write(out/'manifest.json',m);frame=load_labels(out,m);plan={}
    for seed in spec['development_seeds']:
        fv,_=reference(m,frame,seed)
        from .splits import make_folds
        np.testing.assert_array_equal(fv,make_folds(frame,seed).set_index('sample_id').loc[frame.sample_id,'fold'].to_numpy())
        for fold in range(5):
            training=frame.loc[fv!=fold];query=clean(frame.loc[fv==fold]);fitting,calibration=partitions(training)
            plan[f's{seed}-f{fold}']=dict(training=digest(training.sample_id.tolist()),query=digest(query.sample_id.tolist()),
                fitting=digest(fitting.sample_id.tolist()),calibration=digest(calibration.sample_id.tolist()))
    write(out/'activation.json',dict(manifest_sha256=sha(out/'manifest.json'),plan=plan))


def fit_arm(out,unit,fitting,calibration,training,query,target,arm,settings,identity):
    directory=unit/arm;directory.mkdir(exist_ok=False)
    event(out,'procedure_reserved',optimizer_budget=2,arm=arm,**identity)
    original=native.CrossRegressor;starts=[]

    class Recorded(original):
        def initialize(self,frame,y):
            phase='selection' if not starts else 'refit'
            if len(starts)>=2:raise ValueError('Optimizer allocation exceeded')
            starts.append(phase);event(out,'optimizer_started',phase=phase,arm=arm,**identity)
            result=super().initialize(frame,y)
            event(out,'optimizer_initialized',phase=phase,arm=arm,**identity)
            return result

    native.CrossRegressor=Recorded
    try:
        refit,selector,pred,cp=native.fit_partition(fitting,calibration,training,query,target,arm,settings)
    finally:native.CrossRegressor=original
    if starts!=['selection','refit']:raise ValueError('Native optimizer budget differs')
    selector.save(directory/'selector.json');refit.save(directory/'refit.json')
    write(directory/'traces.json',dict(selector=selector.metadata(),refit=refit.metadata()))
    with (directory/'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream,query_ids=query.sample_id.to_numpy(str),query=pred,calibration=cp)
    event(out,'procedure_completed',optimizer_runs=2,arm=arm,**identity)
    return pred,dict(selector=selector.metadata(),refit=refit.metadata())


def worker(out,seed,fold,target):
    out=Path(out);m=read(out/'manifest.json');verify(m['files']);spec=m['spec']
    if runtime()!=m['runtime'] or seed not in spec['development_seeds'] or fold not in range(5) or target not in TARGETS:raise ValueError('Unregistered runtime or unit')
    activation=read(out/'activation.json')
    if activation['manifest_sha256']!=sha(out/'manifest.json'):raise ValueError('Activation differs')
    unit=out/f'{target}-s{seed}-f{fold}';unit.mkdir(exist_ok=False)
    identity=dict(source_directory=str(out),seed=seed,fold=fold,target=target,trial_id='DCN_CROSS_A20')
    event(out,'unit_reserved',**identity)
    try:
        frame=load_labels(out,m);fv,ref=reference(m,frame,seed);j=TARGETS.index(target)
        training=frame.loc[fv!=fold];query=clean(frame.loc[fv==fold]);fitting,calibration=partitions(training)
        plan=dict(training=digest(training.sample_id.tolist()),query=digest(query.sample_id.tolist()),
                  fitting=digest(fitting.sample_id.tolist()),calibration=digest(calibration.sample_id.tolist()))
        if plan!=activation['plan'][f's{seed}-f{fold}']:raise ValueError('Frozen partitions differ')
        outputs={};meta={}
        for arm in ARMS:
            outputs[arm],meta[arm]=fit_arm(out,unit,fitting,calibration,training,query,target,arm,spec['training'],identity)
        for phase in ('selector','refit'):
            a,b=meta['ADDITIVE'][phase],meta['CROSS'][phase]
            if a['parameter_count']!=b['parameter_count'] or a['initial_state_digest']!=b['initial_state_digest']:raise ValueError('Matched arm initialization differs')
        endpoints={arm:.8*ref[fv==fold,j]+.2*pred for arm,pred in outputs.items()}
        if not np.isfinite(list(endpoints.values())).all() or any((p<0).any() for p in endpoints.values()):raise ValueError('Invalid endpoint; no clipping')
        with (unit/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream,query_ids=query.sample_id.to_numpy(str),reference=ref[fv==fold,j],
                candidate=endpoints['CROSS'],control=endpoints['ADDITIVE'])
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak>spec['max_rss_mib']:raise ValueError('Worker memory gate failed')
        verify(m['files'])
        write(unit/'complete.json',dict(identity=identity,partitions=plan,peak_rss_mib=peak,optimizer_runs=4,
            manifest_sha256=sha(out/'manifest.json'),hashes={str(p.relative_to(unit)):sha(p) for p in unit.rglob('*') if p.is_file()}))
        event(out,'unit_completed',**identity)
    except BaseException as error:
        write(unit/'failure.json',dict(error=repr(error)));event(out,'unit_failed',error=repr(error),**identity);raise


def audit_arm(unit,fitting,calibration,training,query,target,arm,settings,hashes):
    directory=unit/arm;traces=read(directory/'traces.json')
    meta={}
    for phase,part in [('selector',fitting),('refit',training)]:
        h=hashes[f'{arm}/{phase}.json'];path=directory/(phase+'.json')
        meta[phase]=verify_model(path,part,part[target].to_numpy(),arm,settings,h,traces[phase])
        width=len(FEATURES)+len(meta[phase]['preprocessing']['categories'])
        if state_digest(CrossNetwork(width,settings,arm).state_dict())!=meta[phase]['initial_state_digest']:raise ValueError('Independent initial state differs')
    if meta['selector']['selected_epoch']!=meta['refit']['selected_epoch']:raise ValueError('Fresh refit selected epoch differs')
    with np.load(directory/'predictions.npz',allow_pickle=False) as a:
        pred,cp=a['query'].copy(),a['calibration'].copy();np.testing.assert_array_equal(a['query_ids'],query.sample_id.to_numpy(str))
    selector_path=directory/'selector.json';refit_path=directory/'refit.json'
    cal=independent_prediction(selector_path,clean(calibration),hashes[f'{arm}/selector.json'])
    checkpoint=float(np.abs(cal-calibration[target].to_numpy()).mean()/meta['selector']['target_std'])
    expected=meta['selector']['history'][meta['selector']['selected_epoch']-1]['calibration_standardized_mae']
    if abs(checkpoint-expected)>1e-10:raise ValueError('Chosen checkpoint validation MAE differs')
    model=CrossRegressor.load(refit_path,hashes[f'{arm}/refit.json'])
    predictions=[model.predict(query),model.predict(query.iloc[::-1])[::-1],
                 np.concatenate([model.predict(query.iloc[i:i+73]) for i in range(0,len(query),73)]),
                 independent_prediction(refit_path,query,hashes[f'{arm}/refit.json'])]
    differences=[float(np.max(abs(p-pred))) for p in predictions]+[float(np.max(abs(cal-cp)))]
    if max(differences)>1e-10:raise ValueError('Independent cold/order/chunk prediction differs')
    return dict(maximum_difference=max(differences),checkpoint_mae_difference=abs(checkpoint-expected),
                selector=meta['selector'],refit=meta['refit']),pred


def cold(out):
    out=Path(out);m=read(out/'manifest.json');verify(m['files']);runtime();frame=load_labels(out,m);records={}
    for seed in m['spec']['development_seeds']:
        fv,ref=reference(m,frame,seed)
        for fold in range(5):
            training=frame.loc[fv!=fold];query=clean(frame.loc[fv==fold]);fitting,calibration=partitions(training)
            for j,target in enumerate(TARGETS):
                unit=out/f'{target}-s{seed}-f{fold}';complete=read(unit/'complete.json')
                if complete['manifest_sha256']!=sha(out/'manifest.json'):raise ValueError('Unit manifest differs')
                verify({str(unit/n):h for n,h in complete['hashes'].items()});values={};details={}
                for arm in ARMS:details[arm],values[arm]=audit_arm(unit,fitting,calibration,training,query,target,arm,m['spec']['training'],complete['hashes'])
                with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['query_ids'],query.sample_id.to_numpy(str));np.testing.assert_array_equal(saved['reference'],ref[fv==fold,j])
                    for name,arm in [('candidate','CROSS'),('control','ADDITIVE')]:np.testing.assert_array_equal(saved[name],.8*saved['reference']+.2*values[arm])
                for phase in ('selector','refit'):
                    a,b=details['ADDITIVE'][phase],details['CROSS'][phase]
                    if a['initial_state_digest']!=b['initial_state_digest'] or a['parameter_count']!=b['parameter_count']:raise ValueError('Cold matched arms differ')
                records[unit.name]=dict(maximum_difference=max(d['maximum_difference'] for d in details.values()),
                    selected_epochs={a:d['selector']['selected_epoch'] for a,d in details.items()},
                    hashes=complete['hashes'],cold_models=4)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>m['spec']['max_rss_mib']:raise ValueError('Cold memory gate failed')
    verify(m['files']);write(out/'cold.json',dict(status='passed',records=records,cold_models=80,new_fits=0,
        optimizer_runs=0,peak_rss_mib=peak,manifest_sha256=sha(out/'manifest.json')))


def decisions(records,spec):
    result={}
    for target,rows in records.items():
        if set(rows)!={'42','3407'} or any(not np.isfinite([v['gain'],v['mechanism_gain'],v['working_score']]).all() for v in rows.values()):
            raise ValueError('Two complete finite split records required')
        mean=float(np.mean([v['gain'] for v in rows.values()]));mechanism=float(np.mean([v['mechanism_gain'] for v in rows.values()]));score=float(np.mean([v['working_score'] for v in rows.values()]))
        checks={
            'both_complete_seeds_positive':all(v['gain']>0 for v in rows.values()),
            'mean_gain_at_least_0_01':mean>=spec['development_mean_gain_minimum'],
            'positive_matched_control_advantage':mechanism>0,'mean_local_working_score_at_least_96_25':score>=spec['minimum_local_working_score']}
        result[target]=dict(eligible=all(checks.values()),checks=checks,mean_gain=mean,mean_mechanism_gain=mechanism,
                            mean_local_working_score=score,failure_reasons=[k for k,v in checks.items() if not v])
    return result


def summarize(out):
    from .candidate_tiers import classify_candidates
    from .v49_run import metric_detail
    out=Path(out);m=read(out/'manifest.json');verify(m['files'])
    if read(out/'cold.json')['status']!='passed':raise ValueError('Cold audit must pass before scoring')
    frame=load_labels(out,m);records={};metrics={t:{a:{} for a in ('Q75','DCN_CROSS_A20')} for t in TARGETS}
    for seed in m['spec']['development_seeds']:
        fv,ref=reference(m,frame,seed)
        for j,target in enumerate(TARGETS):
            a=np.full(len(frame),np.nan);b=a.copy();raw=a.copy()
            for fold in range(5):
                unit=out/f'{target}-s{seed}-f{fold}'
                with np.load(unit/'predictions.npz',allow_pickle=False) as s:a[fv==fold]=s['candidate'];b[fv==fold]=s['control']
                with np.load(unit/'CROSS/predictions.npz',allow_pickle=False) as s:raw[fv==fold]=s['query']
            if not np.isfinite([a,b,raw]).all():raise ValueError('Complete same-seed OOF required')
            y=frame[target].to_numpy();den=math.fsum(map(float,y))
            errors=[math.fsum(abs(float(v)-float(p)) for v,p in zip(y,pred)) for pred in (ref[:,j],a,b)]
            gain=50*(errors[0]-errors[1])/den;control=50*(errors[0]-errors[2])/den
            fast=50*(np.sum(abs(y-ref[:,j]))-np.sum(abs(y-a)))/den
            if abs(fast-gain)>1e-10:raise ValueError('Independent scalar/vector scoring differs')
            other=TARGETS[1-j];oy=frame[other].to_numpy()
            other_wmape=math.fsum(abs(float(v)-float(p)) for v,p in zip(oy,ref[:,1-j]))/math.fsum(map(float,oy))
            records.setdefault(target,{})[str(seed)]=dict(gain=gain,control_gain=control,
                mechanism_gain=50*(errors[2]-errors[1])/den,working_score=100-50*(errors[1]/den+other_wmape),
                standalone_wmape=math.fsum(abs(float(v)-float(p)) for v,p in zip(y,raw))/den)
            for name,pred in [('Q75',ref[:,j]),('DCN_CROSS_A20',a)]:metrics[target][name][str(seed)]=metric_detail(y,pred,fv,frame.spout_no.to_numpy())
    tier_spec=dict(split_seeds=[42,3407],folds=5,candidates={t:['DCN_CROSS_A20'] for t in TARGETS},
        reference_by_target={t:'Q75' for t in TARGETS},tie_preference_by_target={t:['DCN_CROSS_A20'] for t in TARGETS})
    tiers=classify_candidates(metrics,tier_spec,yaml.safe_load((Path(m['spec']['main_root'])/'configs/candidate_tiers.yaml').read_text()))
    events=[json.loads(v) for v in (out/'events.jsonl').read_text().splitlines()]
    expected=dict(unit_reserved=20,unit_completed=20,procedure_reserved=40,procedure_completed=40,optimizer_started=80,optimizer_initialized=80)
    counts={k:sum(e['event']==k for e in events) for k in expected}
    if counts!=expected or any(e['event'].endswith('_failed') for e in events):raise ValueError('Budget did not close')
    write(out/'summary.json',dict(status='complete_development',records=records,decisions=decisions(records,m['spec']),tiers=tiers,
        counts=counts,optimizer_runs=80,saved_states=80,new_reference_fits=0,new_confirmation_seeds=0,
        formal_promoted=False,manifest_sha256=sha(out/'manifest.json'),cold_sha256=sha(out/'cold.json'),
        full_data_fits=0,packages=0,desktop_writes=0,uploads=0))


def execute(out):
    out=Path(out);m=read(out/'manifest.json');started=time.monotonic();event(out,'controller_started',pid=os.getpid())
    try:
        for seed in m['spec']['development_seeds']:
            for fold in range(5):
                for target in TARGETS:
                    with (out/f'{target}-s{seed}-f{fold}.log').open('x') as log:
                        subprocess.run([sys.executable,'-m',__spec__.name,'worker',str(out),'--seed',str(seed),'--fold',str(fold),'--target',target],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/'cold.log').open('x') as log:result=subprocess.run([sys.executable,'-m',__spec__.name,'cold',str(out)],stdout=log,stderr=subprocess.STDOUT)
        write(out/'cold-process-exit.json',dict(exit_code=result.returncode))
        if result.returncode:raise RuntimeError('Independent cold audit failed')
        summarize(out);verify(m['files']);event(out,'controller_completed',pid=os.getpid())
        write(out/'terminal.json',dict(status='passed',controller_pid=os.getpid(),cold_exit_code=result.returncode,
            manifest_sha256=sha(out/'manifest.json'),cold_sha256=sha(out/'cold.json'),summary_sha256=sha(out/'summary.json'),
            events_sha256=sha(out/'events.jsonl'),seconds_descriptive=time.monotonic()-started))
    except BaseException as error:event(out,'controller_failed',pid=os.getpid(),error=repr(error));write(out/'failure.json',dict(error=repr(error)));raise


def synthetic():
    rng=np.random.default_rng(63002);x=rng.normal(size=(2754,len(FEATURES)));frame=pd.DataFrame(x,columns=FEATURES)
    frame.insert(0,'sample_id',[f'dcn-full-synthetic-{i:05d}' for i in range(len(x))]);frame.insert(1,'spout_no',np.arange(len(x))%4+1)
    frame['tap_iron']=200+3*x[:,0]+4*x[:,1]*x[:,2]+2*x[:,3]*x[:,4]*x[:,5]+.1*rng.normal(size=len(x))
    frame['tap_time_len']=40+x[:,6]+.3*x[:,7]*x[:,8]
    return frame.iloc[:1762],frame.iloc[1762:2203],frame.iloc[:2203],clean(frame.iloc[2203:]),frame.tap_iron.to_numpy()[2203:]


def probe(work,out):
    work,out=Path(work).resolve(),Path(out).resolve();out.mkdir(parents=True,exist_ok=False);spec=read(work/SPEC)
    write(out/'manifest.json',dict(spec=spec,sources=sources(work),runtime=runtime()))
    fitting,calibration,training,query,qy=synthetic();unit=out/'pair';unit.mkdir();records={}
    for arm in ARMS:
        _,records[arm]=fit_arm(out,unit,fitting,calibration,training,query,'tap_iron',arm,spec['training'],dict(trial_id='synthetic-full-size',source_directory=str(out)))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>1024:raise ValueError('Synthetic memory gate failed')
    write(unit/'complete.json',dict(optimizer_runs=4,peak_rss_mib=peak,
        hashes={str(p.relative_to(unit)):sha(p) for p in unit.rglob('*') if p.is_file()}))
    with (out/'cold.log').open('x') as log:result=subprocess.run([sys.executable,'-m',__spec__.name,'probe-cold',str(out)],stdout=log,stderr=subprocess.STDOUT)
    write(out/'cold-process-exit.json',dict(exit_code=result.returncode))
    if result.returncode:raise RuntimeError('Synthetic cold audit failed')


def probe_cold(out):
    out=Path(out);m=read(out/'manifest.json');verify(m['sources']);fitting,calibration,training,query,qy=synthetic()
    unit=out/'pair';complete=read(unit/'complete.json');verify({str(unit/n):h for n,h in complete['hashes'].items()});rows={}
    for arm in ARMS:
        a,pred=audit_arm(unit,fitting,calibration,training,query,'tap_iron',arm,m['spec']['training'],complete['hashes'])
        rows[arm]=dict(maximum_difference=a['maximum_difference'],query_mae=float(abs(pred-qy).mean()),
            constant_mae=float(abs(training.tap_iron.mean()-qy).mean()),selected_epoch=a['selector']['selected_epoch'],
            stopped_epoch=a['selector']['stopped_epoch'],parameter_count=a['refit']['parameter_count'],initial_state_digest=a['refit']['initial_state_digest'])
    if rows['ADDITIVE']['initial_state_digest']!=rows['CROSS']['initial_state_digest'] or rows['ADDITIVE']['parameter_count']!=rows['CROSS']['parameter_count']:raise ValueError('Synthetic matched arms differ')
    if any(v['query_mae']>=v['constant_mae'] for v in rows.values()):raise ValueError('Synthetic learnability admission failed')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>1024:raise ValueError('Synthetic cold memory gate failed')
    write(out/'cold.json',dict(status='passed',rows=rows,optimizer_runs=4,cold_models=4,audit_new_fits=0,
        peak_rss_mib=peak,manifest_sha256=sha(out/'manifest.json'),warm_sha256=sha(unit/'complete.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','execute','worker','cold','probe','probe-cold'));p.add_argument('output')
    p.add_argument('--work');p.add_argument('--tests');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);p.add_argument('--target')
    a=p.parse_args();runtime()
    if a.action=='freeze':freeze(a.work,a.output,a.tests)
    elif a.action=='worker':worker(a.output,a.seed,a.fold,a.target)
    elif a.action=='cold':cold(a.output)
    elif a.action=='probe':probe(a.work,a.output)
    elif a.action=='probe-cold':probe_cold(a.output)
    else:execute(a.output)


if __name__=='__main__':main()
