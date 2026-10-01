"""SHORT_SPAN four-split confirmation with frozen matching Q75 references.

Preparation requires an exact-source full test receipt and committed source.
Workers cannot run before activation. This controller never fits all 2754 rows
or produces a submission, and never retries a consumed unit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

from .data import FEATURES,TARGETS
from .ema_average_span import bind_original_sources,candidate_column,old_cache,task_frames
from .ema_evaluation_diagnostics import verify_files
from .ema_reference_artifacts import sha,write_new
from .ema_reference_capture import ReferenceCapture,capture_original_factory
from .ema_reference_ledger import KINDS,binding_sources,reference_bindings
from .ema_reference_plan import reference_plan
from .ema_span_confirmation_models import fit_component
from .ema_training_scale import event,require_memory
from .v3_4_bags import group_safe_inner_folds
from .v5_library import fold_vector,load_v5_training_frame
from .v5_spec import load_v5_spec
from .v7_periodic import digest

SPEC='configs/ema_span_confirmation/SPEC.json'
DEVELOPMENT=[42,3407]
CONFIRMATION=[271828,314159]
NATIVE_PER_REFERENCE=dict(catboost_fit=29,ebm_fit=6,ebm_boost=48,sklearn_mlp_fit=1,torch_optimizer=6)


def validate_scope(spec,original):
    expected=dict(candidate='SHORT_SPAN',target='tap_time_len',thread_target=96.45,
        development_seeds=DEVELOPMENT,confirmation_seeds=CONFIRMATION,folds=5,
        candidate_beta=.9801,control_beta=.99,replacement_weight=.75,
        workers=1,numerical_threads=1,monitor_seconds=600,max_worker_rss_mib=1536,
        cold_predict_atol=.0005,reference_factory_calls=10,reference_pipeline_instances=320,
        new_component_estimators=20,new_confirmation_seeds=2,
        maximum_runtime_seconds=None,automatic_scientific_retries=False,
        full_data_fits=0,packages=0,desktop_writes=0,agent_uploads=0,platform_queue_additions=0)
    if any(spec.get(k)!=v or type(spec.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError('Unregistered confirmation scope')
    if spec['training']!=original['training']['tap_time_len'] or spec['mechanisms']!=original['mechanisms']:
        raise ValueError('Original EMA scientific trainer changed')
    if spec['reference_spec']['reference']!=original['reference'] or spec['reference_spec']['budget']!={
            'reference_workers':1,'b0_pipeline_fit_calls_per_factory':32}:
        raise ValueError('Matching reference recipe or worker scope changed')
    expected_calls={k:10*n for k,n in NATIVE_PER_REFERENCE.items()}
    expected_calls['torch_optimizer']+=40
    if spec['native_budget']!=expected_calls:raise ValueError('Confirmation native budget changed')
    if spec['formal_gate']!=dict(minimum_complete_seeds=4,every_seed_gain_positive=True,
        seed_paired_lcb95_positive=True,lcb_method='one_sided_Student_t_95_percent'):
        raise ValueError('Four-seed confirmation gate changed')


def sources(workspace):
    workspace=Path(workspace)
    paths=list((workspace/'src').rglob('*.py'))+list((workspace/'tests').glob('*.py'))
    for suffix in ('*.yaml','*.json'):paths+=list((workspace/'configs').rglob(suffix))
    paths+=[workspace/p for p in ('uv.lock','pyproject.toml',
        'docs/ema_span_confirmation/PREREGISTRATION.md','scripts/ema_span_confirmation.py',
        'scripts/observe_ema_span_confirmation.py')]
    return {str(p.relative_to(workspace)):sha(p) for p in sorted(set(paths))}


def runtime(spec,*,before_fork=False):
    from .v49_run import check_runtime
    import torch
    if sys.version_info[:2]!=(3,12):raise ValueError('Locked Python3.12 required')
    versions=check_runtime(spec)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    if before_fork and threading.active_count()!=1:
        raise ValueError('Fresh single-thread Python process required before the original fork worker')
    return versions


def partition(training,query,settings):
    from .ema_training_scale import assert_isolated
    assert_isolated(training,query)
    inner=np.asarray(group_safe_inner_folds(training,seed=settings['inner_seed'])['fold'])
    return dict(training=digest(training.sample_id.tolist()),query=digest(query.sample_id.tolist()),
        training_frame=digest(training.to_dict(orient='list')),query_frame=digest(query.to_dict(orient='list')),
        inner_fit=digest(training.loc[inner!=0,'sample_id'].tolist()),
        inner_validation=digest(training.loc[inner==0,'sample_id'].tolist()),training_rows=len(training),query_rows=len(query))


def freeze_composition(main,plan):
    """Propagate basis vectors through original fixed A composition, without fit.

    B36's original nonnegative projection remains explicit, outside the linear
    A graph. It cannot be folded into basis coefficients or added to candidates.
    """
    from .v4_1_reference import V36FixedRecipeFactory
    factory=V36FixedRecipeFactory(main,workers=1);a=factory.a_factory
    slots=[]
    for role in plan['roles']:
        outputs=2 if role['family'] in {'legacy_joint_tree','legacy_joint_mlp_lbfgs'} or role['name']=='V12_joint' else 1
        slots.extend(dict(name=role['name'],column=k,outputs=outputs) for k in range(outputs))
    matrix=np.eye(len(slots));store={}
    for i,slot in enumerate(slots):
        if slot['column']!=0:continue
        count=slot['outputs'];store[slot['name']]=matrix[:,i] if count==1 else matrix[:,i:i+count]
    bases={n:store[n] for n in a._recovery.BASE_NAMES}
    combined={**bases,**a._recovery.comp(bases)};coefficients={}
    for target in TARGETS:
        names=a._l1_pool[target];key='iron_weights' if target=='tap_iron' else 'time_weights'
        weights=np.array([a._l1_weights[key][n] for n in names],float);weights/=weights.sum()
        l1=np.column_stack([a._recovery.get(combined,n,target) for n in names])@weights
        parts=[w*(l1 if n=='L1' else store[n]) for n,w in zip(a._target_members[target],a._target_weights[target])]
        coefficients[target]=np.sum(parts,axis=0).tolist()
    return dict(slots=slots,a_coefficients=coefficients,b_weights={t:factory.weights[t].tolist() for t in TARGETS},
        b_experts={t:list(factory.selected_experts[t]) for t in TARGETS},existing_b36_nonnegative_projection=True)


def manifest_context(workspace,out,*,load_data=True):
    workspace=Path(workspace).resolve();out=Path(out).resolve()
    manifest=json.loads((out/'manifest.json').read_text());spec=manifest['spec']
    if manifest['workspace']!=str(workspace) or manifest['output']!=str(out):
        raise ValueError('Confirmation workspace/output identity changed')
    if digest({k:v for k,v in manifest.items() if k!='identity'})!=manifest['identity']:
        raise ValueError('Confirmation manifest identity changed')
    verify_files(workspace,manifest['sources'])
    for p,h in manifest['model_sources'].items():
        if sha(p)!=h:raise ValueError('Actual matching model source changed')
    original=yaml.safe_load((workspace/'configs/strong_component_regularization/SPEC.yaml').read_text())
    validate_scope(spec,original);runtime(spec)
    if not load_data:return manifest,None,None
    main=Path(spec['main_root']);verify_files(main,spec['inputs'])
    frame=load_v5_training_frame(main)
    folds={s:fold_vector(main,frame,s,load_v5_spec(main)) for s in [*DEVELOPMENT,*CONFIRMATION]}
    if {str(s):digest(f.tolist()) for s,f in folds.items()}!=manifest['fold_digests']:
        raise ValueError('Frozen confirmation fold identity changed')
    return manifest,frame,folds


def development_vectors(spec,frame,folds):
    """Reuse original source/seed/trial identities; never relabel caches as fits."""
    out=Path(spec['development_source']);manifest=json.loads((out/'manifest.json').read_text())
    audit=json.loads((out/'audit.json').read_text());summary=json.loads((out/'summary.json').read_text())
    if (audit['status']!='passed' or audit['manifest_sha256']!=sha(out/'manifest.json')
            or audit['summary_sha256']!=sha(out/'summary.json') or summary['selected_for_confirmation']!='SHORT_SPAN'
            or min(summary['gains']['SHORT_SPAN'].values())<=0):
        raise ValueError('Complete audited positive development qualification required')
    result={}
    for seed in DEVELOPMENT:
        fv=folds[seed];vectors={k:np.full(len(frame),np.nan) for k in ('q75','candidate','iron')}
        for fold in range(5):
            training,query=task_frames(frame,fv,fold);cache=old_cache(manifest['spec'],seed,fold,training,query)
            directory=out/f'SHORT_SPAN-s{seed}-f{fold}'
            complete=json.loads((directory/'complete.json').read_text())
            if complete['manifest_sha256']!=sha(out/'manifest.json'):raise ValueError('Development unit identity changed')
            verify_files(directory,complete['hashes'])
            with np.load(directory/'predictions.npz',allow_pickle=False) as a:
                if a['query_ids'].tolist()!=query.sample_id.tolist():raise ValueError('Development OOF query changed')
                np.testing.assert_array_equal(a['q75'],cache['q75']);np.testing.assert_array_equal(a['old_ema'],cache['ema'])
                np.testing.assert_array_equal(a['iron_reference'],cache['iron'])
                np.testing.assert_array_equal(a['candidate'],candidate_column(cache['q75'],cache['ema'],a['ema']))
                for key,saved in [('q75','q75'),('candidate','candidate'),('iron','iron_reference')]:
                    vectors[key][fv==fold]=a[saved]
        if any(not np.isfinite(v).all() for v in vectors.values()):raise ValueError('Development OOF incomplete')
        result[seed]=vectors
    return result


def prepare(workspace,receipt):
    import psutil
    workspace=Path(workspace).resolve();spec=json.loads((workspace/SPEC).read_text());main=Path(spec['main_root'])
    validate_scope(spec,yaml.safe_load((workspace/'configs/strong_component_regularization/SPEC.yaml').read_text()))
    source_hashes=sources(workspace);tests=json.loads(Path(receipt).read_text())
    if tests.get('status')!='passed' or tests.get('sources')!=source_hashes or tests.get('full_suite') is not True:
        raise ValueError('Exact-source complete locked tests required')
    if subprocess.check_output(['git','status','--porcelain','--',*source_hashes],cwd=workspace,text=True).strip():
        raise ValueError('Commit tested confirmation source before preparation')
    versions=runtime(spec,before_fork=True);require_memory(spec);verify_files(main,spec['inputs'])
    terminal=Path(spec['development_source'])/'terminal-verification-r1.json'
    terminal_record=json.loads(terminal.read_text())
    if terminal_record['status']!='passed' or terminal_record['controller_exit_code']!=0:
        raise ValueError('Actual previous successful terminal required')
    current=json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current.get(k)!=v for k,v in spec['platform_reference'].items()):
        raise ValueError('Current platform reference changed before scientific freeze')
    own=set([os.getpid(),*[p.pid for p in psutil.Process().parents()]])
    names={'ema_average_span.py','observe_ema_average_span.py','ema_span_confirmation.py','observe_ema_span_confirmation.py'}
    for p in psutil.process_iter(['pid','cmdline']):
        if p.info['pid'] not in own and any(Path(x).name in names for x in (p.info['cmdline'] or [])):
            raise ValueError('Serial scientific dependency still active')
    old_manifest=json.loads((Path(spec['development_source'])/'manifest.json').read_text())
    bind_original_sources(workspace,main,old_manifest['sources'])
    frame=load_v5_training_frame(main)
    folds={s:fold_vector(main,frame,s,load_v5_spec(main)) for s in [*DEVELOPMENT,*CONFIRMATION]}
    development_vectors(spec,frame,folds)
    plan=reference_plan(main,spec['reference_spec'],eligible_spouts=[1,2])
    if plan['expected_native_calls_per_factory']!=NATIVE_PER_REFERENCE:raise ValueError('Actual reference native proposal changed')
    partitions={}
    for seed in CONFIRMATION:
        for fold in range(5):
            training,query=task_frames(frame,folds[seed],fold)
            for role in plan['roles']:
                if role['family']=='A_global_spout_shrink':
                    eligible=sorted(int(k) for k,v in training.spout_no.value_counts().items() if v>=role['minimum_spout_samples'])
                    if eligible!=role['eligible_spouts']:raise ValueError('Frozen shrink native population changed')
            partitions[f's{seed}-f{fold}']=partition(training,query,spec['training'])
    model_sources={str((workspace/p).resolve()):h for p,h in source_hashes.items()}
    # The unchanged private recovery script inserts main/src in sys.path.
    # Bind that actual source location as well as this isolated worktree.
    model_sources.update({str(p.resolve()):sha(p) for p in (main/'src').rglob('*.py')})
    for p in spec['inputs']:
        if Path(p).suffix in {'.py','.yaml','.json','.jsonl','.toml','.lock'}:
            model_sources[str((main/p).resolve())]=spec['inputs'][p]
    model_sources.update(binding_sources(reference_bindings()))
    out=Path(spec['output']).resolve()
    if not out.is_relative_to(main/'local/runs') or out.exists():raise ValueError('Fresh private scientific output required')
    manifest=dict(spec=spec,sources=source_hashes,model_sources=model_sources,versions=versions,
        workspace=str(workspace),output=str(out),reference_plan=plan,partitions=partitions,
        reference_composition=freeze_composition(main,plan),
        fold_digests={str(s):digest(f.tolist()) for s,f in folds.items()},tests_receipt_sha256=sha(receipt),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),
        scientific_execution_admitted=True,created_ns=time.time_ns())
    manifest['identity']=digest(manifest);out.mkdir(parents=True,exist_ok=False);write_new(out/'manifest.json',manifest)
    return out


def warm_worker(workspace,out,seed,fold):
    from .v30_reference import fit_b0
    out=Path(out);manifest,frame,folds=manifest_context(workspace,out);spec=manifest['spec']
    activation=json.loads((out/'activation.json').read_text())
    if activation['manifest_sha256']!=sha(out/'manifest.json') or seed not in CONFIRMATION or fold not in range(5):
        raise ValueError('Unknown or unactivated scientific unit')
    runtime(spec,before_fork=True);require_memory(spec)
    unit=out/f's{seed}-f{fold}';unit.mkdir(exist_ok=False)
    write_new(unit/'start.json',dict(seed=seed,fold=fold,pid=os.getpid(),manifest_sha256=sha(out/'manifest.json')))
    event(out/'events.jsonl',dict(event='unit_started',seed=seed,fold=fold))
    try:
        training,query=task_frames(frame,folds[seed],fold)
        if partition(training,query,spec['training'])!=manifest['partitions'][unit.name]:
            raise ValueError('Declared scientific training/query partition changed')
        capture=ReferenceCapture(unit/'reference',source_directory=str(out.resolve()),split_seed=seed,fold=fold,
            training=training,query=query,plan=manifest['reference_plan'],source_hashes=manifest['model_sources'])
        with capture_original_factory(capture):values,reference_metadata=fit_b0(Path(spec['main_root']),training,query,spec['reference_spec'])
        capture.close();components={};receipts={}
        for name,beta in [('OLD_EMA',spec['control_beta']),('SHORT_SPAN',spec['candidate_beta'])]:
            identity=dict(source_directory=str(out.resolve()),split_seed=seed,fold=fold,trial_id=name)
            components[name],_=fit_component(unit/name,training,query,identity=identity,settings=spec['training'],
                mechanisms=dict(spec['mechanisms'],ema_beta=beta),source_hashes=manifest['model_sources'])
            receipts[name]=sha(unit/name/'complete.json')
        q75=values['tap_time_len']+.75*(components['OLD_EMA']-values['v7_time'])
        candidate=candidate_column(q75,components['OLD_EMA'],components['SHORT_SPAN'])
        if any(not np.isfinite(v).all() or (v<0).any() for v in (q75,candidate,values['tap_iron'])):
            raise ValueError('Invalid Q75/candidate extrapolation; no clipping')
        with (unit/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream,query_ids=query.sample_id.to_numpy(str),q75=q75,candidate=candidate,
                iron=values['tap_iron'],old_ema=components['OLD_EMA'],short_ema=components['SHORT_SPAN'],**values)
        peak=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)/1024
        if peak>spec['max_worker_rss_mib']:raise ValueError('Original worker memory gate failed')
        manifest_context(workspace,out)
        payload=dict(seed=seed,fold=fold,pid=os.getpid(),manifest_sha256=sha(out/'manifest.json'),
            reference_receipt_sha256=sha(unit/'reference/complete.json'),component_receipts=receipts,
            reference_metadata=reference_metadata,partition=manifest['partitions'][unit.name],peak_rss_mib=peak,
            predictions_sha256=sha(unit/'predictions.npz'),start_sha256=sha(unit/'start.json'))
        write_new(unit/'warm-complete.json',payload)
        event(out/'events.jsonl',dict(event='unit_warm_completed',seed=seed,fold=fold))
    except BaseException as error:
        write_new(unit/'failure.json',dict(error=repr(error)))
        event(out/'events.jsonl',dict(event='unit_failed',seed=seed,fold=fold,error=repr(error)));raise


def cold_worker(workspace,out,seed,fold,receipt):
    from .ema_reference_capture_audit import audit_capture
    from .ema_span_confirmation_models import audit_component
    out=Path(out);manifest,_,_=manifest_context(workspace,out,load_data=False)
    if seed not in CONFIRMATION or fold not in range(5):raise ValueError('Unknown cold unit')
    unit=out/f's{seed}-f{fold}'
    if sha(unit/'warm-complete.json')!=receipt:raise ValueError('External warm unit identity changed')
    warm=json.loads((unit/'warm-complete.json').read_text())
    if warm['pid']==os.getpid() or warm['manifest_sha256']!=sha(out/'manifest.json'):
        raise ValueError('Independent cold process/manifest required')
    reference=audit_capture(unit/'reference',warm['reference_receipt_sha256'])
    components={name:audit_component(unit/name,h) for name,h in warm['component_receipts'].items()}
    if reference['retained_states']!=40 or set(components)!={'OLD_EMA','SHORT_SPAN'}:
        raise ValueError('Cold retained state coverage changed')
    if sha(unit/'predictions.npz')!=warm['predictions_sha256'] or sha(unit/'start.json')!=warm['start_sha256']:
        raise ValueError('Cold unit prediction or start identity changed')
    arithmetic=verify_cold_composition(unit,manifest['reference_composition'])
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>manifest['spec']['max_worker_rss_mib']:raise ValueError('Cold worker memory gate failed')
    payload=dict(status='passed',warm_receipt_sha256=receipt,retained_states=44,new_fits=0,
        reference_cold_receipt_sha256=sha(unit/'reference/cold-complete.json'),
        component_cold_receipts={n:sha(unit/n/'cold-complete.json') for n in components},cold_peak_rss_mib=peak,
        maximum_composition_difference=arithmetic)
    write_new(unit/'cold-complete.json',payload)
    event(out/'events.jsonl',dict(event='unit_completed',seed=seed,fold=fold))
    return payload


def verify_cold_composition(unit,graph):
    """Independent aggregation of already cold-verified constituent outputs."""
    unit=Path(unit);store={}
    for slot in graph['slots']:
        name=slot['name']
        if name not in store:store[name]=np.load(unit/'reference'/name/'final/observed.npy',allow_pickle=False)
    matrix=np.column_stack([store[s['name']] if s['outputs']==1 else store[s['name']][:,s['column']] for s in graph['slots']])
    b={}
    if graph['existing_b36_nonnegative_projection'] is not True:raise ValueError('Original B36 projection changed')
    for target in TARGETS:
        a=matrix@np.asarray(graph['a_coefficients'][target]);w=graph['b_weights'][target]
        value=w[0]*a
        for weight,name in zip(w[1:],graph['b_experts'][target]):value=value+weight*store[name]
        b[target]=np.maximum(value,0)
    values=dict(v36_iron=b['tap_iron'],v36_time=b['tap_time_len'],v12_iron=store['V12_joint'][:,0],
        n_time=store['N0048'],v7_time=store['V7_periodic'])
    values['tap_iron']=.5*values['v36_iron']+.5*values['v12_iron']
    values['tap_time_len']=.2*values['v36_time']+.3*values['n_time']+.5*values['v7_time']
    for name,key in [('OLD_EMA','old_ema'),('SHORT_SPAN','short_ema')]:
        values[key]=np.load(unit/name/'refit-witness/observed.npy',allow_pickle=False)[:,0]
    values['q75']=values['tap_time_len']+.75*(values['old_ema']-values['v7_time'])
    values['candidate']=values['q75']+.75*(values['short_ema']-values['old_ema'])
    values['iron']=values['tap_iron'];maximum=0.
    with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
        for name,expected in values.items():
            actual=saved[name]
            if actual.shape!=expected.shape or not np.isfinite(expected).all():raise ValueError('Cold composition vector invalid')
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
    if maximum>1e-9:raise ValueError('Cold constituent/composition arithmetic mismatch')
    return maximum


def collect(manifest,frame,folds):
    spec=manifest['spec'];out=Path(manifest['output']);vectors=development_vectors(spec,frame,folds)
    for seed in CONFIRMATION:
        fv=folds[seed];v={k:np.full(len(frame),np.nan) for k in ('q75','candidate','iron')}
        for fold in range(5):
            unit=out/f's{seed}-f{fold}';warm=json.loads((unit/'warm-complete.json').read_text())
            cold=json.loads((unit/'cold-complete.json').read_text())
            if (cold['status']!='passed' or cold['warm_receipt_sha256']!=sha(unit/'warm-complete.json')
                    or warm['manifest_sha256']!=sha(out/'manifest.json') or (unit/'failure.json').exists()
                    or sha(unit/'predictions.npz')!=warm['predictions_sha256']):
                raise ValueError('Completed bound warm/cold scientific unit required')
            with np.load(unit/'predictions.npz',allow_pickle=False) as a:
                if a['query_ids'].tolist()!=frame.loc[fv==fold,'sample_id'].tolist():raise ValueError('Confirmation OOF row order changed')
                np.testing.assert_array_equal(a['q75'],a['tap_time_len']+.75*(a['old_ema']-a['v7_time']))
                np.testing.assert_array_equal(a['candidate'],candidate_column(a['q75'],a['old_ema'],a['short_ema']))
                np.testing.assert_array_equal(a['iron'],a['tap_iron'])
                for k in v:v[k][fv==fold]=a[k]
        if any(not np.isfinite(p).all() for p in v.values()):raise ValueError('Complete same-seed OOF required')
        vectors[seed]=v
    return vectors


def evaluate(frame,folds,vectors,policy):
    from .candidate_tiers import classify_candidates
    from .v49_run import metric_detail
    from .v5_resolution import paired_summary
    if set(vectors)!=set([*DEVELOPMENT,*CONFIRMATION]) or set(folds)!=set(vectors):
        raise ValueError('Exactly four complete distinct seed vectors required')
    y=frame.tap_time_len.to_numpy(float);yi=frame.tap_iron.to_numpy(float)
    if not np.isfinite(y).all() or np.abs(y).sum()<=0:raise ValueError('Valid metric targets required')
    metrics={'tap_time_len':{'Q75':{},'SHORT_SPAN':{}}};gains={};scores={}
    for seed in [*DEVELOPMENT,*CONFIRMATION]:
        v=vectors[seed]
        if set(v)!={'q75','candidate','iron'} or any(p.shape!=(len(frame),) or not np.isfinite(p).all() or (p<0).any() for p in v.values()):
            raise ValueError('Incomplete finite nonnegative same-seed OOF')
        for name,key in [('Q75','q75'),('SHORT_SPAN','candidate')]:
            metrics['tap_time_len'][name][str(seed)]=metric_detail(y,v[key],folds[seed],frame.spout_no.to_numpy())
        gains[str(seed)]=float(50*(np.abs(y-v['q75']).sum()-np.abs(y-v['candidate']).sum())/np.abs(y).sum())
        iron_wmape=float(np.abs(yi-v['iron']).sum()/np.abs(yi).sum())
        scores[str(seed)]={name:float(100-50*(m[str(seed)]['wmape']+iron_wmape)) for name,m in metrics['tap_time_len'].items()}
    paired=paired_summary(list(gains.values()))
    tiers={}
    for label,seeds in [('development',DEVELOPMENT),('confirmation_descriptive',CONFIRMATION)]:
        m={'tap_time_len':{name:{str(s):values[str(s)] for s in seeds} for name,values in metrics['tap_time_len'].items()}}
        tier_spec=dict(split_seeds=seeds,folds=5,candidates={'tap_time_len':['SHORT_SPAN']},
            tie_preference_by_target={'tap_time_len':['SHORT_SPAN']},reference_by_target={'tap_time_len':'Q75'})
        tiers[label]=classify_candidates(m,tier_spec,policy)
    return dict(gains=gains,paired=paired,metrics=metrics,package_scores=scores,tiers=tiers,
        four_seed_gate_passed=all(g>0 for g in gains.values()) and paired['lcb95']>0,
        G0='independent_final_audit_pending',G1='four_complete_seed_evidence',release_authorized=False,
        full_data_fits=0,packages=0,desktop_writes=0,agent_uploads=0,platform_queue_additions=0)


def execute(workspace,out):
    out=Path(out);manifest,_,_=manifest_context(workspace,out);spec=manifest['spec']
    current=json.loads((Path(spec['main_root'])/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current.get(k)!=v for k,v in spec['platform_reference'].items()):
        raise ValueError('Current reference changed before activation')
    write_new(out/'activation.json',dict(manifest_sha256=sha(out/'manifest.json'),started_ns=time.time_ns(),controller_pid=os.getpid()))
    command=[sys.executable,str(Path(workspace)/'scripts/ema_span_confirmation.py'),'--workspace',str(workspace),'--output',str(out)]
    try:
        for seed in CONFIRMATION:
            for fold in range(5):
                subprocess.run(command+['worker','--seed',str(seed),'--fold',str(fold)],check=True)
                receipt=sha(out/f's{seed}-f{fold}'/'warm-complete.json')
                subprocess.run(command+['cold','--seed',str(seed),'--fold',str(fold),'--receipt',receipt],check=True)
        manifest,frame,folds=manifest_context(workspace,out)
        summary=evaluate(frame,folds,collect(manifest,frame,folds),yaml.safe_load((Path(workspace)/'configs/candidate_tiers.yaml').read_text()))
        write_new(out/'summary.json',summary)
        subprocess.run(command+['audit'],check=True)
        write_new(out/'completion-event.json',dict(status='completed',completed_ns=time.time_ns(),
            summary_sha256=sha(out/'summary.json'),audit_sha256=sha(out/'audit.json')))
    except BaseException as error:
        write_new(out/'failure.json',dict(error=repr(error)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--receipt',type=Path,required=True)
    sub.add_parser('execute');sub.add_parser('audit')
    for name in ('worker','cold'):
        w=sub.add_parser(name);w.add_argument('--seed',type=int,required=True);w.add_argument('--fold',type=int,required=True)
        if name=='cold':w.add_argument('--receipt',required=True)
    a=p.parse_args()
    if a.command=='prepare':print(prepare(a.workspace,a.receipt))
    elif a.command=='execute':execute(a.workspace,a.output)
    elif a.command=='worker':warm_worker(a.workspace,a.output,a.seed,a.fold)
    elif a.command=='cold':print(json.dumps(cold_worker(a.workspace,a.output,a.seed,a.fold,a.receipt)))
    else:
        from .ema_span_confirmation_audit import audit
        print(json.dumps(audit(a.workspace,a.output)))
