"""Zero-fit current-reference review of retained V10 predictions, not model replay."""
import argparse
import hashlib
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

from .candidate_tiers import classify_candidates
from .data import FEATURES
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .q75_gaussian_confirmation import read
from .q75_gaussian_time import compose, forbidden, reference_order
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v7_periodic import PeriodicRegressor, digest
from .v10_periodic_loss import LossRegressor
from .v33_run import metric_detail

WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/q75_retained_v10_mae/SPEC.json'
PROTOCOL = 'docs/q75_retained_v10_mae/PREREGISTRATION.md'
SEEDS = (42,3407)
CANDIDATE = 'V10_MAE_Q75_A20'
CONTROL = 'V7_MSE_Q75_A20'
SOURCE_NAMES = ('v10_periodic_loss.py','v7_periodic.py','v3_6_networks.py',
                'v3_4_bags.py','v5_library.py','v5_spec.py','data.py')


def validate_spec(spec):
    expected=dict(candidate=CANDIDATE,control=CONTROL,split_seeds=list(SEEDS),folds=5,weight=.2,
        new_fits=0,optimizer_runs=0,new_states=0,cold_states=0,engineering_optimizer_runs=0,
        confirmation_fits=0,full_data_fits=0,packages=0,desktop_writes=0,agent_uploads=0,
        workers=1,numerical_threads=1,torch_interop_threads=1,monitor_seconds=600,
        maximum_runtime_seconds=None,automatic_retries=False,automatic_confirmation=False)
    if any(spec.get(k)!=v for k,v in expected.items()):raise ValueError('Unregistered zero-fit scope')
    if spec['candidates']!={'tap_time_len':[CANDIDATE]}:raise ValueError('Single frozen candidate required')


def decision(gains,over_ready,over_mse):
    for values in (gains,over_ready,over_mse):
        if set(values)!=set(map(str,SEEDS)) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Two complete finite paired seeds required')
    failed=[]
    if not all(v>0 for v in gains.values()):failed.append('not_both_positive_vs_Q75')
    if not all(v>0 for v in over_ready.values()):failed.append('not_both_positive_vs_ready_Laplace')
    if math.fsum(over_mse.values())/2<=0:failed.append('nonpositive_mean_vs_matched_MSE')
    return dict(fresh_engineered_development_worth_registering=not failed,failed_conditions=failed,
        confirmation_eligible=False,formal_promoted=False,release_eligible=False,
        evidence_limit='retained_predictions_only_no_scientific_checkpoint_replay')


def ledger_entries(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    keys=[row.get('key') for row in rows]
    if len(keys)!=len(set(keys)) or any(row.get('event')!='complete' for row in rows):
        raise ValueError('Complete unique original ledger entries required')
    return {row['key']:row for row in rows}


def checked_vector(path,entry,expected_rows):
    if sha(path)!=entry['prediction_sha256']:raise ValueError('Original prediction identity changed')
    values=np.load(path,allow_pickle=False)
    if values.shape!=(expected_rows,) or not np.isfinite(values).all():
        raise ValueError('Aligned finite original fold vector required')
    return values


def prohibit_training():
    torch.optim.Optimizer.__init__=torch.optim.AdamW=torch.optim.Adam=forbidden
    PeriodicRegressor.fit=PeriodicRegressor._initialize=PeriodicRegressor._train=forbidden
    LossRegressor.fit=LossRegressor._train=NumericPreprocessor.fit=forbidden


def access(out,stage):
    with (out/'access.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage,time_ns=time.time_ns(),
            frozen_files_sha256=sha(out/'frozen-files.json'),scope='authorized_round2_only',
            protected_prelim_targets_read=False,new_fits=0))+'\n')
        stream.flush();os.fsync(stream.fileno())


def freeze(out,checks):
    spec=read(WORK/SPEC);validate_spec(spec);main=Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main/'local/runs'):raise ValueError('Fresh private output required')
    versions=runtime(spec);torch.set_num_interop_threads(1);prohibit_training()
    current=read(main/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(current[k]!=v for k,v in spec['reference'].items()):raise ValueError('Current reference changed')
    receipt=read(checks)
    if (receipt['exit_code']!=0 or receipt['optimizer_constructor_attempts']!=0 or receipt['optimizer_runs']!=0
            or not receipt['python_version'].startswith('3.12') or sha(receipt['junit_path'])!=receipt['junit_sha256']):
        raise ValueError('Locked zero-optimizer checks required')
    verify_files(receipt['source_hashes'])
    mae,mse,ref=[main/spec[key] for key in ('old_mae','old_mse','reference_run')]
    original=read(mae/'manifest.json');ma=read(mae/'audit-r1.json');vm=read(mse/'manifest.json');va=read(mse/'audit-r1.json')
    if (ma['status']!='passed' or ma['prediction_count']!=20 or ma['selected_confirmation']!='smooth_l1_01'
            or va['ledger_completions']!=120 or va['verified_prediction_hashes']!=120):
        raise ValueError('Original complete prediction audit required')
    if sha(WORK/spec['original_specification'])!=original['spec_sha256'] or sha(WORK/spec['original_mse_specification'])!=vm['spec_sha256']:
        raise ValueError('Historical specifications changed')
    for name in SOURCE_NAMES:
        if sha(WORK/'src/bf_tap_r2'/name)!=original['dependency_code_hashes'][name]:
            raise ValueError('Historical scientific source changed: '+name)
    if (sha(WORK/'src/bf_tap_r2/v10_periodic_loss.py')!=original['code_sha256']
            or sha(WORK/'src/bf_tap_r2/v7_periodic.py')!=vm['code_sha256']
            or any(versions[k]!=v for old in (original,vm) for k,v in old['versions'].items())):
        raise ValueError('Original code/runtime identity differs')
    rm=read(ref/'manifest.json');rt=read(ref/'terminal.json')
    if rt['status']!='passed' or rt['actual_exit_codes']!=[0]*4:raise ValueError('Complete audited current Q75/Laplace required')
    verify_files({str(ref/n):h for n,h in rt['artifacts'].items()})
    paths=list((WORK/'src').rglob('*.py'))+[WORK/p for p in (SPEC,PROTOCOL,'uv.lock','pyproject.toml',
        spec['original_specification'],spec['original_mse_specification'],'configs/candidate_tiers.yaml',
        'scripts/review_q75_retained_v10_mae.py','scripts/check_q75_retained_v10_mae.py',
        'scripts/observe_ema_fusion_selection.py','tests/test_q75_retained_v10_mae.py')]
    paths += [checks,Path(receipt['junit_path'])]
    for old in (mae,mse):paths += [old/n for n in ('manifest.json','audit-r1.json','summary.json','fit_ledger.jsonl')]
    for seed in SEEDS:
        paths.append(ref/f'oof-{seed}.npz')
        for fold in range(5):
            paths += [mae/f'tap_time_len-mae-s{seed}-f{fold}.npy',mse/f'tap_time_len-tabm_plr001-s{seed}-f{fold}.npy']
    paths += [ref/n for n in ('manifest.json','terminal.json','report.json','independent-score.json')]
    files={**rm['files'],**{str(p):sha(p) for p in paths}}
    verify_files(files)
    out.mkdir(parents=True,exist_ok=False);write_new(out/'frozen-files.json',files)
    write_new(out/'manifest.json',dict(spec=spec,files=files,versions=versions,
        sources={str(p):sha(p) for p in paths if p.is_relative_to(WORK)},
        frozen_files_sha256=sha(out/'frozen-files.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()))


def context(out):
    m=read(out/'manifest.json');validate_spec(m['spec']);verify_files(m['files'])
    if sha(out/'frozen-files.json')!=m['frozen_files_sha256']:raise ValueError('Frozen identity changed')
    runtime(m['spec']);torch.set_num_interop_threads(1);prohibit_training();access(out,'read_for_zero_fit_review')
    return m,m['spec']


def audit_metadata(frame,folds,fold,entry,settings,is_mae):
    training=frame.loc[folds!=fold].reset_index(drop=True)
    query=frame.loc[folds==fold].reset_index(drop=True)
    inner=group_safe_inner_folds(training,seed=settings['inner_seed'])['fold']
    fitting=training.loc[inner!=0].reset_index(drop=True);calibration=training.loc[inner==0].reset_index(drop=True)
    groups=lambda f:set(pd.util.hash_pandas_object(f[list(FEATURES)],index=False))
    if groups(training)&groups(query) or groups(fitting)&groups(calibration):raise ValueError('Group partition overlap')
    md=entry['metadata']
    if (md['fit_ids_digest']!=digest(training.sample_id.tolist()) or md['inner_ids_digest']!=digest(fitting.sample_id.tolist())
            or md['fit_rows']!=len(training) or md['inner_fit_rows']!=len(fitting) or md['optimizer_runs']!=2
            or not 1<=md['selected_epoch']<=settings['max_epochs']):
        raise ValueError('Original fit metadata does not match native partition')
    for name,part in [('inner_feature_means',fitting),('outer_feature_means',training)]:
        np.testing.assert_allclose(md[name],part[list(FEATURES)].to_numpy(float).mean(0),rtol=0,atol=1e-10)
    if is_mae and (md['loss']!='mae' or md['beta'] is not None or md['budget_limited']
            or md['selection_stopped_epoch']!=md['selected_epoch']+settings['patience']):
        raise ValueError('Original MAE recipe or recorded stop metadata differs')
    return dict(fit_rows=len(training),query_rows=len(query),inner_fit_rows=len(fitting),selected_epoch=md['selected_epoch'],
        checkpoint_available=False,selector_trace_available=False)


def evaluate(out):
    manifest,spec=context(out);main=Path(spec['main_root']);mae,mse,ref=[main/spec[k] for k in ('old_mae','old_mse','reference_run')]
    frame=load_v5_training_frame(main);originals=[read(p/'manifest.json') for p in (mae,mse)]
    data_digest=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    if len(frame)!=2754 or frame.sample_id.duplicated().any() or any(m['data_digest']!=data_digest for m in originals):
        raise ValueError('Original complete frame identity differs')
    ledgers=[ledger_entries(p/'fit_ledger.jsonl') for p in (mae,mse)]
    settings=yaml.safe_load((WORK/spec['original_specification']).read_text())['training']
    mse_settings=yaml.safe_load((WORK/spec['original_mse_specification']).read_text())['training']
    if settings['inner_seed']!=mse_settings['inner_seed']:raise ValueError('Matched inner protocol differs')
    audit=read(mae/'audit-r1.json');gains={};over_ready={};over_mse={};units=[];hashes={}
    metrics={'tap_time_len':{name:{} for name in ('Q75','READY_LAPLACE',CANDIDATE,CONTROL)}}
    y=frame.tap_time_len.to_numpy(float);spouts=frame.spout_no.to_numpy()
    for seed in SEEDS:
        key=str(seed);folds=fold_vector(main,frame,seed,load_v5_spec(main))
        if any(m['fold_digests'][key]!=digest(folds.tolist()) for m in originals):raise ValueError('Original fold identity differs')
        with np.load(ref/f'oof-{seed}.npz',allow_pickle=False) as old:
            order=reference_order(frame.sample_id,old['ids'])
            np.testing.assert_array_equal(folds,old['folds'][order]);np.testing.assert_array_equal(y,old['y'][order])
            base=old['Q75'][order];ready=old['LAPLACE_FIXED_A20'][order]
        members=[]
        for directory,ledger,recipe,training_settings in [(mae,ledgers[0],'mae',settings),(mse,ledgers[1],'tabm_plr001',mse_settings)]:
            member=np.full(len(frame),np.nan)
            for fold in range(5):
                name=f'tap_time_len-{recipe}-s{seed}-f{fold}'
                entry=ledger[name];path=directory/(name+'.npy')
                if recipe=='mae' and audit['hashes'][name]!=entry['prediction_sha256']:raise ValueError('Original audit prediction binding differs')
                record=audit_metadata(frame,folds,fold,entry,training_settings,recipe=='mae')
                member[folds==fold]=checked_vector(path,entry,int((folds==fold).sum()))
                units.append(dict(source_directory=str(directory),split_seed=seed,trial_id=name,prediction_sha256=sha(path),**record))
            members.append(member)
        endpoint=compose(base,members[0]);control=compose(base,members[1])
        for dest,reference in [(gains,base),(over_ready,ready),(over_mse,control)]:
            dest[key]=float(50*(np.abs(y-reference)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        for name,p in [('Q75',base),('READY_LAPLACE',ready),(CANDIDATE,endpoint),(CONTROL,control)]:
            metrics['tap_time_len'][name][key]=metric_detail(y,p,folds,spouts)
        with (out/f'oof-{seed}.npz').open('xb') as stream:np.savez_compressed(stream,ids=frame.sample_id.to_numpy(str),y=y,
            folds=folds,spouts=spouts,Q75=base,ready=ready,member=members[0],mse_member=members[1],endpoint=endpoint,control=control)
        hashes[key]=sha(out/f'oof-{seed}.npz')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if rss>spec['max_worker_rss_mib'] or len(units)!=20:raise ValueError('Memory or complete provenance count failed')
    policy=yaml.safe_load((WORK/'configs/candidate_tiers.yaml').read_text())
    write_new(out/'report.json',dict(gains=gains,gains_over_ready=over_ready,gains_over_mse=over_mse,
        **decision(gains,over_ready,over_mse),metrics=metrics,units=units,
        tiers=classify_candidates(metrics,spec,policy),new_fits=0,optimizer_runs=0,cold_states=0,packages=0,
        peak_rss_mib=rss,independent_audit='pending',oof_sha256=hashes,manifest_sha256=sha(out/'manifest.json')))


def audit(out):
    manifest,spec=context(out);report=read(out/'report.json');diff=[];gains={};ready_gain={};mse_gain={}
    for seed in SEEDS:
        key=str(seed);path=out/f'oof-{seed}.npz'
        if sha(path)!=report['oof_sha256'][key]:raise ValueError('OOF identity changed')
        with np.load(path,allow_pickle=False) as v:
            if len(v['ids'])!=2754 or len(set(v['ids']))!=2754 or set(v['folds'])!=set(range(5)):
                raise ValueError('Complete OOF required')
            y=v['y'].tolist();base=v['Q75'].tolist();ready=v['ready'].tolist();den=math.fsum(map(abs,y))
            endpoint=[.8*b+.2*m for b,m in zip(base,v['member'])];control=[.8*b+.2*m for b,m in zip(base,v['mse_member'])]
            for actual,saved in [(endpoint,v['endpoint']),(control,v['control'])]:diff.extend(abs(a-b) for a,b in zip(actual,saved))
            for dest,reference,field in [(gains,base,'gains'),(ready_gain,ready,'gains_over_ready'),(mse_gain,control,'gains_over_mse')]:
                value=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,reference,endpoint))/den
                dest[key]=value;diff.append(abs(value-report[field][key]))
            for name,p in [('Q75',base),('READY_LAPLACE',ready),(CANDIDATE,endpoint),(CONTROL,control)]:
                m=report['metrics']['tap_time_len'][name][key];groups=[('wmape',np.ones(len(y),bool))]
                groups += [(('by_fold',str(f)),v['folds']==f) for f in range(5)]
                groups += [(('by_spout',str(s)),v['spouts']==s) for s in sorted(set(v['spouts']))]
                for field,mask in groups:
                    ii=np.flatnonzero(mask);actual=math.fsum(abs(y[i]-p[i]) for i in ii)/math.fsum(abs(y[i]) for i in ii)
                    expected=m[field] if isinstance(field,str) else m[field[0]][field[1]];diff.append(abs(actual-expected))
    result=decision(gains,ready_gain,mse_gain)
    if max(diff)>spec['scalar_atol'] or any(report[k]!=v for k,v in result.items()):raise ValueError('Independent scalar/decision mismatch')
    write_new(out/'independent-score.json',dict(status='passed',gains=gains,gains_over_ready=ready_gain,gains_over_mse=mse_gain,
        **result,scalar_checks=len(diff),max_difference=max(diff),new_fits=0,optimizer_runs=0,cold_states=0,
        manifest_sha256=sha(out/'manifest.json'),report_sha256=sha(out/'report.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','evaluate','audit'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checks',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.operation=='freeze':
        if args.checks is None:raise ValueError('Checks receipt required')
        freeze(out,args.checks.resolve())
    else:globals()[args.operation](out)


if __name__=='__main__':main()
