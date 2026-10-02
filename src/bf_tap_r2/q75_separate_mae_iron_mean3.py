"""Three fixed training seeds for independent joint-MAE iron, paired with seed42."""
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

from .candidate_tiers import classify_candidates
from .data import FEATURES, TARGETS
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .ema_reference_ledger import KINDS, NativeLedger, binding_sources, native_hooks
from .joint_mae_model import JointMAERegressor, TARGET_ORDER
from .q75_joint_mae import joint_cold_state
from .v3_6_networks import NumericPreprocessor
from .q75_gaussian_confirmation import hooks, read
from .q75_gaussian_time import compose, forbidden, reference_order, selected_epoch
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v33_run import metric_detail, partitions

WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/q75_separate_mae_iron_mean3/SPEC.json'
PROTOCOL = 'docs/q75_separate_mae_iron_mean3/PREREGISTRATION.md'
SEEDS = (42,3407)
ARMS = ('SEPARATE_INIT1042','SEPARATE_INIT2042')
TRAINING_SEEDS = {'SEPARATE_INIT1042':1042,'SEPARATE_INIT2042':2042}


def validate_spec(spec):
    required=dict(candidate_arms=list(ARMS),split_seeds=list(SEEDS),training_seeds=[42,1042,2042],
        new_training_seeds=[1042,2042],weight=.2,folds=5,outer_estimators=20,optimizer_runs=40,
        new_states=40,reused_states=20,engineering_optimizer_runs=0,reference_fits=0,
        target_order=list(TARGET_ORDER),monitor_seconds=600,
        confirmation_fits=0,full_data_fits=0,workers=1,numerical_threads=1,torch_interop_threads=1,
        maximum_runtime_seconds=None,automatic_retries=False,packages=0,desktop_writes=0,agent_uploads=0)
    if any(spec.get(k)!=v for k,v in required.items()):raise ValueError('Unregistered mean3 scope or budget')
    if spec['candidates']!={'tap_iron':['SEPARATE_MAE_IRON_MEAN3_A20']}:raise ValueError('Single frozen candidate required')


def admission(gains,over_single):
    for values in (gains,over_single):
        if set(values)!=set(map(str,SEEDS)) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Complete two-seed paired gains required')
    failed=[]
    if not all(v>0 for v in gains.values()):failed.append('not_both_positive_vs_Q75')
    if not all(v>0 for v in over_single.values()):failed.append('not_both_positive_vs_single_seed_SEPARATE_MAE')
    return dict(confirmation_finalist='SEPARATE_MAE_IRON_MEAN3_A20' if not failed else None,
        failed_conditions=failed,formal_promoted=False)


def mean_three(original,extra1042,extra2042):
    arrays=[np.asarray(v,float) for v in (original,extra1042,extra2042)]
    if arrays[0].ndim!=1 or any(v.shape!=arrays[0].shape for v in arrays) or not np.isfinite(arrays).all():
        raise ValueError('Aligned finite within-split members required')
    return np.mean(np.stack(arrays),axis=0)


def iron_column(prediction):
    prediction=np.asarray(prediction,float)
    if prediction.ndim!=2 or prediction.shape[1]!=2 or not np.isfinite(prediction).all():
        raise ValueError('Finite two-target prediction required before iron projection')
    return prediction[:,0]


def calibration_value(prediction,calibration,metadata):
    return float((np.abs(prediction-calibration[list(TARGET_ORDER)].to_numpy(float))/np.asarray(metadata['target_std'])).mean())


def unit_settings(manifest,arm):
    return dict(manifest['native']['training'],random_seed=TRAINING_SEEDS[arm])


def access(out,stage):
    with (out/'access.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage,time_ns=time.time_ns(),
            frozen_files_sha256=sha(out/'frozen-files.json'),scope='synthetic_or_authorized_round2_only',
            protected_prelim_targets_read=False))+'\n')
        stream.flush();os.fsync(stream.fileno())


def freeze(out,checks):
    spec=read(WORK/SPEC);validate_spec(spec);main=Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main/'local/runs'):raise ValueError('Fresh private directory required')
    versions=runtime(spec);torch.set_num_interop_threads(1)
    current=read(main/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(current[k]!=v for k,v in spec['reference'].items()):raise ValueError('Reference changed')
    checked=read(checks)
    if checked['exit_code']!=0 or checked['optimizer_constructor_attempts']!=0 or checked['optimizer_runs']!=0:
        raise ValueError('Successful zero-optimizer locked checks required')
    if not checked['python_version'].startswith('3.12') or sha(checked['junit_path'])!=checked['junit_sha256']:
        raise ValueError('Locked Python checks identity differs')
    verify_files(checked['source_hashes'])
    old=main/spec['joint_development'];dm=read(old/'manifest.json');dt=read(old/'terminal.json')
    if dt['status']!='passed' or dt['actual_exit_codes']!=[0]*4:raise ValueError('Original two complete seeds required')
    verify_files({str(old/n):h for n,h in dt['artifacts'].items()})
    engineering=main/spec['previous_engineering'];et=read(engineering/'terminal.json');em=read(engineering/'manifest.json')
    if (et['status']!='passed' or et['optimizer_runs']!=4 or et['cold_states']!=4
            or et['actual_exit_codes']!=[0,0] or dm['native']!=em['native']
            or dm['native']['training']['random_seed']!=42):
        raise ValueError('Original joint MAE engineering evidence required')
    verify_files({str(engineering/n):h for n,h in et['artifacts'].items()})
    for rel in ['src/bf_tap_r2/joint_mae_model.py','src/bf_tap_r2/v33_mixture.py','src/bf_tap_r2/v3_6_networks.py']:
        old_file=str(Path(dm['spec']['main_root'])/'local/worktrees/q75-joint-mae-development'/rel)
        if sha(WORK/rel)!=dm['files'][old_file] or sha(WORK/rel)!=em['files'][old_file]:
            raise ValueError('Original tested scientific source changed')
    paths=list((WORK/'src').rglob('*.py'))+[WORK/p for p in (SPEC,PROTOCOL,'uv.lock','pyproject.toml',
        'configs/candidate_tiers.yaml','scripts/q75_separate_mae_iron_mean3.py','scripts/check_q75_separate_mae_iron_mean3.py',
        'scripts/observe_ema_fusion_selection.py','tests/test_q75_separate_mae_iron_mean3.py')]
    paths += [Path(p) for p in checked['source_hashes']]
    paths += [checks,Path(checked['junit_path'])]
    paths += [old/n for n in ('manifest.json','terminal.json','training-complete.json','cold-audit.json','report.json','independent-score.json','oof-42.npz','oof-3407.npz')]
    paths += [engineering/n for n in ('manifest.json','terminal.json','cold-audit.json')]
    completed=read(old/'training-complete.json')
    for seed in SEEDS:
        for fold in range(5):
            unit=old/f's{seed}-f{fold}-SEPARATE_MAE';warm=read(unit/'warm-complete.json')
            if completed['warm_hashes'][unit.name]!=sha(unit/'warm-complete.json'):
                raise ValueError('Original model chain changed')
            verify_files({str(unit/n):h for n,h in warm['hashes'].items()})
            if read(unit/'cold-complete.json')['warm_sha256']!=sha(unit/'warm-complete.json'):
                raise ValueError('Original cold receipt changed')
            paths += [p for p in unit.rglob('*') if p.is_file()]
    native_sources=binding_sources(hooks())
    native_sources.update({str(WORK/p):sha(WORK/p) for p in ('src/bf_tap_r2/joint_mae_model.py',
        'src/bf_tap_r2/q75_separate_mae_iron_mean3.py','src/bf_tap_r2/v33_mixture.py','src/bf_tap_r2/v3_6_networks.py')})
    files={**dm['files'],**native_sources,**{str(p):sha(p) for p in paths}};verify_files(files)
    out.mkdir(parents=True,exist_ok=False);write_new(out/'frozen-files.json',files);access(out,'freeze_development')
    frame=load_v5_training_frame(main)
    if len(frame)!=2754 or frame.sample_id.duplicated().any():raise ValueError('Official complete rows required')
    arrays=dict(x=frame[list(FEATURES)].to_numpy(float),ids=frame.sample_id.to_numpy(str),spouts=frame.spout_no.to_numpy(),y=frame.tap_iron.to_numpy(float),y_time=frame.tap_time_len.to_numpy(float))
    for seed in SEEDS:
        fv=fold_vector(main,frame,seed,load_v5_spec(main));arrays[f'folds_{seed}']=fv
        with np.load(old/f'oof-{seed}.npz',allow_pickle=False) as ref:
            order=reference_order(frame.sample_id,ref['ids'])
            np.testing.assert_array_equal(ref['folds'][order],fv);np.testing.assert_array_equal(ref['tap_iron_y'][order],arrays['y'])
            np.testing.assert_array_equal(ref['tap_time_len_y'][order],arrays['y_time'])
            arrays[f'q75_{seed}']=ref['tap_iron_Q75'][order]
            arrays[f'original42_{seed}']=iron_column(ref['member_SEPARATE_MAE'][order])
            arrays[f'single_{seed}']=ref['tap_iron_SEPARATE_MAE_A20'][order]
            np.testing.assert_array_equal(arrays[f'single_{seed}'],compose(arrays[f'q75_{seed}'],arrays[f'original42_{seed}']))
    with (out/'inputs.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    units=[dict(seed=seed,fold=fold,arm=arm) for seed in SEEDS for fold in range(5) for arm in ARMS]
    write_new(out/'manifest.json',dict(phase='development',spec=spec,native=dm['native'],files=files,
        sources={str(p):sha(p) for p in paths if p.is_relative_to(WORK)},native_sources=native_sources,
        versions=versions,python=sys.version,units=units,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        frozen_files_sha256=sha(out/'frozen-files.json'),inputs_sha256=sha(out/'inputs.npz')))


def context(out, *, no_fit=False):
    manifest = read(out / 'manifest.json')
    validate_spec(manifest['spec'])
    verify_files(manifest['files'])
    if sha(out / 'inputs.npz') != manifest['inputs_sha256'] or sha(out / 'frozen-files.json') != manifest['frozen_files_sha256']:
        raise ValueError('Frozen input identity changed')
    runtime(manifest['spec'])
    torch.set_num_interop_threads(1)
    if no_fit:
        JointMAERegressor.initialize = JointMAERegressor.train = forbidden
        torch.optim.Optimizer.__init__ = torch.optim.Adam = torch.optim.AdamW = NumericPreprocessor.fit = forbidden
    access(out, 'read_' + manifest['phase'])
    data = np.load(out / 'inputs.npz', allow_pickle=False)
    frame = pd.DataFrame(data['x'], columns=FEATURES)
    frame['sample_id'], frame['spout_no'], frame['tap_iron'], frame['tap_time_len'] = data['ids'], data['spouts'], data['y'], data['y_time']
    return manifest, frame, data


def unit_parts(manifest, frame, data, seed, fold):
    training, query, fitting, calibration = partitions(frame, data[f'folds_{seed}'], fold, manifest['native'])
    group = lambda part: set(pd.util.hash_pandas_object(part[list(FEATURES)], index=False))
    if group(training) & group(query) or group(fitting) & group(calibration):
        raise ValueError('Group partition overlap')
    return training, query, fitting, calibration


def train(out):
    manifest, frame, data = context(out)
    for item in manifest['units']:
        seed, fold = item['seed'], item['fold']
        settings=unit_settings(manifest,item['arm'])
        unit = out / f"s{seed}-f{fold}-{item['arm']}"
        unit.mkdir(exist_ok=False)
        write_new(unit / 'start.json', dict(seed=seed, fold=fold, pid=os.getpid(), manifest_sha256=sha(out / 'manifest.json')))
        training, query, fitting, calibration = unit_parts(manifest, frame, data, seed, fold)
        expected = {key: 2 if key == 'torch_optimizer' else 0 for key in KINDS}
        ledger = NativeLedger(unit / 'native', identity=dict(source_directory=str(out), split_seed=seed,
            fold=fold, trial_id=item['arm']), expected=expected, training_ids=training.sample_id.tolist(),
            query_ids=query.sample_id.tolist(), source_hashes=manifest['native_sources'])
        try:
            with native_hooks(hooks()):
                with ledger.partition(fitting.sample_id.tolist(), calibration.sample_id.tolist()):
                    selector = JointMAERegressor('SEPARATE_MAE', settings).initialize(fitting, fitting[list(TARGET_ORDER)].to_numpy(float))
                    epoch = selector.train(settings['max_epochs'], (calibration, calibration[list(TARGET_ORDER)].to_numpy(float)))
                cp = selector.predict(calibration)
                selector.save(unit / 'calibration_model.pt')
                with ledger.partition(training.sample_id.tolist()):
                    model = JointMAERegressor('SEPARATE_MAE', settings).initialize(training, training[list(TARGET_ORDER)].to_numpy(float))
                    model.train(epoch)
                prediction = model.predict(query)
                model.save(unit / 'model.pt')
            native_receipt = ledger.close()
            metadata = dict(seed=seed, fold=fold, target_order=list(TARGET_ORDER), recipe=item['arm'],
                            training_seed=settings['random_seed'],calibration=selector.metadata(), refit=model.metadata())
            write_new(unit / 'metadata.json', metadata)
            with (unit / 'predictions.npz').open('xb') as stream:
                np.savez_compressed(stream, prediction=prediction, calibration_prediction=cp,
                    query_ids=query.sample_id.to_numpy(str), calibration_ids=calibration.sample_id.to_numpy(str))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if rss > manifest['spec']['max_worker_rss_mib']:
                raise ValueError('Training RSS exceeds frozen gate')
            hashes = {str(p.relative_to(unit)): sha(p) for p in unit.rglob('*') if p.is_file()}
            write_new(unit / 'warm-complete.json', dict(hashes=hashes, counts=native_receipt['counts'], peak_rss_mib=rss,
                manifest_sha256=sha(out / 'manifest.json'), selected_epoch=epoch, pid=os.getpid()))
        except BaseException as error:
            write_new(unit / 'failure.json', dict(error=repr(error), automatic_retry=False))
            raise
    write_new(out / 'training-complete.json', dict(units=len(manifest['units']), optimizer_runs=2*len(manifest['units']),
        warm_hashes={f"s{i['seed']}-f{i['fold']}-{i['arm']}": sha(out / f"s{i['seed']}-f{i['fold']}-{i['arm']}" / 'warm-complete.json') for i in manifest['units']}))


def cold(out):
    manifest, frame, data = context(out, no_fit=True)
    completion = read(out / 'training-complete.json')
    differences, count, learnability = [], 0, {}
    for item in manifest['units']:
        seed, fold = item['seed'], item['fold']
        settings=unit_settings(manifest,item['arm'])
        unit = out / f"s{seed}-f{fold}-{item['arm']}"
        warm = read(unit / 'warm-complete.json')
        if completion['warm_hashes'][unit.name] != sha(unit / 'warm-complete.json') or warm['pid'] == os.getpid():
            raise ValueError('Independent cold process and warm identity required')
        verify_files({str(unit / p): h for p, h in warm['hashes'].items()})
        training, query, fitting, calibration = unit_parts(manifest, frame, data, seed, fold)
        meta = read(unit / 'metadata.json')
        if (meta['seed'],meta['fold'],meta['recipe'],meta['target_order']) != (seed,fold,item['arm'],list(TARGET_ORDER)):
            raise ValueError('Unit model identity differs')
        if any(meta[p]['joint_mae_arm']!='SEPARATE_MAE' for p in ('calibration','refit')):
            raise ValueError('Checkpoint arm differs from registered unit')
        if meta['training_seed']!=settings['random_seed']:raise ValueError('Training seed identity differs')
        chosen, best = selected_epoch(meta['calibration']['history'], settings)
        if (chosen != warm['selected_epoch'] or chosen != meta['calibration']['selected_epoch']
                or chosen != meta['refit']['selected_epoch'] or chosen != meta['refit']['stopped_epoch']
                or meta['calibration']['stopped_epoch'] != len(meta['calibration']['history'])
                or [r['epoch'] for r in meta['refit']['history']] != list(range(1, chosen+1))):
            raise ValueError('Cold selector/refit epochs mismatch')
        native = read(unit / 'native/scope-complete.json')
        if native['counts'] != warm['counts'] or native['counts']['torch_optimizer'] != 2:
            raise ValueError('Native budget mismatch')
        for index, fit, cal in [(1, fitting, calibration), (2, training, None)]:
            call = read(unit / f'native/call-{index:04d}/start.json')
            if call['partition']['training_ids'] != fit.sample_id.tolist() or call['partition']['calibration_ids'] != ([] if cal is None else cal.sample_id.tolist()):
                raise ValueError('Native optimizer partition mismatch')
        with np.load(unit / 'predictions.npz', allow_pickle=False) as saved:
            for file, fit, held, field, ids, metadata in [
                ('calibration_model.pt', fitting, calibration, 'calibration_prediction', 'calibration_ids', meta['calibration']),
                ('model.pt', training, query, 'prediction', 'query_ids', meta['refit'])]:
                np.testing.assert_array_equal(saved[ids], held.sample_id.to_numpy(str))
                prediction, difference = joint_cold_state(unit / file, fit, held.drop(columns=list(TARGETS), errors='ignore'),
                    metadata, settings, saved[field])
                if difference > manifest['spec']['cold_predict_atol']:
                    raise ValueError('Cold/order/chunk tolerance exceeded')
                if file == 'calibration_model.pt':
                    value = calibration_value(prediction,calibration,metadata)
                    if abs(value-best) > 1e-10:
                        raise ValueError('Selected state is not the traced best checkpoint')
                differences.append(difference)
                count += 1
        write_new(unit / 'cold-complete.json', dict(status='passed', warm_sha256=sha(unit / 'warm-complete.json'),
                  cold_states=2, native_counts=native['counts']))
    new_count=count
    old=Path(manifest['spec']['main_root'])/manifest['spec']['joint_development']
    for seed in SEEDS:
        fv=data[f'folds_{seed}'];rebuilt=np.full(len(frame),np.nan)
        for fold in range(5):
            unit=old/f's{seed}-f{fold}-SEPARATE_MAE';meta=read(unit/'metadata.json')
            training,query,fitting,calibration=unit_parts(manifest,frame,data,seed,fold)
            if ((meta['seed'],meta['fold'],meta['recipe'],meta['target_order'])
                    !=(seed,fold,'SEPARATE_MAE',list(TARGET_ORDER))
                    or any(meta[p]['joint_mae_arm']!='SEPARATE_MAE' for p in ('calibration','refit'))):
                raise ValueError('Original control target or model identity differs')
            warm=read(unit/'warm-complete.json');native=read(unit/'native/scope-complete.json')
            if native['counts']!=warm['counts'] or native['counts']['torch_optimizer']!=2:
                raise ValueError('Original native budget differs')
            for index,fit,cal in [(1,fitting,calibration),(2,training,None)]:
                call=read(unit/f'native/call-{index:04d}/start.json')
                if (call['partition']['training_ids']!=fit.sample_id.tolist()
                        or call['partition']['calibration_ids']!=([] if cal is None else cal.sample_id.tolist())):
                    raise ValueError('Original optimizer partition differs')
            chosen,best=selected_epoch(meta['calibration']['history'],manifest['native']['training'])
            if (meta['refit']['selected_epoch']!=chosen or meta['refit']['stopped_epoch']!=chosen
                    or meta['calibration']['selected_epoch']!=chosen or warm['selected_epoch']!=chosen
                    or meta['calibration']['stopped_epoch']!=len(meta['calibration']['history'])
                    or [r['epoch'] for r in meta['refit']['history']]!=list(range(1,chosen+1))):
                raise ValueError('Original control selection differs')
            with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
                for file,fit,held,field,ids,md in [
                    ('calibration_model.pt',fitting,calibration,'calibration_prediction','calibration_ids',meta['calibration']),
                    ('model.pt',training,query,'prediction','query_ids',meta['refit'])]:
                    np.testing.assert_array_equal(saved[ids],held.sample_id.to_numpy(str))
                    prediction,difference=joint_cold_state(unit/file,fit,held.drop(columns=list(TARGETS),errors='ignore'),
                        md,manifest['native']['training'],saved[field])
                    if difference>manifest['spec']['cold_predict_atol']:raise ValueError('Original control cold mismatch')
                    if file=='model.pt':rebuilt[fv==fold]=iron_column(prediction)
                    elif abs(calibration_value(prediction,calibration,md)-best)>1e-10:
                        raise ValueError('Original selected checkpoint mismatch')
                    differences.append(difference);count+=1
        np.testing.assert_array_equal(rebuilt,data[f'original42_{seed}'])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss > manifest['spec']['max_worker_rss_mib'] or new_count!=40 or count!=60:
        raise ValueError('Cold RSS/count gate failed')
    write_new(out / 'cold-audit.json', dict(status='passed', cold_states=count, new_cold_states=new_count,reused_cold_states=count-new_count, max_difference=max(differences),
        learnability=learnability, peak_rss_mib=rss, new_fits=0, manifest_sha256=sha(out / 'manifest.json')))


def evaluate(out):
    manifest,frame,data=context(out,no_fit=True)
    if read(out/'cold-audit.json')['cold_states']!=60:raise ValueError('All new and reused states must be cold checked')
    spec=manifest['spec'];metrics={'tap_iron':{n:{} for n in ['Q75','SEPARATE_MAE_IRON_A20','SEPARATE_MAE_IRON_MEAN3_A20']}}
    gains={};over={};hashes={};members={}
    for seed in SEEDS:
        key=str(seed);fv=data[f'folds_{seed}'];y=data['y'];base=data[f'q75_{seed}'];single=data[f'single_{seed}']
        arrays=dict(ids=data['ids'],y=y,folds=fv,spouts=data['spouts'],Q75=base,SEPARATE_MAE_IRON_A20=single,member42=data[f'original42_{seed}'])
        for arm in ARMS:
            member=np.full(len(frame),np.nan)
            for fold in range(5):
                unit=out/f's{seed}-f{fold}-{arm}';warm=read(unit/'warm-complete.json')
                if read(unit/'cold-complete.json')['warm_sha256']!=sha(unit/'warm-complete.json'):raise ValueError('Cold state binding changed')
                verify_files({str(unit/p):h for p,h in warm['hashes'].items()})
                with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['query_ids'],frame.loc[fv==fold,'sample_id'].to_numpy(str))
                    member[fv==fold]=iron_column(saved['prediction'])
            arrays[f'member{TRAINING_SEEDS[arm]}']=member
        mean=mean_three(*[arrays[f'member{s}'] for s in [42,1042,2042]]);endpoint=compose(base,mean)
        arrays['member_mean3']=mean;arrays['SEPARATE_MAE_IRON_MEAN3_A20']=endpoint
        gains[key]=float(50*(np.abs(y-base)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        over[key]=float(50*(np.abs(y-single)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        members[key]={str(s):float(50*(np.abs(y-base)-np.abs(y-compose(base,arrays[f'member{s}']))).sum()/np.abs(y).sum()) for s in [42,1042,2042]}
        for name,p in [('Q75',base),('SEPARATE_MAE_IRON_A20',single),('SEPARATE_MAE_IRON_MEAN3_A20',endpoint)]:
            metrics['tap_iron'][name][key]=metric_detail(y,p,fv,data['spouts'])
        with (out/f'oof-{seed}.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
        hashes[key]=sha(out/f'oof-{seed}.npz')
    policy=yaml.safe_load((WORK/'configs/candidate_tiers.yaml').read_text())
    write_new(out/'report.json',dict(gains=gains,gains_over_single=over,member_gains_descriptive=members,
        metrics=metrics,**admission(gains,over),tiers=classify_candidates(metrics,spec,policy),
        new_estimators=20,optimizer_runs=40,new_states=40,reused_states=20,reference_fits=0,
        confirmation_fits=0,full_fits=0,packages=0,desktop_writes=0,uploads=0,
        oof_sha256=hashes,manifest_sha256=sha(out/'manifest.json')))


def audit(out):
    manifest,frame,data=context(out,no_fit=True);report=read(out/'report.json');diff=[];gains={};over={}
    for seed in SEEDS:
        key=str(seed);path=out/f'oof-{seed}.npz'
        if sha(path)!=report['oof_sha256'][key]:raise ValueError('OOF identity differs')
        with np.load(path,allow_pickle=False) as v:
            if len(v['ids'])!=2754 or len(set(v['ids']))!=2754 or set(v['folds'])!=set(range(5)):
                raise ValueError('Incomplete OOF')
            for field,expected in [('ids',data['ids']),('y',data['y']),('folds',data[f'folds_{seed}']),
                ('Q75',data[f'q75_{seed}']),('SEPARATE_MAE_IRON_A20',data[f'single_{seed}']),('member42',data[f'original42_{seed}'])]:
                np.testing.assert_array_equal(v[field],expected)
            y,base,single=[v[k].tolist() for k in ['y','Q75','SEPARATE_MAE_IRON_A20']]
            mean=[math.fsum(row)/3 for row in zip(v['member42'],v['member1042'],v['member2042'])]
            endpoint=[.8*b+.2*m for b,m in zip(base,mean)]
            diff.extend(abs(a-b) for a,b in zip(mean,v['member_mean3']));diff.extend(abs(a-b) for a,b in zip(endpoint,v['SEPARATE_MAE_IRON_MEAN3_A20']))
            denominator=math.fsum(map(abs,y))
            gains[key]=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,base,endpoint))/denominator
            over[key]=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,single,endpoint))/denominator
            diff += [abs(gains[key]-report['gains'][key]),abs(over[key]-report['gains_over_single'][key])]
            for seed0 in [42,1042,2042]:
                p=[.8*b+.2*m for b,m in zip(base,v[f'member{seed0}'])]
                g=50*math.fsum(abs(a-b)-abs(a-c) for a,b,c in zip(y,base,p))/denominator
                diff.append(abs(g-report['member_gains_descriptive'][key][str(seed0)]))
            for name,p in [('Q75',base),('SEPARATE_MAE_IRON_A20',single),('SEPARATE_MAE_IRON_MEAN3_A20',endpoint)]:
                m=report['metrics']['tap_iron'][name][key]
                groups=[('wmape',np.ones(len(y),bool))]+[(('by_fold',str(f)),v['folds']==f) for f in range(5)]+[(('by_spout',str(s)),v['spouts']==s) for s in sorted(set(v['spouts']))]
                for field,mask in groups:
                    ii=np.flatnonzero(mask);actual=math.fsum(abs(y[i]-p[i]) for i in ii)/math.fsum(abs(y[i]) for i in ii)
                    expected=m[field] if isinstance(field,str) else m[field[0]][field[1]];diff.append(abs(actual-expected))
    decision=admission(gains,over)
    if any(report[k]!=v for k,v in decision.items()) or max(diff)>manifest['spec']['scalar_atol']:
        raise ValueError('Independent mean/scalar/admission mismatch')
    write_new(out/'independent-score.json',dict(status='passed',gains=gains,gains_over_single=over,
        **decision,scalar_checks=len(diff),max_difference=max(diff),report_sha256=sha(out/'report.json'),manifest_sha256=sha(out/'manifest.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','train','cold','evaluate','audit'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checks',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.operation=='freeze':
        if args.checks is None:raise ValueError('Locked checks receipt required')
        freeze(out,args.checks.resolve())
    else:globals()[args.operation](out)


if __name__=='__main__':main()
