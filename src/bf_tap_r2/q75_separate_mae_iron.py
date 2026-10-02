"""Prospective confirmation of the observed independent MAE iron control."""
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
from .joint_mae_model import TARGET_ORDER, JointMAERegressor
from .fixed_point_model import FixedPointRegressor
from .q75_joint_mae import joint_cold_state
from .q75_fixed_point_losses import point_cold_state
from .q75_gaussian_confirmation import gate
from scipy.stats import t
from .v7_periodic import digest
from .v3_6_networks import NumericPreprocessor
from .ema_average_span import old_cache
from .q75_gaussian_confirmation import hooks, read
from .q75_gaussian_time import cold_state, compose, forbidden, reference_order, selected_epoch
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v33_run import metric_detail, partitions

WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/q75_separate_mae_iron/SPEC.json'
PROTOCOL = 'docs/q75_separate_mae_iron/PREREGISTRATION.md'
SEEDS = (271828,314159)
ALL_SEEDS = (42,3407,271828,314159)
ARMS = ('SEPARATE_MAE','SINGLE_MAE')
CANDIDATE = 'SEPARATE_MAE_IRON_A20'


def validate_spec(spec):
    required=dict(candidate=CANDIDATE,candidate_arms=list(ARMS),target_order=list(TARGET_ORDER),
        split_seeds=list(SEEDS),development_seeds=[42,3407],weight=.2,folds=5,outer_estimators=20,
        optimizer_runs=40,new_states=40,engineering_optimizer_runs=0,reference_fits=0,
        confirmation_fits=20,repeated_development_fits=0,full_data_fits=0,workers=1,numerical_threads=1,
        torch_interop_threads=1,monitor_seconds=600,maximum_runtime_seconds=None,
        automatic_retries=False,packages=0,desktop_writes=0,agent_uploads=0)
    if any(spec.get(k)!=v for k,v in required.items()):raise ValueError('Unregistered separate iron scope or budget')
    if spec['candidates']!={'tap_iron':[CANDIDATE]}:raise ValueError('Only one iron candidate registered')


def admission(gains,over_single):
    q=gate(gains);m=gate(over_single)
    return dict(formal_promoted=q['formal_promoted'],release_eligible=q['formal_promoted'] and m['formal_promoted'],
        paired=q['paired'],paired_over_single=m['paired'],failed_conditions=q['failed_conditions'],
        mechanism_failed_conditions=m['failed_conditions'])


def model_for(arm,settings):
    if arm=='SEPARATE_MAE':return JointMAERegressor(arm,settings)
    if arm=='SINGLE_MAE':return FixedPointRegressor('LAPLACE_FIXED',settings)
    raise ValueError('Unknown frozen arm')


def target_values(frame,arm):
    return frame[list(TARGET_ORDER)].to_numpy(float) if arm=='SEPARATE_MAE' else frame.tap_iron.to_numpy(float)


def unit_settings(manifest,arm):
    return manifest['native_by_arm'][arm]['training']


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
    if (checked['exit_code']!=0 or checked['optimizer_constructor_attempts']!=0 or checked['optimizer_runs']!=0
            or not checked['python_version'].startswith('3.12') or sha(checked['junit_path'])!=checked['junit_sha256']):
        raise ValueError('Locked zero-optimizer checks required')
    verify_files(checked['source_hashes'])
    paths=list((WORK/'src').rglob('*.py'))+[WORK/p for p in (SPEC,PROTOCOL,'uv.lock','pyproject.toml',
        'scripts/q75_separate_mae_iron.py','scripts/check_q75_separate_mae_iron.py',
        'scripts/observe_ema_fusion_selection.py','tests/test_q75_separate_mae_iron.py')]
    paths += [checks,Path(checked['junit_path'])]
    files={};natives={};roots={}
    for arm,devkey,engkey,trial,expected in [('SEPARATE_MAE','development','joint_engineering','SEPARATE_MAE',40),
                                           ('SINGLE_MAE','single_development','single_engineering','IRON_LAPLACE_FIXED',60)]:
        dev=main/spec[devkey];eng=main/spec[engkey];roots[arm]=dev
        dm=read(dev/'manifest.json');dt=read(dev/'terminal.json');et=read(eng/'terminal.json');em=read(eng/'manifest.json')
        if (dt['status']!='passed' or dt['actual_exit_codes']!=[0]*4 or dt['optimizer_runs']!=expected
                or dt['cold_states']!=expected or et['status']!='passed' or et['actual_exit_codes']!=[0,0]):
            raise ValueError('Completed native development and engineering required')
        for directory,terminal in [(dev,dt),(eng,et)]:
            verify_files({str(directory/n):h for n,h in terminal['artifacts'].items()})
            paths += [directory/n for n in terminal['artifacts']]+[directory/'terminal.json']
        source_names=['v33_mixture.py','v3_6_networks.py','joint_mae_model.py'] if arm=='SEPARATE_MAE' else ['v33_mixture.py','v3_6_networks.py','laplace_time_model.py','fixed_point_model.py']
        for name in source_names:
            wanted=sha(WORK/'src/bf_tap_r2'/name)
            for old in [dm,em]:
                candidates=[h for p,h in old['sources'].items() if p.endswith('/src/bf_tap_r2/'+name)]
                if candidates!=[wanted]:raise ValueError('G0/development scientific source mismatch: '+name)
        natives[arm]=dm['native'];files.update(dm['files'])
        if dm['native']['training']!=em['native']['training']:raise ValueError('Native settings differ from full-shape engineering')
        completion=read(dev/'training-complete.json')
        for seed in (42,3407):
            paths.append(dev/f'oof-{seed}.npz')
            for fold in range(5):
                unit=dev/f's{seed}-f{fold}-{trial}';warm=read(unit/'warm-complete.json');cold=read(unit/'cold-complete.json')
                if completion['warm_hashes'][unit.name]!=sha(unit/'warm-complete.json') or cold['warm_sha256']!=sha(unit/'warm-complete.json'):
                    raise ValueError('Original control warm/cold chain changed')
                verify_files({str(unit/n):h for n,h in warm['hashes'].items()})
                paths += [p for p in unit.rglob('*') if p.is_file()]
    reference=main/spec['reference_confirmation'];rt=read(reference/'terminal-verification-r1.json');rm=read(reference/'manifest.json')
    if rt['status']!='passed' or rt['actual_controller_exit_code']!=0:raise ValueError('Q75 reference terminal missing')
    for name,key in [('manifest.json','manifest_sha256'),('audit.json','audit_sha256'),('summary.json','summary_sha256'),('process-terminal.json','process_terminal_sha256')]:
        if sha(reference/name)!=rt[key]:raise ValueError('Q75 reference terminal identity differs')
        paths.append(reference/name)
    if rm['spec']['reference_spec']['reference']['iron_weights']!={'v36':.5,'v12_joint':.5}:
        raise ValueError('Q75 iron recipe changed')
    paths.append(reference/'terminal-verification-r1.json')
    for seed in SEEDS:
        for fold in range(5):
            unit=reference/f's{seed}-f{fold}';warm=read(unit/'warm-complete.json');cold=read(unit/'cold-complete.json')
            if (cold['status']!='passed' or warm['manifest_sha256']!=sha(reference/'manifest.json')
                    or cold['warm_receipt_sha256']!=sha(unit/'warm-complete.json')
                    or warm['predictions_sha256']!=sha(unit/'predictions.npz')
                    or warm['reference_receipt_sha256']!=sha(unit/'reference/complete.json')
                    or cold['reference_cold_receipt_sha256']!=sha(unit/'reference/cold-complete.json')):
                raise ValueError('Q75 same-fold warm/cold reference identity differs')
            paths += [unit/n for n in ('warm-complete.json','cold-complete.json','predictions.npz','reference/complete.json','reference/cold-complete.json')]
    native_sources=binding_sources(hooks())
    native_sources.update({str(WORK/'src/bf_tap_r2'/n):sha(WORK/'src/bf_tap_r2'/n) for n in
        ['q75_separate_mae_iron.py','joint_mae_model.py','fixed_point_model.py','laplace_time_model.py','v33_mixture.py','v3_6_networks.py']})
    files.update(native_sources);files.update({str(p):sha(p) for p in paths});verify_files(files)
    out.mkdir(parents=True,exist_ok=False);write_new(out/'frozen-files.json',files);access(out,'freeze_confirmation')
    frame=load_v5_training_frame(main)
    if len(frame)!=2754 or frame.sample_id.duplicated().any():raise ValueError('Complete official rows required')
    arrays=dict(x=frame[list(FEATURES)].to_numpy(float),ids=frame.sample_id.to_numpy(str),spouts=frame.spout_no.to_numpy(),
        y=frame.tap_time_len.to_numpy(float),y_iron=frame.tap_iron.to_numpy(float))
    for seed in ALL_SEEDS:
        fv=fold_vector(main,frame,seed,load_v5_spec(main));arrays[f'folds_{seed}']=fv
        if seed in (42,3407):
            for arm,dev in roots.items():
                with np.load(dev/f'oof-{seed}.npz',allow_pickle=False) as old:
                    order=reference_order(frame.sample_id,old['ids'])
                    np.testing.assert_array_equal(old['folds'][order],fv);np.testing.assert_array_equal(old['tap_iron_y'][order],arrays['y_iron'])
                    base=old['tap_iron_Q75'][order]
                    if f'q75_{seed}' in arrays:np.testing.assert_array_equal(base,arrays[f'q75_{seed}'])
                    arrays[f'q75_{seed}']=base
                    member=old['member_SEPARATE_MAE'][order,0] if arm=='SEPARATE_MAE' else old['member_IRON_LAPLACE_FIXED'][order]
                    arrays[f'{arm}_{seed}']=member
                    field='tap_iron_SEPARATE_MAE_A20' if arm=='SEPARATE_MAE' else 'IRON_LAPLACE_FIXED_A20'
                    np.testing.assert_array_equal(compose(base,member),old[field][order])
            y=arrays['y_iron'];endpoint=compose(base,arrays[f'SEPARATE_MAE_{seed}']);control=compose(base,arrays[f'SINGLE_MAE_{seed}'])
            if any(float((np.abs(y-r)-np.abs(y-endpoint)).sum())<=0 for r in (base,control)):
                raise ValueError('Two complete positive development gains versus both references required')
        else:
            base=np.full(len(frame),np.nan)
            for fold in range(5):
                unit=reference/f's{seed}-f{fold}';warm=read(unit/'warm-complete.json')
                training,query,_,_=partitions(frame,fv,fold,natives['SEPARATE_MAE'])
                if (warm['partition']['training']!=digest(training.sample_id.tolist())
                        or warm['partition']['query']!=digest(query.sample_id.tolist())):
                    raise ValueError('Reference training/query identity differs')
                with np.load(unit/'predictions.npz',allow_pickle=False) as old:
                    np.testing.assert_array_equal(old['query_ids'],query.sample_id.to_numpy(str))
                    np.testing.assert_array_equal(old['iron'],old['tap_iron'])
                    np.testing.assert_array_equal(old['iron'],.5*old['v36_iron']+.5*old['v12_iron'])
                    base[fv==fold]=old['iron']
            if not np.isfinite(base).all() or (base<0).any():raise ValueError('Invalid complete Q75 iron')
            arrays[f'q75_{seed}']=base
    with (out/'inputs.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    units=[dict(seed=seed,fold=fold,arm=arm) for seed in SEEDS for fold in range(5) for arm in ARMS]
    write_new(out/'manifest.json',dict(phase='confirmation',spec=spec,native=natives['SEPARATE_MAE'],native_by_arm=natives,
        files=files,sources={str(p):sha(p) for p in paths if p.is_relative_to(WORK)},native_sources=native_sources,
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
        JointMAERegressor.initialize = JointMAERegressor.train = FixedPointRegressor.initialize = FixedPointRegressor.train = forbidden
        torch.optim.Optimizer.__init__ = torch.optim.Adam = torch.optim.AdamW = NumericPreprocessor.fit = forbidden
    access(out, 'read_' + manifest['phase'])
    data = np.load(out / 'inputs.npz', allow_pickle=False)
    frame = pd.DataFrame(data['x'], columns=FEATURES)
    frame['sample_id'], frame['spout_no'], frame['tap_time_len'], frame['tap_iron'] = data['ids'], data['spouts'], data['y'], data['y_iron']
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
                    selector = model_for(item['arm'],settings).initialize(fitting, target_values(fitting,item['arm']))
                    epoch = selector.train(settings['max_epochs'], (calibration, target_values(calibration,item['arm'])))
                cp = selector.predict(calibration)
                selector.save(unit / 'calibration_model.pt')
                with ledger.partition(training.sample_id.tolist()):
                    model = model_for(item['arm'],settings).initialize(training, target_values(training,item['arm']))
                    model.train(epoch)
                prediction = model.predict(query)
                model.save(unit / 'model.pt')
            native_receipt = ledger.close()
            metadata = dict(seed=seed, fold=fold, target_order=list(TARGET_ORDER) if item['arm']=='SEPARATE_MAE' else ['tap_iron'], recipe=item['arm'],
                            calibration=selector.metadata(), refit=model.metadata())
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
        if (meta['seed'],meta['fold'],meta['recipe'],meta['target_order']) != (seed,fold,item['arm'],list(TARGET_ORDER) if item['arm']=='SEPARATE_MAE' else ['tap_iron']):
            raise ValueError('Unit model identity differs')
        identity_field='joint_mae_arm' if item['arm']=='SEPARATE_MAE' else 'point_loss_arm'
        identity_value=item['arm'] if item['arm']=='SEPARATE_MAE' else 'LAPLACE_FIXED'
        if any(meta[p][identity_field]!=identity_value for p in ('calibration','refit')):
            raise ValueError('Checkpoint arm differs from registered unit')
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
                clean=held.drop(columns=list(TARGETS),errors='ignore')
                if item['arm']=='SEPARATE_MAE':
                    prediction,difference=joint_cold_state(unit/file,fit,clean,metadata,settings,saved[field])
                else:
                    prediction,difference=point_cold_state(unit/file,fit,clean,'tap_iron',metadata,settings,saved[field])
                if difference>manifest['spec']['cold_predict_atol']:raise ValueError('Cold/order/chunk tolerance exceeded')
                if file=='calibration_model.pt':
                    value=float((np.abs(prediction-target_values(calibration,item['arm']))/np.array(metadata['target_std'])).mean())
                    if abs(value-best)>1e-10:raise ValueError('Selected state is not the traced best checkpoint')
                differences.append(difference)
                count += 1
        write_new(unit / 'cold-complete.json', dict(status='passed', warm_sha256=sha(unit / 'warm-complete.json'),
                  cold_states=2, native_counts=native['counts']))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss > manifest['spec']['max_worker_rss_mib'] or count != 2*len(manifest['units']):
        raise ValueError('Cold RSS/count gate failed')
    write_new(out / 'cold-audit.json', dict(status='passed', cold_states=count, max_difference=max(differences),
        learnability=learnability, peak_rss_mib=rss, new_fits=0, manifest_sha256=sha(out / 'manifest.json')))


def evaluate(out):
    manifest,frame,data=context(out,no_fit=True)
    if read(out/'cold-audit.json')['cold_states']!=40:raise ValueError('Complete confirmation cold required')
    gains={};over={};metrics={};hashes={}
    for seed in ALL_SEEDS:
        key=str(seed);fv=data[f'folds_{seed}'];y=data['y_iron'];base=data[f'q75_{seed}'];members={}
        for arm in ARMS:
            if seed in (42,3407):members[arm]=data[f'{arm}_{seed}']
            else:
                member=np.full(len(frame),np.nan)
                for fold in range(5):
                    unit=out/f's{seed}-f{fold}-{arm}';warm=read(unit/'warm-complete.json');cold=read(unit/'cold-complete.json')
                    if cold['warm_sha256']!=sha(unit/'warm-complete.json'):raise ValueError('Cold identity changed')
                    verify_files({str(unit/n):h for n,h in warm['hashes'].items()})
                    with np.load(unit/'predictions.npz',allow_pickle=False) as v:
                        np.testing.assert_array_equal(v['query_ids'],frame.loc[fv==fold,'sample_id'].to_numpy(str))
                        member[fv==fold]=v['prediction'][:,0] if arm=='SEPARATE_MAE' else v['prediction']
                members[arm]=member
        endpoint=compose(base,members['SEPARATE_MAE']);control=compose(base,members['SINGLE_MAE'])
        gains[key]=float(50*(np.abs(y-base)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        over[key]=float(50*(np.abs(y-control)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        metrics[key]={name:metric_detail(y,p,fv,data['spouts']) for name,p in [('Q75',base),(CANDIDATE,endpoint),('SINGLE_MAE_A20',control)]}
        with (out/f'oof-{seed}.npz').open('xb') as stream:np.savez_compressed(stream,ids=data['ids'],folds=fv,spouts=data['spouts'],
            y=y,Q75=base,member=members['SEPARATE_MAE'],single_member=members['SINGLE_MAE'],endpoint=endpoint,single_endpoint=control)
        hashes[key]=sha(out/f'oof-{seed}.npz')
    write_new(out/'report.json',dict(candidate=CANDIDATE,selection_origin=manifest['spec']['selection_origin'],gains=gains,gains_over_single=over,
        **admission(gains,over),metrics=metrics,optimizer_runs=40,new_states=40,repeated_development_fits=0,
        reference_fits=0,full_fits=0,packages=0,desktop_writes=0,uploads=0,independent_audit='pending',
        oof_sha256=hashes,manifest_sha256=sha(out/'manifest.json')))


def audit(out):
    manifest,frame,data=context(out,no_fit=True);report=read(out/'report.json');differences=[];gains={};over={}
    for seed in ALL_SEEDS:
        key=str(seed);path=out/f'oof-{seed}.npz'
        if sha(path)!=report['oof_sha256'][key]:raise ValueError('OOF hash changed')
        with np.load(path,allow_pickle=False) as v:
            np.testing.assert_array_equal(v['ids'],data['ids']);np.testing.assert_array_equal(v['folds'],data[f'folds_{seed}'])
            np.testing.assert_array_equal(v['y'],data['y_iron']);np.testing.assert_array_equal(v['Q75'],data[f'q75_{seed}'])
            if len(set(v['ids']))!=2754 or set(v['folds'])!=set(range(5)):raise ValueError('Complete coverage required')
            y=v['y'].tolist();base=v['Q75'].tolist();den=math.fsum(map(abs,y))
            endpoint=[.8*b+.2*m for b,m in zip(base,v['member'])];control=[.8*b+.2*m for b,m in zip(base,v['single_member'])]
            for actual,saved in [(endpoint,v['endpoint']),(control,v['single_endpoint'])]:differences.extend(abs(a-b) for a,b in zip(actual,saved))
            gains[key]=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,base,endpoint))/den
            over[key]=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,control,endpoint))/den
            differences += [abs(gains[key]-report['gains'][key]),abs(over[key]-report['gains_over_single'][key])]
            for name,p in [('Q75',base),(CANDIDATE,endpoint),('SINGLE_MAE_A20',control)]:
                m=report['metrics'][key][name];groups=[('wmape',np.ones(len(y),bool))]
                groups += [(('by_fold',str(f)),v['folds']==f) for f in range(5)]
                groups += [(('by_spout',str(s)),v['spouts']==s) for s in sorted(set(v['spouts']))]
                for field,mask in groups:
                    positions=np.flatnonzero(mask);expected=m[field] if isinstance(field,str) else m[field[0]][field[1]]
                    actual=math.fsum(abs(y[i]-p[i]) for i in positions)/math.fsum(abs(y[i]) for i in positions)
                    differences.append(abs(actual-expected))
    for values,field in [(gains,'paired'),(over,'paired_over_single')]:
        mean=math.fsum(values.values())/4;sd=math.sqrt(math.fsum((v-mean)**2 for v in values.values())/3)
        for k,value in dict(mean=mean,sd=sd,se=sd/2,lcb95=mean-float(t.ppf(.95,3))*sd/2).items():differences.append(abs(value-report[field][k]))
    decision=admission(gains,over)
    if any(report[k]!=v for k,v in decision.items()) or max(differences)>manifest['spec']['scalar_atol']:
        raise ValueError('Independent scalar or gate mismatch')
    write_new(out/'independent-score.json',dict(status='passed',gains=gains,gains_over_single=over,**decision,
        scalar_checks=len(differences),max_difference=max(differences),report_sha256=sha(out/'report.json'),manifest_sha256=sha(out/'manifest.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','train','cold','evaluate','audit'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checks',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.operation=='freeze':
        if args.checks is None:raise ValueError('Locked checks receipt required')
        freeze(out,args.checks.resolve())
    else:globals()[args.operation](out)


if __name__=='__main__':main()
