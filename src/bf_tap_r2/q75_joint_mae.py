"""Matched shared/separate MAE representations; immutable same-split references."""
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
from .joint_mae_model import ARMS, TARGET_ORDER, JointMAERegressor, numpy_predict
from .v7_periodic import digest
from .v3_6_networks import NumericPreprocessor
from .ema_average_span import old_cache
from .q75_gaussian_confirmation import hooks, read
from .q75_gaussian_time import cold_state, compose, forbidden, reference_order, selected_epoch
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v33_run import metric_detail, partitions

WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/q75_joint_mae/SPEC.json'
PROTOCOL = 'docs/q75_joint_mae/PREREGISTRATION.md'
SEEDS = (42,3407)
CANDIDATES = ('SHARED_MAE_IRON_A20', 'SHARED_MAE_TIME_A20')


def validate_spec(spec):
    required = dict(candidate_arms=list(ARMS),target_order=list(TARGET_ORDER),split_seeds=list(SEEDS),
        weight=.2,fixed_scale=.5,folds=5,outer_estimators=20,optimizer_runs=40,new_states=40,
        engineering_estimators=2,engineering_optimizer_runs=4,engineering_rows=2754,reference_fits=0,
        confirmation_fits=0,full_data_fits=0,workers=1,numerical_threads=1,torch_interop_threads=1,
        monitor_seconds=600,maximum_runtime_seconds=None,automatic_retries=False,packages=0,
        desktop_writes=0,agent_uploads=0)
    if any(spec.get(k)!=v for k,v in required.items()):
        raise ValueError('Unregistered joint MAE scope or budget')
    if spec['candidates'] != {t:[c] for t,c in zip(TARGET_ORDER,CANDIDATES)}:
        raise ValueError('Frozen target/candidate map required')


def admission(gains,over_ready,over_separate):
    if any(set(v)!=set(CANDIDATES) for v in (gains,over_ready,over_separate)):
        raise ValueError('All registered candidates required')
    eligible=[];failed={}
    for candidate in CANDIDATES:
        for values in (gains[candidate],over_ready[candidate],over_separate[candidate]):
            if set(values)!=set(map(str,SEEDS)) or not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Complete two-seed paired gains required')
        reasons=[]
        if not all(v>0 for v in gains[candidate].values()):reasons.append('not_both_positive_vs_Q75')
        if not all(v>0 for v in over_separate[candidate].values()):reasons.append('not_both_positive_vs_SEPARATE_MAE')
        if candidate==CANDIDATES[1] and not all(v>0 for v in over_ready[candidate].values()):
            reasons.append('not_both_positive_vs_ready_LAPLACE_FIXED_A20')
        failed[candidate]=reasons
        if not reasons:eligible.append(candidate)
    finalist=min(eligible,key=lambda c:(-sum(gains[c].values())/2,CANDIDATES.index(c))) if eligible else None
    return dict(confirmation_finalist=finalist,failed_conditions=failed,formal_promoted=False)


def access(out,stage):
    with (out/'access.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage,time_ns=time.time_ns(),
            frozen_files_sha256=sha(out/'frozen-files.json'),scope='synthetic_or_authorized_round2_only',
            protected_prelim_targets_read=False))+'\n')
        stream.flush();os.fsync(stream.fileno())


def joint_cold_state(path,training,query,meta,settings,expected):
    if any(t in query.columns for t in TARGET_ORDER):raise ValueError('Query labels prohibited')
    model=JointMAERegressor.load(path)
    if model.arm!=meta['joint_mae_arm'] or model.settings!=settings or meta['target_order']!=list(TARGET_ORDER):
        raise ValueError('Model arm/settings/target order mismatch')
    if (meta['fit_ids_digest']!=digest(training.sample_id.astype(str).tolist())
            or meta['fit_rows']!=len(training) or model.preprocessor_.metadata()!=meta['preprocessing']):
        raise ValueError('Training/preprocessing identity mismatch')
    x=training[list(FEATURES)].to_numpy(float);y=training[list(TARGET_ORDER)].to_numpy(float)
    for actual,wanted in [(model.preprocessor_.means_,x.mean(0)),(model.preprocessor_.stds_,x.std(0)),
                         (model.mean_,y.mean(0)),(model.std_,y.std(0))]:
        np.testing.assert_allclose(actual,wanted,rtol=1e-12,atol=1e-10)
    vocabulary={int(v):i+1 for i,v in enumerate(sorted(training.spout_no.unique()))}
    if model.preprocessor_.spout_to_index_!=vocabulary or model.preprocessor_.n_spout_categories_!=len(vocabulary)+1:
        raise ValueError('Training vocabulary mismatch')
    np.testing.assert_array_equal(model.mean_,meta['target_mean']);np.testing.assert_array_equal(model.std_,meta['target_std'])
    if any(not torch.isfinite(v).all() for v in model.model_.state_dict().values()):raise ValueError('Nonfinite checkpoint')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    prediction=model.predict(query)
    alternatives=[prediction,model.predict(query.iloc[::-1])[::-1],
        np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)]),
        numpy_predict(payload,query)]
    if expected.shape!=(len(query),2):raise ValueError('Two-output state identity required')
    return prediction,max(float(np.max(np.abs(p-expected))) for p in alternatives)


def freeze(out,phase,checks,engineering=None):
    spec=read(WORK/SPEC);validate_spec(spec)
    main=Path(spec['main_root'])
    if phase not in ('engineering','development') or out.exists() or not out.is_relative_to(main/'local/runs'):
        raise ValueError('Fresh registered private phase required')
    versions=runtime(spec);torch.set_num_interop_threads(1)
    current=read(main/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(current[k]!=v for k,v in spec['reference'].items()):raise ValueError('Current platform reference changed')
    checked=read(checks)
    if (checked['exit_code']!=0 or checked['optimizer_constructor_attempts']!=0 or checked['optimizer_runs']!=0
            or not checked['python_version'].startswith('3.12') or sha(checked['junit_path'])!=checked['junit_sha256']):
        raise ValueError('Locked zero-optimizer checks required')
    verify_files(checked['source_hashes'])
    reference=main/spec['gaussian_development']
    dm,dt=[read(reference/name) for name in ('manifest.json','terminal.json')]
    if dt['status']!='passed' or dt['actual_exit_codes']!=[0,0]:raise ValueError('Audited Q75/Gaussian development required')
    for name,key in [('manifest.json','manifest_sha256'),('report.json','report_sha256'),('independent-score.json','independent_score_sha256')]:
        if sha(reference/name)!=dt[key]:raise ValueError('Reference terminal identity differs')
    paths=list((WORK/'src').rglob('*.py'))+[WORK/p for p in (SPEC,PROTOCOL,'uv.lock','pyproject.toml',
        spec['original_specification'],'configs/candidate_tiers.yaml','scripts/q75_joint_mae.py',
        'scripts/check_q75_joint_mae.py','scripts/observe_ema_fusion_selection.py','tests/test_q75_joint_mae.py')]
    paths += [reference/n for n in ('manifest.json','report.json','independent-score.json','terminal.json','oof-42.npz','oof-3407.npz')]
    laplace_reference=main/spec['laplace_development']
    lm=read(laplace_reference/'manifest.json');lt=read(laplace_reference/'terminal.json')
    if lt['status']!='passed' or lt['actual_exit_codes']!=[0,0,0,0]:
        raise ValueError('Audited complete Laplace development required')
    verify_files({str(laplace_reference/n):h for n,h in lt['artifacts'].items()})
    paths += [laplace_reference/n for n in ('manifest.json','terminal.json','report.json','independent-score.json','oof-42.npz','oof-3407.npz')]
    paths += [checks,Path(checked['junit_path'])]
    if phase=='development':
        if engineering is None:raise ValueError('Full engineering run required')
        et=read(engineering/'terminal.json')
        if et['status']!='passed' or et['actual_exit_codes']!=[0,0] or et['optimizer_runs']!=4 or et['cold_states']!=4:
            raise ValueError('Successful independent engineering terminal required')
        verify_files({str(engineering/n):h for n,h in et['artifacts'].items()})
        for p,h in read(engineering/'manifest.json')['sources'].items():
            if sha(p)!=h:raise ValueError('Engineering-tested source changed')
        paths += [p for p in engineering.rglob('*') if p.is_file()]
    native_sources=binding_sources(hooks())
    native_sources.update({str(WORK/p):sha(WORK/p) for p in ('src/bf_tap_r2/joint_mae_model.py',
        'src/bf_tap_r2/q75_joint_mae.py','src/bf_tap_r2/laplace_time_model.py','src/bf_tap_r2/v33_mixture.py','src/bf_tap_r2/v3_6_networks.py')})
    files={**dm['files'],**lm['files'],**native_sources,**{str(p):sha(p) for p in paths}}
    verify_files(files)
    out.mkdir(parents=True,exist_ok=False);write_new(out/'frozen-files.json',files)
    access(out,'freeze_'+phase)
    native=yaml.safe_load((WORK/spec['original_specification']).read_text())
    native['training'].update(objective='sum of two fixed b=.5 Laplace NLLs',prediction='two target locations')
    if native['training']['initial_scale']!=spec['fixed_scale']:raise ValueError('Fixed scale identity changed')
    if phase=='engineering':
        rng=np.random.default_rng(spec['engineering_seed']);n=spec['engineering_rows']
        x=rng.normal(size=(n,len(FEATURES)))
        frame=pd.DataFrame(x,columns=FEATURES)
        frame['sample_id']=[f'fixed-point-synthetic-{i}' for i in range(n)]
        frame['spout_no']=1+np.arange(n)%2
        frame['tap_time_len']=120+20*x[:,0]+10*x[:,1]+rng.laplace(size=n)*(.3+.3/(1+np.exp(-x[:,2])))
        frame['tap_iron']=500+40*x[:,0]+30*x[:,3]+rng.laplace(size=n)
        folds={-1:np.arange(n)%5};refs={};gauss={};iron={};laplace={};laplace_scale={}
        units=[dict(seed=-1,fold=0,arm=arm) for arm in ARMS]
    else:
        frame=load_v5_training_frame(main)
        if len(frame)!=2754 or frame.sample_id.duplicated().any():raise ValueError('Complete official training identity required')
        folds={};refs={};gauss={};iron={};laplace={};laplace_scale={}
        for seed in SEEDS:
            fv=fold_vector(main,frame,seed,load_v5_spec(main));folds[seed]=fv
            with np.load(reference/f'oof-{seed}.npz',allow_pickle=False) as previous:
                order=reference_order(frame.sample_id,previous['ids'])
                np.testing.assert_array_equal(fv,previous['folds'][order])
                np.testing.assert_array_equal(frame.tap_time_len.to_numpy(),previous['y'][order])
                refs[seed]=previous['Q75'][order];gauss[seed]=previous['endpoint'][order]
                np.testing.assert_array_equal(gauss[seed],compose(refs[seed],previous['member'][order]))
            with np.load(laplace_reference/f'oof-{seed}.npz',allow_pickle=False) as previous:
                order=reference_order(frame.sample_id,previous['ids'])
                np.testing.assert_array_equal(fv,previous['folds'][order])
                np.testing.assert_array_equal(frame.tap_time_len.to_numpy(),previous['y'][order])
                np.testing.assert_array_equal(refs[seed],previous['Q75'][order])
                laplace[seed]=previous['LAPLACE_FIXED_A20'][order]
                laplace_scale[seed]=previous['LAPLACE_SCALE_A20'][order]
            iron[seed]=np.full(len(frame),np.nan)
            parent_spec=read(main/dm['spec']['reference_run']/'manifest.json')['spec']
            for fold in range(5):
                training,query,_,_=partitions(frame,fv,fold,native)
                parent=old_cache(parent_spec,seed,fold,training,query)
                np.testing.assert_array_equal(parent['q75'],refs[seed][fv==fold])
                iron[seed][fv==fold]=parent['iron']
            if not np.isfinite(iron[seed]).all():raise ValueError('Incomplete same-split iron reference')
        units=[dict(seed=seed,fold=fold,arm=arm) for seed in SEEDS for fold in range(5) for arm in ARMS]
    arrays=dict(x=frame[list(FEATURES)].to_numpy(float),ids=frame.sample_id.to_numpy(str),spouts=frame.spout_no.to_numpy(),y=frame.tap_time_len.to_numpy(float),y_iron=frame.tap_iron.to_numpy(float))
    arrays.update({f'folds_{s}':v for s,v in folds.items()});arrays.update({f'q75_{s}':v for s,v in refs.items()});arrays.update({f'gaussian_{s}':v for s,v in gauss.items()})
    arrays.update({f'iron_{s}':v for s,v in iron.items()});arrays.update({f'laplace_{s}':v for s,v in laplace.items()});arrays.update({f'laplace_scale_{s}':v for s,v in laplace_scale.items()})
    with (out/'inputs.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    sources={str(p):sha(p) for p in paths if p.is_relative_to(WORK)}
    write_new(out/'manifest.json',dict(phase=phase,spec=spec,native=native,files=files,sources=sources,
        native_sources=native_sources,versions=versions,python=sys.version,units=units,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
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
                    selector = JointMAERegressor(item['arm'], manifest['native']['training']).initialize(fitting, fitting[list(TARGET_ORDER)].to_numpy(float))
                    epoch = selector.train(manifest['native']['training']['max_epochs'], (calibration, calibration[list(TARGET_ORDER)].to_numpy(float)))
                cp = selector.predict(calibration)
                selector.save(unit / 'calibration_model.pt')
                with ledger.partition(training.sample_id.tolist()):
                    model = JointMAERegressor(item['arm'], manifest['native']['training']).initialize(training, training[list(TARGET_ORDER)].to_numpy(float))
                    model.train(epoch)
                prediction = model.predict(query)
                model.save(unit / 'model.pt')
            native_receipt = ledger.close()
            metadata = dict(seed=seed, fold=fold, target_order=list(TARGET_ORDER), recipe=item['arm'],
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
        unit = out / f"s{seed}-f{fold}-{item['arm']}"
        warm = read(unit / 'warm-complete.json')
        if completion['warm_hashes'][unit.name] != sha(unit / 'warm-complete.json') or warm['pid'] == os.getpid():
            raise ValueError('Independent cold process and warm identity required')
        verify_files({str(unit / p): h for p, h in warm['hashes'].items()})
        training, query, fitting, calibration = unit_parts(manifest, frame, data, seed, fold)
        meta = read(unit / 'metadata.json')
        if (meta['seed'],meta['fold'],meta['recipe'],meta['target_order']) != (seed,fold,item['arm'],list(TARGET_ORDER)):
            raise ValueError('Unit model identity differs')
        if any(meta[p]['joint_mae_arm']!=item['arm'] for p in ('calibration','refit')):
            raise ValueError('Checkpoint arm differs from registered unit')
        chosen, best = selected_epoch(meta['calibration']['history'], manifest['native']['training'])
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
                    metadata, manifest['native']['training'], saved[field])
                if difference > manifest['spec']['cold_predict_atol']:
                    raise ValueError('Cold/order/chunk tolerance exceeded')
                if file == 'calibration_model.pt':
                    value = float((np.abs(prediction-calibration[list(TARGET_ORDER)].to_numpy(float))/np.array(metadata['target_std'])).mean())
                    if abs(value-best) > 1e-10:
                        raise ValueError('Selected state is not the traced best checkpoint')
                elif manifest['phase'] == 'engineering':
                    learnability[item['arm']]={}
                    for index,target in enumerate(TARGET_ORDER):
                        y=frame[target].to_numpy()[data[f'folds_{seed}']==fold]
                        result=dict(model_mae=float(np.abs(prediction[:,index]-y).mean()),
                            median_constant_mae=float(np.abs(np.median(training[target])-y).mean()))
                        learnability[item['arm']][target]=result
                        if result['model_mae']>=result['median_constant_mae']:
                            raise ValueError('Synthetic learnability failed')
                differences.append(difference)
                count += 1
        write_new(unit / 'cold-complete.json', dict(status='passed', warm_sha256=sha(unit / 'warm-complete.json'),
                  cold_states=2, native_counts=native['counts']))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss > manifest['spec']['max_worker_rss_mib'] or count != 2*len(manifest['units']):
        raise ValueError('Cold RSS/count gate failed')
    write_new(out / 'cold-audit.json', dict(status='passed', cold_states=count, max_difference=max(differences),
        learnability=learnability, peak_rss_mib=rss, new_fits=0, manifest_sha256=sha(out / 'manifest.json')))


def reference_vectors(data, seed, target):
    if target=='tap_time_len':
        return data['y'],data[f'q75_{seed}'],data[f'laplace_{seed}']
    return data['y_iron'],data[f'iron_{seed}'],data[f'iron_{seed}']


def evaluate(out):
    manifest,frame,data=context(out,no_fit=True)
    if manifest['phase']!='development' or read(out/'cold-audit.json')['cold_states']!=40:
        raise ValueError('Complete independent development cold audit required')
    spec=manifest['spec'];metrics={t:{'Q75':{},'READY':{},'SEPARATE_MAE_A20':{}} for t in TARGET_ORDER}
    gains={c:{} for c in CANDIDATES};over={c:{} for c in CANDIDATES};paired={c:{} for c in CANDIDATES};hashes={}
    for seed in SEEDS:
        key=str(seed);fv=data[f'folds_{seed}'];arrays=dict(ids=data['ids'],folds=fv,spouts=data['spouts'])
        members={arm:np.full((len(frame),2),np.nan) for arm in ARMS}
        for arm in ARMS:
            for fold in range(5):
                unit=out/f's{seed}-f{fold}-{arm}';warm=read(unit/'warm-complete.json');c=read(unit/'cold-complete.json')
                if c['warm_sha256']!=sha(unit/'warm-complete.json'):raise ValueError('Cold unit identity changed')
                verify_files({str(unit/p):h for p,h in warm['hashes'].items()})
                with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['query_ids'],frame.loc[fv==fold,'sample_id'].to_numpy(str))
                    members[arm][fv==fold]=saved['prediction']
            arrays['member_'+arm]=members[arm]
        for index,(target,candidate) in enumerate(zip(TARGET_ORDER,CANDIDATES)):
            y,base,ready=reference_vectors(data,seed,target)
            endpoint=compose(base,members['SHARED_MAE'][:,index]);control=compose(base,members['SEPARATE_MAE'][:,index])
            arrays.update({target+'_y':y,target+'_Q75':base,target+'_READY':ready,target+'_SEPARATE_MAE_A20':control,candidate:endpoint})
            for name,p in [('Q75',base),('READY',ready),('SEPARATE_MAE_A20',control),(candidate,endpoint)]:
                metrics[target].setdefault(name,{})[key]=metric_detail(y,p,fv,data['spouts'])
            for dest,reference in [(gains,base),(over,ready),(paired,control)]:
                dest[candidate][key]=float(50*(np.abs(y-reference)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        with (out/f'oof-{seed}.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
        hashes[key]=sha(out/f'oof-{seed}.npz')
    policy=yaml.safe_load((WORK/'configs/candidate_tiers.yaml').read_text())
    write_new(out/'report.json',dict(gains=gains,gains_over_ready=over,gains_over_separate=paired,metrics=metrics,
        **admission(gains,over,paired),tiers=classify_candidates(metrics,spec,policy),
        new_estimators=20,optimizer_runs=40,new_states=40,reference_fits=0,confirmation_fits=0,full_fits=0,
        packages=0,desktop_writes=0,uploads=0,independent_audit='pending',oof_sha256=hashes,
        manifest_sha256=sha(out/'manifest.json')))


def audit(out):
    manifest,frame,data=context(out,no_fit=True);report=read(out/'report.json')
    differences=[];gains={c:{} for c in CANDIDATES};over={c:{} for c in CANDIDATES};paired={c:{} for c in CANDIDATES}
    for seed in SEEDS:
        key=str(seed);path=out/f'oof-{seed}.npz'
        if sha(path)!=report['oof_sha256'][key]:raise ValueError('OOF identity changed')
        with np.load(path,allow_pickle=False) as v:
            if len(v['ids'])!=2754 or len(set(v['ids']))!=2754 or set(v['folds'])!=set(range(5)):
                raise ValueError('Incomplete OOF coverage')
            np.testing.assert_array_equal(v['ids'],data['ids']);np.testing.assert_array_equal(v['folds'],data[f'folds_{seed}'])
            for index,(target,candidate) in enumerate(zip(TARGET_ORDER,CANDIDATES)):
                yt,bt,rt=reference_vectors(data,seed,target)
                for field,expected in [('y',yt),('Q75',bt),('READY',rt)]:np.testing.assert_array_equal(v[target+'_'+field],expected)
                y,base,ready=[a.tolist() for a in (yt,bt,rt)];denominator=math.fsum(map(abs,y))
                endpoint=[.8*b+.2*m for b,m in zip(base,v['member_SHARED_MAE'][:,index])]
                control=[.8*b+.2*m for b,m in zip(base,v['member_SEPARATE_MAE'][:,index])]
                differences.extend(abs(a-b) for a,b in zip(endpoint,v[candidate]))
                differences.extend(abs(a-b) for a,b in zip(control,v[target+'_SEPARATE_MAE_A20']))
                for dest,reference,field in [(gains,base,'gains'),(over,ready,'gains_over_ready'),(paired,control,'gains_over_separate')]:
                    actual=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,reference,endpoint))/denominator
                    dest[candidate][key]=actual;differences.append(abs(actual-report[field][candidate][key]))
                for name,p in [('Q75',base),('READY',ready),('SEPARATE_MAE_A20',control),(candidate,endpoint)]:
                    metric=report['metrics'][target][name][key]
                    groups=[('wmape',np.ones(len(y),bool))]
                    groups += [(('by_fold',str(f)),v['folds']==f) for f in range(5)]
                    groups += [(('by_spout',str(s)),v['spouts']==s) for s in sorted(set(v['spouts']))]
                    for field,mask in groups:
                        positions=np.flatnonzero(mask)
                        expected=metric[field] if isinstance(field,str) else metric[field[0]][field[1]]
                        actual=math.fsum(abs(y[i]-p[i]) for i in positions)/math.fsum(abs(y[i]) for i in positions)
                        differences.append(abs(actual-expected))
    decision=admission(gains,over,paired)
    if any(report[k]!=v for k,v in decision.items()) or max(differences)>manifest['spec']['scalar_atol']:
        raise ValueError('Independent metric or admission mismatch')
    write_new(out/'independent-score.json',dict(status='passed',gains=gains,gains_over_ready=over,gains_over_separate=paired,
        **decision,scalar_checks=len(differences),max_difference=max(differences),
        report_sha256=sha(out/'report.json'),manifest_sha256=sha(out/'manifest.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','train','cold','evaluate','audit'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['engineering','development'])
    parser.add_argument('--engineering',type=Path);parser.add_argument('--checks',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.operation=='freeze':
        if args.checks is None:raise ValueError('Locked checks receipt required')
        freeze(out,args.phase,args.checks.resolve(),args.engineering.resolve() if args.engineering else None)
    else:globals()[args.operation](out)


if __name__=='__main__':main()
