"""Complete matched Laplace development against immutable Q75/Gaussian OOF."""
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
from .laplace_time_model import ARMS, LaplaceRegressor
from .q75_gaussian_confirmation import hooks, read
from .q75_gaussian_time import cold_state, compose, forbidden, reference_order, selected_epoch
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v33_run import metric_detail, partitions

WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/q75_laplace_time/SPEC.json'
PROTOCOL = 'docs/q75_laplace_time/PREREGISTRATION.md'
SEEDS = (42,3407)


def validate_spec(spec):
    required = dict(candidate_arms=list(ARMS),split_seeds=list(SEEDS),weight=.2,fixed_scale=.5,
        folds=5,outer_estimators=20,optimizer_runs=40,new_states=40,engineering_estimators=2,
        engineering_optimizer_runs=4,engineering_rows=2754,reference_fits=0,confirmation_fits=0,
        full_data_fits=0,workers=1,numerical_threads=1,torch_interop_threads=1,
        maximum_runtime_seconds=None,automatic_retries=False,packages=0,desktop_writes=0,agent_uploads=0)
    if any(spec.get(k)!=v for k,v in required.items()):
        raise ValueError('Unregistered Laplace scope or budget')


def admission(gains,over_gaussian):
    if set(gains)!=set(ARMS) or set(over_gaussian)!=set(ARMS):
        raise ValueError('Both registered arms required')
    eligible=[]; failed={}
    for arm in ARMS:
        for values in (gains[arm],over_gaussian[arm]):
            if set(values)!=set(map(str,SEEDS)) or not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Complete two-seed paired gains required')
        reasons=[]
        if not all(v>0 for v in gains[arm].values()): reasons.append('not_both_development_seeds_positive_vs_Q75')
        if sum(over_gaussian[arm].values())/2<=0: reasons.append('nonpositive_mean_advantage_over_GAUSS1_A20')
        failed[arm]=reasons
        if not reasons: eligible.append(arm)
    finalist=min(eligible,key=lambda arm:(-sum(gains[arm].values())/2,ARMS.index(arm))) if eligible else None
    return dict(confirmation_finalist=finalist,failed_conditions=failed,formal_promoted=False)


def access(out,stage):
    with (out/'access.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage,time_ns=time.time_ns(),
            frozen_files_sha256=sha(out/'frozen-files.json'),scope='synthetic_or_authorized_round2_only',
            protected_prelim_targets_read=False))+'\n')
        stream.flush();os.fsync(stream.fileno())


def laplace_cold_state(path,training,query,target,meta,settings,expected):
    model=LaplaceRegressor.load(path)
    if model.arm!=meta['laplace_arm'] or meta['likelihood']!='Laplace':
        raise ValueError('Saved Laplace arm differs')
    prediction,difference=cold_state(path,training,query,target,meta,settings,expected)
    if not np.array_equal(prediction,model.predict(query)):
        raise ValueError('Tagged Laplace and generic location readers differ')
    return prediction,difference


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
        spec['original_specification'],'configs/candidate_tiers.yaml','scripts/q75_laplace_time.py',
        'scripts/check_q75_laplace_time.py','scripts/observe_ema_fusion_selection.py','tests/test_q75_laplace_time.py')]
    paths += [reference/n for n in ('manifest.json','report.json','independent-score.json','terminal.json','oof-42.npz','oof-3407.npz')]
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
    native_sources.update({str(WORK/p):sha(WORK/p) for p in ('src/bf_tap_r2/laplace_time_model.py',
        'src/bf_tap_r2/q75_laplace_time.py','src/bf_tap_r2/v33_mixture.py','src/bf_tap_r2/v3_6_networks.py')})
    files={**dm['files'],**native_sources,**{str(p):sha(p) for p in paths}}
    verify_files(files)
    out.mkdir(parents=True,exist_ok=False);write_new(out/'frozen-files.json',files)
    access(out,'freeze_'+phase)
    native=yaml.safe_load((WORK/spec['original_specification']).read_text())
    native['training'].update(objective='Laplace negative log likelihood',prediction='conditional location/median')
    if native['training']['initial_scale']!=spec['fixed_scale']:raise ValueError('Fixed scale identity changed')
    if phase=='engineering':
        rng=np.random.default_rng(spec['engineering_seed']);n=spec['engineering_rows']
        x=rng.normal(size=(n,len(FEATURES)))
        frame=pd.DataFrame(x,columns=FEATURES)
        frame['sample_id']=[f'laplace-synthetic-{i}' for i in range(n)]
        frame['spout_no']=1+np.arange(n)%2
        frame['tap_time_len']=120+20*x[:,0]+10*x[:,1]+rng.laplace(size=n)*(.3+.3/(1+np.exp(-x[:,2])))
        folds={-1:np.arange(n)%5};refs={};gauss={}
        units=[dict(seed=-1,fold=0,arm=arm) for arm in ARMS]
    else:
        frame=load_v5_training_frame(main)
        if len(frame)!=2754 or frame.sample_id.duplicated().any():raise ValueError('Complete official training identity required')
        folds={};refs={};gauss={}
        for seed in SEEDS:
            fv=fold_vector(main,frame,seed,load_v5_spec(main));folds[seed]=fv
            with np.load(reference/f'oof-{seed}.npz',allow_pickle=False) as previous:
                order=reference_order(frame.sample_id,previous['ids'])
                np.testing.assert_array_equal(fv,previous['folds'][order])
                np.testing.assert_array_equal(frame.tap_time_len.to_numpy(),previous['y'][order])
                refs[seed]=previous['Q75'][order];gauss[seed]=previous['endpoint'][order]
                np.testing.assert_array_equal(gauss[seed],compose(refs[seed],previous['member'][order]))
        units=[dict(seed=seed,fold=fold,arm=arm) for seed in SEEDS for fold in range(5) for arm in ARMS]
    arrays=dict(x=frame[list(FEATURES)].to_numpy(float),ids=frame.sample_id.to_numpy(str),spouts=frame.spout_no.to_numpy(),y=frame.tap_time_len.to_numpy(float))
    arrays.update({f'folds_{s}':v for s,v in folds.items()});arrays.update({f'q75_{s}':v for s,v in refs.items()});arrays.update({f'gaussian_{s}':v for s,v in gauss.items()})
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
        LaplaceRegressor.initialize = LaplaceRegressor.train = forbidden
        torch.optim.Adam = torch.optim.AdamW = forbidden
    access(out, 'read_' + manifest['phase'])
    data = np.load(out / 'inputs.npz', allow_pickle=False)
    frame = pd.DataFrame(data['x'], columns=FEATURES)
    frame['sample_id'], frame['spout_no'], frame['tap_time_len'] = data['ids'], data['spouts'], data['y']
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
                    selector = LaplaceRegressor(item['arm'], manifest['native']['training']).initialize(fitting, fitting.tap_time_len.to_numpy())
                    epoch = selector.train(manifest['native']['training']['max_epochs'], (calibration, calibration.tap_time_len.to_numpy()))
                cp = selector.predict(calibration)
                selector.save(unit / 'calibration_model.pt')
                with ledger.partition(training.sample_id.tolist()):
                    model = LaplaceRegressor(item['arm'], manifest['native']['training']).initialize(training, training.tap_time_len.to_numpy())
                    model.train(epoch)
                prediction = model.predict(query)
                model.save(unit / 'model.pt')
            native_receipt = ledger.close()
            metadata = dict(seed=seed, fold=fold, target='tap_time_len', recipe=item['arm'],
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
        if (meta['seed'],meta['fold'],meta['recipe'],meta['target']) != (seed,fold,item['arm'],'tap_time_len'):
            raise ValueError('Unit model identity differs')
        if any(meta[p]['laplace_arm']!=item['arm'] for p in ('calibration','refit')):
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
                prediction, difference = laplace_cold_state(unit / file, fit, held.drop(columns=list(TARGETS), errors='ignore'),
                    'tap_time_len', metadata, manifest['native']['training'], saved[field])
                if difference > manifest['spec']['cold_predict_atol']:
                    raise ValueError('Cold/order/chunk tolerance exceeded')
                if file == 'calibration_model.pt':
                    value = np.abs(prediction-calibration.tap_time_len.to_numpy()).mean()/metadata['target_std']
                    if abs(value-best) > 1e-10:
                        raise ValueError('Selected state is not the traced best checkpoint')
                elif manifest['phase'] == 'engineering':
                    y = frame.tap_time_len.to_numpy()[data[f'folds_{seed}'] == fold]
                    learnability[item['arm']] = dict(model_mae=float(np.abs(prediction-y).mean()),
                        median_constant_mae=float(np.abs(np.median(training.tap_time_len)-y).mean()))
                    if learnability[item['arm']]['model_mae'] >= learnability[item['arm']]['median_constant_mae']:
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


def evaluate(out):
    manifest,frame,data=context(out,no_fit=True)
    if manifest['phase']!='development' or read(out/'cold-audit.json')['cold_states']!=40:
        raise ValueError('Complete independent development cold audit required')
    spec=manifest['spec'];metrics={'tap_time_len':{name:{} for name in ('Q75','GAUSS1_A20',*[arm+'_A20' for arm in ARMS])}}
    gains={a:{} for a in ARMS};over={a:{} for a in ARMS};hashes={}
    for seed in SEEDS:
        key=str(seed);fv=data[f'folds_{seed}'];base=data[f'q75_{seed}'];gauss=data[f'gaussian_{seed}'];y=data['y']
        arrays=dict(ids=data['ids'],y=y,folds=fv,spouts=data['spouts'],Q75=base,GAUSS1_A20=gauss)
        predictions={'Q75':base,'GAUSS1_A20':gauss}
        for arm in ARMS:
            member=np.full(len(frame),np.nan)
            for fold in range(5):
                unit=out/f's{seed}-f{fold}-{arm}';warm=read(unit/'warm-complete.json');c=read(unit/'cold-complete.json')
                if c['warm_sha256']!=sha(unit/'warm-complete.json'):raise ValueError('Cold unit identity changed')
                verify_files({str(unit/p):h for p,h in warm['hashes'].items()})
                with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['query_ids'],frame.loc[fv==fold,'sample_id'].to_numpy(str))
                    member[fv==fold]=saved['prediction']
            endpoint=compose(base,member)
            gains[arm][key]=float(50*(np.abs(y-base)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
            over[arm][key]=float(50*(np.abs(y-gauss)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
            arrays['member_'+arm]=member;arrays[arm+'_A20']=endpoint;predictions[arm+'_A20']=endpoint
        for name,p in predictions.items():metrics['tap_time_len'][name][key]=metric_detail(y,p,fv,data['spouts'])
        with (out/f'oof-{seed}.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
        hashes[key]=sha(out/f'oof-{seed}.npz')
    policy=yaml.safe_load((WORK/'configs/candidate_tiers.yaml').read_text())
    write_new(out/'report.json',dict(gains=gains,gains_over_gaussian=over,metrics=metrics,
        scale_minus_fixed={str(s):gains[ARMS[1]][str(s)]-gains[ARMS[0]][str(s)] for s in SEEDS},
        **admission(gains,over),tiers=classify_candidates(metrics,spec,policy),
        new_estimators=20,optimizer_runs=40,new_states=40,reference_fits=0,confirmation_fits=0,full_fits=0,
        packages=0,desktop_writes=0,uploads=0,independent_audit='pending',oof_sha256=hashes,
        manifest_sha256=sha(out/'manifest.json')))


def audit(out):
    manifest,frame,data=context(out,no_fit=True);report=read(out/'report.json')
    differences=[];gains={a:{} for a in ARMS};over={a:{} for a in ARMS}
    for seed in SEEDS:
        key=str(seed);path=out/f'oof-{seed}.npz'
        if sha(path)!=report['oof_sha256'][key]:raise ValueError('OOF identity changed')
        with np.load(path,allow_pickle=False) as values:
            if len(values['ids'])!=2754 or len(set(values['ids']))!=2754 or set(values['folds'])!=set(range(5)):
                raise ValueError('Incomplete OOF coverage')
            for field,expected in [('ids',data['ids']),('y',data['y']),('folds',data[f'folds_{seed}']),
                ('Q75',data[f'q75_{seed}']),('GAUSS1_A20',data[f'gaussian_{seed}'])]:
                np.testing.assert_array_equal(values[field],expected)
            y,base,gauss=[values[k].tolist() for k in ('y','Q75','GAUSS1_A20')]
            denominator=math.fsum(map(abs,y));predictions={'Q75':base,'GAUSS1_A20':gauss}
            for arm in ARMS:
                endpoint=[.8*b+.2*m for b,m in zip(base,values['member_'+arm])]
                differences.extend(abs(a-b) for a,b in zip(endpoint,values[arm+'_A20']))
                gains[arm][key]=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,base,endpoint))/denominator
                over[arm][key]=50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,gauss,endpoint))/denominator
                differences += [abs(gains[arm][key]-report['gains'][arm][key]),abs(over[arm][key]-report['gains_over_gaussian'][arm][key])]
                predictions[arm+'_A20']=endpoint
            for name,p in predictions.items():
                metric=report['metrics']['tap_time_len'][name][key]
                groups=[('wmape',np.ones(len(y),bool))]
                groups += [(('by_fold',str(f)),values['folds']==f) for f in range(5)]
                groups += [(('by_spout',str(s)),values['spouts']==s) for s in sorted(set(values['spouts']))]
                for field,mask in groups:
                    positions=np.flatnonzero(mask)
                    expected=metric[field] if isinstance(field,str) else metric[field[0]][field[1]]
                    actual=math.fsum(abs(y[i]-p[i]) for i in positions)/math.fsum(abs(y[i]) for i in positions)
                    differences.append(abs(actual-expected))
            differences.append(abs((gains[ARMS[1]][key]-gains[ARMS[0]][key])-report['scale_minus_fixed'][key]))
    decision=admission(gains,over)
    if any(report[k]!=v for k,v in decision.items()) or max(differences)>manifest['spec']['scalar_atol']:
        raise ValueError('Independent metric or admission mismatch')
    write_new(out/'independent-score.json',dict(status='passed',gains=gains,gains_over_gaussian=over,
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
