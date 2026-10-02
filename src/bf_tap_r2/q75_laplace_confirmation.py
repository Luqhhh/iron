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
from .q75_gaussian_confirmation import hooks, read, gate
from scipy.stats import t
from .q75_gaussian_time import cold_state, compose, forbidden, reference_order, selected_epoch
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v5_resolution import paired_summary
from .v33_run import metric_detail, partitions

WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/q75_laplace_confirmation/SPEC.json'
PROTOCOL = 'docs/q75_laplace_confirmation/PREREGISTRATION.md'
SEEDS = (42,3407,271828,314159)


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


def validate_spec(spec):
    expected=dict(candidate='LAPLACE_FIXED_A20',arm='LAPLACE_FIXED',weight=.2,
        development_seeds=[42,3407],confirmation_seeds=[271828,314159],folds=5,
        outer_estimators=10,optimizer_runs=20,new_states=20,engineering_optimizer_runs=0,
        reference_fits=0,repeated_development_fits=0,workers=1,numerical_threads=1,
        torch_interop_threads=1,maximum_runtime_seconds=None,automatic_retries=False,
        full_data_fits=0,packages=0,desktop_writes=0,agent_uploads=0,training_changes=[])
    if any(spec.get(k)!=v for k,v in expected.items()):raise ValueError('Unregistered fixed Laplace confirmation scope')


def freeze(out,checks):
    spec=read(WORK/SPEC);validate_spec(spec);main=Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main/'local/runs'):raise ValueError('Fresh private output required')
    versions=runtime(spec);torch.set_num_interop_threads(1)
    current=read(main/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(current[k]!=v for k,v in spec['reference'].items()):raise ValueError('Current reference changed')
    checked=read(checks)
    if (checked['exit_code']!=0 or checked['optimizer_constructor_attempts']!=0
            or checked['optimizer_runs']!=0 or not checked['python_version'].startswith('3.12')
            or sha(checked['junit_path'])!=checked['junit_sha256']):raise ValueError('Locked zero-fit checks required')
    verify_files(checked['source_hashes'])
    development=main/spec['development'];gaussian=main/spec['gaussian_confirmation']
    dm,dr,dt=[read(development/n) for n in ('manifest.json','report.json','terminal.json')]
    gm,gt=[read(gaussian/n) for n in ('manifest.json','terminal.json')]
    if (dt['status']!='passed' or dt['actual_exit_codes']!=[0]*4 or dt['optimizer_runs']!=40
            or dt['cold_states']!=40 or dr['confirmation_finalist']!='LAPLACE_FIXED'
            or any(v<=0 for v in dr['gains']['LAPLACE_FIXED'].values())
            or sum(dr['gains_over_gaussian']['LAPLACE_FIXED'].values())<=0):
        raise ValueError('Complete admitted fixed-scale development required')
    if gt['status']!='passed' or gt['actual_exit_codes']!=[0]*4:raise ValueError('Audited four-seed references required')
    for directory,terminal in ((development,dt),(gaussian,gt)):
        verify_files({str(directory/n):h for n,h in terminal['artifacts'].items()})
    for rel in ('src/bf_tap_r2/laplace_time_model.py','src/bf_tap_r2/v33_mixture.py','src/bf_tap_r2/v3_6_networks.py'):
        old=next(h for p,h in dm['sources'].items() if p.endswith('/'+rel))
        if sha(WORK/rel)!=old:raise ValueError('Scientific training implementation changed')
    paths=list((WORK/'src').rglob('*.py'))+[WORK/p for p in (SPEC,PROTOCOL,'uv.lock','pyproject.toml',
        'scripts/q75_laplace_confirmation.py','scripts/check_q75_laplace_confirmation.py',
        'scripts/observe_ema_fusion_selection.py','tests/test_q75_laplace_confirmation.py')]
    paths += [checks,Path(checked['junit_path'])]
    for directory,terminal,seeds in ((development,dt,(42,3407)),(gaussian,gt,SEEDS)):
        paths += [directory/n for n in terminal['artifacts']]+[directory/'terminal.json']
        paths += [directory/f'oof-{seed}.npz' for seed in seeds]
    native_sources=binding_sources(hooks())
    native_sources.update({str(WORK/p):sha(WORK/p) for p in ('src/bf_tap_r2/laplace_time_model.py',
        'src/bf_tap_r2/q75_laplace_confirmation.py','src/bf_tap_r2/v33_mixture.py','src/bf_tap_r2/v3_6_networks.py')})
    files={**dm['files'],**gm['files'],**native_sources,**{str(p):sha(p) for p in paths}}
    verify_files(files);out.mkdir(parents=True,exist_ok=False);write_new(out/'frozen-files.json',files)
    access(out,'freeze_confirmation_before_authorized_labels')
    frame=load_v5_training_frame(main);arrays=dict(x=frame[list(FEATURES)].to_numpy(float),ids=frame.sample_id.to_numpy(str),
        spouts=frame.spout_no.to_numpy(),y=frame.tap_time_len.to_numpy(float))
    if len(frame)!=2754 or frame.sample_id.duplicated().any():raise ValueError('Complete official training identity required')
    for seed in SEEDS:
        fv=fold_vector(main,frame,seed,load_v5_spec(main));arrays[f'folds_{seed}']=fv
        with np.load(gaussian/f'oof-{seed}.npz',allow_pickle=False) as saved:
            order=reference_order(frame.sample_id,saved['ids'])
            np.testing.assert_array_equal(fv,saved['folds'][order]);np.testing.assert_array_equal(arrays['y'],saved['y'][order])
            arrays[f'q75_{seed}']=saved['Q75'][order];arrays[f'gaussian_{seed}']=saved['endpoint'][order]
            np.testing.assert_array_equal(arrays[f'gaussian_{seed}'],compose(arrays[f'q75_{seed}'],saved['member'][order]))
        if seed in (42,3407):
            with np.load(development/f'oof-{seed}.npz',allow_pickle=False) as saved:
                order=reference_order(frame.sample_id,saved['ids'])
                for key,value in [('y',arrays['y']),('folds',fv),('Q75',arrays[f'q75_{seed}']),('GAUSS1_A20',arrays[f'gaussian_{seed}'])]:
                    np.testing.assert_array_equal(saved[key][order],value)
    with (out/'inputs.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    write_new(out/'manifest.json',dict(phase='confirmation',spec=spec,native=dm['native'],files=files,
        sources={str(p):sha(p) for p in paths if p.is_relative_to(WORK)},native_sources=native_sources,
        versions=versions,python=sys.version,units=[dict(seed=s,fold=f,arm='LAPLACE_FIXED') for s in spec['confirmation_seeds'] for f in range(5)],
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
    manifest, frame, data = context(out, no_fit=True)
    if manifest['phase'] != 'confirmation' or read(out / 'cold-audit.json')['cold_states'] != 20:
        raise ValueError('Complete independent cold confirmation required')
    spec = manifest['spec']
    dev = Path(spec['main_root']) / spec['development']
    metrics, gains, over_gaussian, hashes = {}, {}, {}, {}
    for seed in SEEDS:
        fv, base = data[f'folds_{seed}'], data[f'q75_{seed}']
        if seed in (42, 3407):
            with np.load(dev / f'oof-{seed}.npz', allow_pickle=False) as previous:
                order = reference_order(frame.sample_id, previous['ids'])
                member = previous['member_LAPLACE_FIXED'][order]
        else:
            member = np.full(len(frame), np.nan)
            for fold in range(5):
                unit = out / f's{seed}-f{fold}-LAPLACE_FIXED'
                warm, cold_receipt = read(unit / 'warm-complete.json'), read(unit / 'cold-complete.json')
                if cold_receipt['warm_sha256'] != sha(unit / 'warm-complete.json'):
                    raise ValueError('Cold candidate binding changed')
                verify_files({str(unit / p): h for p, h in warm['hashes'].items()})
                with np.load(unit / 'predictions.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['query_ids'], frame.loc[fv == fold, 'sample_id'].to_numpy(str))
                    member[fv == fold] = saved['prediction']
        endpoint = compose(base, member)
        gaussian = data[f'gaussian_{seed}']
        y = data['y']
        gains[str(seed)] = float(50*(np.abs(y-base)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        over_gaussian[str(seed)] = float(50*(np.abs(y-gaussian)-np.abs(y-endpoint)).sum()/np.abs(y).sum())
        metrics[str(seed)] = {name: metric_detail(y, prediction, fv, data['spouts']) for name, prediction in [('Q75', base), ('GAUSS1_A20', gaussian), ('LAPLACE_FIXED_A20', endpoint)]}
        path = out / f'oof-{seed}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, ids=data['ids'], y=y, folds=fv, spouts=data['spouts'], Q75=base, GAUSS1_A20=gaussian, member=member, endpoint=endpoint)
        hashes[str(seed)] = sha(path)
    write_new(out / 'report.json', dict(gains=gains, gains_over_gaussian=over_gaussian,
        paired_over_gaussian=paired_summary([over_gaussian[str(s)] for s in SEEDS]), metrics=metrics, **gate(gains),
        oof_sha256=hashes, candidate=spec['candidate'], new_estimators=10, new_states=20, optimizer_runs=20,
        reference_fits=0, repeated_development_fits=0, full_fits=0, packages=0, desktop_writes=0, uploads=0,
        independent_audit='pending', manifest_sha256=sha(out / 'manifest.json')))


def audit(out):
    manifest, _, _ = context(out, no_fit=True)
    report = read(out / 'report.json')
    differences, gains, over_gaussian = [], {}, {}
    for seed in SEEDS:
        path = out / f'oof-{seed}.npz'
        if sha(path) != report['oof_sha256'][str(seed)]:
            raise ValueError('OOF file identity changed')
        with np.load(path, allow_pickle=False) as values:
            if len(values['ids']) != 2754 or len(set(values['ids'])) != 2754 or set(values['folds']) != set(range(5)):
                raise ValueError('Incomplete four-seed OOF')
            y, base, member, gaussian = [values[k].tolist() for k in ('y', 'Q75', 'member', 'GAUSS1_A20')]
            endpoint = [(.8*b)+(.2*m) for b, m in zip(base, member)]
            differences.extend(abs(a-b) for a,b in zip(endpoint, values['endpoint']))
            gain = 50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,base,endpoint))/math.fsum(map(abs,y))
            gains[str(seed)] = gain
            over_gaussian[str(seed)] = 50*math.fsum(abs(a-b)-abs(a-p) for a,b,p in zip(y,gaussian,endpoint))/math.fsum(map(abs,y))
            differences.append(abs(over_gaussian[str(seed)]-report['gains_over_gaussian'][str(seed)]))
            differences.append(abs(gain-report['gains'][str(seed)]))
            for name, prediction in [('Q75', base), ('GAUSS1_A20', gaussian), ('LAPLACE_FIXED_A20', endpoint)]:
                record = report['metrics'][str(seed)][name]
                groups = [('wmape', np.ones(len(y), bool))]
                groups += [(('by_fold',str(f)),values['folds']==f) for f in range(5)]
                groups += [(('by_spout',str(s)),values['spouts']==s) for s in sorted(set(values['spouts']))]
                for key, mask in groups:
                    ids = np.flatnonzero(mask)
                    actual = math.fsum(abs(y[i]-prediction[i]) for i in ids)/math.fsum(abs(y[i]) for i in ids)
                    expected = record[key] if isinstance(key,str) else record[key[0]][key[1]]
                    differences.append(abs(actual-expected))
    mean = math.fsum(gains.values())/4
    se = math.sqrt(math.fsum((gain-mean)**2 for gain in gains.values())/3)/2
    lcb = mean-float(t.ppf(.95,3))*se
    gaussian_mean=math.fsum(over_gaussian.values())/4
    gaussian_se=math.sqrt(math.fsum((v-gaussian_mean)**2 for v in over_gaussian.values())/3)/2
    gaussian_lcb=gaussian_mean-float(t.ppf(.95,3))*gaussian_se
    differences.extend([abs(gaussian_mean-report['paired_over_gaussian']['mean']),abs(gaussian_lcb-report['paired_over_gaussian']['lcb95'])])
    differences.extend([abs(mean-report['paired']['mean']),abs(lcb-report['paired']['lcb95'])])
    passed = all(gain>0 for gain in gains.values()) and lcb>0
    if passed != report['formal_promoted'] or max(differences)>manifest['spec']['scalar_atol']:
        raise ValueError('Independent score or formal gate mismatch')
    write_new(out / 'independent-score.json', dict(status='passed', gains=gains, gains_over_gaussian=over_gaussian,
        mean_gain_over_gaussian=gaussian_mean,seed_lcb95_over_gaussian=gaussian_lcb,mean_gain=mean, seed_lcb95=lcb,
        formal_promoted=passed, scalar_checks=len(differences), max_difference=max(differences),
        report_sha256=sha(out / 'report.json'), manifest_sha256=sha(out / 'manifest.json')))



def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','train','cold','evaluate','audit'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checks',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.operation=='freeze':
        if args.checks is None:raise ValueError('Locked checks required')
        freeze(out,args.checks.resolve())
    else:globals()[args.operation](out)


if __name__=='__main__':main()
