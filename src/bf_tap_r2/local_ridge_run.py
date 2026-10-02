"""Frozen serial development, cold audit and target-specific Q75 comparison."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import yaml

from .data import FEATURES, TARGETS
from .local_ridge_neighborhood import LocalRidgeNeighborhood, independent_predict
from .v3_run import load_training_frame, load_fold_vector
from .v7_periodic import digest

SPEC = 'configs/local_ridge_neighborhood/SPEC.json'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def event(out, name, **details):
    with (Path(out)/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=name, time_ns=time.time_ns(), **details), allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Frozen file changed: '+path)


def runtime():
    if sys.version_info[:2] != (3, 12) or any(os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('Python3.12 and four single-thread variables required')
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'pandas', 'scipy', 'scikit-learn')}
    if versions != dict(numpy='2.2.6', pandas='2.3.3', scipy='1.18.1', **{'scikit-learn':'1.8.0'}):
        raise ValueError('Verified CPU environment changed')
    return dict(python=sys.version, executable=sys.executable, versions=versions,
                threads={k:os.environ[k] for k in THREADS})


def sources(work):
    paths = list((work/'src').rglob('*.py'))
    paths += [work/SPEC, work/'docs/local_ridge_neighborhood/PREREGISTRATION.md',
              work/'tests/test_local_ridge_neighborhood.py', work/'uv.lock', work/'pyproject.toml',
              work/'configs/protection.yaml', work/'configs/candidate_tiers.yaml',
              work/'scripts/observe_local_ridge.py']
    return {str(p.resolve()): sha(p) for p in paths}


def load_labels(out, manifest):
    main = Path(manifest['spec']['main_root'])
    event(out, 'label_access', scope='round2_V2_only_no_initial_round_November_targets',
          manifest_sha256=sha(Path(out)/'manifest.json'), protection_sha256=sha(main/'configs/protection.yaml'),
          file=str(main/'复赛_train/train_samples.csv'), process_id=os.getpid())
    return load_training_frame(main)


def reference(manifest, frame, seed):
    binding = read(manifest['binding_path'])
    exported_ids = np.load(binding['ids']['path'], allow_pickle=False)
    if len(set(exported_ids)) != 2754 or set(exported_ids) != set(frame.sample_id):
        raise ValueError('Reference population differs')
    pos = {v:i for i,v in enumerate(exported_ids)}
    order = [pos[v] for v in frame.sample_id]
    fv = np.load(binding['columns'][str(seed)]['folds']['path'], allow_pickle=False)[order]
    native = load_fold_vector(Path(manifest['spec']['main_root']), frame, seed)
    np.testing.assert_array_equal(fv, native)
    time_column = np.load(binding['columns'][str(seed)]['time']['path'], allow_pickle=False)[order]
    iron = np.full(len(frame), np.nan)
    for fold in range(5):
        info = manifest['reference_units'][f's{seed}-f{fold}']
        with np.load(info['path'], allow_pickle=False) as a:
            held = fv == fold
            np.testing.assert_array_equal(a['query_ids'], frame.loc[held,'sample_id'].to_numpy(str))
            np.testing.assert_array_equal(a['q75'], time_column[held])
            iron[held] = a[info['iron_field']]
    result = np.column_stack([iron, time_column])
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid reference')
    return fv, result


def freeze(work, out, tests):
    work, out = Path(work).resolve(), Path(out).resolve()
    spec = read(work/SPEC); main = Path(spec['main_root'])
    if out.exists() or not out.is_relative_to(main/'local/runs'):
        raise ValueError('Fresh private run directory required')
    source = sources(work)
    if read(tests)['sources'] != source or read(tests)['status'] != 'passed':
        raise ValueError('Exact-source test receipt required')
    if subprocess.check_output(['git','status','--porcelain','--',*source],cwd=work,text=True).strip():
        raise ValueError('Commit tested science source before freezing')
    best = read(main/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best[k] != v for k,v in spec['reference'].items()):
        raise ValueError('Current reference changed')
    binding_path = main/spec['reference_binding']
    if sha(binding_path) != spec['reference_binding_sha256']:
        raise ValueError('Q75 export binding changed')
    binding = read(binding_path)
    files = dict(source)
    files.update(binding['frozen_original_evidence'])
    for item in [binding['ids'], *[v for a in binding['columns'].values() for v in a.values()]]:
        files[item['path']] = item['sha256']
    files[str(binding_path)] = sha(binding_path)
    units = {}
    for seed in spec['development_seeds']+spec['confirmation_seeds']:
        for fold in range(5):
            development = seed in spec['development_seeds']
            p = Path(binding['original_development_source'] if development else binding['original_confirmation_source'])
            p /= f'SHORT_SPAN-s{seed}-f{fold}' if development else f's{seed}-f{fold}'
            units[f's{seed}-f{fold}'] = dict(path=str(p/'predictions.npz'),
                iron_field='iron_reference' if development else 'iron')
    for path in [main/'configs/protection.yaml', main/'configs/candidate_tiers.yaml', Path(tests),
                 *sorted((main/'复赛_train').glob('*.csv'))]:
        files[str(path.resolve())] = sha(path)
    for seed in spec['development_seeds']:
        p = main/f'local/runs/round2-v2/comparison-r1/folds-{seed}.csv'
        files[str(p)] = sha(p)
    probe = out.parent/'engineering-r1'
    if read(probe/'cold.json')['status'] != 'passed' or read(probe/'actual-exit.json')['exit_code'] != 0:
        raise ValueError('Successful complete-size synthetic cold admission required')
    if read(probe/'warm.json')['sources'] != source or read(probe/'warm.json')['runtime'] != runtime():
        raise ValueError('Synthetic engineering identity differs')
    files.update({str(p):sha(p) for p in probe.iterdir() if p.is_file()})
    verify(files); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(spec=spec, files=files, sources=source, workspace=str(work), binding_path=str(binding_path),
                    reference_units=units, runtime=runtime(),
                    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip())
    write(out/'manifest.json', manifest)
    frame = load_labels(out,manifest)
    plan = {}
    for seed in spec['development_seeds']:
        fv, _ = reference(manifest, frame, seed)
        from .splits import make_folds
        expected = make_folds(frame,seed).set_index('sample_id').loc[frame.sample_id,'fold'].to_numpy()
        np.testing.assert_array_equal(fv,expected)
        for fold in range(5):
            plan[f's{seed}-f{fold}'] = dict(training=digest(frame.loc[fv!=fold,'sample_id'].tolist()),
                query=digest(frame.loc[fv==fold,'sample_id'].tolist()),rows=int(sum(fv==fold)))
    write(out/'activation.json',dict(manifest_sha256=sha(out/'manifest.json'),plan=plan))


def worker(out, seed, fold):
    out = Path(out); manifest = read(out/'manifest.json'); verify(manifest['files'])
    if runtime() != manifest['runtime'] or seed not in manifest['spec']['development_seeds'] or fold not in range(5):
        raise ValueError('Unregistered unit or runtime')
    activation = read(out/'activation.json')
    if activation['manifest_sha256'] != sha(out/'manifest.json'):
        raise ValueError('Activation differs')
    unit = out/f's{seed}-f{fold}'; unit.mkdir(exist_ok=False)
    identity = dict(source_directory=str(out.resolve()),seed=seed,trial_id='LOCAL_RIDGE64_A20',fold=fold)
    event(out,'unit_reserved',**identity)
    try:
        frame = load_labels(out,manifest); fv, ref = reference(manifest,frame,seed)
        train = frame.loc[fv!=fold,['sample_id','spout_no',*FEATURES]]
        query = frame.loc[fv==fold,['sample_id','spout_no',*FEATURES]]
        plan = activation['plan'][unit.name]
        if plan['training'] != digest(train.sample_id.tolist()) or plan['query'] != digest(query.sample_id.tolist()):
            raise ValueError('Frozen partitions differ')
        event(out,'preprocessing_started',**identity)
        model = LocalRidgeNeighborhood().fit(train,frame.loc[fv!=fold,list(TARGETS)].to_numpy())
        model.save(unit/'state.npz'); event(out,'preprocessing_completed',**identity)
        event(out,'local_systems_reserved',count=len(query),target_rhs=2*len(query),**identity)
        predictions = model.predict(query)
        event(out,'local_systems_completed',count=len(query),target_rhs=2*len(query),**identity)
        endpoint = .8*ref[fv==fold]+.2*predictions['linear']
        control = .8*ref[fv==fold]+.2*predictions['knn']
        if (endpoint < 0).any() or (control < 0).any() or not np.isfinite([endpoint,control]).all():
            raise ValueError('Invalid fixed endpoint; no clipping')
        with (unit/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream,query_ids=query.sample_id.to_numpy(str),reference=ref[fv==fold],
                candidate=endpoint,control=control,**predictions)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > manifest['spec']['max_rss_mib']:
            raise ValueError('Memory gate failed')
        verify(manifest['files'])
        write(unit/'complete.json',dict(identity=identity,partitions=plan,peak_rss_mib=peak,
            manifest_sha256=sha(out/'manifest.json'),hashes={p.name:sha(p) for p in unit.iterdir()}))
        event(out,'unit_completed',**identity)
    except BaseException as error:
        write(unit/'failure.json',dict(error=repr(error))); event(out,'unit_failed',error=repr(error),**identity)
        raise


def audit_state(model, train, y):
    np.testing.assert_array_equal(model.ids,train.sample_id.to_numpy(str))
    raw = train[list(FEATURES)].to_numpy(float)
    mean = np.array([math.fsum(v)/len(v) for v in raw.T])
    scale = np.array([math.sqrt(math.fsum((float(a)-b)**2 for a in v)/len(v)) for v,b in zip(raw.T,mean)])
    scale[scale==0] = 1.
    ym = np.array([math.fsum(v)/len(v) for v in y.T])
    ys = np.array([math.sqrt(math.fsum((float(a)-b)**2 for a in v)/len(v)) for v,b in zip(y.T,ym)])
    ys[ys==0] = 1.
    for actual,expected in [(model.mean,mean),(model.scale,scale),(model.ymean,ym),(model.yscale,ys)]:
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
    categories = np.unique(train.spout_no.to_numpy(np.int64))
    np.testing.assert_array_equal(model.categories,categories)
    onehot = np.column_stack([np.zeros(len(train)), *[train.spout_no.to_numpy()==c for c in categories]])
    np.testing.assert_allclose(model.x,np.column_stack([(raw-mean)/scale,onehot]),rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(model.y,(y-ym)/ys,rtol=1e-12,atol=1e-10)


def compare(model, query, saved):
    warm = model.predict(query)
    reverse = model.predict(query.iloc[::-1])
    chunks = [model.predict(query.iloc[i:i+73]) for i in range(0,len(query),73)]
    alternate = independent_predict(model,query)
    maximum = 0.
    for name in ('linear','knn','neighbors','beta'):
        vectors = [warm[name],reverse[name][::-1],np.concatenate([a[name] for a in chunks]),alternate[name]]
        for actual in vectors:
            if name=='neighbors': np.testing.assert_array_equal(actual,saved[name])
            else:
                difference = float(np.max(np.abs(actual-saved[name])))
                maximum = max(maximum,difference)
                np.testing.assert_allclose(actual,saved[name],rtol=0,atol=1e-8)
    np.testing.assert_allclose(warm['condition'],saved['condition'],rtol=0,atol=1e-8)
    return maximum


def cold(out):
    out = Path(out); manifest = read(out/'manifest.json'); verify(manifest['files'])
    if runtime()!=manifest['runtime']: raise ValueError('Cold runtime differs')
    frame = load_labels(out,manifest); records = {}
    for seed in manifest['spec']['development_seeds']:
        fv, ref = reference(manifest,frame,seed)
        for fold in range(5):
            unit = out/f's{seed}-f{fold}'; complete = read(unit/'complete.json')
            if complete['manifest_sha256']!=sha(out/'manifest.json'): raise ValueError('Unit manifest differs')
            verify({str(unit/name):h for name,h in complete['hashes'].items()})
            model = LocalRidgeNeighborhood.load(unit/'state.npz')
            train = frame.loc[fv!=fold,['sample_id','spout_no',*FEATURES]]
            query = frame.loc[fv==fold,['sample_id','spout_no',*FEATURES]]
            if set(train.sample_id)&set(query.sample_id): raise ValueError('Partition overlap')
            audit_state(model,train,frame.loc[fv!=fold,list(TARGETS)].to_numpy())
            with np.load(unit/'predictions.npz',allow_pickle=False) as a:
                saved = {k:a[k].copy() for k in a.files}
            np.testing.assert_array_equal(saved['query_ids'],query.sample_id.to_numpy(str))
            np.testing.assert_array_equal(saved['reference'],ref[fv==fold])
            for name,member in [('candidate','linear'),('control','knn')]:
                np.testing.assert_array_equal(saved[name],.8*saved['reference']+.2*saved[member])
            difference = compare(model,query,saved)
            records[unit.name] = dict(maximum_difference=difference,
                maximum_condition=float(saved['condition'].max()),maximum_coefficient=float(abs(saved['beta']).max()),
                inference_audit_local_systems=4*len(query),model_sha256=sha(unit/'state.npz'),
                predictions_sha256=sha(unit/'predictions.npz'))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>manifest['spec']['max_rss_mib']: raise ValueError('Cold memory gate failed')
    verify(manifest['files'])
    write(out/'cold.json',dict(status='passed',manifest_sha256=sha(out/'manifest.json'),records=records,
        preprocessing_refits=0,optimizer_runs=0,peak_rss_mib=peak))


def summarize(out):
    from .candidate_tiers import classify_candidates
    from .v49_run import metric_detail
    out=Path(out); manifest=read(out/'manifest.json'); verify(manifest['files'])
    if read(out/'cold.json')['status']!='passed': raise ValueError('Cold audit required before scoring')
    frame=load_labels(out,manifest); records={}; metrics={t:{a:{} for a in ('Q75','LOCAL_RIDGE64_A20')} for t in TARGETS}
    y=frame[list(TARGETS)].to_numpy(); spouts=frame.spout_no.to_numpy()
    for seed in manifest['spec']['development_seeds']:
        fv,ref=reference(manifest,frame,seed); a=np.full_like(ref,np.nan); b=np.full_like(ref,np.nan)
        for fold in range(5):
            with np.load(out/f's{seed}-f{fold}'/'predictions.npz',allow_pickle=False) as saved:
                a[fv==fold]=saved['candidate']; b[fv==fold]=saved['control']
        if not np.isfinite([a,b]).all(): raise ValueError('Full same-seed OOF required')
        for j,t in enumerate(TARGETS):
            base=math.fsum(abs(float(v)-float(p)) for v,p in zip(y[:,j],ref[:,j]))
            denom=math.fsum(map(float,y[:,j]))
            errors={name:math.fsum(abs(float(v)-float(p)) for v,p in zip(y[:,j],pred[:,j])) for name,pred in [('candidate',a),('control',b)]}
            records.setdefault(t,{})[str(seed)] = dict(gain=50*(base-errors['candidate'])/denom,
                control_gain=50*(base-errors['control'])/denom,
                mechanism_gain=50*(errors['control']-errors['candidate'])/denom)
            for name,pred in [('Q75',ref),('LOCAL_RIDGE64_A20',a)]:
                metrics[t][name][str(seed)]=metric_detail(y[:,j],pred[:,j],fv,spouts)
            # Separate vectorized arithmetic must agree with scalar fsum scoring.
            fast=50*(np.sum(abs(y[:,j]-ref[:,j]))-np.sum(abs(y[:,j]-a[:,j])))/denom
            if abs(fast-records[t][str(seed)]['gain'])>1e-10: raise ValueError('Independent score differs')
    decisions={t:dict(eligible=all(v['gain']>0 for v in rows.values()) and
        np.mean([v['mechanism_gain'] for v in rows.values()])>0,
        mean_gain=float(np.mean([v['gain'] for v in rows.values()])),
        mean_mechanism_gain=float(np.mean([v['mechanism_gain'] for v in rows.values()]))) for t,rows in records.items()}
    tier_spec=dict(split_seeds=[42,3407],folds=5,candidates={t:['LOCAL_RIDGE64_A20'] for t in TARGETS},
        reference_by_target={t:'Q75' for t in TARGETS},tie_preference_by_target={t:['LOCAL_RIDGE64_A20'] for t in TARGETS})
    tiers=classify_candidates(metrics,tier_spec,yaml.safe_load((Path(manifest['spec']['main_root'])/'configs/candidate_tiers.yaml').read_text()))
    events=[json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
    counts={k:sum(e['event']==k for e in events) for k in ('unit_reserved','unit_completed','unit_failed','preprocessing_started','preprocessing_completed')}
    systems={k:sum(e['count'] for e in events if e['event']==k) for k in ('local_systems_reserved','local_systems_completed')}
    if counts!={'unit_reserved':10,'unit_completed':10,'unit_failed':0,'preprocessing_started':10,'preprocessing_completed':10} or any(v!=5508 for v in systems.values()):
        raise ValueError('Frozen development budget did not close')
    write(out/'summary.json',dict(status='complete_development',records=records,decisions=decisions,tiers=tiers,
        counts=counts,local_systems=systems,optimizer_runs=0,formal_promoted=False,new_confirmation_seeds=0,
        manifest_sha256=sha(out/'manifest.json'),cold_sha256=sha(out/'cold.json'),
        full_data_fits=0,packages=0,desktop_writes=0,uploads=0))


def execute(out):
    out=Path(out); manifest=read(out/'manifest.json'); start=time.monotonic()
    event(out,'controller_started',pid=os.getpid())
    try:
        for seed in manifest['spec']['development_seeds']:
            for fold in range(5):
                with (out/f'worker-s{seed}-f{fold}.log').open('x') as log:
                    subprocess.run([sys.executable,'-m',__spec__.name,'worker',str(out),'--seed',str(seed),'--fold',str(fold)],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/'cold.log').open('x') as log:
            result=subprocess.run([sys.executable,'-m',__spec__.name,'cold',str(out)],stdout=log,stderr=subprocess.STDOUT)
        write(out/'cold-process-exit.json',dict(exit_code=result.returncode))
        if result.returncode: raise RuntimeError('Independent cold process failed')
        summarize(out); verify(manifest['files'])
        event(out,'controller_completed',pid=os.getpid(),seconds_descriptive=time.monotonic()-start)
        write(out/'terminal.json',dict(status='passed',seconds_descriptive=time.monotonic()-start,
            manifest_sha256=sha(out/'manifest.json'),cold_sha256=sha(out/'cold.json'),summary_sha256=sha(out/'summary.json'),
            events_sha256=sha(out/'events.jsonl'),controller_pid=os.getpid(),cold_exit_code=result.returncode))
    except BaseException as error:
        event(out,'controller_failed',error=repr(error),pid=os.getpid())
        write(out/'failure.json',dict(error=repr(error))); raise


def synthetic():
    import pandas as pd
    rng=np.random.default_rng(62002)
    x=rng.normal(size=(2754,len(FEATURES)))
    # Deliberate near-collinearity and exact constant dimensions.
    x[:,1]=x[:,0]+1e-10*x[:,1];x[:,-1]=0
    frame=pd.DataFrame(x,columns=FEATURES)
    frame.insert(0,'sample_id',[f'synthetic-{i:05d}' for i in range(len(x))])
    frame.insert(1,'spout_no',np.arange(len(x))%4+1)
    y=np.column_stack([200+3*x[:,0]+x[:,2]*x[:,3],40+x[:,4]+.5*x[:,5]**2])
    return frame.iloc[:2203],y[:2203],frame.iloc[2203:]


def probe(work,out):
    work,out=Path(work).resolve(),Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    train,y,query=synthetic();model=LocalRidgeNeighborhood().fit(train,y)
    model.save(out/'state.npz');pred=model.predict(query)
    with (out/'predictions.npz').open('xb') as stream: np.savez_compressed(stream,**pred)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>1024: raise ValueError('Synthetic memory gate failed')
    write(out/'warm.json',dict(sources=sources(work),runtime=runtime(),preprocessing_states=1,
        local_systems=551,target_rhs=1102,optimizer_runs=0,peak_rss_mib=peak,
        maximum_condition=float(pred['condition'].max()),maximum_coefficient=float(abs(pred['beta']).max()),
        hashes={str(p):sha(p) for p in (out/'state.npz',out/'predictions.npz')}))
    with (out/'cold.log').open('x') as log:
        result=subprocess.run([sys.executable,'-m',__spec__.name,'probe-cold',str(out)],stdout=log,stderr=subprocess.STDOUT)
    write(out/'cold-process-exit.json',dict(exit_code=result.returncode))
    if result.returncode: raise RuntimeError('Synthetic independent cold process failed')


def probe_cold(out):
    out=Path(out);warm=read(out/'warm.json');verify(warm['hashes'])
    model=LocalRidgeNeighborhood.load(out/'state.npz');train,y,query=synthetic()
    audit_state(model,train,y)
    with np.load(out/'predictions.npz',allow_pickle=False) as a: saved={k:a[k].copy() for k in a.files}
    difference=compare(model,query,saved)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>1024: raise ValueError('Synthetic cold memory gate failed')
    write(out/'cold.json',dict(status='passed',maximum_difference=difference,preprocessing_refits=0,
        inference_audit_local_systems=2204,optimizer_runs=0,peak_rss_mib=peak,warm_sha256=sha(out/'warm.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','execute','worker','cold','probe','probe-cold'))
    parser.add_argument('output');parser.add_argument('--work');parser.add_argument('--tests')
    parser.add_argument('--seed',type=int);parser.add_argument('--fold',type=int)
    args=parser.parse_args();runtime()
    if args.action=='freeze':freeze(args.work,args.output,args.tests)
    elif args.action=='worker':worker(args.output,args.seed,args.fold)
    elif args.action=='cold':cold(args.output)
    elif args.action=='probe':probe(args.work,args.output)
    elif args.action=='probe-cold':probe_cold(args.output)
    else:execute(args.output)


if __name__=='__main__': main()
