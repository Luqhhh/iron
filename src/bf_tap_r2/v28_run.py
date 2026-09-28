"""V28 append-only complete-coverage runner and hash-bound cold audit."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import time
import numpy as np
import yaml
from .data import TARGETS
from .v7_periodic import digest, file_hash, write_new
from .v23_run import load_references, data_digest, score_predictions, worker_init
from .v28_protocol import select_finalist, promotion_decision, require_audited_development

SPEC = 'configs/round2_v28_edge_kan/SPEC.yaml'
RUNS = 'local/runs/round2-v28-edge-spline-network'
ARMS = ('SHARED', 'EDGE')


def source_hashes(root):
    paths = sorted(list((root/'src').rglob('*.py')) + [p for p in (root/'configs').rglob('*') if p.is_file()])
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def environment(spec):
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError('Set '+key+'=1 before process launch')
    actual = {k: importlib.metadata.version(k) for k in spec['runtime_versions']}
    if actual != spec['runtime_versions']:
        raise ValueError('Frozen runtime mismatch')
    return actual


def append_event(directory, event):
    payload = (json.dumps(event, allow_nan=False)+'\n').encode()
    fd = os.open(directory/'fit_ledger.jsonl', os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
    try:
        # One atomic append; fsync precedes optimizer creation / completed-file claim.
        if os.write(fd, payload) != len(payload):
            raise OSError('Incomplete ledger write')
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_events(events, expected):
    result = {}
    for key in sorted(expected):
        rows = [e for e in events if e['key'] == key]
        signature = [(r['event'], r.get('stage')) for r in rows]
        if signature != [('fit_started',None), ('optimizer_started','inner'),
                         ('optimizer_started','outer'), ('complete',None)]:
            raise ValueError('Missing, failed or duplicate fit/optimizer events: '+key)
        result[key] = rows[-1]
    if {e['key'] for e in events} != set(expected):
        raise ValueError('Unexpected ledger fit identity')
    return result


def independent_score(frame, refs, b0, values, target):
    y = frame[target].to_numpy(dtype=float)
    mixed = .8*refs[target]+.2*values
    errors = {t: np.abs(frame[t].to_numpy()-(mixed if t == target else refs[t])).sum()
              / np.abs(frame[t].to_numpy()).sum() for t in TARGETS}
    old_error = np.abs(y-refs[target]).sum()/np.abs(y).sum()
    baseline_error = np.abs(y-b0[target]).sum()/np.abs(y).sum()
    return {'candidate_score': 100.-50.*sum(errors.values()),
        'gain': 50.*(old_error-errors[target]),
        'same_candidate_column_gain_vs_B0': 50.*(baseline_error-errors[target])}


def private_directory(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to((root/RUNS).resolve()) or resolved == (root/RUNS).resolve():
        raise ValueError('New private V28 run subdirectory required')
    return resolved


def require_phase_paths(root, output, development):
    parent = (root/RUNS).resolve()
    expected = parent/('development-r1' if development is None else 'confirmation-r1')
    if output.resolve() != expected or (development is not None and development.resolve() != parent/'development-r1'):
        raise ValueError('Exact canonical phase paths required; no retry directories')


def preflight(root, output):
    spec = yaml.safe_load((root/SPEC).read_text())
    versions = environment(spec)
    base = root/'local/v28-strategy-20260928'
    names = ['G0-resource-decision.json','G0-golden.json','G0-learnability-SHARED.json','G0-learnability-EDGE.json']
    reports = {n: json.loads((base/n).read_text()) for n in names}
    if (not reports[names[0]]['passed'] or reports[names[1]]['status'] != 'passed'
            or any(not reports[n]['passed'] for n in names[2:])):
        raise ValueError('All G0 checks must pass')
    starts = list(base.glob('G0-*-started.json'))
    if {p.name for p in starts} != {f'G0-{stage}-{arm}-started.json'
                                  for stage in ('resource','learnability') for arm in ARMS}:
        raise ValueError('Exactly four accounted G0 optimizers required')
    ledger = list(map(json.loads,(base/'G0-ledger.jsonl').read_text().splitlines()))
    if len(ledger) != 4 or {e['key'] for e in ledger} != {p.name for p in starts}:
        raise ValueError('G0 ledger coverage mismatch')
    frame, folds, b0, refs, hashes = load_references(root,[42,3407])
    from .v3_4_bags import group_safe_inner_folds
    for fv in folds.values():
        for f in range(5):
            training = frame.loc[fv != f].reset_index(drop=True)
            inner = group_safe_inner_folds(training,seed=42)['fold'] != 0
            if len(training)>2204 or inner.sum()>1764 or (~inner).sum()>441:
                raise ValueError('Resource projection row bounds exceeded')
    paths = [p for p in base.glob('G0-*') if p.is_file()] + [base/'g0_probe.py',base/'golden.py']
    paths += [base/'author-source/src__efficient_kan__kan.py',base/'author-source/LICENSE']
    result = {'status':'passed','spec_sha256':file_hash(root/SPEC),'source_hashes':source_hashes(root),
        'versions':versions,'data_digest':data_digest(frame),
        'fold_digests':{str(s):digest(fv.tolist()) for s,fv in folds.items()},'reference_hashes':hashes,
        'G0_hashes':{str(p.relative_to(root)):file_hash(p) for p in paths},
        'G0_resource':reports[names[0]],'official_fits':0,'G0_optimizer_runs':4}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_new(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','reference_hashes','G0_hashes')},indent=2))


def verify_gate(root, path):
    spec = yaml.safe_load((root/SPEC).read_text())
    versions = environment(spec)
    gate = json.loads(path.read_text())
    if (gate['status'] != 'passed' or gate['spec_sha256'] != file_hash(root/SPEC)
            or gate['source_hashes'] != source_hashes(root) or gate['versions'] != versions
            or any(file_hash(root/p) != h for p,h in gate['G0_hashes'].items())):
        raise ValueError('Successful frozen G0/source/spec required')
    return gate, versions


def expected_keys(manifest):
    return {f'{t}-{a}-s{s}-f{f}' for t in manifest['targets'] for a in ARMS
            for s in manifest['seeds'] for f in range(5)}


def artifacts_hash(directory, manifest):
    return digest({key+suffix:file_hash(directory/(key+suffix)) for key in sorted(expected_keys(manifest))
                   for suffix in ('.npy','.pkl')})


def evidence_identity(directory, manifest):
    return {k:manifest[k] for k in ('spec_sha256','source_hashes','gate_sha256')} | {
        'manifest_sha256':file_hash(directory/'manifest.json'),
        'ledger_sha256':file_hash(directory/'fit_ledger.jsonl'),
        'artifacts_digest':artifacts_hash(directory,manifest)}


def fit_job(directory, frame, fv, seed, fold, target, arm):
    from .v28_kan import KANRegressor
    key = f'{target}-{arm}-s{seed}-f{fold}'
    event = {'event':'fit_started','key':key,'time_ns':time.time_ns()}
    write_new(directory/(key+'.started.json'),event)
    append_event(directory,event)
    def started(row):
        row = {**row,'key':key,'time_ns':time.time_ns()}
        write_new(directory/(key+'.'+row['stage']+'-started.json'),row)
        append_event(directory,row)
    training = frame.loc[fv != fold].reset_index(drop=True)
    query = frame.loc[fv == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = KANRegressor(arm).fit(training,training[target].to_numpy(),started)
    values = model.predict(query)
    import torch
    model.metadata_['threads'] = {'intra':torch.get_num_threads(),'inter':torch.get_num_interop_threads()}
    model.session_.optimizer = None
    model.session_.model.zero_grad(set_to_none=True)
    model.session_.x = model.session_.y = None
    with (directory/(key+'.npy')).open('xb') as stream:
        np.save(stream,values,allow_pickle=False)
    with (directory/(key+'.pkl')).open('xb') as stream:
        pickle.dump(model,stream)
    return {'event':'complete','key':key,'metadata':model.metadata_,
        'prediction_sha256':file_hash(directory/(key+'.npy')),
        'model_sha256':file_hash(directory/(key+'.pkl'))}


def assemble(directory, frame, folds, target, arm):
    result = {}
    for s,fv in folds.items():
        vector = np.full(len(frame),np.nan)
        for f in range(5):
            value = np.load(directory/f'{target}-{arm}-s{s}-f{f}.npy',allow_pickle=False)
            if value.shape != (int((fv == f).sum()),) or not np.isfinite(value).all():
                raise ValueError('Incomplete fold prediction')
            vector[fv == f] = value
        if not np.isfinite(vector).all():
            raise ValueError('Incomplete OOF')
        result[s] = vector
    return result


def records_for(directory, frame, folds, b0, refs, targets):
    rows = []
    for target in targets:
        for arm in ARMS:
            values = assemble(directory,frame,folds,target,arm)
            scores = score_predictions(frame,folds,b0,refs,values,target)
            rows.append({'target':target,'arm':arm,'seed_results':scores,
                'seed_gains':{s:r['gain'] for s,r in scores.items()},
                'mean_gain':float(np.mean([r['gain'] for r in scores.values()]))})
    return rows


def confirmation_records(development_records, rows):
    result = []
    for row in rows:
        dev = next(r for r in development_records if (r['target'],r['arm']) == (row['target'],row['arm']))
        gains = {**dev['seed_gains'],**row['seed_gains']}
        scores = [dev['seed_results'][str(s)]['candidate_score'] for s in (42,3407)]
        decision = promotion_decision(gains,scores)
        decision['promoted'] = bool(row['arm']=='EDGE' and decision['promoted'])
        result.append({**row,'seed_results':{**dev['seed_results'],**row['seed_results']},
            'seed_gains':gains,'four_seed_decision':decision})
    return result


def run(root, output, gate_path, development=None, workers=4):
    output = private_directory(root,output)
    require_phase_paths(root,output,development)
    if not 1 <= workers <= 4:
        raise ValueError('Workers must be 1..4')
    gate, versions = verify_gate(root,gate_path)
    identity = {'spec_sha256':gate['spec_sha256'],'source_hashes':gate['source_hashes'],
                'gate_sha256':file_hash(gate_path)}
    dev_summary = None
    if development is None:
        if output.name != 'development-r1':
            raise ValueError('Frozen development-r1 required; retries forbidden')
        targets,seeds = list(TARGETS),[42,3407]
    else:
        development = private_directory(root,development)
        if output.name != 'confirmation-r1' or development.name != 'development-r1':
            raise ValueError('Frozen phase directories required')
        dev_manifest = json.loads((development/'manifest.json').read_text())
        if any(dev_manifest[k] != v for k,v in identity.items()):
            raise ValueError('Development identity changed')
        target = require_audited_development(development,evidence_identity(development,dev_manifest))
        audit(root,development,persist=False)
        dev_summary = json.loads((development/'summary.json').read_text())
        targets,seeds = [target],[7777,12011]
    frame,folds,b0,refs,hashes = load_references(root,seeds)
    if data_digest(frame) != gate['data_digest']:
        raise ValueError('Data changed')
    fold_hashes = {str(s):digest(v.tolist()) for s,v in folds.items()}
    if development is None and (fold_hashes != gate['fold_digests'] or hashes != gate['reference_hashes']):
        raise ValueError('Development fold/reference identity changed')
    output.mkdir(parents=True,exist_ok=False)
    manifest = {**identity,'gate_path':str(gate_path.relative_to(root)),'versions':versions,
        'data_digest':data_digest(frame),'fold_digests':fold_hashes,'reference_hashes':hashes,
        'targets':targets,'seeds':seeds,'workers':workers,'development':str(development.relative_to(root)) if development else None,
        'development_summary_sha256':file_hash(development/'summary.json') if development else None,
        'development_audit_sha256':file_hash(development/'audit-r1.json') if development else None}
    write_new(output/'manifest.json',manifest)
    started = time.time()
    jobs = [(s,f,t,a) for t in targets for a in ARMS for s in seeds for f in range(5)]
    failed = []
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks = {pool.submit(fit_job,output,frame,folds[s],s,f,t,a):f'{t}-{a}-s{s}-f{f}' for s,f,t,a in jobs}
        for future in as_completed(tasks):
            key = tasks[future]
            try:
                event = future.result()
            except Exception as exc:
                failed.append(key)
                event = {'event':'failed','key':key,'error':repr(exc)}
            append_event(output,event)
            print(json.dumps({k:v for k,v in event.items() if k!='metadata'}),flush=True)
    if failed:
        write_new(output/'failure.json',{'failed_fits':failed,'retry':False})
        raise RuntimeError('Failed fits retained; no retry or classification')
    if source_hashes(root) != manifest['source_hashes']:
        raise ValueError('Source changed during fits')
    rows = records_for(output,frame,folds,b0,refs,targets)
    if development is None:
        result = {'status':'development_complete','records':rows,'selected_for_confirmation':select_finalist(rows)}
    else:
        result = {'status':'confirmation_complete','records':confirmation_records(dev_summary['records'],rows),
                  'target':targets[0]}
    result.update({'fits':len(jobs),'failed_fits':0,'wall_seconds':time.time()-started,'packages':0,'uploads':0})
    write_new(output/'summary.json',result)
    print(json.dumps(result),flush=True)


def audit(root, directory, persist=True):
    directory = private_directory(root,directory)
    worker_init()
    manifest = json.loads((directory/'manifest.json').read_text())
    gate_path = root/manifest['gate_path']
    gate,versions = verify_gate(root,gate_path)
    if (manifest['source_hashes'] != gate['source_hashes'] or manifest['spec_sha256'] != gate['spec_sha256']
            or manifest['gate_sha256'] != file_hash(gate_path) or manifest['versions'] != versions):
        raise ValueError('Audit source/spec/gate changed')
    is_dev = manifest['development'] is None
    if (manifest['seeds'] != ([42,3407] if is_dev else [7777,12011])
            or manifest['targets'] != (list(TARGETS) if is_dev else [manifest['targets'][0]])
            or len(expected_keys(manifest)) != (40 if is_dev else 20)):
        raise ValueError('Frozen phase coverage required')
    frame,folds,b0,refs,hashes = load_references(root,manifest['seeds'])
    if (hashes != manifest['reference_hashes'] or data_digest(frame) != manifest['data_digest']
            or {str(s):digest(fv.tolist()) for s,fv in folds.items()} != manifest['fold_digests']):
        raise ValueError('Audit reference/data/fold identity changed')
    events = list(map(json.loads,(directory/'fit_ledger.jsonl').read_text().splitlines()))
    completed = validate_events(events,expected_keys(manifest))
    worst = 0.
    from .v28_kan import KANPreprocessor
    from .v3_4_bags import group_safe_inner_folds
    for key,event in completed.items():
        t,a,s,f = key.rsplit('-',3)
        seed,fold = int(s[1:]),int(f[1:])
        mask = folds[seed] == fold
        for suffix,field in (('.npy','prediction_sha256'),('.pkl','model_sha256')):
            if file_hash(directory/(key+suffix)) != event[field]:
                raise ValueError('Model/prediction hash changed')
        model = pickle.loads((directory/(key+'.pkl')).read_bytes())
        meta = event['metadata']
        if model.arm != a or model.metadata_ != meta or meta['threads'] != {'intra':1,'inter':1}:
            raise ValueError('Model arm/metadata/thread identity mismatch')
        training = frame.loc[~mask].reset_index(drop=True)
        y = training[t].to_numpy()
        split = group_safe_inner_folds(training,seed=42)
        inner = split['fold'] != 0
        if (meta['fit_ids_digest'] != digest(training.sample_id.tolist())
                or meta['inner_ids_digest'] != digest(training.loc[inner,'sample_id'].tolist())
                or meta['validation_ids_digest'] != digest(training.loc[~inner,'sample_id'].tolist())
                or meta['group_hash'] != split['group_hash'] or meta['inner_fold_hash'] != split['inner_fold_hash']):
            raise ValueError('Train/inner/group identity mismatch')
        for index,snap in ((inner,meta['inner']),(np.ones(len(y),dtype=bool),meta)):
            pre = KANPreprocessor().fit(training.loc[index])
            if (snap['target_mean'] != float(y[index].mean()) or snap['target_std'] != max(float(y[index].std()),1e-8)
                    or snap['vocabulary'] != pre.vocabulary_.tolist()
                    or snap['quantiles'] != pre.quantiles_.quantiles_.tolist()
                    or snap['quantile_references'] != pre.quantiles_.references_.tolist()):
                raise ValueError('Train-only preprocessing/scaling audit failure')
        query = frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
        saved = np.load(directory/(key+'.npy'),allow_pickle=False)
        delta = max(float(np.max(np.abs(model.predict(query,chunk_rows=c)-saved))) for c in (256,73))
        if delta>1e-6:
            raise ValueError('Cold/chunk inference audit failure')
        worst = max(worst,delta)
    rows = records_for(directory,frame,folds,b0,refs,manifest['targets'])
    for row in rows:
        values = assemble(directory,frame,folds,row['target'],row['arm'])
        for s in manifest['seeds']:
            independent = independent_score(frame,refs[s],b0[s],values[s],row['target'])
            for k,v in independent.items():
                if abs(v-row['seed_results'][str(s)][k])>1e-11:
                    raise ValueError('Independent pooled arithmetic mismatch')
    summary = json.loads((directory/'summary.json').read_text())
    if is_dev:
        if rows != summary['records'] or select_finalist(rows) != summary['selected_for_confirmation']:
            raise ValueError('Independent development selection mismatch')
    else:
        development = private_directory(root,root/manifest['development'])
        dev_manifest = json.loads((development/'manifest.json').read_text())
        target = require_audited_development(development,evidence_identity(development,dev_manifest))
        if (manifest['targets'] != [target] or file_hash(development/'summary.json') != manifest['development_summary_sha256']
                or file_hash(development/'audit-r1.json') != manifest['development_audit_sha256']):
            raise ValueError('Development chain changed')
        audit(root,development,persist=False)
        dev = json.loads((development/'summary.json').read_text())
        if confirmation_records(dev['records'],rows) != summary['records']:
            raise ValueError('Independent four-seed decision mismatch')
    report = {'status':'passed','fits':len(completed),'optimizer_runs':2*len(completed),
        'cold_chunk_max_difference':worst,'summary_sha256':file_hash(directory/'summary.json'),
        **evidence_identity(directory,manifest),'packages':0,'uploads':0}
    if persist:
        write_new(directory/'audit-r1.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--gate',type=Path)
    parser.add_argument('--development',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = private_directory(root,root/args.output)
    if args.preflight:
        preflight(root,output)
    elif args.audit:
        audit(root,output)
    else:
        if args.gate is None:
            parser.error('--gate required')
        run(root,output,(root/args.gate).resolve(),(root/args.development).resolve() if args.development else None,args.workers)


if __name__ == '__main__':
    main()