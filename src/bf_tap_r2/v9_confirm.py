"""Confirm frozen RealMLP iron using verified V8 same-fold baseline caches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import nested_blend
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, references, write_new
from .v8_confirm import strongest, check_control
from .v9_realmlp import _job


def read_reference(path, event, expected_positions, total_rows):
    if file_hash(path)!=event['sha256']:
        raise ValueError('Reference prediction hash mismatch')
    metadata = event['reference_metadata']
    if metadata['n_train']!=total_rows-len(expected_positions) or metadata['n_query']!=len(expected_positions) or metadata['weights_reselected']:
        raise ValueError('Reference fit identity mismatch')
    with np.load(path,allow_pickle=False) as cached:
        if cached['positions'].dtype.kind not in 'iu' or not np.array_equal(cached['positions'],expected_positions):
            raise ValueError('Reference row positions mismatch')
        iron = np.asarray(cached['iron'],dtype=float)
        if iron.shape!=(len(expected_positions),) or not np.isfinite(iron).all():
            raise ValueError('Reference iron coverage mismatch')
    return iron


def _setup(root):
    import pytabkit
    spec_path, config_path = root/'configs/round2_v9/SPEC.yaml', root/'configs/round2_v9/CONFIRMATION.yaml'
    spec, config = yaml.safe_load(spec_path.read_text()),yaml.safe_load(config_path.read_text())
    dev = root/config['development_run']; reference = root/config['reference']['run']
    summary, manifest, audit = [json.loads((dev/name).read_text()) for name in ['summary.json','manifest.json','audit-r1.json']]
    selected = strongest(summary,spec)
    if {k:selected[k] for k in ['target','recipe']}!=config['selected'] or selected['target']!='tap_iron':
        raise ValueError('Frozen V9 iron finalist mismatch')
    if audit['status']!='passed' or audit['prediction_count']!=40 or file_hash(spec_path)!=manifest['spec_sha256']:
        raise ValueError('Verified frozen V9 development required')
    for name,sha in manifest['dependency_code_hashes'].items():
        if file_hash(root/'src/bf_tap_r2'/name)!=sha:raise ValueError('Development source changed')
    for relative,sha in manifest['author_source_hashes'].items():
        if file_hash(Path(pytabkit.__file__).parent/relative)!=sha:raise ValueError('Author source changed')
    for key,sha in audit['hashes'].items():
        if file_hash(dev/(key+'.npy'))!=sha:raise ValueError('Development prediction changed')
    for relative,sha in manifest['primary_reference_hashes'].items():
        if file_hash(root/relative)!=sha:raise ValueError('Primary reference changed')
    versions = {p:importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions!=manifest['versions']:raise ValueError('Runtime changed')
    frame = load_v5_training_frame(root)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    if data_hash!=manifest['data_digest']:raise ValueError('Training snapshot changed')
    seeds = [*spec['split_seeds'],*config['confirmation_seeds']]
    folds = {s:fold_vector(root,frame,s,load_v5_spec(root)) for s in seeds}
    reference_manifest = json.loads((reference/'manifest.json').read_text())
    if reference_manifest['data_digest']!=data_hash:raise ValueError('Reference data mismatch')
    for seed in seeds:
        if digest(folds[seed].tolist())!=reference_manifest['fold_digests'][str(seed)]:
            raise ValueError('Reference fold mismatch')
    for relative,sha in reference_manifest['source_hashes'].items():
        if file_hash(root/relative)!=sha:raise ValueError('Frozen reference source changed')
    control = json.loads((reference/'control.json').read_text())
    if control['status']!='passed' or control['baseline_sha256']!=file_hash(reference/'control-baseline.npz'):
        raise ValueError('Verified same-fold reference control required')
    with np.load(reference/'control-baseline.npz',allow_pickle=False) as c:
        check_control(c['time'],c['expected_time'],1e-9)
    return spec,config,selected,frame,folds,dev,reference,versions


def _reference_events(reference, config):
    path = reference/'fit_ledger.jsonl'
    # The producing job is append-only; ignore a not-yet-terminated last line.
    rows = [json.loads(l) for l in path.read_text().splitlines(keepends=True) if l.endswith('\n')] if path.exists() else []
    completed = [r for r in rows if r['event']=='complete']
    events = {(r['seed'],r['fold']):r for r in completed}
    if len(events)!=len(completed):raise ValueError('Duplicate reference completion')
    expected = {(s,f) for s in config['confirmation_seeds'] for f in config['folds']}
    if set(events)-expected:raise ValueError('Unexpected reference fold')
    return events if set(events)==expected else None


def evaluate(root, output):
    root = Path(root).resolve()
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v9-realmlp'):
        raise ValueError('Private V9 output required')
    spec,config,selected,frame,folds,dev,reference,versions = _setup(root)
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['selected']!=config['selected'] or manifest['confirmation_spec_sha256']!=file_hash(root/'configs/round2_v9/CONFIRMATION.yaml'):
        raise ValueError('Confirmation specification changed')
    if (manifest['parent_spec_sha256']!=file_hash(root/'configs/round2_v9/SPEC.yaml')
            or manifest['development_summary_sha256']!=file_hash(dev/'summary.json')
            or manifest['versions']!=versions):
        raise ValueError('Confirmation parent identity changed')
    for seed,vector in folds.items():
        if manifest['fold_digests'][str(seed)]!=digest(vector.tolist()):
            raise ValueError('Confirmation fold identity changed')
    for relative,sha in manifest['source_hashes'].items():
        if file_hash(root/relative)!=sha:raise ValueError('Confirmation source changed')
    if manifest['reference_manifest_sha256']!=file_hash(reference/'manifest.json'):
        raise ValueError('Reference manifest changed')
    events = _reference_events(reference,config)
    if events is None:
        print(json.dumps({'event':'awaiting_existing_reference_job','candidate_fits_complete':True}),flush=True)
        return False
    ledger = [json.loads(l) for l in (out/'fit_ledger.jsonl').read_text().splitlines()]
    expected = {(s,f) for s in config['confirmation_seeds'] for f in config['folds']}
    if len(ledger)!=10 or any(r['event']!='complete' for r in ledger) or {(r['seed'],r['fold']) for r in ledger}!=expected:
        raise ValueError('Complete candidate confirmation coverage required')
    fitted = {(r['seed'],r['fold']):r for r in ledger}
    recorded,_ = references(root,frame,spec)
    base,candidate = {s:np.full(len(frame),np.nan) for s in folds},{s:np.full(len(frame),np.nan) for s in folds}
    hashes = {}
    for seed in folds:
        if seed in spec['split_seeds']:
            base[seed] = recorded[seed]['tap_iron']
        for fold in range(5):
            mask = folds[seed]==fold
            if seed in spec['split_seeds']:
                p = dev/f'tap_iron-{selected["recipe"]}-s{seed}-f{fold}.npy'
            else:
                p = out/f'seed-{seed}-fold-{fold}.npy'
                if file_hash(p)!=fitted[(seed,fold)]['prediction_sha256']:
                    raise ValueError('Confirmation prediction changed')
                ref_path = reference/f'seed-{seed}-fold-{fold}.npz'
                base[seed][mask] = read_reference(ref_path,events[(seed,fold)],np.flatnonzero(mask),len(frame))
                hashes[str(ref_path.relative_to(root))] = file_hash(ref_path)
            values = np.load(p)
            if values.shape!=(int(mask.sum()),) or not np.isfinite(values).all():raise ValueError('Candidate coverage mismatch')
            candidate[seed][mask] = values
    result = nested_blend(frame.tap_iron.values,folds,base,candidate,spec['blend_grid'])
    passed = len(folds)>=4 and all(g>0 for g in result['seed_gains'].values()) and result['seed_summary']['lcb95']>0
    audit = json.loads((dev/'audit-r1.json').read_text())
    local_score = next(r['development_package_score'] for r in audit['records'] if r['target']==selected['target'] and r['recipe']==selected['recipe'])
    write_new(out/'summary.json',{'status':'confirmation_complete','selected':config['selected'],'A35_comparison':result,
        'four_seed_pass':bool(passed),'fold_level':'descriptive_only','development_package_score':local_score,
        'local_working_gate_basis':'two_development_seeds_frozen_before_confirmation',
        'local_working_gate_met':local_score>=spec['promotion']['local_working_gate'],
        'verified_reference_hashes':hashes,'candidate_outer_fits':10,'new_baseline_outer_fits':0,
        'packages':0,'uploads':0,'release_authorized':False})
    print(json.dumps({'event':'confirmation_complete','four_seed_pass':bool(passed),'seed_summary':result['seed_summary']}),flush=True)
    return True


def fit(root, output):
    root = Path(root).resolve()
    spec,config,selected,frame,folds,dev,reference,versions = _setup(root)
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v9-realmlp'):raise ValueError('Private V9 output required')
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'manifest.json',{'selected':config['selected'],'parent_spec_sha256':file_hash(root/'configs/round2_v9/SPEC.yaml'),
        'confirmation_spec_sha256':file_hash(root/'configs/round2_v9/CONFIRMATION.yaml'),
        'development_summary_sha256':file_hash(dev/'summary.json'),'reference_manifest_sha256':file_hash(reference/'manifest.json'),
        'versions':versions,'fold_digests':{s:digest(f.tolist()) for s,f in folds.items()},
        'source_hashes':{str(p.relative_to(root)):file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    failures = 0
    with ProcessPoolExecutor(max_workers=config['budget']['workers']) as pool:
        jobs = {pool.submit(_job,frame,folds[s],selected['target'],spec['recipes'][selected['recipe']],spec['inner_seed'],f):(s,f)
                for s in config['confirmation_seeds'] for f in config['folds']}
        for future in as_completed(jobs):
            seed,fold = jobs[future]
            try:
                prediction,metadata = future.result()
                path = out/f'seed-{seed}-fold-{fold}.npy'
                with path.open('xb') as stream:np.save(stream,prediction)
                event = {'event':'complete','seed':seed,'fold':fold,'prediction_sha256':file_hash(path),'metadata':metadata}
            except Exception as exc:
                failures += 1
                event = {'event':'failed','seed':seed,'fold':fold,'error':repr(exc)}
            with (out/'fit_ledger.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(json.dumps({k:v for k,v in event.items() if k!='metadata'}),flush=True)
    if failures:raise RuntimeError('Failed confirmation evidence retained')
    write_new(out/'fit-phase.json',{'status':'complete','candidate_outer_fits':10,'baseline_outer_fits':0})
    evaluate(root,out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluate',action='store_true')
    args = parser.parse_args()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        if os.environ.get(key)!='1':parser.error(f'Set {key}=1')
    (evaluate if args.evaluate else fit)(Path.cwd(),args.output)


if __name__=='__main__':main()
