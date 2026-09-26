"""Confirm the frozen V8 iron finalist with newly fitted same-fold references."""
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

from .data import TARGETS
from .v5_library import fold_vector, load_v5_training_frame
from .v5_replicate import _identity
from .v5_resolution import nested_blend, package_score, wmape
from .v5_spec import load_v5_spec
from .v7_confirm import read_reference_fold
from .v7_periodic import digest, file_hash, references, write_new
from .v8_attention import AttentionRegressor


def strongest(summary, spec):
    expected = {(t, r) for t, recipes in spec['candidates'].items() for r in recipes}
    rows = summary['records']
    if (summary['status'] != 'development_complete' or len(rows) != len(expected)
            or {(r['target'], r['recipe']) for r in rows} != expected):
        raise ValueError('Complete frozen development pool required')
    seeds = {str(s) for s in spec['split_seeds']}
    eligible = []
    for row in rows:
        required = ['A35'] + (['V7_candidate_pool'] if row['target'] == 'tap_time_len' else [])
        if any(set(row['comparisons'][label]['seed_gains']) != seeds for label in required):
            raise ValueError('Complete development seed coverage required')
        if all(float(g) > 0 for label in required for g in row['comparisons'][label]['seed_gains'].values()):
            label = required[-1]
            mean = float(np.mean(list(row['comparisons'][label]['seed_gains'].values())))
            eligible.append((row, mean))
    if not eligible:
        raise ValueError('No eligible V8 candidate')
    return min(eligible, key=lambda pair: (-pair[1], spec['tie_preference_by_target'][pair[0]['target']].index(pair[0]['recipe']), pair[0]['target']))[0]


def reference_fold(root, frame, folds, fold, workers):
    from .v4_1_reference import V36FixedRecipeFactory
    train = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    bundle = V36FixedRecipeFactory(root, workers=workers).fit_predict(train, query)
    values = {target: np.asarray(bundle['b36'][target], dtype=float) for target in TARGETS}
    if any(v.shape != (len(query),) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Invalid baseline fold output')
    return values, bundle['meta']


def candidate_fold(frame, folds, fold, spec, selected):
    train = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = AttentionRegressor(spec['recipes'][selected['recipe']], spec['training'])
    model.fit(train, train[selected['target']].values)
    return model.predict(query), model.metadata_


def _job(root, frame, folds, fold, workers, spec, selected):
    baseline, metadata = reference_fold(root, frame, folds, fold, workers)
    pred, fitted = candidate_fold(frame, folds, fold, spec, selected)
    return baseline, pred, metadata, fitted


def check_control(actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid reference control arrays')
    difference = float(np.max(np.abs(actual - expected)))
    if difference > tolerance:
        raise ValueError(f'Reference control difference {difference} exceeds {tolerance}')
    return difference


def run(root, output):
    root = Path(root).resolve()
    spec_path, confirm_path = root/'configs/round2_v8/SPEC.yaml', root/'configs/round2_v8/CONFIRMATION.yaml'
    spec, config = yaml.safe_load(spec_path.read_text()), yaml.safe_load(confirm_path.read_text())
    dev = root/config['development_run']
    summary, manifest, audit = [json.loads((dev/name).read_text()) for name in ['summary.json','manifest.json','audit-r1.json']]
    selected = strongest(summary, spec)
    if {k: selected[k] for k in ['target','recipe']} != config['selected'] or selected['target'] != 'tap_iron':
        raise ValueError('Frozen iron finalist mismatch')
    if audit['status'] != 'passed' or audit['prediction_count'] != 40:
        raise ValueError('Complete development audit required')
    if file_hash(spec_path) != manifest['spec_sha256']:
        raise ValueError('Development specification changed')
    for name, sha in manifest['dependency_code_hashes'].items():
        if file_hash(root/'src/bf_tap_r2'/name) != sha:
            raise ValueError(f'Development source changed: {name}')
    for key, sha in audit['hashes'].items():
        if file_hash(dev/(key+'.npy')) != sha:
            raise ValueError('Development prediction changed')
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != manifest['versions']:
        raise ValueError('Development runtime changed')
    frame = load_v5_training_frame(root)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    if data_hash != manifest['data_digest']:
        raise ValueError('Development data changed')
    seeds = [*spec['split_seeds'], *config['confirmation_seeds']]
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in seeds}
    for seed in spec['split_seeds']:
        if digest(folds[seed].tolist()) != manifest['fold_digests'][str(seed)]:
            raise ValueError('Development folds changed')
    control = config['control']; cs, cf = control['seed'], control['fold']
    cache_path = root/control['reference']
    meta = {'kind':'v36','trial_id':'v36-s1-N-0048','target':'tap_time_len','line':'N'}
    expected_time, _ = read_reference_fold(cache_path, _identity(frame,folds[cs],cs,cf,meta), folds[cs]==cf)
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v8-feature-attention'):
        raise ValueError('Private V8 output required')
    out.mkdir(parents=True, exist_ok=False)
    private_sources = [root / name for name in [
        'local/runs/round2-v3.4-ebm-and-constrained-composition/run_l1_oof.py',
        'local/runs/round2-v3-local-search/coarse-catboost-r1/fit_ledger.jsonl',
        'local/runs/round2-v3.1-directed-search/s1-coarse-r1/fit_ledger.jsonl',
        'local/runs/round2-v3.2-ensemble-and-target-search/coarse-r1/fit_ledger.jsonl',
        'local/runs/round2-v3.3-structure-search/coarse-r1/fit_ledger.jsonl',
        'local/runs/round2-v3.1-directed-search/prepared-packages-r1/manifest.json',
        'local/runs/round2-v3.4-ebm-and-constrained-composition/release-candidates-r1/01_V34_A_MECHANICAL_CONSTRAINED/manifest.json',
        'local/runs/round2-v3.6-loss-training-and-numeric-encoding/v36-summary.json',
        'local/runs/round2-v3.6-loss-training-and-numeric-encoding/fixed-r2-final/fit_ledger.jsonl',
    ]] + [cache_path]
    config_sources = sorted((root/'configs').rglob('*.yaml'))
    write_new(out/'manifest.json', {'selected':config['selected'],'parent_spec_sha256':file_hash(spec_path),
        'confirmation_spec_sha256':file_hash(confirm_path),'development_summary_sha256':file_hash(dev/'summary.json'),
        'data_digest':data_hash,'fold_digests':{s:digest(f.tolist()) for s,f in folds.items()},
        'versions':versions,'source_hashes':{str(p.relative_to(root)):file_hash(p) for p in [*sorted((root/'src/bf_tap_r2').glob('*.py')),*config_sources,*private_sources]}})
    base, candidate = {s:np.full(len(frame),np.nan) for s in seeds}, {s:np.full(len(frame),np.nan) for s in seeds}
    recorded, _ = references(root,frame,spec)
    for seed in spec['split_seeds']:
        base[seed] = recorded[seed]['tap_iron']
        for fold in range(5):
            candidate[seed][folds[seed]==fold] = np.load(dev/f'tap_iron-{selected["recipe"]}-s{seed}-f{fold}.npy')
    def save(seed, fold, baseline, prediction, reference_metadata, fitted):
        mask = folds[seed]==fold
        path = out/f'seed-{seed}-fold-{fold}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, positions=np.flatnonzero(mask), iron=baseline['tap_iron'], time=baseline['tap_time_len'], candidate=prediction)
        base[seed][mask], candidate[seed][mask] = baseline['tap_iron'], prediction
        event = {'event':'complete','seed':seed,'fold':fold,'sha256':file_hash(path),'reference_metadata':reference_metadata,'candidate_metadata':fitted}
        with (out/'fit_ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps({k:event[k] for k in ['event','seed','fold','sha256']}),flush=True)
    # The engineering control is one of the ten required reference fits.
    baseline, metadata = reference_fold(root,frame,folds[cs],cf,config['budget']['baseline_workers_per_outer'])
    with (out/'control-baseline.npz').open('xb') as stream:
        np.savez_compressed(stream,iron=baseline['tap_iron'],time=baseline['tap_time_len'],expected_time=expected_time)
    try:
        difference = check_control(baseline['tap_time_len'],expected_time,control['maximum_absolute_difference'])
    except Exception as exc:
        write_new(out/'control-failure.json',{'status':'failed','error':repr(exc),'candidate_fits':0})
        raise
    write_new(out/'control.json',{'status':'passed','maximum_absolute_difference':difference,'reference_metadata':metadata,'baseline_sha256':file_hash(out/'control-baseline.npz')})
    print(json.dumps({'event':'reference_control_passed','maximum_absolute_difference':difference}),flush=True)
    prediction, fitted = candidate_fold(frame,folds[cs],cf,spec,selected)
    save(cs,cf,baseline,prediction,metadata,fitted)
    failures = 0
    with ProcessPoolExecutor(max_workers=config['budget']['outer_workers']) as pool:
        jobs = {pool.submit(_job,root,frame,folds[s],f,config['budget']['baseline_workers_per_outer'],spec,selected):(s,f)
                for s in config['confirmation_seeds'] for f in config['folds'] if (s,f)!=(cs,cf)}
        for future in as_completed(jobs):
            seed, fold = jobs[future]
            try:
                save(seed,fold,*future.result())
            except Exception as exc:
                failures += 1
                with (out/'fit_ledger.jsonl').open('a') as stream:
                    stream.write(json.dumps({'event':'failed','seed':seed,'fold':fold,'error':repr(exc)})+'\n')
    if failures or any(not np.isfinite(v).all() for v in [*base.values(),*candidate.values()]):
        raise RuntimeError('Incomplete confirmation; failure evidence retained')
    result = nested_blend(frame.tap_iron.values,folds,base,candidate,spec['blend_grid'])
    passed = (len(seeds)>=4 and all(g>0 for g in result['seed_gains'].values()) and result['seed_summary']['lcb95']>0)
    local_scores = []
    for seed in spec['split_seeds']:
        alpha = selected['comparisons']['A35']['alphas'][str(seed)]
        pred = (1-alpha)*base[seed]+alpha*candidate[seed]
        local_scores.append(package_score(wmape(frame.tap_iron.values,pred),wmape(frame.tap_time_len.values,recorded[seed]['tap_time_len'])))
    local_score = float(np.mean(local_scores))
    write_new(out/'summary.json',{'status':'confirmation_complete','selected':config['selected'],'A35_comparison':result,
        'four_seed_pass':bool(passed),'fold_level':'descriptive_only','development_package_score':local_score,
        'local_working_gate_met':local_score>=spec['promotion']['local_working_gate'],
        'baseline_outer_recipe_fits':10,'candidate_outer_fits':10,'packages':0,'uploads':0,'release_authorized':False})
    print(json.dumps({'event':'confirmation_complete','four_seed_pass':bool(passed),'seed_summary':result['seed_summary'],'local_score':local_score}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        if os.environ.get(key)!='1':parser.error(f'Set {key}=1')
    run(Path.cwd(),args.output)


if __name__=='__main__':
    main()
