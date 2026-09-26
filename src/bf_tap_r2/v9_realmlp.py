"""From-scratch author RealMLP recipes with explicit group-safe selection/refit."""
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
import torch
import yaml

from .data import FEATURES, TARGETS
from .v3_4_bags import group_safe_inner_folds
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import nested_blend, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, references, score_detail, write_new
from .v8_attention import v7_reference


class InputEncoder:
    """Train-only numeric imputation and spout vocabulary; no IDs or labels."""
    def fit(self, frame):
        x = frame[list(FEATURES)].to_numpy(dtype=float, copy=True)
        self.medians_ = np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
        spout = frame.spout_no.to_numpy(dtype=float)
        self.categories_ = np.unique(spout[np.isfinite(spout)])
        if not np.isfinite(self.medians_).all() or not len(self.categories_):
            raise ValueError('Cannot fit an empty feature or spout vocabulary')
        return self

    def transform(self, frame):
        x = frame[list(FEATURES)].to_numpy(dtype=float, copy=True)
        x = np.where(np.isfinite(x),x,self.medians_)
        cat = frame.spout_no.to_numpy(dtype=float)[:,None] == self.categories_[None,:]
        return np.ascontiguousarray(np.column_stack([x,cat]),dtype=np.float32)


def make_estimator(recipe, **overrides):
    from pytabkit import RealMLP_TD_Regressor, RealMLP_TD_S_Regressor
    classes = {c.__name__:c for c in [RealMLP_TD_Regressor,RealMLP_TD_S_Regressor]}
    cls = classes[recipe['class']]
    if cls(**recipe['constructor']).get_config() != recipe['resolved']:
        raise ValueError('Author resolved configuration changed')
    return cls(**{**recipe['constructor'],**overrides})


class RealMLPRegressor:
    def __init__(self, recipe, inner_seed=42):
        self.recipe, self.inner_seed = recipe, inner_seed

    def fit(self, frame, y):
        y = np.asarray(y,dtype=float)
        if y.shape!=(len(frame),) or not np.isfinite(y).all() or not np.std(y)>0:
            raise ValueError('Invalid RealMLP training target')
        torch.set_num_threads(1)
        mask = group_safe_inner_folds(frame,seed=self.inner_seed)['fold'] != 0
        inner = frame.loc[mask].reset_index(drop=True)
        encoder = InputEncoder().fit(inner)
        selector = make_estimator(self.recipe)
        selector.fit(encoder.transform(inner),y[mask],encoder.transform(frame.loc[~mask]),y[~mask])
        epoch = int(selector.fit_params_['stop_epoch']['mae'])
        if not 1<=epoch<=self.recipe['constructor']['n_epochs']:
            raise ValueError('Invalid selected epoch')
        self.encoder_ = InputEncoder().fit(frame)
        self.model_ = make_estimator(self.recipe,stop_epoch=epoch,val_fraction=0.)
        self.model_.fit(self.encoder_.transform(frame),y)
        self.metadata_ = {'selected_epoch':epoch,'schedule_horizon':self.recipe['constructor']['n_epochs'],
                          'fit_rows':len(frame),'inner_fit_rows':len(inner),'optimizer_runs':2,
                          'fit_ids_digest':digest(frame.sample_id.tolist()),'inner_ids_digest':digest(inner.sample_id.tolist()),
                          'inner_numeric_medians':encoder.medians_.tolist(),'outer_numeric_medians':self.encoder_.medians_.tolist(),
                          'inner_spout_vocabulary':encoder.categories_.tolist(),'outer_spout_vocabulary':self.encoder_.categories_.tolist(),
                          'input_columns':list(FEATURES)+[f'spout_{v}' for v in self.encoder_.categories_],
                          'native_shuffled_drop_last':True,'author_class':self.recipe['class']}
        return self

    def predict(self, frame):
        p = np.asarray(self.model_.predict(self.encoder_.transform(frame)),dtype=float)
        if p.shape!=(len(frame),) or not np.isfinite(p).all():
            raise ValueError('Invalid RealMLP prediction')
        return p


def _job(frame, folds, target, recipe, inner_seed, fold):
    train = frame.loc[folds!=fold].reset_index(drop=True)
    query = frame.loc[folds==fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = RealMLPRegressor(recipe,inner_seed).fit(train,train[target].values)
    return model.predict(query),model.metadata_


def v8_iron_reference(root, frame, folds, spec, base):
    directory = root/spec['v8_iron_diagnostic']['prediction_directory']
    manifest = json.loads((directory/'manifest.json').read_text())
    if hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()!=manifest['data_digest']:
        raise ValueError('V8 diagnostic data mismatch')
    ledger = {r['key']:r for r in [json.loads(l) for l in (directory/'fit_ledger.jsonl').read_text().splitlines()] if r['event']=='complete'}
    result, hashes = {},{}
    for seed in folds:
        if digest(folds[seed].tolist())!=manifest['fold_digests'][str(seed)]:
            raise ValueError('V8 diagnostic fold mismatch')
        pred = np.full(len(frame),np.nan)
        for fold in range(5):
            key = f'tap_iron-{spec["v8_iron_diagnostic"]["recipe"]}-s{seed}-f{fold}'
            path = directory/(key+'.npy'); sha = file_hash(path)
            if sha!=ledger[key]['prediction_sha256']:
                raise ValueError('V8 diagnostic prediction mismatch')
            pred[folds[seed]==fold] = np.load(path)
            hashes[str(path.relative_to(root))] = sha
        if not np.isfinite(pred).all():
            raise ValueError('V8 diagnostic incomplete coverage')
        alpha = spec['v8_iron_diagnostic']['blend_weight']
        result[seed] = {'tap_iron':(1-alpha)*base[seed]['tap_iron']+alpha*pred,
                        'tap_time_len':base[seed]['tap_time_len']}
    return result,hashes


def run(root, output):
    import pytabkit
    from .candidate_tiers import classify_candidates
    root = Path(root).resolve(); spec_path = root/'configs/round2_v9/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p:importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions!=spec['runtime_versions']:
        raise ValueError('V9 runtime mismatch')
    for recipe in spec['recipes'].values():
        make_estimator(recipe)
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v9-realmlp'):
        raise ValueError('Private V9 output required')
    out.mkdir(parents=True,exist_ok=False)
    frame = load_v5_training_frame(root)
    folds = {s:fold_vector(root,frame,s,load_v5_spec(root)) for s in spec['split_seeds']}
    reference_audit_path = root/spec['v8_iron_diagnostic']['prediction_directory']/'reference-audit-r2.json'
    reference_audit = json.loads(reference_audit_path.read_text())
    if reference_audit['status']!='passed':
        raise ValueError('Verified primary references required')
    for relative,sha in reference_audit['file_sha256'].items():
        if file_hash(root/relative)!=sha:
            raise ValueError('Historical primary reference changed')
    base,q20 = references(root,frame,spec)
    v7,v7_hashes = v7_reference(root,frame,folds,spec,base)
    v8,v8_hashes = v8_iron_reference(root,frame,folds,spec,base)
    author_root = Path(pytabkit.__file__).parent
    write_new(out/'manifest.json',{'spec_sha256':file_hash(spec_path),'code_sha256':file_hash(__file__),
              'versions':versions,'primary_reference_hashes':reference_audit['file_sha256'],'data_digest':hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
              'fold_digests':{s:digest(f.tolist()) for s,f in folds.items()},'v7_reference_hashes':v7_hashes,'v8_diagnostic_hashes':v8_hashes,
              'author_source_hashes':{str(p.relative_to(author_root)):file_hash(p) for p in sorted(author_root.rglob('*.py'))},
              'dependency_code_hashes':{p.name:file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    failures = 0
    with ProcessPoolExecutor(max_workers=spec['budget']['workers']) as pool:
        jobs = {pool.submit(_job,frame,folds[s],target,recipe,spec['inner_seed'],f):(target,name,s,f)
                for target in TARGETS for name,recipe in spec['recipes'].items() for s in folds for f in range(5)}
        for future in as_completed(jobs):
            target,name,seed,fold = jobs[future]
            key = f'{target}-{name}-s{seed}-f{fold}'
            try:
                prediction,metadata = future.result()
                path = out/(key+'.npy')
                with path.open('xb') as stream:np.save(stream,prediction)
                event = {'event':'complete','key':key,'metadata':metadata,'prediction_sha256':file_hash(path)}
            except Exception as exc:
                failures += 1
                event = {'event':'failed','key':key,'error':repr(exc)}
            with (out/'fit_ledger.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(json.dumps({k:v for k,v in event.items() if k!='metadata'}),flush=True)
    if failures:
        raise RuntimeError(f'V9 failures retained: {failures}')
    records,metrics = [],{}
    for target in TARGETS:
        y = frame[target].values
        metrics[target] = {spec['reference_by_target'][target]:{str(s):score_detail(y,base[s][target],folds[s],frame.spout_no.values) for s in folds}}
        for name in spec['recipes']:
            pred = {s:np.full(len(frame),np.nan) for s in folds}
            for s in folds:
                for f in range(5):pred[s][folds[s]==f] = np.load(out/f'{target}-{name}-s{s}-f{f}.npy')
                if not np.isfinite(pred[s]).all():raise ValueError('Incomplete V9 OOF coverage')
            comparisons = {label:nested_blend(y,folds,{s:ref[s][target] for s in folds},pred,spec['blend_grid'])
                           for label,ref in [('A35',base),('Q20',q20),('V7_candidate_pool',v7),('V8_iron_diagnostic',v8)]}
            eligible = all(g>0 for g in comparisons['A35']['seed_gains'].values())
            if target=='tap_time_len':eligible &= all(g>0 for g in comparisons['V7_candidate_pool']['seed_gains'].values())
            records.append({'target':target,'recipe':name,'comparisons':comparisons,'confirmation_eligible':bool(eligible),
                            'single_wmape':{s:wmape(y,pred[s]) for s in folds}})
            metrics[target][name] = {}
            for s in folds:
                alpha = comparisons['A35']['alphas'][s]
                metrics[target][name][str(s)] = score_detail(y,(1-alpha)*base[s][target]+alpha*pred[s],folds[s],frame.spout_no.values)
    tiers = classify_candidates(metrics,spec,yaml.safe_load((root/'configs/candidate_tiers.yaml').read_text()))
    write_new(out/'summary.json',{'status':'development_complete','records':records,'candidate_tiers':tiers,'packages':0,'uploads':0,'release_authorized':False})
    for row in records:
        print(json.dumps({'target':row['target'],'recipe':row['recipe'],'a35':row['comparisons']['A35']['seed_gains'],
                          'v7':row['comparisons']['V7_candidate_pool']['seed_gains'],'confirmation_eligible':row['confirmation_eligible']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        if os.environ.get(key)!='1':parser.error(f'Set {key}=1')
    run(Path.cwd(),args.output)


if __name__=='__main__':main()
