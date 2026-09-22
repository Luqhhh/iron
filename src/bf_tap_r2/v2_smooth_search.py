"""Frozen V2.4 smooth-residual comparison, replay and isolated release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .candidate_tiers import classify_candidates
from .cv import target_metrics
from .data import TARGETS
from .low_signal_cv import sensitivity
from .smooth_residual import make_model
from .submission import ZIP_NAME, deny_training_reads, package, validate_result
from .v2_release import load_v2
from .v2_refinement import fit_once, fold_vector, isolated_payload
from .v2_robust_joint import checked_manifest, id_digest


def check(root, output):
    manifest = json.loads((output/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    return manifest


def aligned(path, train, folds):
    frame = pd.read_csv(path)
    if not frame.sample_id.is_unique or set(frame.sample_id) != set(train.sample_id):
        raise ValueError('OOF ID mismatch')
    frame = frame.set_index('sample_id').loc[train.sample_id]
    np.testing.assert_array_equal(frame.fold, folds)
    return frame


def initialize(root, output, config):
    if not output.is_relative_to(root/'local/runs/round2-v2.4'):
        raise ValueError('Fresh private output required')
    spec = yaml.safe_load(config.read_text())
    old = checked_manifest(root, root/spec['source_run'])
    files = dict(old['files'])
    extra = [config, Path(__file__), Path(__file__).with_name('smooth_residual.py'),
             root/spec['policy'], root/'uv.lock', root/spec['source_run']/'selection/summary.json',
             root/spec['source_run']/'selection/independent_verification.json']
    for key in ('source_run','v22_run'):
        extra.extend((root/spec[key]/'oof').glob('*.csv'))
    # Preserve both existing pending packages and the removed T1 artifact.
    prior = root/'local/runs/round2-v2.3/candidate-release-r1'
    delivery = json.loads((prior/'release_manifest.json').read_text())
    extra.append(prior/'release_manifest.json')
    for name, record in delivery['packages'].items():
        for filename in (ZIP_NAME,'result.csv'):
            extra.extend([prior/'release'/name/filename, Path(record['desktop'])/filename])
    for path in extra:
        files[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)] = digest(path)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('models','oof','selection','release'):
        (output/folder).mkdir()
    write_json(output/'manifest.json', {'spec':spec,'files':files,
        'policy':yaml.safe_load((root/spec['policy']).read_text()),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()})
    return spec


def collect(root, output, replay=False):
    manifest = check(root,output)
    spec = manifest['spec']
    train = load_v2(root/'复赛_train','train',2754)
    metrics, predictions = {}, {}
    for target in TARGETS:
        reference = spec['reference_by_target'][target]
        comparisons = spec['comparison_by_target'][target]
        routes = [reference,*comparisons,*spec['candidates'][target]]
        metrics[target] = {r:{} for r in routes}
        predictions[target] = {r:{} for r in routes}
        for split in spec['split_seeds']:
            folds = fold_vector(root/spec['fold_run'],train,split)
            old = aligned(root/spec['source_run']/'oof'/f'{split}-{target}.csv',train,folds)
            values = {r:old[r].to_numpy() for r in [reference,*comparisons]}
            np.testing.assert_allclose(old[target],train[target],rtol=1e-14)
            for route in spec['routes']:
                values[route] = np.full(len(train),np.nan)
                for fold in range(spec['folds']):
                    training, valid = train.loc[folds!=fold],train.loc[folds==fold]
                    path = output/'models'/f'{split}-{target}-{fold}-{route}.joblib'
                    identity = {'route':route,'target':target,'split':split,'heldout_fold':fold,
                                'train_ids':id_digest(training),'valid_ids':id_digest(valid)}
                    if replay:
                        model = joblib.load(path)
                        if model.model_identity_ != identity:
                            raise ValueError('Held-out identity mismatch')
                        scale = model.target_scale_ if route=='QRC' else model.target_scales_[0]
                        np.testing.assert_allclose(scale,training[target].mean(),rtol=1e-14)
                    else:
                        model = make_model(route,target,spec)
                        model.model_identity_ = identity
                        fit_once(model,training,training[target],path,output,'cv',identity,spec['budget']['cv_wrapper_fits'])
                        write_json(path.with_suffix('.json'),{'identity':identity,'actual_parameters':model.actual_parameters_,
                            'model_sha256':digest(path),'regressor_fits':2 if route=='QRC' else 1})
                    record = json.loads(path.with_suffix('.json').read_text())
                    if digest(path)!=record['model_sha256'] or model.actual_parameters_!=record['actual_parameters']:
                        raise ValueError('Model or parsed parameters mismatch')
                    with threadpool_limits(limits=1):
                        pred = model.predict(valid)
                        if replay:
                            np.testing.assert_allclose(pred,model.predict(valid.iloc[::-1])[::-1],rtol=1e-12,atol=1e-10)
                            batched=np.concatenate([model.predict(valid.iloc[i:i+127]) for i in range(0,len(valid),127)])
                            np.testing.assert_allclose(pred,batched,rtol=1e-12,atol=1e-10)
                    values[route][folds==fold] = pred
                values['A'+route] = .5*values[reference]+.5*values[route]
            if target=='tap_iron':
                values['DJ'] = .5*values['I2_C2_D4_IRON_EQUAL']+.5*values['AJ']
            saved = train[['sample_id','spout_no',target]].copy()
            saved['fold'] = folds
            for route in routes:
                predictions[target][route][split]=values[route]
                metrics[target][route][str(split)]=target_metrics(train,target,values[route],folds)
                saved[route]=values[route]
            path=output/'oof'/f'{split}-{target}.csv'
            if replay:
                stored=aligned(path,train,folds)
                np.testing.assert_allclose(stored[saved.columns.drop('sample_id')],saved.drop(columns='sample_id'),rtol=1e-12,atol=1e-10)
            else:
                saved.to_csv(path,index=False,mode='x',float_format='%.17g')
    return train,metrics,predictions


def summarize(root,output,train,metrics,predictions,replay=False):
    manifest=check(root,output)
    spec=manifest['spec']
    tiers=classify_candidates(metrics,spec,manifest['policy'])
    comparisons=[]
    for target in TARGETS:
        for route in spec['candidates'][target]:
            d=tiers['decisions'][target][route]
            row={'target':target,'candidate':route,'mean_wmape':d['mean_wmape'],'delta_current':-d['mean_gain'],
                 'improved_folds':d['improved_folds'],'worst_spout_delta':d['worst_spout_delta'],
                 'tier':d['tier'],'failed_conditions':','.join(d['failed_conditions'])}
            for split in spec['split_seeds']:
                row[f'wmape_{split}']=metrics[target][route][str(split)]['wmape']
            for baseline in spec['comparison_by_target'][target]:
                row['delta_'+baseline]=d['mean_wmape']-np.mean([metrics[target][baseline][str(s)]['wmape'] for s in spec['split_seeds']])
            comparisons.append(row)
    bootstrap={}
    for record in tiers['submission_priority']:
        target,route=record['target'],record['candidate']
        distribution,intervals=sensitivity(train,target,predictions[target][route],predictions[target][spec['reference_by_target'][target]],spec['bootstrap'])
        key=f'{target}-{route}'
        bootstrap[key]={'intervals':intervals,'fraction_negative':float((distribution['mean']<0).mean())}
        path=output/'selection'/f'bootstrap-{key}.csv'
        if replay:
            np.testing.assert_allclose(distribution,pd.read_csv(path),rtol=1e-10,atol=1e-14)
        else:
            distribution.to_csv(path,index=False,mode='x')
    result={'metrics':metrics,'tiers':tiers,'comparisons':comparisons,'bootstrap':bootstrap,
            'cv_wrapper_fits':60,'cv_regressor_fits':80,'reused_development_splits':True}
    if replay:
        if result!=json.loads((output/'selection/summary.json').read_text()):
            raise ValueError('Independent metrics/selection mismatch')
    else:
        write_json(output/'selection/summary.json',result)
        pd.DataFrame(comparisons).to_csv(output/'selection/comparison.csv',index=False,mode='x')
    return result


def verify(root,output):
    manifest=check(root,output)
    events=[json.loads(line) for line in (output/'fit_ledger.jsonl').read_text().splitlines()]
    starts=[e for e in events if e['event']=='start' and e['stage']=='cv']
    done=[e for e in events if e['event']=='complete' and e['stage']=='cv']
    if len(starts)!=60 or len(done)!=60 or len({e['model'] for e in done})!=60:
        raise ValueError('CV fit ledger mismatch')
    for record in done:
        if digest(Path(record['model']))!=record['model_sha256']:
            raise ValueError('Ledger SHA mismatch')
    train,metrics,predictions=collect(root,output,True)
    summarize(root,output,train,metrics,predictions,True)
    write_json(output/'selection/independent_verification.json',{'status':'PASS','replayed_models':60,'new_fits':0,
        'scale_and_fold_identity_verified':True,'metrics_tiers_bootstrap_reproduced':True})


def evaluate(root,output,config):
    initialize(root,output,config)
    try:
        train,metrics,predictions=collect(root,output)
        summarize(root,output,train,metrics,predictions)
        subprocess.run([sys.executable,'-m','bf_tap_r2.v2_smooth_search','verify','--output',str(output)],cwd=root,check=True)
        write_json(output/'COMPLETE_EVALUATION.json',{'status':'PASS','cv_wrapper_fits':60,'cv_regressor_fits':80})
    except Exception as exc:
        write_json(output/'FAILED.json',{'error':str(exc)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['evaluate','verify'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/round2_v2_4/experiment.yaml'))
    args=parser.parse_args()
    root=Path.cwd().resolve()
    if args.action=='evaluate':
        evaluate(root,args.output.resolve(),args.config.resolve())
    else:
        verify(root,args.output.resolve())


if __name__=='__main__':
    main()
