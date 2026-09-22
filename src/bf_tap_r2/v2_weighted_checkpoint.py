"""Frozen positive-kernel-geometry OOF comparison with independent replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest,write_json
from .candidate_tiers import classify_candidates
from .cv import target_metrics
from .data import TARGETS,FEATURES
from .low_signal_cv import sensitivity
from .kernel_geometry import GeometryRegressor, numeric_weights
from .v2_refinement import fit_once,fold_vector,parent_model
from .v2_release import load_v2
from .v2_robust_joint import id_digest
from .v2_smooth_search import aligned,check


def initialize(root,output):
    if not output.is_relative_to(root/'local/runs/round2-v2.9'):
        raise ValueError('Fresh private V2.9 directory required')
    path=root/'configs/round2_v2_9/experiment.yaml'
    spec=yaml.safe_load(path.read_text())
    frozen=check(root,root/spec['source_run'])
    files=dict(frozen['files'])
    extras=[path,Path(__file__),Path(__file__).with_name('kernel_geometry.py'),root/spec['policy'],root/'uv.lock']
    for control_run in spec['controls'].values():
        extras.extend((root/control_run/'oof').glob('*.csv'))
        extras.extend([root/control_run/'selection/summary.json',root/control_run/'selection/independent_verification.json'])
    for key in ('reference_run','time_best_run'):
        extras.extend((root/spec[key]/'oof').glob('*.csv'))
        extras.extend([root/spec[key]/'selection/summary.json',root/spec[key]/'selection/independent_verification.json'])
    # Bind all newly queued packages as well as the previously frozen originals.
    for directory in ('round2-v2.4/release-r1','round2-v2.5/release-r1'):
        r=root/'local/runs'/directory
        extras.append(r/'release_manifest.json')
        for name,record in json.loads((r/'release_manifest.json').read_text())['packages'].items():
            extras.extend([r/'release'/name/'result.csv',r/'release'/name/'Luqhhh_bf_tap_predict_round2.zip',Path(record['desktop'])/'Luqhhh_bf_tap_predict_round2.zip'])
    for p in extras:
        files[str(p.relative_to(root)) if p.is_relative_to(root) else str(p)]=digest(p)
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('models','oof','selection'):
        (output/folder).mkdir()
    write_json(output/'manifest.json',{'files':files,'spec':spec,'policy':yaml.safe_load((root/spec['policy']).read_text()),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()})


def compute(root,output,replay=False):
    frozen=check(root,output)
    spec=frozen['spec']
    train=load_v2(root/'复赛_train','train',2754)
    metrics={t:{r:{} for r in [spec['reference_by_target'][t],spec['best_by_target'][t],*spec['candidates'][t],*spec['controls']]} for t in TARGETS}
    predictions={t:{r:{} for r in metrics[t]} for t in TARGETS}
    for split in spec['split_seeds']:
        folds=fold_vector(root/spec['fold_run'],train,split)
        values={t:{r:np.full(len(train),np.nan) for r in spec['models']} for t in TARGETS}
        for fold in range(5):
            training,valid=train.loc[folds!=fold],train.loc[folds==fold]
            for route,target in [(r,t) for r in spec['models'] for t in TARGETS]:
                path=output/'models'/f'{split}-{fold}-{route}-{target}.joblib'
                identity={'route':route,'target':target,'split':split,'heldout_fold':fold,'train_ids':id_digest(training),'valid_ids':id_digest(valid)}
                fields=(target,)
                source_path=parent_model(root/spec['fold_run'],split,target,fold,'C2_42')
                source_sha=digest(source_path)
                if frozen['files'].get(str(source_path.relative_to(root)))!=source_sha:
                    raise ValueError('Geometry source missing from frozen manifest')
                teacher=joblib.load(source_path)
                if teacher.config['execution']['model_seed']!=42 or teacher.estimator_.tree_count_!=1500 or teacher.route!='C2':
                    raise ValueError('Invalid geometry teacher identity')
                if teacher.estimator_.feature_names_ != [*FEATURES,'spout_no']:
                    raise ValueError('Teacher feature order mismatch')
                original=aligned(root/spec['fold_run']/'oof'/f'{split}-{target}.csv',train,folds)
                np.testing.assert_allclose(teacher.predict(valid),original.C2.to_numpy()[folds==fold],rtol=1e-12,atol=1e-10)
                importance=teacher.estimator_.get_feature_importance(type='PredictionValuesChange')[:len(FEATURES)]
                weights=numeric_weights(importance,spec['geometry']['numeric_uniform_floor']) if route=='KW2' else np.ones(len(FEATURES))
                desired=spec['models'][route]
                identity['geometry_source_sha256']=source_sha
                if replay:
                    model=joblib.load(path)
                    if model.model_identity_!=identity or model.spec!=desired or tuple(model.target_fields_)!=fields:
                        raise ValueError('Model identity/spec mismatch')
                else:
                    model=GeometryRegressor('KWT',desired,target,weights)
                    model.model_identity_=identity
                    with threadpool_limits(limits=1):
                        fit_once(model,training,training[target],path,output,'cv',identity,20)
                    write_json(path.with_suffix('.json'),{'model_sha256':digest(path),'parameters':model.actual_parameters_,
                        'numeric_weights':weights.tolist(),'target_scales':model.scales_.tolist()})
                    print(f'Completed {split} fold={fold} {route} {target}',flush=True)
                record=json.loads(path.with_suffix('.json').read_text())
                if digest(path)!=record['model_sha256'] or model.actual_parameters_!=record['parameters']:
                    raise ValueError('Saved fit metadata mismatch')
                np.testing.assert_array_equal(model.numeric_weights_,weights)
                np.testing.assert_array_equal(record['numeric_weights'],weights)
                np.testing.assert_array_equal(record['target_scales'],model.scales_)
                np.testing.assert_allclose(model.scales_,training[list(fields)].mean(),rtol=1e-14)
                np.testing.assert_allclose(model.preprocessor_.named_transformers_['numeric'].mean_,training[list(FEATURES)].mean(),rtol=1e-12,atol=1e-10)
                with threadpool_limits(limits=1):
                    features=valid.drop(columns=list(TARGETS))
                    pred=model.predict(features)
                    if replay:
                        np.testing.assert_allclose(pred,model.predict(features.iloc[::-1])[::-1],rtol=1e-11,atol=1e-9)
                        batches=np.concatenate([model.predict(features.iloc[i:i+127]) for i in range(0,len(valid),127)])
                        np.testing.assert_allclose(pred,batches,rtol=1e-11,atol=1e-9)
                        np.testing.assert_allclose(pred[:1],model.predict(features.iloc[:1]),rtol=1e-11,atol=1e-9)
                values[target][route][folds==fold]=pred
        for target in TARGETS:
            reference,best=spec['reference_by_target'][target],spec['best_by_target'][target]
            previous=aligned(root/spec['reference_run']/'oof'/f'{split}-{target}.csv',train,folds)
            np.testing.assert_allclose(previous[target],train[target],rtol=1e-14)
            values[target][reference]=previous[reference].to_numpy()
            source=previous if target=='tap_iron' else aligned(root/spec['time_best_run']/'oof'/f'{split}-{target}.csv',train,folds)
            values[target][best]=source[best].to_numpy()
            for control,control_run in spec['controls'].items():
                old_control=aligned(root/control_run/'oof'/f'{split}-{target}.csv',train,folds)
                np.testing.assert_allclose(old_control[target],train[target],rtol=1e-14)
                values[target][control]=old_control[control].to_numpy()
            for route in spec['models']:
                values[target]['A'+route]=.5*values[target][reference]+.5*values[target][route]
            saved=train[['sample_id','spout_no',target]].copy()
            saved['fold']=folds
            for route,pred in values[target].items():
                saved[route]=pred
                metrics[target][route][str(split)]=target_metrics(train,target,pred,folds)
                predictions[target][route][split]=pred
            path=output/'oof'/f'{split}-{target}.csv'
            if replay:
                old=aligned(path,train,folds)
                np.testing.assert_allclose(old[saved.columns.drop('sample_id')],saved.drop(columns='sample_id'),rtol=1e-11,atol=1e-9)
            else:
                saved.to_csv(path,index=False,mode='x',float_format='%.17g')
    tiers=classify_candidates(metrics,spec,frozen['policy'])
    rows=[]
    for target in TARGETS:
        best=spec['best_by_target'][target]
        for route,d in tiers['decisions'][target].items():
            rows.append({'target':target,'candidate':route,'mean_wmape':d['mean_wmape'],'delta_current':-d['mean_gain'],
                'delta_best':d['mean_wmape']-np.mean([metrics[target][best][str(s)]['wmape'] for s in spec['split_seeds']]),
                'delta_KR1':d['mean_wmape']-np.mean([metrics[target]['KR1'][str(s)]['wmape'] for s in spec['split_seeds']]),
                'delta_KWT':d['mean_wmape']-np.mean([metrics[target]['KWT'][str(s)]['wmape'] for s in spec['split_seeds']]),
                'improved_folds':d['improved_folds'],'worst_spout_delta':d['worst_spout_delta'],'tier':d['tier'],
                'failed_conditions':','.join(d['failed_conditions'])})
    bootstrap={}
    for d in tiers['submission_priority']:
        target,route=d['target'],d['candidate']
        distribution,intervals=sensitivity(train,target,predictions[target][route],predictions[target][spec['reference_by_target'][target]],spec['bootstrap'])
        key=f'{target}-{route}'
        bootstrap[key]={'intervals':intervals,'fraction_negative':float((distribution['mean']<0).mean())}
        path=output/'selection'/f'bootstrap-{key}.csv'
        if replay:
            np.testing.assert_allclose(distribution,pd.read_csv(path),rtol=1e-10,atol=1e-14)
        else:
            distribution.to_csv(path,index=False,mode='x')
    result={'metrics':metrics,'tiers':tiers,'comparisons':rows,'bootstrap':bootstrap,'new_cv_fits':20,
            'new_full_fits':0,'new_packages':0,'reused_development_splits':True,
            'total_fit_attempts':20}
    if replay:
        if result!=json.loads((output/'selection/summary.json').read_text()):
            raise ValueError('Independent summary mismatch')
    else:
        write_json(output/'selection/summary.json',result)
        pd.DataFrame(rows).to_csv(output/'selection/comparison.csv',index=False,mode='x')


def verify(root,output):
    es=[json.loads(s) for s in (output/'fit_ledger.jsonl').read_text().splitlines()]
    starts=[e for e in es if e['event']=='start'];done=[e for e in es if e['event']=='complete']
    if len(starts)!=20 or len(done)!=20 or len({e['model'] for e in done})!=20:
        raise ValueError('Geometry fit budget mismatch')
    for e in done:
        if digest(Path(e['model']))!=e['model_sha256']:
            raise ValueError('Fit ledger SHA mismatch')
    compute(root,output,True)
    write_json(output/'selection/independent_verification.json',{'status':'PASS','replayed_models':20,'new_fits':0,
        'training_only_transforms_and_fold_ids_checked':True,'metrics_tiers_bootstrap_reproduced':True})



def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['evaluate','verify'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path.cwd().resolve();output=a.output.resolve()
    if a.action=='verify':
        verify(root,output)
    else:
        initialize(root,output)
        try:
            compute(root,output)
            subprocess.run([sys.executable,'-m','bf_tap_r2.v2_weighted_checkpoint','verify','--output',str(output)],cwd=root,check=True)
            write_json(output/'COMPLETE.json',{'status':'PASS','completed_cv_models':20,'total_fit_attempts':20,
                'new_full_fits':0,'new_packages':0})
        except Exception as e:
            write_json(output/'FAILED.json',{'error':str(e)})
            raise


if __name__=='__main__':
    main()
