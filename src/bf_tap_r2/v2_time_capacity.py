"""Frozen follow-up for time capacity after V2.4's negative time results."""
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
from .low_signal_cv import sensitivity
from .smooth_residual import make_model
from .v2_refinement import fit_once,fold_vector
from .v2_release import load_v2
from .v2_robust_joint import id_digest
from .v2_smooth_search import aligned,check


def initialize(root,output):
    if not output.is_relative_to(root/'local/runs/round2-v2.5'):
        raise ValueError('Private new V2.5 output required')
    config=root/'configs/round2_v2_5/experiment.yaml'
    spec=yaml.safe_load(config.read_text())
    source=root/spec['source_run']
    frozen=check(root,source)
    if json.loads((source/'selection/independent_verification.json').read_text())['status']!='PASS':
        raise ValueError('V2.4 replay required')
    files=dict(frozen['files'])
    for p in [config,Path(__file__),root/'src/bf_tap_r2/smooth_residual.py',root/spec['policy'],
              source/'selection/summary.json',source/'selection/independent_verification.json',
              *[source/'oof'/f'{s}-tap_time_len.csv' for s in spec['split_seeds']]]:
        files[str(p.relative_to(root))]=digest(p)
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('models','oof','selection'):
        (output/folder).mkdir()
    write_json(output/'manifest.json',{'files':files,'spec':spec,'policy':yaml.safe_load((root/spec['policy']).read_text()),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()})


def evaluate_predictions(root,output,replay=False):
    manifest=check(root,output)
    spec=manifest['spec']
    target='tap_time_len'
    reference=spec['reference_by_target'][target]
    train=load_v2(root/'复赛_train','train',2754)
    metrics={target:{r:{} for r in [reference,'T1_B3_PREFIX1000',*spec['candidates'][target]]}}
    predictions={r:{} for r in metrics[target]}
    for split in spec['split_seeds']:
        folds=fold_vector(root/spec['fold_run'],train,split)
        previous=aligned(root/spec['source_run']/'oof'/f'{split}-{target}.csv',train,folds)
        values={r:previous[r].to_numpy() for r in (reference,'T1_B3_PREFIX1000')}
        for route in spec['routes']:
            values[route]=np.full(len(train),np.nan)
            for fold in range(spec['folds']):
                training,valid=train.loc[folds!=fold],train.loc[folds==fold]
                path=output/'models'/f'{split}-{target}-{fold}-{route}.joblib'
                identity={'route':route,'target':target,'split':split,'heldout_fold':fold,'train_ids':id_digest(training),'valid_ids':id_digest(valid)}
                if replay:
                    model=joblib.load(path)
                    if model.model_identity_!=identity:
                        raise ValueError('Fold identity mismatch')
                    np.testing.assert_allclose(model.target_scales_,[training[target].mean()],rtol=1e-14)
                else:
                    model=make_model(route,target,spec)
                    model.model_identity_=identity
                    fit_once(model,training,training[target],path,output,'cv',identity,30)
                    write_json(path.with_suffix('.json'),{'actual_parameters':model.actual_parameters_,'model_sha256':digest(path)})
                record=json.loads(path.with_suffix('.json').read_text())
                if digest(path)!=record['model_sha256'] or model.actual_parameters_!=record['actual_parameters']:
                    raise ValueError('Model identity mismatch')
                with threadpool_limits(limits=1):
                    pred=model.predict(valid)
                    if replay:
                        np.testing.assert_array_equal(pred,model.predict(valid.iloc[::-1])[::-1])
                        np.testing.assert_array_equal(pred,np.concatenate([model.predict(valid.iloc[i:i+127]) for i in range(0,len(valid),127)]))
                values[route][folds==fold]=pred
            values['A'+route]=.5*values[reference]+.5*values[route]
        stored=train[['sample_id','spout_no',target]].copy()
        stored['fold']=folds
        for route,pred in values.items():
            metrics[target][route][str(split)]=target_metrics(train,target,pred,folds)
            predictions[route][split]=pred
            stored[route]=pred
        path=output/'oof'/f'{split}-{target}.csv'
        if replay:
            old=aligned(path,train,folds)
            np.testing.assert_allclose(old[stored.columns.drop('sample_id')],stored.drop(columns='sample_id'),rtol=1e-12,atol=1e-10)
        else:
            stored.to_csv(path,index=False,mode='x',float_format='%.17g')
    tiers=classify_candidates(metrics,spec,manifest['policy'])
    bootstrap={}
    for r in tiers['submission_priority']:
        route=r['candidate']
        distribution,intervals=sensitivity(train,target,predictions[route],predictions[reference],spec['bootstrap'])
        path=output/'selection'/f'bootstrap-{route}.csv'
        if replay:
            np.testing.assert_allclose(distribution,pd.read_csv(path),rtol=1e-10,atol=1e-14)
        else:
            distribution.to_csv(path,index=False,mode='x')
        bootstrap[route]={'intervals':intervals,'fraction_negative':float((distribution['mean']<0).mean())}
    result={'metrics':metrics,'tiers':tiers,'bootstrap':bootstrap,'cv_regressor_fits':30,'reused_development_splits':True}
    if replay:
        if result!=json.loads((output/'selection/summary.json').read_text()):
            raise ValueError('Summary mismatch')
    else:
        write_json(output/'selection/summary.json',result)


def verify(root,output):
    events=[json.loads(s) for s in (output/'fit_ledger.jsonl').read_text().splitlines()]
    starts=[e for e in events if e['event']=='start']
    done=[e for e in events if e['event']=='complete']
    if len(starts)!=30 or len(done)!=30 or len({e['model'] for e in done})!=30:
        raise ValueError('Fit budget mismatch')
    for e in done:
        if digest(Path(e['model']))!=e['model_sha256']:
            raise ValueError('Ledger digest mismatch')
    evaluate_predictions(root,output,True)
    write_json(output/'selection/independent_verification.json',{'status':'PASS','replayed_models':30,'new_fits':0})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['evaluate','verify'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root,output=Path.cwd().resolve(),a.output.resolve()
    if a.action=='verify':
        verify(root,output)
        return
    initialize(root,output)
    try:
        evaluate_predictions(root,output)
        subprocess.run([sys.executable,'-m','bf_tap_r2.v2_time_capacity','verify','--output',str(output)],cwd=root,check=True)
        write_json(output/'COMPLETE_EVALUATION.json',{'status':'PASS','cv_regressor_fits':30})
    except Exception as e:
        write_json(output/'FAILED.json',{'error':str(e)})
        raise


if __name__=='__main__':
    main()
