"""Fixed J1 three-training-seed follow-up, using only held-out members."""
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

from .audit import digest,write_json
from .candidate_tiers import classify_candidates
from .cv import target_metrics
from .data import TARGETS
from .low_signal_cv import sensitivity
from .normalized_models import JointSnapshotRegressor
from .submission import ZIP_NAME,package,validate_result
from .v2_refinement import fit_once,fold_vector,isolated_payload
from .v2_release import load_v2
from .v2_robust_joint import id_digest
from .v2_smooth_search import aligned,check
from .v2_selected_release import BASE,file_node,checked_prediction


def old_fold(root,spec,split,fold):
    return root/spec['joint_cv_run']/'models'/f'{split}-{fold}-J1-joint.joblib'


def old_full(root,spec):
    manifest=json.loads((root/spec['joint_full_run']/'release_manifest.json').read_text())
    r=manifest['packages']['V23_AJ_I_ONLY']['rules']
    path=root/r['joint_model']
    if digest(path)!=r['joint_model_sha256']:
        raise ValueError('Original full J1 changed')
    return path


def initialize(root,output):
    if not output.is_relative_to(root/'local/runs/round2-v2.6'):
        raise ValueError('Private fresh V2.6 output required')
    path=root/'configs/round2_v2_6/experiment.yaml'
    spec=yaml.safe_load(path.read_text())
    source=root/spec['source_run']
    frozen=check(root,source)
    files=dict(frozen['files'])
    extra=[path,Path(__file__),root/'src/bf_tap_r2/v2_selected_release.py',root/spec['policy'],old_full(root,spec),
           source/'selection/summary.json',source/'selection/independent_verification.json']
    for split in spec['split_seeds']:
        extra.append(source/'oof'/f'{split}-tap_iron.csv')
        for fold in range(5):
            p=old_fold(root,spec,split,fold)
            if digest(p)!=json.loads(p.with_suffix('.json').read_text())['model_sha256']:
                raise ValueError('Original joint fold SHA mismatch')
            extra.extend([p,p.with_suffix('.json')])
    for p in extra:
        files[str(p.relative_to(root))]=digest(p)
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('models','oof','selection','release'):
        (output/folder).mkdir()
    write_json(output/'manifest.json',{'spec':spec,'files':files,'policy':yaml.safe_load((root/spec['policy']).read_text()),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()})


def cv(root,output,replay=False):
    frozen=check(root,output)
    spec=frozen['spec']
    target='tap_iron'
    reference=spec['reference_by_target'][target]
    train=load_v2(root/'复赛_train','train',2754)
    metrics={target:{r:{} for r in [reference,'DJ',*spec['candidates'][target]]}}
    predictions={r:{} for r in metrics[target]}
    for split in spec['split_seeds']:
        folds=fold_vector(root/spec['fold_run'],train,split)
        previous=aligned(root/spec['source_run']/'oof'/f'{split}-{target}.csv',train,folds)
        j3=np.full(len(train),np.nan)
        for fold in range(5):
            training,valid=train.loc[folds!=fold],train.loc[folds==fold]
            values=[]
            for seed in spec['model_seeds']:
                identity={'route':'J1','model_seed':seed,'split':split,'heldout_fold':fold,
                          'train_ids':id_digest(training),'valid_ids':id_digest(valid)}
                path=old_fold(root,spec,split,fold) if seed==42 else output/'models'/f'{split}-{fold}-J1-{seed}.joblib'
                if seed==42 or replay:
                    model=joblib.load(path)
                    if seed!=42 and model.model_identity_!=identity:
                        raise ValueError('Seed member fold mismatch')
                    if seed==42:
                        old=model.model_identity_
                        if old['heldout_fold']!=fold or old['split']!=split or old['train_ids_sha256']!=id_digest(training) or old['valid_ids_sha256']!=id_digest(valid):
                            raise ValueError('Reused member held-out identity mismatch')
                else:
                    model=JointSnapshotRegressor(dict(spec['common'],random_seed=seed))
                    model.model_identity_=identity
                    fit_once(model,training,training[list(TARGETS)],path,output,'cv',identity,20)
                    write_json(path.with_suffix('.json'),{'model_sha256':digest(path),'actual_parameters':model.actual_parameters_})
                if model.parameters['random_seed']!=seed or model.estimator_.tree_count_!=1500 or tuple(model.target_fields_)!=TARGETS:
                    raise ValueError('Joint seed/tree/target mismatch')
                np.testing.assert_allclose(model.target_scales_,training[list(TARGETS)].mean(),rtol=1e-14)
                if digest(path)!=json.loads(path.with_suffix('.json').read_text())['model_sha256']:
                    raise ValueError('Member SHA mismatch')
                with threadpool_limits(limits=1):
                    pred=model.predict(valid)
                    if replay:
                        np.testing.assert_array_equal(pred,model.predict(valid.iloc[::-1])[::-1])
                        np.testing.assert_array_equal(pred,np.concatenate([model.predict(valid.iloc[i:i+127]) for i in range(0,len(valid),127)]))
                values.append(pred[:,0])
            j3[folds==fold]=np.mean(np.stack(values),axis=0)
        arrays={reference:previous[reference].to_numpy(),'DJ':previous.DJ.to_numpy(),'J3':j3}
        arrays['AJ3']=.5*arrays[reference]+.5*j3
        arrays['DJ3']=.5*previous.I2_C2_D4_IRON_EQUAL.to_numpy()+.5*arrays['AJ3']
        stored=train[['sample_id','spout_no',target]].copy()
        stored['fold']=folds
        for route,pred in arrays.items():
            metrics[target][route][str(split)]=target_metrics(train,target,pred,folds)
            predictions[route][split]=pred
            stored[route]=pred
        path=output/'oof'/f'{split}-{target}.csv'
        if replay:
            old=aligned(path,train,folds)
            np.testing.assert_allclose(old[stored.columns.drop('sample_id')],stored.drop(columns='sample_id'),rtol=1e-12,atol=1e-10)
        else:
            stored.to_csv(path,index=False,mode='x',float_format='%.17g')
    tiers=classify_candidates(metrics,spec,frozen['policy'])
    eligible=[r for r in tiers['formal_selected'] if all(metrics[target][r['candidate']][str(s)]['wmape']<metrics[target]['DJ'][str(s)]['wmape']-frozen['policy']['promotion']['numerical_equality_tolerance'] for s in spec['split_seeds'])]
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
    result={'metrics':metrics,'tiers':tiers,'delivery_selected':eligible,'bootstrap':bootstrap,'new_cv_fits':20}
    if replay:
        if result!=json.loads((output/'selection/summary.json').read_text()):
            raise ValueError('Independent selection mismatch')
    else:
        write_json(output/'selection/summary.json',result)


def verify(root,output):
    events=[json.loads(s) for s in (output/'fit_ledger.jsonl').read_text().splitlines()]
    done=[e for e in events if e['event']=='complete' and e['stage']=='cv']
    starts=[e for e in events if e['event']=='start' and e['stage']=='cv']
    if len(starts)!=20 or len(done)!=20 or len({e['model'] for e in done})!=20:
        raise ValueError('CV budget mismatch')
    for e in done:
        if digest(Path(e['model']))!=e['model_sha256']:
            raise ValueError('Ledger SHA mismatch')
    cv(root,output,True)
    write_json(output/'selection/independent_verification.json',{'status':'PASS','new_fits':0,'new_models_replayed':20,'old_models_reused':10})


def release(root,output):
    frozen=check(root,output)
    spec=frozen['spec']
    if json.loads((output/'selection/independent_verification.json').read_text())['status']!='PASS':
        raise ValueError('Verified CV required')
    summary=json.loads((output/'selection/summary.json').read_text())
    if len(summary['delivery_selected'])!=1:
        raise ValueError('No formal winner beating DJ in both splits')
    route=summary['delivery_selected'][0]['candidate']
    name='V26_FORMAL_IRON_'+route
    desktop=Path('/mnt/c/Users/lqh22/Desktop/submission/round2-v2.6')/name
    if desktop.exists() or (output/'release'/name).exists():
        raise FileExistsError('Fresh release destination required')
    training=load_v2(root/'复赛_train','train',2754)
    nodes=[file_node(root,old_full(root,spec),'tap_iron','joint_iron')]
    for seed in spec['model_seeds'][1:]:
        model=JointSnapshotRegressor(dict(spec['common'],random_seed=seed))
        model.model_identity_={'route':'J1','target':'tap_iron','stage':'full','model_seed':seed,'train_ids':id_digest(training)}
        path=output/'models'/f'full-J1-{seed}.joblib'
        fit_once(model,training,training[list(TARGETS)],path,output,'full',model.model_identity_,2)
        write_json(path.with_suffix('.json'),{'model_sha256':digest(path),'actual_parameters':model.actual_parameters_,
                   'target_scales':model.target_scales_.tolist(),'target_fields':list(model.target_fields_)})
        nodes.append(file_node(root,path,'tap_iron','joint_iron'))
    node={'kind':'mean','weights':[1/3]*3,'members':nodes,'aggregation':'arithmetic_mean'}
    base=root/BASE
    if route in ('AJ3','DJ3'):
        node={'kind':'mean','weights':[.5,.5],'members':[file_node(root,base,'tap_iron'),node]}
    if route=='DJ3':
        node={'kind':'mean','weights':[.5,.5],'members':[file_node(root,root/'local/runs/round2-v2.2/checkpoint-r1/release/V22_I_ONLY/result.csv','tap_iron'),node]}
    test=load_v2(root/'复赛_test','test',322)
    pred=checked_prediction(root,node,test)
    payload=isolated_payload(base.read_bytes(),test.sample_id,'tap_iron',pred)
    prior=validate_result(base.read_bytes(),test.sample_id)
    now=validate_result(payload,test.sample_id)
    if any(a['pred_tap_time_len']!=b['pred_tap_time_len'] for a,b in zip(prior,now)):
        raise ValueError('Time column changed')
    folder=output/'release'/name
    folder.mkdir()
    package(folder,payload,test.sample_id)
    write_json(output/'release_manifest.json',{'base_sha256':digest(base),'packages':{name:{'target':'tap_iron','candidate':route,
        'tier':'formal','node':node,'zip_sha256':digest(folder/ZIP_NAME),'desktop':str(desktop),
        'changed_rows':{'pred_tap_iron':sum(a['pred_tap_iron']!=b['pred_tap_iron'] for a,b in zip(prior,now)),'pred_tap_time_len':0}}},
        'new_full_regressor_fits':2,'new_full_wrapper_fits':2,'agent_uploads':0})
    subprocess.run([sys.executable,'-m','bf_tap_r2.v2_selected_release','infer','--output',str(output)],cwd=root,check=True)
    if (folder/'cold.csv').read_bytes()!=payload:
        raise ValueError('Cold release byte mismatch')
    check(root,output)
    desktop.mkdir(parents=True,exist_ok=False)
    for filename in (ZIP_NAME,'result.csv'):
        shutil.copy2(folder/filename,desktop/filename)
    if digest(desktop/ZIP_NAME)!=digest(folder/ZIP_NAME):
        raise ValueError('Desktop copy mismatch')
    write_json(output/'COMPLETE_RELEASE.json',{'status':'PASS','package':name,'new_full_fits':2,'cold_byte_identical':True,
        'time_strings_unchanged':True,'agent_uploads':0})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['evaluate','verify','release'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root,output=Path.cwd().resolve(),a.output.resolve()
    if a.action=='verify':
        verify(root,output)
    elif a.action=='release':
        release(root,output)
    else:
        initialize(root,output)
        try:
            cv(root,output)
            subprocess.run([sys.executable,'-m','bf_tap_r2.v2_joint_seeds','verify','--output',str(output)],cwd=root,check=True)
            write_json(output/'COMPLETE_EVALUATION.json',{'status':'PASS','new_cv_fits':20})
        except Exception as e:
            write_json(output/'FAILED.json',{'error':str(e)})
            raise


if __name__=='__main__':
    main()
