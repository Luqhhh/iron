"""Publish verified single-target V2.4 recommendations, never alter pending ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .audit import digest,write_json
from .data import TARGETS
from .smooth_residual import make_model
from .submission import ZIP_NAME,deny_training_reads,package,validate_result
from .v2_refinement import fit_once,isolated_payload
from .v2_release import load_v2
from .v2_smooth_search import check
from .v2_robust_joint import id_digest

BASE='local/runs/round2-v2.1/refinement-r1/release/B/result.csv'


def file_node(root,path,target,kind='csv'):
    return {'kind':kind,'path':str(path.relative_to(root)), 'sha256':digest(path),'target':target}


def predict_node(root,node,frame):
    if node['kind']=='mean':
        weights=np.asarray(node['weights'],dtype=float)
        if len(weights)!=len(node['members']) or (weights<0).any() or not np.isfinite(weights).all() or abs(weights.sum()-1)>1e-14:
            raise ValueError('Invalid fixed blend')
        return sum(w*predict_node(root,m,frame) for w,m in zip(weights,node['members']))
    path=root/node['path']
    if digest(path)!=node['sha256']:
        raise ValueError('Prediction member changed')
    if node['kind']=='csv':
        source=pd.read_csv(path,float_precision="round_trip")
        if not source.sample_id.is_unique or len(source)!=322:
            raise ValueError('Source CSV identity mismatch')
        return source.set_index('sample_id').loc[frame.sample_id,'pred_'+node['target']].to_numpy()
    if node['kind']!='model':
        raise ValueError('Unknown prediction node')
    model=joblib.load(path)
    if model.model_identity_['target']!=node['target']:
        raise ValueError('Model target mismatch')
    return model.predict(frame)


def checked_prediction(root,node,frame):
    with threadpool_limits(limits=1):
        pred=predict_node(root,node,frame)
        np.testing.assert_allclose(pred,predict_node(root,node,frame.iloc[::-1])[::-1],rtol=1e-12,atol=1e-10)
        chunks=np.concatenate([predict_node(root,node,frame.iloc[i:i+111]) for i in range(0,len(frame),111)])
        np.testing.assert_allclose(pred,chunks,rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(pred[:1],predict_node(root,node,frame.iloc[:1]),rtol=1e-12,atol=1e-10)
    if pred.shape!=(len(frame),) or not np.isfinite(pred).all():
        raise ValueError('Invalid prediction shape/values')
    return pred


def infer(root,output):
    sys.addaudithook(deny_training_reads)
    manifest=json.loads((output/'release_manifest.json').read_text())
    base=root/BASE
    if digest(base)!=manifest['base_sha256']:
        raise ValueError('Parent changed')
    frame=load_v2(root/'复赛_test','test',322)
    for name,record in manifest['packages'].items():
        pred=checked_prediction(root,record['node'],frame)
        payload=isolated_payload(base.read_bytes(),frame.sample_id,record['target'],pred)
        with (output/'release'/name/'cold.csv').open('xb') as stream:
            stream.write(payload)


def release(root,source,output,formal_only=False):
    frozen=check(root,source)
    spec=frozen['spec']
    namespace=spec.get('run_namespace','round2-v2.4')
    if namespace not in ('round2-v2.4','round2-v2.5') or not output.is_relative_to(root/'local/runs'/namespace):
        raise ValueError('Private output required')
    if json.loads((source/'selection/independent_verification.json').read_text())['status']!='PASS':
        raise ValueError('Independent replay required')
    summary=json.loads((source/'selection/summary.json').read_text())
    chosen=summary['tiers']['formal_selected'] if formal_only else summary['tiers']['submission_priority']
    if not chosen:
        raise ValueError('No recommended candidates')
    prefix=spec.get("release_prefix","V24")
    names=[f"{prefix}_{r['tier'].upper()}_{'IRON' if r['target']=='tap_iron' else 'TIME'}_{r['candidate']}" for r in chosen]
    desktop_base=Path('/mnt/c/Users/lqh22/Desktop/submission')/namespace
    for name in names:
        if (desktop_base/name).exists():
            raise FileExistsError(desktop_base/name)
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('models','release'):
        (output/folder).mkdir()
    write_json(output/'source_manifest.json',{'source':str(source.relative_to(root)),'source_manifest_sha256':digest(source/'manifest.json'),
        'verified_summary_sha256':digest(source/'selection/summary.json'),'source_sha256':digest(Path(__file__)),
        'spec':spec,'frozen_choices':chosen})
    try:
        test=load_v2(root/'复赛_test','test',322)
        base=root/BASE
        anchor_rows=validate_result(base.read_bytes(),test.sample_id)
        packages,cache={},{}
        fits,regressor_fits=0,0
        seen={digest(root/'local/runs/round2-v2.2/checkpoint-r1/release/V22_I_ONLY/result.csv'),
              digest(root/'local/runs/round2-v2.3/candidate-release-r1/release/V23_AJ_I_ONLY/result.csv')}
        for name,record in zip(names,chosen):
            target,route=record['target'],record['candidate']
            if route=='DJ':
                node={'kind':'mean','weights':[.5,.5],'members':[
                    file_node(root,root/'local/runs/round2-v2.2/checkpoint-r1/release/V22_I_ONLY/result.csv',target),
                    file_node(root,root/'local/runs/round2-v2.3/candidate-release-r1/release/V23_AJ_I_ONLY/result.csv',target)]}
            else:
                member=route[1:] if route.startswith('A') else route
                key=(target,member)
                if key not in cache:
                    train=load_v2(root/'复赛_train','train',2754)
                    model=make_model(member,target,spec)
                    model.model_identity_={'target':target,'route':member,'stage':'full','train_ids':id_digest(train)}
                    path=output/'models'/f'full-{target}-{member}.joblib'
                    fit_once(model,train,train[target],path,output,'full',model.model_identity_,spec['budget']['full_wrapper_fits_max'])
                    write_json(path.with_suffix('.json'),{'identity':model.model_identity_,'actual_parameters':model.actual_parameters_,
                        'model_sha256':digest(path),'regressor_fits':2 if member=='QRC' else 1})
                    cache[key]=file_node(root,path,target,'model')
                    fits+=1
                    regressor_fits+=2 if member=='QRC' else 1
                node=cache[key]
                if route.startswith('A'):
                    node={'kind':'mean','weights':[.5,.5],'members':[file_node(root,base,target),node]}
            prediction=checked_prediction(root,node,test)
            payload=isolated_payload(base.read_bytes(),test.sample_id,target,prediction)
            sha=hashlib.sha256(payload).hexdigest()
            if sha in seen:
                raise ValueError('Duplicate pending package')
            seen.add(sha)
            folder=output/'release'/name
            folder.mkdir()
            package(folder,payload,test.sample_id)
            rows=validate_result(payload,test.sample_id)
            changed={'pred_'+t:sum(a['pred_'+t]!=b['pred_'+t] for a,b in zip(anchor_rows,rows)) for t in TARGETS}
            if changed['pred_'+next(t for t in TARGETS if t!=target)]!=0:
                raise ValueError('Untouched column changed')
            packages[name]={'target':target,'candidate':route,'tier':record['tier'],'node':node,'changed_rows':changed,
                            'zip_sha256':digest(folder/ZIP_NAME),'desktop':str(desktop_base/name)}
        write_json(output/'release_manifest.json',{'base_sha256':digest(base),'packages':packages,
                   'new_full_wrapper_fits':fits,'new_full_regressor_fits':regressor_fits,'agent_uploads':0})
        subprocess.run([sys.executable,'-m','bf_tap_r2.v2_selected_release','infer','--output',str(output)],cwd=root,check=True)
        for name,r in packages.items():
            folder=output/'release'/name
            if (folder/'cold.csv').read_bytes()!=(folder/'result.csv').read_bytes():
                raise ValueError('Cold inference byte mismatch')
        check(root,source)
        for name,r in packages.items():
            desktop=Path(r['desktop'])
            desktop.mkdir(parents=True,exist_ok=False)
            for filename in (ZIP_NAME,'result.csv'):
                shutil.copy2(output/'release'/name/filename,desktop/filename)
            if digest(desktop/ZIP_NAME)!=r['zip_sha256']:
                raise ValueError('Desktop SHA mismatch')
        write_json(output/'COMPLETE.json',{'status':'PASS','new_packages':list(packages),'new_full_wrapper_fits':fits,
            'new_full_regressor_fits':regressor_fits,'cold_byte_identical':True,'unchanged_column_strings':True,'old_artifacts_preserved':True})
    except Exception as exc:
        write_json(output/'FAILED.json',{'error':str(exc)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['release','infer'])
    parser.add_argument('--source',type=Path,default=Path('local/runs/round2-v2.4/smooth-r1'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--formal-only',action='store_true')
    args=parser.parse_args()
    root=Path.cwd().resolve()
    if args.action=='release':
        release(root,args.source.resolve(),args.output.resolve(),args.formal_only)
    else:
        infer(root,args.output.resolve())


if __name__=='__main__':
    main()
