"""Descriptive, zero-fit residual grouping for the frozen best V2 OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from .audit import digest, write_json
from .data import FEATURES, TARGETS
from .models import inputs
from .v2_refinement import fold_vector
from .v2_release import load_v2
from .v2_smooth_search import aligned, check


def diagnostic_inputs(frame):
    x = inputs(frame).drop(columns='spout_no')
    if (x[['air_volume','total_press_diff']] <= 0).any().any():
        raise ValueError('Diagnostic ratios require positive denominators')
    x['oxygen_per_air_volume'] = x.oxygen/(60*x.air_volume)
    x['pressure_per_air_volume'] = x.total_press_diff/x.air_volume
    x['thermal_difference'] = x.hot_air_temp-x.furnace_throat_temp
    x['upper_pressure_fraction'] = x.upper_press_diff/x.total_press_diff
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Nonfinite diagnostic inputs')
    return x


def grouped_summary(values, labels, predictions, spouts, bins):
    x, y, pred = np.asarray(values), np.asarray(labels), np.asarray(predictions)
    if pred.shape != (len(y),2) or x.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all() or not np.isfinite(x).all() or y.sum() <= 0:
        raise ValueError('Invalid paired diagnostic arrays')
    groups = pd.qcut(x,bins,labels=False,duplicates='drop')
    if pd.isna(groups).all():
        groups=np.zeros(len(y),dtype=int)
    else:
        groups=np.asarray(groups,dtype=int)
    residual=pred-y[:,None]
    average=residual.mean(axis=1)
    undefined=bool(np.ptp(x)==0 or np.ptp(average)==0)
    corr=0.0 if undefined else float(pd.Series(x).rank().corr(pd.Series(average).rank()))
    rows=[]
    for group in np.unique(groups):
        mask=groups==group;means=residual[mask].mean(axis=0)
        rows.append({'group':int(group),'rows':int(mask.sum()),'input_min':float(x[mask].min()),'input_max':float(x[mask].max()),
            'mean_residual_split42':float(means[0]),'mean_residual_split3407':float(means[1]),
            'mean_residual':float(average[mask].mean()),'mean_absolute_error':float(np.abs(residual[mask]).mean()),
            'same_bias_sign':bool(means[0]*means[1]>0),
            'spout_counts':{str(k):int(np.sum(np.asarray(spouts)[mask]==k)) for k in np.unique(spouts)}})
    return {'spearman_mean_residual':corr,'spearman_undefined':undefined,
        'weighted_absolute_group_bias_relative':float(sum(r['rows']*abs(r['mean_residual']) for r in rows)/y.sum()),
        'same_sign_groups':sum(r['same_bias_sign'] for r in rows),'groups':rows}


def initialize(root, output):
    if not output.is_relative_to(root/'local/runs/round2-v2.11'):
        raise ValueError('Fresh private V2.11 output required')
    config=root/'configs/round2_v2_11/experiment.yaml';spec=yaml.safe_load(config.read_text())
    prior=check(root,root/spec['source_run']);files=dict(prior['files'])
    paths=[config,Path(__file__),root/'复赛_test/data_dictionary.xlsx',root/'复赛_test/README.txt']
    for t,run in spec['reference_runs'].items():
        paths.extend(root/run/'oof'/f'{s}-{t}.csv' for s in spec['split_seeds'])
        paths.extend([root/run/'selection/summary.json',root/run/'selection/independent_verification.json'])
    for path in paths:
        files[str(path.relative_to(root))]=digest(path)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',{'files':files,'spec':spec,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})


def compute(root,output,replay=False):
    spec=check(root,output)['spec'];train=load_v2(root/'复赛_train','train',2754)
    x=diagnostic_inputs(train)
    if list(x.columns)!=[*FEATURES,*spec['additional_diagnostics']]:
        raise ValueError('Diagnostic feature pool mismatch')
    details={};rows=[];overall={}
    for target in TARGETS:
        pred=[]
        for split in spec['split_seeds']:
            folds=fold_vector(root/spec['fold_run'],train,split)
            old=aligned(root/spec['reference_runs'][target]/'oof'/f'{split}-{target}.csv',train,folds)
            np.testing.assert_allclose(old[target],train[target],rtol=1e-14)
            pred.append(old[spec['reference_by_target'][target]].to_numpy())
        pred=np.column_stack(pred);y=train[target].to_numpy()
        residual=pred-y[:,None]
        overall[target]={'mean_residual':residual.mean(axis=0).tolist(),'median_residual':np.median(residual,axis=0).tolist(),
            'pooled_wmape':(np.abs(residual).sum(axis=0)/y.sum()).tolist()}
        details[target]={}
        for name in x:
            summary=grouped_summary(x[name],y,pred,train.spout_no,spec['bins'])
            details[target][name]=summary
            rows.append({'target':target,'feature':name,**{k:v for k,v in summary.items() if k!='groups'}})
            if replay:
                # Separate groupby path checks paired means, sizes and absolute error.
                b=pd.qcut(x[name],spec['bins'],duplicates='drop')
                audit=pd.DataFrame({'group':b,'r42':residual[:,0],'r3407':residual[:,1],
                    'mean':residual.mean(axis=1),'absolute':np.abs(residual).mean(axis=1)})
                grouped=audit.groupby('group',observed=True).agg(rows=('mean','size'),mean=('mean','mean'),absolute=('absolute','mean'),r42=('r42','mean'),r3407=('r3407','mean'))
                for expected,(_,actual) in zip(summary['groups'],grouped.iterrows(),strict=True):
                    np.testing.assert_allclose([expected['rows'],expected['mean_residual'],expected['mean_absolute_error'],expected['mean_residual_split42'],expected['mean_residual_split3407']],actual[['rows','mean','absolute','r42','r3407']],rtol=1e-12,atol=1e-10)
    result={'overall':overall,'details':details,'new_fits':0,'new_packages':0,'descriptive_development_evidence_only':True}
    if replay:
        if result!=json.loads((output/'summary.json').read_text()):
            raise ValueError('Independent diagnostic replay mismatch')
        write_json(output/'verification.json',{'status':'PASS','targets':2,'features_per_target':25,'new_fits':0,'groupby_crosscheck':True})
    else:
        write_json(output/'summary.json',result)
        pd.DataFrame(rows).to_csv(output/'feature_summary.csv',index=False,mode='x')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','verify']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path.cwd().resolve();out=a.output.resolve()
    if a.action=='verify':
        compute(root,out,True)
        return
    initialize(root,out)
    try:
        compute(root,out)
        subprocess.run([sys.executable,'-m','bf_tap_r2.v2_residual_structure','verify','--output',str(out)],check=True)
        write_json(out/'COMPLETE.json',{'status':'PASS','new_fits':0,'new_packages':0})
    except Exception as e:
        write_json(out/'FAILED.json',{'error':str(e)})
        raise


if __name__=='__main__':
    main()
