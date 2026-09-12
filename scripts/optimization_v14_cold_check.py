"""Independent original-model reload and stored-coefficient validation; zero fits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json, verify_file_identities
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import META, PRED, read_json
from bf_tap.optimization.dual_ratio_common import frame
from bf_tap.optimization.structural import INPUT, apply_correction
from bf_tap.optimization.structural_run import predict_inputs
from bf_tap.optimization.trajectory_run import builder_for, equal
from bf_tap.optimization.v13_common import zero_fit
from bf_tap.optimization.horizon_calibration import (CANDIDATE, CONTROL, read_bank,verify_bank,
    verify_pair,select_matched,predict_time,lad_certificate,month_start)
from optimization_v14_h2_calibration import verify, original_fold


def check(root):
    if (root/'cold_validation.json').exists():
        raise ContractError('cold evidence already exists; preserve it')
    manifest=read_json(root/'manifest.json');verify(manifest);reg=manifest['registration']
    verify_file_identities(read_json(root/'predictions_complete.json')['identities'])
    verify_file_identities(read_json(root/'OOF_provenance.json')['forecast_bank_identities'])
    h2=read_bank(root/'paired_bank/H2.csv');h1=read_bank(root/'paired_bank/H1_same_calendar.csv')
    verify_bank(h2,2,manifest['inventory']);verify_bank(h1,1,manifest['inventory']);verify_pair(h2,h1)
    builder=builder_for(manifest);checks=[]
    with zero_fit() as counts:
        folds={m:original_fold(manifest,m)[0] for m in reg['oof_months']}
        for bank,horizon in ((h2,2),(h1,1)):
            for cutoff,g in bank.groupby('fold_cutoff',sort=True):
                month=cutoff.month
                fresh=predict_inputs(g[META],folds[month],builder,reg)
                diff=equal(fresh,g,PRED+['pred_rate'],reg['prediction_tolerance'])
                # Rebuild original features under reversed order as a model-level check.
                reverse=predict_inputs(g[META].iloc[::-1],folds[month],builder,reg)
                reverse_diff=equal(fresh,reverse,PRED+['pred_rate'],0.)
                checks.append({'bank_horizon':horizon,'model_month':month,'rows':len(g),'max_delta':diff,'reverse_delta':reverse_diff})
                builder.cache.clear()
                print(f'OPT31 cold bank H{horizon} model {month}: replay + reverse PASS; fits=0',flush=True)
        for month,n in reg['origins'].items():
            samples=read_bank(root/'outer_inputs'/f'{month}_metadata.csv')
            base=predict_inputs(samples,folds[month],builder,reg)
            saved_base=frame(root/'outer_inputs'/f'{month}_inputs.csv')
            diff=equal(base,saved_base,PRED+['pred_rate'],reg['prediction_tolerance'])
            original=frame(root/'predictions'/f'{month}_V1.csv')
            legacy=apply_correction(base,read_json(Path(reg['source_v8'])/'corrections'/f'{month}.json')['alpha'])
            equal(legacy,original,PRED,reg['prediction_tolerance'])
            selected=select_matched(h2,h1,folds[month][3],month_start(month),reg['minimum_OOF_rows'])
            for role,s in zip((CANDIDATE,CONTROL),selected):
                stored=read_json(root/'coefficients'/f'{role}-{month}.json')
                if lad_certificate(s,stored['beta']) != stored['certificate']:
                    raise ContractError('cold certificate differs')
                prediction,_=predict_time(saved_base,original,stored['beta'])
                saved=frame(root/'predictions'/f'{month}_{role}.csv')
                delta=equal(prediction,saved,PRED,0.)
                # Scalar replay is exact across serialization, order, chunks, subset, single.
                reverse,_=predict_time(saved_base.iloc[::-1],original.iloc[::-1],stored['beta'])
                equal(prediction,reverse,PRED,0.)
                parts=[]
                for start in range(0,len(saved_base),127):
                    b=saved_base.iloc[start:start+127];v=original.loc[original.sample_id.isin(b.sample_id)]
                    p,_=predict_time(b,v,stored['beta']);parts.append(p)
                equal(prediction,pd.concat(parts),PRED,0.)
                for b in (saved_base.iloc[::17],saved_base.iloc[[0]]):
                    p,_=predict_time(b,original.loc[original.sample_id.isin(b.sample_id)],stored['beta'])
                    equal(p,prediction.loc[prediction.sample_id.isin(b.sample_id)],PRED,0.)
                if not np.array_equal(prediction[PRED[0]],original.set_index('sample_id').loc[prediction.sample_id,PRED[0]]):
                    raise ContractError('cold iron exact equality failed')
                checks.append({'outer_month':month,'role':role,'prediction_delta':delta,'coefficient_certificate':True,
                               'reverse_chunk_subset_single_exact':True,'iron_exact':True,'original_component_delta':diff})
            # Original model chunk/subset/single agreement on each origin.
            subset=samples.iloc[::17]
            fresh=predict_inputs(subset,folds[month],builder,reg)
            equal(fresh,base.loc[base.sample_id.isin(subset.sample_id)],PRED+['pred_rate'],0.)
            chunks=pd.concat([predict_inputs(samples.iloc[start:start+127],folds[month],builder,reg) for start in range(0,len(samples),127)],ignore_index=True)
            equal(chunks,base,PRED+['pred_rate'],0.)
            single=predict_inputs(samples.iloc[[0]],folds[month],builder,reg)
            equal(single,base.iloc[[0]],PRED+['pred_rate'],0.)
            builder.cache.clear()
            print(f'OPT31 cold outer {month}: model replay + stored V7/D1 + chunks/subsets PASS; fits=0',flush=True)
    if any(counts.values()): raise ContractError('cold fitting attempted')
    verify(manifest)
    atomic_write_json(root/'cold_validation.json',{'engineering_valid':True,'status':'PASS_INDEPENDENT_COLD',
        'checks':checks,'all_origins_iron_exact':True,'official_data_identity_verified':False,
        'quality_evaluated':False,'platform_verified':False,**counts})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    check(parser.parse_args().run)
