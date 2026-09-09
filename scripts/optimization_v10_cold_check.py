"""Independent no-fit reload, source reconstruction and temporal OOF audit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, file_sha256, stable_digest
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import META, PRED, read_json, forbid_fit
from bf_tap.optimization.dual_ratio_common import frame, verify_manifest
from bf_tap.optimization.dual_ratio_run import assert_equal
from bf_tap.optimization.refresh_factorial import stamp
from bf_tap.optimization.residual_stack import (CANDIDATE, CONTEXT, RATIOS, ResidualModel, predict)
from bf_tap.optimization.residual_run import (builder_for, load_fold, context_for, dated,
    training_at, outer_inputs)


def same_context(a,b):
    a=a.set_index('sample_id').sort_index();b=b.set_index('sample_id').sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.spout_no.astype(str),b.spout_no.astype(str)):
        raise ContractError('cold context metadata mismatch')
    if not np.array_equal(a[CONTEXT[1:]].to_numpy(),b[CONTEXT[1:]].to_numpy(),equal_nan=True):
        raise ContractError('cold public asof features differ')


def check(source,output):
    output.mkdir(parents=True,exist_ok=False)
    try:
        manifest=read_json(source/'manifest.json');verify_manifest(manifest)
        reg=manifest['registration'];reg['origins']={int(k):v for k,v in reg['origins'].items()}
        if read_json(source/'final_status.json')['G0']!='PASS_OPT23_EXECUTION':
            raise ContractError('complete development run required')
        builder=builder_for(manifest);counter={'attempted_target_fits':0}
        fold_checks=[];outer_checks=[];folds=[];contexts=[];loaded={}
        records={r['month']:r for r in read_json(source/'training_records.json')}
        with forbid_fit(counter):
            for month in reg['oof_months']:
                q,h,entry,oof=load_fold(reg,month)
                ctx=context_for(builder,oof[META],entry,h)
                fresh=pd.DataFrame({'sample_id':oof.sample_id.to_numpy(),
                    'pred_inverse_rate':q.predict(builder.X(oof[META],entry,h))})
                assert_equal(fresh,oof,['pred_inverse_rate'])
                fold_checks.append(dict(month=month,rows=len(oof),q_OOF_max_difference=0.))
                loaded[month]=(q,h,entry);folds.append(oof);contexts.append(ctx)
                builder.cache.clear()
                print(f'OPT23 cold OOF {month}: q identity and original asof context PASS',flush=True)
            combined=pd.concat(folds,ignore_index=True);context=pd.concat(contexts,ignore_index=True)
            same_context(context,frame(source/'OOF_context.csv'))
            for month in reg['origins']:
                q,h,entry=loaded[month];cutoff=stamp(month);root=source/'models'/str(month)
                expected=training_at(combined,h,cutoff,reg['residual']['minimum_OOF_rows'])
                saved=dated(root/'training_OOF.csv');record=records[month];tr=record['training']
                assert_equal(saved,expected,['tap_iron','tap_time_len',*PRED,*RATIOS])
                for date in ('reference_time','label_available_at','fold_cutoff','inverse_fit_cutoff'):
                    a=saved.set_index('sample_id').sort_index()[date]
                    b=expected.set_index('sample_id').sort_index()[date]
                    if not a.equals(b):raise ContractError('cold OOF date mismatch')
                if stable_digest(expected.sample_id.tolist())!=tr['OOF_ids_sha256']:
                    raise ContractError('cold residual training IDs differ')
                if file_sha256(root/'training_OOF.csv')!=tr['OOF_sha256'] or file_sha256(root/'training_context.csv')!=tr['context_sha256']:
                    raise ContractError('cold residual training bytes differ')
                trainctx=context.loc[context.sample_id.isin(expected.sample_id)]
                same_context(trainctx,frame(root/'training_context.csv'))
                models={}
                for target in ('tap_iron','tap_time_len'):
                    if file_sha256(root/target/'bundle.json')!=record['bundle_sha256'][target]:
                        raise ContractError('cold residual bundle changed')
                    model=ResidualModel.load(root/target)
                    if model.metadata['metadata']!=tr:raise ContractError('cold residual provenance differs')
                    models[target]=model
                samples,parts,ctx=outer_inputs(reg,month,combined,builder,h)
                freshq=pd.DataFrame({'sample_id':samples.sample_id.to_numpy(),
                    'pred_inverse_rate':q.predict(builder.X(samples[META],entry,h))})
                assert_equal(freshq,parts,['pred_inverse_rate'])
                assert_equal(parts,frame(source/'predictions'/f'{month}_inputs.csv'),PRED+RATIOS)
                same_context(ctx,frame(source/'predictions'/f'{month}_context.csv'))
                fresh=predict(models,parts,ctx)
                savedpred=frame(source/'predictions'/f'{month}_V4.csv')
                assert_equal(fresh,savedpred,PRED)
                reversectx=context_for(builder,samples.iloc[::-1],entry,h)
                assert_equal(fresh,predict(models,parts.iloc[::-1],reversectx),PRED)
                if file_sha256(source/'predictions'/f'{month}_V4.csv')!=record['prediction_sha256']:
                    raise ContractError('cold candidate bytes changed')
                if set(samples.sample_id)&set(expected.sample_id):raise ContractError('cold meta fit/eval overlap')
                outer_checks.append(dict(month=month,rows=len(fresh),OOF_rows=len(expected),
                    max_prediction_difference=0.,reversal_exact=True,causal_training_identity=True))
                builder.cache.clear()
                print(f'OPT23 cold origin {month}: residual models, OOF labels, forecast and reversal PASS',flush=True)
        verify_manifest(manifest)
        atomic_write_json(output/'validation.json',dict(status='PASS',candidate=CANDIDATE,
            source_manifest_sha256=file_sha256(source/'manifest.json'),script_sha256=file_sha256(__file__),
            folds=fold_checks,origins=outer_checks,**counter,official_target_paths_present=False))
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(error=str(exc),fit_attempts=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    check(args.source,args.output)
