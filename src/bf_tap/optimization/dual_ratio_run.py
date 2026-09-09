"""OPT-22: eight new q models, six shared time LAD fits, fixed V2/V3 comparison."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json,file_sha256,stable_digest,build_inference_source_contract
from ..exceptions import ContractError
from ..offline import _load_process_sources
from ..schema import validate_history
from .component_export import META,PRED,ComponentFeatures,read_json,forbid_fit
from .final_lifecycle import algorithm
from .refresh_factorial import Context,stamp
from .horizon_router import load_component
from .rate_model import RateModel,schema
from .structural import apply_correction
from .structural_run import predict_inputs
from .dual_ratio import V2,V3,predict_dual,acceptance
from .dual_ratio_common import frame,registry,freeze,verify_manifest,score,week_intervals
from .inverse_rate_model import InverseRateModel,InverseFitBudget
from .inverse_oof import fit_certified_time


def history_frame(path):
    h=frame(path)
    for c in ('reference_time','tap_end_time','available_at'):h[c]=pd.to_datetime(h[c])
    validate_history(h);return h.sort_values(['reference_time','sample_id'],kind='mergesort').reset_index(drop=True)


def source_oof(source,month):
    oof=frame(source/'oof'/f'{month}.csv')
    for c in ('reference_time','label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max'):oof[c]=pd.to_datetime(oof[c])
    return oof


def assert_equal(left,right,columns,tolerance=0.):
    a=left.set_index('sample_id').sort_index();b=right.set_index('sample_id').sort_index()
    if not a.index.equals(b.index):raise ContractError('prediction ID alignment failed')
    maximum=float(np.abs(a[columns].to_numpy()-b[columns].to_numpy()).max())
    if maximum>tolerance:raise ContractError(f'prediction reproduction failed: {maximum}')
    return maximum


def run(ablation,output):
    if read_json(ablation/'final_status.json')['G0']!='PASS_OPT21':raise ContractError('OPT21 must finish before OPT22')
    output.mkdir(parents=True,exist_ok=False);budget=InverseFitBudget(8);scalar_fits=0
    try:
        reg=registry();source=Path(reg['source_v8']);parent=read_json(source/'manifest.json');a=algorithm()
        paths={k:v['path'] for k,v in parent['inputs'].items() if k in ('operation_hourly','burden_change','data_dictionary')}
        evidence=[ablation/'final_status.json',ablation/'diagnosis.json']
        for month in reg['oof_months']:
            evidence.extend([source/'oof'/f'{month}.csv',source/'models'/str(month)/'fit_record.json'])
            evidence.extend((source/'models'/str(month)/'rate').glob('*'))
        for month in reg['origins']:
            evidence.extend([source/'predictions'/f'{month}_inputs.csv',source/'predictions'/f'{month}_V1.csv',source/'corrections'/f'{month}.json'])
        manifest=freeze(output,'OPT22_inverse_rate_training_and_causal_time_LAD',evidence,paths)
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features']);builder=ComponentFeatures(a,op,burden)
        contract=build_inference_source_contract(manifest['inputs'],semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        models={};q_oof=[];records=[]
        with budget:
            for month in reg['oof_months']:
                cutoff=stamp(month);old_rate=RateModel.load(source/'models'/str(month)/'rate')
                old_record=read_json(source/'models'/str(month)/'fit_record.json')
                if file_sha256(source/'models'/str(month)/'rate'/'bundle.json')!=old_record['rate_bundle_sha256']:raise ContractError('parent rate identity mismatch')
                h=history_frame(source/'models'/str(month)/'rate'/'history_snapshot.csv')
                tr=old_rate.metadata_['training']
                if old_rate.metadata_['inference_source_contract']!=contract:raise ContractError('public source contract differs')
                if any(pd.Timestamp(tr[k])!=cutoff for k in ('fit_cutoff','history_cutoff','label_available_cutoff')) or (h.reference_time>=cutoff).any() or (h.available_at>cutoff).any():raise ContractError('future q training data')
                if stable_digest(h.sample_id.tolist())!=tr['sample_ids_sha256']:raise ContractError('training IDs differ from V1')
                X=Context.X(builder,h[META],cutoff,'E09_PROCESS_CHANGE_E02','R2',h)
                if schema(X)!=old_rate.schema_:raise ContractError('q must use exactly the rate feature schema')
                model=InverseRateModel().fit(X,h.tap_iron,h.tap_time_len,a['baseline']['parameters'],budget)
                dest=output/'models'/str(month)
                training=dict(tr,reference_max=str(h.reference_time.max()),label_available_max=str(h.available_at.max()),history_available_max=str(h.available_at.max()))
                model.save(dest,dict(parameters=a['baseline']['parameters'],training=training,inference_source_contract=contract,
                    inverse_spec=reg['OPT22'],source_v8_rate_bundle_sha256=old_record['rate_bundle_sha256'],
                    positive_iron_ids_sha256=stable_digest(h.loc[h.tap_iron>0,'sample_id'].tolist()),manifest_sha256=file_sha256(output/'manifest.json')),h)
                loaded=InverseRateModel.load(dest)
                if not np.array_equal(model.predict(X),loaded.predict(X)):raise ContractError('q model serialization changed predictions')
                entry=old_record['base_components']['OR'];models[month]=(loaded,h,entry)
                oof=source_oof(source,month)
                if not (oof.fold_cutoff==cutoff).all() or set(oof.sample_id)&set(h.sample_id):raise ContractError('q OOF overlaps fit rows')
                X_eval=builder.X(oof[META],entry,h);q=loaded.predict(X_eval)
                if not np.array_equal(q,loaded.predict(X_eval.iloc[::-1])[::-1]):raise ContractError('q OOF order dependence')
                oof['pred_inverse_rate']=q
                for c,value in [('inverse_fit_cutoff',cutoff),('inverse_train_reference_max',h.reference_time.max()),('inverse_train_available_max',h.available_at.max()),('inverse_history_available_max',h.available_at.max())]:oof[c]=value
                d=output/'oof';d.mkdir(exist_ok=True);oof.to_csv(d/f'{month}.csv',index=False);q_oof.append(oof)
                record=dict(month=month,training=training,inverse_training_rows=loaded.metadata_['inverse_training_rows'],excluded_zero_iron=loaded.metadata_['excluded_zero_iron'],
                    q_bundle_sha256=file_sha256(dest/'bundle.json'),OOF_rows=len(oof),OOF_sha256=file_sha256(d/f'{month}.csv'))
                atomic_write_json(dest/'fit_record.json',record);records.append(record);builder.cache.clear()
                print(f'OPT22 q cutoff {month}: {record["inverse_training_rows"]} rows, excluded iron=0: {record["excluded_zero_iron"]}; fits={budget.completed}/8',flush=True)
            if budget.completed!=8 or budget.forbidden:raise ContractError('inverse fit budget not satisfied')
            combined=pd.concat(q_oof,ignore_index=True);coefficients={};predictions={};alignment=[];inference_counter={'attempted_target_fits':0}
            identities=read_json(Path('local/runs/optimization-v0.5-opt14-export-r1')/'component_identities.json')
            with forbid_fit(inference_counter):
                for month,n in reg['origins'].items():
                    cutoff=stamp(month);model,h,entry=models[month]
                    labeled=combined.merge(h[['sample_id','tap_iron','tap_time_len']],on='sample_id',validate='one_to_one')
                    avail=h.set_index('sample_id').available_at
                    if not np.array_equal(labeled.label_available_at.astype(str),avail.loc[labeled.sample_id].astype(str)):raise ContractError('OOF label availability differs from eligible history')
                    beta,used=fit_certified_time(labeled,cutoff,reg['OPT22']['minimum_causal_OOF_rows']);scalar_fits+=1
                    d=output/'coefficients';d.mkdir(exist_ok=True);used.to_csv(d/f'{month}_training.csv',index=False)
                    coefficient=dict(beta=beta,cutoff=str(cutoff),OOF_rows=len(used),OOF_ids_sha256=stable_digest(used.sample_id.tolist()),
                        OOF_label_available_max=str(used.label_available_at.max()),OOF_training_sha256=file_sha256(d/f'{month}_training.csv'))
                    atomic_write_json(d/f'{month}.json',coefficient);coefficients[str(month)]=coefficient
                    samples=combined.loc[(combined.reference_time>=cutoff)&(combined.reference_time<cutoff+pd.DateOffset(months=n)),META]
                    if samples.sample_id.duplicated().any():raise ContractError('duplicate evaluation metadata')
                    entries={r:identities[f'{month}/{r}'] for r in ('OR','HR')};base_models={r:load_component(e,a,contract) for r,e in entries.items()}
                    old_rate=RateModel.load(source/'models'/str(month)/'rate')
                    fresh=predict_inputs(samples,(entries,base_models,old_rate,h),builder,reg)
                    cached=frame(source/'predictions'/f'{month}_inputs.csv')
                    base_delta=assert_equal(fresh,cached,PRED+['pred_rate'])
                    frozen_alpha=read_json(source/'corrections'/f'{month}.json')['alpha']
                    v1=apply_correction(fresh,frozen_alpha);saved_v1=frame(source/'predictions'/f'{month}_V1.csv')
                    v1_delta=assert_equal(v1,saved_v1,PRED)
                    q=pd.DataFrame({'sample_id':samples.sample_id.to_numpy(),'pred_inverse_rate':model.predict(builder.X(samples,entry,h))})
                    q_reverse=pd.DataFrame({'sample_id':samples.iloc[::-1].sample_id.to_numpy(),'pred_inverse_rate':model.predict(builder.X(samples.iloc[::-1],entry,h))})
                    assert_equal(q,q_reverse,['pred_inverse_rate'])
                    parts=predict_dual(fresh[['sample_id',*PRED]],v1,q,beta)
                    reversed_parts=predict_dual(fresh[['sample_id',*PRED]].iloc[::-1],v1.iloc[::-1],q_reverse,beta)
                    for candidate in (V2,V3):assert_equal(parts[candidate],reversed_parts[candidate],PRED)
                    d=output/'predictions';d.mkdir(exist_ok=True);q.to_csv(d/f'{month}_q.csv',index=False)
                    for candidate,p in parts.items():p.to_csv(d/f'{month}_{candidate}.csv',index=False)
                    predictions[month]=parts;alignment.append(dict(origin=month,R2_rate_max_delta=base_delta,V1_max_delta=v1_delta,reversal_exact=True))
                    builder.cache.clear()
                    print(f'OPT22 outer {month}: beta={beta:.8f}, OOF={len(used)}, R2/V1 exact reproduction and reversal PASS',flush=True)
            atomic_write_json(output/'predictions_complete.json',dict(**budget.counts(),time_LAD_fits=scalar_fits,inference_fit_attempts=inference_counter['attempted_target_fits']))
        # Scoring labels stay outside the training/prediction path.
        def provider(unit,cutoff,metadata,parts):
            return {c:p.loc[p.sample_id.isin(metadata.sample_id)] for c,p in predictions[cutoff.month].items()}
        metrics,summary,errors,iron_exact=score(output,reg,provider)
        gate=acceptance(metrics,summary,reg,iron_exact)
        intervals={c:week_intervals(errors,c,'V1',**reg['bootstrap']) for c in (V2,V3)}
        atomic_write_json(output/'acceptance.json',gate);atomic_write_json(output/'week_bootstrap.json',intervals)
        atomic_write_json(output/'coefficients.json',coefficients);atomic_write_json(output/'training_records.json',records);atomic_write_json(output/'endpoint_alignment.json',alignment)
        verify_manifest(manifest)
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT22',G1=gate['status'],selected=gate['selected'],**budget.counts(),
            time_LAD_fits=scalar_fits,new_R2_fits=0,new_rate_fits=0,units=20,active_V1_and_R2_unchanged=True,challenger_generated=False))
        print(gate,flush=True)
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(error=str(exc),**budget.counts(),time_LAD_fits=scalar_fits));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ablation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.ablation,a.output)
