"""Independently reload q models, reconstruct OOF provenance and reproduce V2/V3."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json,file_sha256,stable_digest,verify_file_identities,build_inference_source_contract
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.component_export import META,PRED,ComponentFeatures,read_json,forbid_fit
from bf_tap.optimization.final_lifecycle import algorithm
from bf_tap.optimization.dual_ratio_common import frame,verify_manifest
from bf_tap.optimization.dual_ratio_run import history_frame,assert_equal
from bf_tap.optimization.inverse_rate_model import InverseRateModel
from bf_tap.optimization.inverse_oof import fit_certified_time
from bf_tap.optimization.dual_ratio import V2,V3,predict_dual

DATES=['reference_time','label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max',
    'inverse_fit_cutoff','inverse_train_reference_max','inverse_train_available_max','inverse_history_available_max']


def dated(path):
    f=frame(path)
    for c in DATES:f[c]=pd.to_datetime(f[c])
    return f


def check(source,output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=read_json(source/'manifest.json');verify_manifest(manifest)
    reg=manifest['registration'];reg['origins']={int(k):v for k,v in reg['origins'].items()}
    if read_json(source/'final_status.json')['G0']!='PASS_OPT22':raise ValueError('complete OPT22 required')
    parent=Path(reg['source_v8']);a=algorithm()
    paths={k:v['path'] for k,v in manifest['inputs'].items()}
    contract=build_inference_source_contract(manifest['inputs'],semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
    op,burden,_=_load_process_sources(paths,a['semantic'],a['features']);builder=ComponentFeatures(a,op,burden)
    counter={'attempted_target_fits':0};models={};folds={};fold_checks=[];outer_checks=[]
    with forbid_fit(counter):
        for month in reg['oof_months']:
            root=source/'models'/str(month);model=InverseRateModel.load(root);h=history_frame(root/'history_snapshot.csv')
            cutoff=pd.Timestamp(f'2024-{month:02d}-01',tz='Asia/Shanghai');md=model.metadata_
            if pd.Timestamp(md['training']['fit_cutoff'])!=cutoff or md['inference_source_contract']!=contract:raise ValueError('q source/cutoff mismatch')
            assert (h.reference_time<cutoff).all() and (h.available_at<=cutoff).all()
            assert stable_digest(h.sample_id.tolist())==md['training']['sample_ids_sha256']
            assert stable_digest(h.loc[h.tap_iron>0,'sample_id'].tolist())==md['positive_iron_ids_sha256']
            assert int((h.tap_iron==0).sum())==md['excluded_zero_iron']
            assert file_sha256(parent/'models'/str(month)/'rate'/'bundle.json')==md['source_v8_rate_bundle_sha256']
            entry=read_json(parent/'models'/str(month)/'fit_record.json')['base_components']['OR']
            oof=dated(source/'oof'/f'{month}.csv');old=frame(parent/'oof'/f'{month}.csv')
            assert not set(h.sample_id)&set(oof.sample_id)
            assert (oof.inverse_fit_cutoff==cutoff).all()
            assert (oof.inverse_train_reference_max==h.reference_time.max()).all()
            assert (oof.inverse_train_available_max==h.available_at.max()).all()
            assert (oof.inverse_history_available_max==h.available_at.max()).all()
            assert_equal(oof,old,PRED+['pred_rate'])
            fresh=pd.DataFrame({'sample_id':oof.sample_id.to_numpy(),'pred_inverse_rate':model.predict(builder.X(oof[META],entry,h))})
            difference=assert_equal(fresh,oof,['pred_inverse_rate'])
            models[month]=(model,h,entry);folds[month]=oof
            fold_checks.append(dict(month=month,rows=len(oof),OOF_max_delta=difference))
            print(f'OPT22 cold q OOF {month}: exact reload PASS',flush=True)
        metadata=pd.concat([f[META] for f in folds.values()],ignore_index=True)
        for month,n in reg['origins'].items():
            cutoff=pd.Timestamp(f'2024-{month:02d}-01',tz='Asia/Shanghai');model,h,entry=models[month]
            used=dated(source/'coefficients'/f'{month}_training.csv');coeff=read_json(source/'coefficients'/f'{month}.json')
            assert file_sha256(source/'coefficients'/f'{month}_training.csv')==coeff['OOF_training_sha256']
            assert stable_digest(used.sample_id.tolist())==coeff['OOF_ids_sha256']
            indexed=h.set_index('sample_id')
            for fold,part in used.groupby('inverse_fit_cutoff'):
                archive=folds[fold.month].set_index('sample_id').loc[part.sample_id].reset_index()
                assert_equal(part,archive,PRED+['pred_rate','pred_inverse_rate'])
                for t in ('tap_iron','tap_time_len'):assert np.array_equal(part[t],indexed.loc[part.sample_id,t])
                assert np.array_equal(part.label_available_at.astype(str),indexed.loc[part.sample_id,'available_at'].astype(str))
            beta,selected=fit_certified_time(used,cutoff,reg['OPT22']['minimum_causal_OOF_rows'])
            assert beta==coeff['beta'] and len(selected)==coeff['OOF_rows']
            samples=metadata.loc[(metadata.reference_time>=cutoff)&(metadata.reference_time<cutoff+pd.DateOffset(months=n))]
            fresh_q=pd.DataFrame({'sample_id':samples.sample_id.to_numpy(),'pred_inverse_rate':model.predict(builder.X(samples[META],entry,h))})
            q_delta=assert_equal(fresh_q,frame(source/'predictions'/f'{month}_q.csv'),['pred_inverse_rate'])
            base=frame(parent/'predictions'/f'{month}_inputs.csv')[['sample_id',*PRED]]
            v1=frame(parent/'predictions'/f'{month}_V1.csv')
            outputs=predict_dual(base,v1,fresh_q,beta)
            deltas={c:assert_equal(p,frame(source/'predictions'/f'{month}_{c}.csv'),PRED) for c,p in outputs.items()}
            assert np.array_equal(outputs[V2].set_index('sample_id').sort_index()[PRED[0]],v1.set_index('sample_id').sort_index()[PRED[0]])
            assert np.array_equal(outputs[V3].set_index('sample_id').sort_index()[PRED[0]],base.set_index('sample_id').sort_index()[PRED[0]])
            outer_checks.append(dict(month=month,rows=len(samples),OOF_rows=len(used),q_max_delta=q_delta,candidate_max_deltas=deltas))
            print(f'OPT22 cold outer {month}: beta provenance, V2/V3 and iron identity PASS',flush=True)
    verify_manifest(manifest)
    atomic_write_json(output/'validation.json',dict(status='PASS',source_manifest_sha256=file_sha256(source/'manifest.json'),
        script_sha256=file_sha256(__file__),folds=fold_checks,origins=outer_checks,**counter,official_target_paths_present=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check(a.source,a.output)
