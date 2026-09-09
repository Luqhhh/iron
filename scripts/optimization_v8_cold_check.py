"""Cold-process verification of saved OPT-20 models, OOF provenance and correction."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json,file_sha256,stable_digest,verify_file_identities,build_inference_source_contract
from bf_tap.io import read_csv
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.component_export import META,PRED,ComponentFeatures,read_json,forbid_fit
from bf_tap.optimization.horizon_router import load_component
from bf_tap.optimization.structural import INPUT,fit_correction,apply_correction
from bf_tap.optimization.structural_run import predict_inputs
from bf_tap.optimization.rate_model import RateModel


def frame(path):
    return pd.read_csv(path,float_precision='round_trip',dtype={'sample_id':str})


def main(root,output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=read_json(root/'manifest.json');reg=manifest['registration'];a=manifest['algorithm']
    reg['origins']={int(k):v for k,v in reg['origins'].items()}
    assert read_json(root/'final_status.json')['G0']=='PASS_OPT20'
    for k in ('inputs','sources','evidence'):verify_file_identities(manifest[k])
    # Only public process paths are available to cold feature construction.
    paths={k:v['path'] for k,v in manifest['inputs'].items() if k in ('operation_hourly','burden_change','data_dictionary')}
    op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
    contract=build_inference_source_contract(manifest['inputs'],semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
    identities=read_json(Path(reg['source_export'])/'component_identities.json')
    builder=ComponentFeatures(a,op,burden);counter={'attempted_target_fits':0};checks=[]
    with forbid_fit(counter):
        for month in reg['origins']:
            cutoff=pd.Timestamp(f'2024-{month:02d}-01',tz='Asia/Shanghai')
            rate=RateModel.load(root/'models'/str(month)/'rate')
            assert rate.metadata_['manifest_sha256']==file_sha256(root/'manifest.json')
            assert rate.metadata_['inference_source_contract']==contract
            assert pd.Timestamp(rate.metadata_['training']['fit_cutoff'])==cutoff
            history=frame(root/'models'/str(month)/'rate'/'history_snapshot.csv')
            for c in ('reference_time','tap_end_time','available_at'):
                history[c]=pd.to_datetime(history[c])
            assert (history.reference_time<cutoff).all() and (history.available_at<=cutoff).all()
            assert stable_digest(history.sort_values(['reference_time','sample_id']).sample_id.astype(str).tolist())==rate.metadata_['training']['sample_ids_sha256']
            used=frame(root/'oof'/f'correction_training_{month}.csv')
            for c in ('reference_time','label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max'):
                used[c]=pd.to_datetime(used[c])
            coeff=read_json(root/'corrections'/f'{month}.json')
            assert stable_digest(used.sample_id.astype(str).tolist())==coeff['OOF_ids_sha256']
            # Recover every OOF prediction from its earlier saved fold, and its target
            # from the current origin's eligible base history, never scoring files.
            h=history.set_index('sample_id')
            for fold,part in used.groupby('fold_cutoff'):
                saved=frame(root/'oof'/f'{fold.month}.csv').set_index('sample_id')
                expected=saved.loc[part.sample_id, PRED+['pred_rate']].to_numpy()
                assert np.array_equal(expected,part[PRED+['pred_rate']].to_numpy())
                fold_history=frame(root/'models'/str(fold.month)/'rate'/'history_snapshot.csv')
                assert not set(part.sample_id)&set(fold_history.sample_id)
                fold_references=pd.to_datetime(fold_history.reference_time)
                fold_available=pd.to_datetime(fold_history.available_at)
                assert (fold_references<fold).all() and (fold_available<=fold).all()
                assert (part.train_reference_max==fold_references.max()).all()
                assert (part.train_available_max==fold_available.max()).all()
                assert (part.history_available_max==fold_available.max()).all()
                record=read_json(root/'models'/str(fold.month)/'fit_record.json')
                for entry in record['base_components'].values():
                    bundle=Path(entry['path'])
                    assert file_sha256(bundle/'bundle.json')==entry['bundle_sha256']
                    base_md=read_json(bundle/'bundle.json')
                    assert base_md['training']['sample_ids_sha256']==record['sample_ids_sha256']
                    assert all(pd.Timestamp(base_md['training'][k])==fold for k in ('fit_cutoff','history_cutoff','label_available_cutoff'))
                for t in ('tap_iron','tap_time_len'):
                    assert np.array_equal(part[t].to_numpy(),h.loc[part.sample_id,t].to_numpy())
                assert np.array_equal(part.label_available_at.astype(str).to_numpy(),h.loc[part.sample_id,'available_at'].astype(str).to_numpy())
            alpha,_=fit_correction(used,cutoff,reg['correction']['minimum_OOF_rows'],reg['rate']['unusable_predicted_rate_max'])
            assert alpha==coeff['alpha']
            entries={r:identities[f'{month}/{r}'] for r in ('OR','HR')}
            loaded={r:load_component(e,a,contract) for r,e in entries.items()}
            saved=frame(root/'predictions'/f'{month}_inputs.csv').set_index('sample_id').sort_index()
            # Reconstruct metadata from fold metadata only; no target-bearing sample table.
            pieces=[frame(root/'oof'/f'{m}.csv')[META] for m in range(month,min(month+reg['origins'][month],12))]
            samples=pd.concat(pieces,ignore_index=True);samples.reference_time=pd.to_datetime(samples.reference_time)
            samples=samples.loc[samples.sample_id.isin(saved.index)]
            fresh=predict_inputs(samples,(entries,loaded,rate,history),builder,reg)
            aligned=fresh.set_index('sample_id').sort_index()
            assert aligned.index.equals(saved.index)
            difference=float(np.abs(aligned[PRED+['pred_rate']]-saved[PRED+['pred_rate']]).to_numpy().max())
            if difference>1e-10:raise AssertionError('cold structural inputs changed')
            corrected=apply_correction(fresh,alpha,reg['rate']['unusable_predicted_rate_max']).set_index('sample_id').sort_index()
            expected=frame(root/'predictions'/f'{month}_V1.csv').set_index('sample_id').sort_index()
            delta=float(np.abs(corrected[PRED]-expected[PRED]).to_numpy().max())
            if delta>1e-10:raise AssertionError('cold corrected outputs changed')
            checks.append(dict(origin=month,OOF_rows=len(used),prediction_rows=len(samples),input_max_delta=difference,output_max_delta=delta))
            print(f'OPT20 cold origin {month}: OOF provenance and inference PASS',flush=True)
    atomic_write_json(output/'validation.json',dict(status='PASS',checks=checks,**counter,
        corrected_max_delta=max(c['output_max_delta'] for c in checks),official_label_paths_in_feature_builder=False,
        script_sha256=file_sha256(__file__),source_manifest_sha256=file_sha256(root/'manifest.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();main(a.run,a.output)
