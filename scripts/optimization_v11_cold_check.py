"""OPT-24 cold audit: no supervised or LAD fitting, no official target paths."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json,file_sha256,stable_digest,verify_file_identities
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import META,PRED,read_json,forbid_fit
from bf_tap.optimization.dual_ratio_common import verify_manifest
from bf_tap.optimization.structural import INPUT,directions,select_oof
from bf_tap.optimization.trajectory_candidate import TimeModel,predict
from bf_tap.optimization.trajectory_run import dated,builder_for,load_fold,candidate_inputs,equal
from bf_tap.optimization.refresh_factorial import stamp


def check_lad_optimal(selected, alpha):
    """Validate the objective's one-sided derivatives; do not estimate a coefficient."""
    delta,_=directions(selected[INPUT]); d=delta[:,1]
    residual=selected.pred_tap_time_len.to_numpy()+alpha*d-selected.tap_time_len.to_numpy()
    zero=np.isclose(residual,0.,atol=1e-10,rtol=0.)
    fixed=float(np.sum(np.sign(residual[~zero])*d[~zero])); kink=float(np.abs(d[zero]).sum())
    if not (0 <= alpha <= 1) or (alpha > 0 and fixed-kink > 1e-8) or (alpha < 1 and fixed+kink < -1e-8):
        raise ContractError('saved LAD fails constrained optimality certificate')


def main(source):
    dest=source/'cold_validation.json'
    if dest.exists():
        raise ContractError('cold evidence already exists')
    manifest=read_json(source/'manifest.json'); reg=manifest['registration']
    reg['origins']={int(k):v for k,v in reg['origins'].items()}
    verify_manifest(manifest)
    receipt=read_json(source/'predictions_complete.json');verify_file_identities(receipt['predictions'])
    if set(manifest['inputs']) != {'operation_hourly','burden_change','data_dictionary'}:
        raise ContractError('cold feature paths contain training labels')
    builder=builder_for(manifest);counter={'attempted_target_fits':0};folds={};models={};checks=[];oofs=[]
    with forbid_fit(counter):
        for month in reg['oof_months']:
            fold,old=load_fold(manifest,month);folds[month]=fold
            root=source/'models'/str(month)/'time';model=TimeModel.load(root);models[month]=model
            tr=model.metadata['training'];h=fold[1]['OR'][1]
            if tr['sample_ids_sha256'] != stable_digest(h.sample_id.tolist()) or tr['cutoff']!=str(stamp(month)) or tr['manifest_sha256']!=file_sha256(source/'manifest.json'):
                raise ContractError('new time training identity differs')
            oof=dated(source/'oof'/f'{month}.csv')
            equal(oof,old,[PRED[0],'pred_rate'],tolerance=0.)
            if not oof[META+['label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max']].equals(old[META+['label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max']]):
                raise ContractError('new OOF certificates differ')
            fresh=candidate_inputs(builder,oof[META],fold,model)
            delta=equal(fresh,oof,PRED+['pred_rate'])
            checks.append(dict(fold=month,OOF_max_delta=delta,rows=len(oof)))
            oofs.append(oof);builder.cache.clear()
            print(f'OPT24 cold fold {month}: provenance and predictions PASS',flush=True)
        combined=pd.concat(oofs,ignore_index=True)
        for month,n in reg['origins'].items():
            fold=folds[month];h=fold[1]['OR'][1];cutoff=stamp(month)
            labeled=combined.merge(h[['sample_id','tap_iron','tap_time_len','available_at']],on='sample_id',validate='one_to_one')
            if not (labeled.label_available_at==labeled.available_at).all():raise ContractError('LAD label availability changed')
            selected=select_oof(labeled,cutoff,reg['minimum_OOF_rows'])
            saved=dated(source/'corrections'/f'{month}_OOF.csv')
            for col in ('available_at',):
                saved[col]=pd.to_datetime(saved[col])
            if not selected.reset_index(drop=True).equals(saved.reset_index(drop=True)):
                raise ContractError('saved LAD rows differ from causal history join')
            coeff=read_json(source/'corrections'/f'{month}.json')
            if stable_digest(selected.sample_id.tolist())!=coeff['OOF_ids_sha256'] or file_sha256(source/'corrections'/f'{month}_OOF.csv')!=coeff['OOF_sha256']:
                raise ContractError('LAD provenance hash differs')
            check_lad_optimal(selected,coeff['alpha_T'])
            samples=combined.loc[(combined.reference_time>=cutoff)&(combined.reference_time<cutoff+pd.DateOffset(months=n)),META]
            original=dated(Path(reg['source_v8'])/'predictions'/f'{month}_V1.csv')
            inputs=candidate_inputs(builder,samples,fold,models[month])
            equal(inputs,dated(source/'predictions'/f'{month}_inputs.csv'),PRED+['pred_rate'])
            pred,_=predict(inputs,original,coeff['alpha_T'])
            delta=equal(pred,dated(source/'predictions'/f'{month}_V5.csv'),PRED)
            equal(pred,original,[PRED[0]],tolerance=0.)
            rev_inputs=candidate_inputs(builder,samples.iloc[::-1],fold,models[month])
            rev,_=predict(rev_inputs,original.iloc[::-1],coeff['alpha_T'])
            equal(pred,rev,PRED)
            checks.append(dict(origin=month,rows=len(pred),prediction_max_delta=delta,iron_exact=True,reversal=True))
            builder.cache.clear()
            print(f'OPT24 cold origin {month}: LAD certificate, iron exact and reversal PASS',flush=True)
    verify_manifest(manifest)
    atomic_write_json(dest,dict(status='PASS',checks=checks,**counter,LAD_fits=0,
        manifest_sha256=file_sha256(source/'manifest.json'),script_sha256=file_sha256(__file__),
        official_training_label_paths_in_feature_inputs=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);main(p.parse_args().source)
