"""Audit final coefficient training against saved causal folds and eligible history."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json,file_sha256,stable_digest
from bf_tap.optimization.component_export import PRED,read_json,forbid_fit
from bf_tap.optimization.structural import fit_correction


def frame(path):return pd.read_csv(path,float_precision='round_trip',dtype={'sample_id':str})


def main(development,release,output):
    if output.exists():raise FileExistsError(output)
    m=read_json(release/'bundle'/'structural.json');cutoff=pd.Timestamp(m['training']['fit_cutoff'])
    used=frame(release/'final_correction_OOF.csv')
    for c in ('reference_time','label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max'):
        used[c]=pd.to_datetime(used[c])
    assert len(used)==m['correction_OOF_rows']
    assert file_sha256(release/'final_correction_OOF.csv')==m['correction_OOF_sha256']
    assert stable_digest(used.sample_id.tolist())==m['correction_OOF_ids_sha256']
    history=frame(release/'bundle'/'rate'/'history_snapshot.csv').set_index('sample_id')
    counter={'attempted_target_fits':0};checks=[]
    with forbid_fit(counter):
        for fold,part in used.groupby('fold_cutoff'):
            prior=frame(development/'oof'/f'{fold.month}.csv').set_index('sample_id')
            assert np.array_equal(part[PRED+['pred_rate']].to_numpy(),prior.loc[part.sample_id,PRED+['pred_rate']].to_numpy())
            trained=frame(development/'models'/str(fold.month)/'rate'/'history_snapshot.csv')
            assert not set(part.sample_id)&set(trained.sample_id)
            assert (pd.to_datetime(trained.reference_time)<fold).all()
            assert (pd.to_datetime(trained.available_at)<=fold).all()
            assert (part.train_reference_max==pd.to_datetime(trained.reference_time).max()).all()
            assert (part.train_available_max==pd.to_datetime(trained.available_at).max()).all()
            assert (part.history_available_max==pd.to_datetime(trained.available_at).max()).all()
            for target in ('tap_iron','tap_time_len'):
                assert np.array_equal(part[target].to_numpy(),history.loc[part.sample_id,target].to_numpy())
            assert np.array_equal(part.label_available_at.astype(str).to_numpy(),pd.to_datetime(history.loc[part.sample_id,'available_at']).astype(str).to_numpy())
            if fold.month>=6:
                # Outer inputs have already been cold-reproduced; bind final OOF
                # rows to that independent check, including the November fold.
                outer=frame(development/'predictions'/f'{fold.month}_inputs.csv').set_index('sample_id')
                assert np.array_equal(prior.loc[part.sample_id,PRED+['pred_rate']].to_numpy(),outer.loc[part.sample_id,PRED+['pred_rate']].to_numpy())
            checks.append(dict(fold=str(fold),rows=len(part)))
        alpha,selected=fit_correction(used,cutoff,100,m['registration']['rate']['unusable_predicted_rate_max'])
        assert alpha==m['alpha'] and len(selected)==len(used)
    atomic_write_json(output,dict(status='PASS',rows=len(used),folds=checks,alpha=alpha,**counter,
        script_sha256=file_sha256(__file__),final_composite_sha256=file_sha256(release/'bundle'/'structural.json')))
    print(f'PASS final OOF audit: {len(used)} rows; exact coefficient reproduction; no fits',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.development,a.release,a.output)
