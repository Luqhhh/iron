"""Independent root/worker cold recovery, fixed hashes, no model/calibration fitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json, stable_digest, verify_file_identities
from bf_tap.optimization.component_export import META, PRED, read_json
from bf_tap.optimization.dual_ratio_common import frame
from bf_tap.optimization.qrf_time_run import (restore_manifest, verify, worker, fold_for, raw_matrix)
from bf_tap.optimization.rate_model import schema
from bf_tap.optimization.structural import apply_correction
from bf_tap.optimization.structural_run import predict_inputs
from bf_tap.optimization.trajectory_run import builder_for, equal
from bf_tap.optimization.v13_common import zero_fit


def check(root):
    manifest=restore_manifest(root);verify(manifest)
    verify_file_identities(read_json(root/'predictions_complete.json')['identities'])
    reg=manifest['registration']; builder=builder_for(manifest); checks={}
    with zero_fit() as counter:
        for month in reg['origins']:
            fold,_=fold_for(manifest,month)
            samples=frame(Path(reg['source_v8'])/'predictions'/f'{month}_inputs.csv',usecols=META)
            samples.reference_time=pd.to_datetime(samples.reference_time,utc=True).dt.tz_convert('Asia/Shanghai')
            x=raw_matrix(builder,samples,fold)
            info=read_json(root/'features'/str(month)/'evaluation.npz.json')
            digest=stable_digest(dict(schema=schema(x),rows=pd.util.hash_pandas_object(x,index=False).astype(str).tolist()))
            if digest!=info['raw_matrix_sha256']: raise ValueError('cold raw feature matrix changed')
            parts=predict_inputs(samples,fold,builder,{'rate':{'unusable_predicted_rate_max':1e-6}})
            v1=apply_correction(parts,read_json(Path(reg['source_v8'])/'corrections'/f'{month}.json')['alpha'],1e-6)
            diff=equal(v1,frame(root/'original_predictions'/f'{month}.csv'),PRED)
            worker(reg,'cold','--root',root,'--month',month,'--output',root/'cold_worker'/f'{month}.npz')
            report=read_json(root/'cold_worker'/f'{month}.npz.json')
            if any(report['zero_fit'].values()) or not report['exact_cold_checks']: raise ValueError('worker inference fit/cold checks failed')
            checks[str(month)]=dict(rows=len(samples),original_V1_max_difference=diff,
                forest_fit_attempts=report['zero_fit']['forest_fit_attempts'],preprocessor_fit_attempts=report['zero_fit']['preprocessor_fit_attempts'],
                full_reverse_chunk_subset_single_exact=True)
            print(f'cold cutoff {month}: original V1 + independent QRF exact checks pass',flush=True)
            builder.cache.clear()
    verify(manifest)
    atomic_write_json(root/'cold_validation.json',dict(status='PASS',zero_fit=counter,folds=checks,QRF_max_difference=0.,
        official_data_identity_verified=False,quality_evaluated=False,platform_verified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    check(parser.parse_args().run.resolve())
