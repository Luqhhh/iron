"""Fresh-process model and arithmetic audit; no import of calibration trainer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    from bf_tap_r2.component_regularization_audit import verify_saved
    from bf_tap_r2.data import FEATURES,TARGETS
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    from bf_tap_r2.v5_library import load_v5_training_frame,fold_vector
    from bf_tap_r2.v5_spec import load_v5_spec
    from bf_tap_r2.v7_periodic import digest
    from bf_tap_r2.v49_run import check_runtime
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text());spec=manifest['spec']
    root=Path(spec['main_root']);workspace=Path(manifest['workspace'])
    check_runtime(spec)
    for base,files in [(workspace,manifest['sources']),(root,spec['inputs'])]:
        for name,expected in files.items():
            if sha(base/name)!=expected:raise ValueError('Frozen identity changed: '+name)
    frame=load_v5_training_frame(root);summary=json.loads((out/'summary.json').read_text())
    maximum=0.;cold_max=0.;models=0;all_gains={n:{} for n in spec['families']};checked=0

    def close(a,b):
        nonlocal maximum,checked
        a,b=np.asarray(a,float),np.asarray(b,float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid audit vectors')
        d=float(np.max(np.abs(a-b))) if a.size else 0.
        maximum=max(maximum,d);checked+=a.size
        if d>1e-9:raise ValueError(f'Independent arithmetic mismatch: {d}')

    for seed in spec['split_seeds']:
        fv=fold_vector(root,frame,seed,load_v5_spec(root));vectors={n:np.full(len(frame),np.nan) for n in ['Q75',*spec['families']]}
        for fold in range(5):
            unit=out/f's{seed}-f{fold}';complete=json.loads((unit/'complete.json').read_text())
            assert complete['manifest_sha256']==sha(out/'manifest.json')
            for name,expected in complete['hashes'].items():
                if sha(unit/name)!=expected:raise ValueError('Saved unit changed')
            training=frame.loc[fv!=fold].reset_index(drop=True)
            query=frame.loc[fv==fold,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
            inner=np.asarray(group_safe_inner_folds(training,seed=27001,n_splits=5)['fold'])
            fitting=training.loc[inner!=0].reset_index(drop=True);calibration=training.loc[inner==0].reset_index(drop=True)
            native_inner=np.asarray(group_safe_inner_folds(fitting,seed=spec['training']['inner_seed'])['fold'])
            native_fit=fitting.loc[native_inner!=0].reset_index(drop=True);native_validation=fitting.loc[native_inner==0]
            selector=verify_saved(unit/'selection.pt',native_fit,native_fit[['tap_time_len']].to_numpy(),
                'EMA',spec['training'],spec['mechanisms'],native_validation)
            model=verify_saved(unit/'refit.pt',fitting,fitting[['tap_time_len']].to_numpy(),
                'EMA',spec['training'],spec['mechanisms'],expected_epoch=selector.saved['trace']['selected_epoch'])
            models+=2
            meta=json.loads((unit/'metadata.json').read_text())
            for phase,m in [('selection',selector),('refit',model)]:
                if m.saved['trace']!=meta['model']['traces'][phase]:raise ValueError('Saved/metadata trace mismatch')
            if meta['peak_rss_mib']>spec['max_worker_rss_mib']:raise ValueError('Memory gate')
            for label,data in [('fitting',fitting),('calibration',calibration),('outer_training',training),('query',query)]:
                if digest(data.sample_id.tolist())!=meta['partitions'][label]:raise ValueError('Partition digest mismatch')
            # Labels are excluded from every inference frame; check numerical duplicate separation.
            ids=[set(data.sample_id) for data in [fitting,calibration,query]]
            if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('ID overlap')
            groups=[set(__import__('pandas').util.hash_pandas_object(data[list(FEATURES)],index=False)) for data in [fitting,calibration,query]]
            if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Duplicate overlap')
            cal_query=calibration.drop(columns=list(TARGETS))
            with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
                a={k:saved[k].copy() for k in saved.files}
            assert a['query_ids'].tolist()==query.sample_id.tolist()
            assert a['calibration_ids'].tolist()==calibration.sample_id.tolist()
            pred=model.predict(cal_query)[:,0]
            if not np.array_equal(pred,a['ema_calibration']):raise ValueError('Full-batch cold difference')
            variants=[model.predict(cal_query.iloc[::-1])[::-1,0],np.concatenate([model.predict(cal_query.iloc[i:i+37])[:,0] for i in range(0,len(cal_query),37)])]
            cold_max=max(cold_max,*[float(np.max(np.abs(v-pred))) for v in variants])
            if cold_max>spec['cold_predict_atol']:raise ValueError('Order/chunk cold difference')
            cache=root/spec['reference_cache']/f'reference-calibration-s{seed}-f{fold}'
            old_meta=json.loads((cache/'metadata.json').read_text())
            if old_meta['fit_ids_digest']!=digest(fitting.sample_id.tolist()) or old_meta['query_ids_digest']!=digest(calibration.sample_id.tolist()):
                raise ValueError('Base calibration partition mismatch')
            with np.load(cache/'predictions.npz',allow_pickle=False) as base:
                assert base['query_ids'].tolist()==calibration.sample_id.tolist()
                b=.2*base['v36_time']+.3*base['n_time']+.5*base['v7_time']
                close(a['calibration_prediction'],b+.75*(pred-base['v7_time']))
            with np.load(root/spec['diagnostic_output']/f'error-map-{seed}.npz',allow_pickle=False) as old:
                close(a['reference'],old['reference'][fv==fold,1])
            cuts=np.quantile(fitting.total_press_diff,[.25,.5,.75])
            cal_bins=np.searchsorted(cuts,calibration.total_press_diff.to_numpy(),side='right')
            query_bins=np.searchsorted(cuts,query.total_press_diff.to_numpy(),side='right')
            correction_cv=np.asarray(group_safe_inner_folds(cal_query,seed=spec['correction']['cv_seed'],n_splits=3)['fold'])
            residual=calibration.tap_time_len.to_numpy()-a['calibration_prediction']
            for family in spec['families']:
                fitted=meta['fitted'][family]
                close(fitted['cuts'],cuts);close(fitted['correction_cv_folds'],correction_cv)
                def offsets(mask):
                    fallback=float(np.median(residual[mask]));values=[]
                    for q in range(4):
                        m=mask&(cal_bins==q)
                        values.append(float(np.median(residual[m])) if family=='PRESSURE' and m.sum()>=spec['correction']['minimum_bin_rows'] else fallback)
                    return np.asarray(values)
                oof=np.empty(len(calibration))
                for f in range(3):
                    held=correction_cv==f
                    oof[held]=offsets(~held)[cal_bins[held]]
                    assert fitted['correction_cv_fit_ids'][str(f)]==digest(cal_query.loc[~held,'sample_id'].tolist())
                losses=[math.fsum(np.abs(residual-g*oof)) for g in spec['correction']['gamma_grid']]
                close(fitted['cv_absolute_error'],losses)
                gamma=next(g for g,l in zip(spec['correction']['gamma_grid'],losses) if l<=min(losses)+1e-12)
                assert fitted['gamma']==gamma
                values=offsets(np.ones(len(calibration),bool));close(fitted['offsets'],values)
                close(a[family],a['reference']+gamma*values[query_bins])
                if not np.isfinite(a[family]).all() or (a[family]<0).any():raise ValueError('Invalid correction')
                vectors[family][fv==fold]=a[family]
            vectors['Q75'][fv==fold]=a['reference']
        y=frame.tap_time_len.to_numpy(float)
        for name,p in vectors.items():
            if not np.isfinite(p).all():raise ValueError('Incomplete full seed')
            m=summary['metrics']['tap_time_len'][name][str(seed)]
            close(m['wmape'],math.fsum(np.abs(y-p))/math.fsum(y))
            for f in range(5):
                mask=fv==f;close(m['by_fold'][str(f)],math.fsum(np.abs(y[mask]-p[mask]))/math.fsum(y[mask]))
            for s in (1,2):
                mask=frame.spout_no.to_numpy()==s;close(m['by_spout'][str(s)],math.fsum(np.abs(y[mask]-p[mask]))/math.fsum(y[mask]))
        for name in spec['families']:
            value=50*math.fsum(np.abs(y-vectors['Q75'])-np.abs(y-vectors[name]))/math.fsum(y)
            close(summary['gains'][name][str(seed)],value);all_gains[name][str(seed)]=value
    eligible=[n for n in spec['families'] if min(all_gains[n].values())>0]
    selected=min(eligible,key=lambda n:(-np.mean(list(all_gains[n].values())),spec['families'].index(n))) if eligible else None
    assert summary['selected_for_confirmation']==selected
    events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    counts={e:sum(v['event']==e for v in events) for e in ['unit_started','unit_completed','optimizer_started','optimizer_completed']}
    assert counts==dict(unit_started=10,unit_completed=10,optimizer_started=20,optimizer_completed=20)
    payload=dict(status='passed',saved_models=models,cold_full_batch_difference=0,maximum_order_chunk_difference=cold_max,
        independent_arithmetic_cells=checked,maximum_arithmetic_difference=maximum,actual_counts=counts,
        manifest_sha256=sha(out/'manifest.json'),summary_sha256=sha(out/'summary.json'),auditor_sha256=sha(__file__),
        outer_isolation_verified=True,selected_for_confirmation=selected,new_fits=0,packages=0)
    with (out/'audit.json').open('x') as h:json.dump(payload,h,indent=2,allow_nan=False);h.write('\n')
    print(json.dumps(payload),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
