"""Independent read-only arithmetic replay; no import of the diagnostic calculator."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    spec=json.loads((ROOT/'configs/ema_evaluation_diagnostics/SPEC.json').read_text())
    main_root=Path(spec['main_root']);out=main_root/spec['output']
    report=json.loads((out/'report.json').read_text())
    manifest=json.loads((out/'execution-manifest.json').read_text())
    if spec!=manifest['spec']:
        raise ValueError('Execution specification differs')
    for name,expected in spec['inputs'].items():
        if sha(main_root/name)!=expected:
            raise ValueError('Original evidence changed: '+name)
    for name,expected in manifest['execution_files'].items():
        if sha(name)!=expected:
            raise ValueError('Execution source changed')
    if sha(main_root/'复赛_test/test_samples.csv')!=report['test_input_sha256']:
        raise ValueError('Test inputs changed')
    from bf_tap_r2.v5_library import load_v5_training_frame,fold_vector
    from bf_tap_r2.v5_spec import load_v5_spec
    frame=load_v5_training_frame(main_root)
    max_error=0.;checked=0
    arrays={}
    for seed in spec['seeds']:
        with np.load(out/f'paired-oof-{seed}.npz',allow_pickle=False) as saved:
            arrays[seed]={k:saved[k].copy() for k in saved.files}
        a=arrays[seed]
        np.testing.assert_array_equal(a['query_ids'],frame.sample_id.to_numpy(str))
        np.testing.assert_array_equal(a['y'],frame.tap_time_len.to_numpy(float))
        np.testing.assert_array_equal(a['folds'],fold_vector(main_root,frame,seed,load_v5_spec(main_root)))
        for fold in range(5):
            held=a['folds']==fold
            d=main_root/'local/runs/strong-component-regularization/development-r2'
            with np.load(d/f'reference-s{seed}-f{fold}/predictions.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(a['V32'][held],z['tap_time_len'])
                old=z['v7_time'].copy()
            with np.load(d/f'tap_time_len-EMA-s{seed}-f{fold}/predictions.npz',allow_pickle=False) as z:
                delta=z['prediction'][:,0]-old
                np.testing.assert_array_equal(a['EMA'][held],a['V32'][held]+spec['incumbent_q']*delta)
            with np.load(Path(spec['ptarl_root'])/f'development/units/tap_time_len-s{seed}-f{fold}/predictions.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(a['PTARL_RAW'][held],z['PTARL_AUX'])
        np.testing.assert_array_equal(a['PTARL_Q20'],.8*a['V32']+.2*a['PTARL_RAW'])
        np.testing.assert_array_equal(a['PTARL_ORIGINAL_Q20'],.8*a['PTARL_ORIGINAL_PARENT']+.2*a['PTARL_RAW'])
        np.testing.assert_array_equal(a['EMA_PLUS_PTARL_INCREMENT_Q20'],a['EMA']+.2*(a['PTARL_RAW']-a['V32']))
        np.testing.assert_array_equal(a['EMA_PTARL_ENDPOINT_HALF'],.5*a['EMA']+.5*a['PTARL_Q20'])
    for record in report['comparisons']:
        a=arrays[record['seed']];y=a['y'];ref=a[record['reference']];candidate=a[record['candidate']]
        reductions=[abs(float(v-r))-abs(float(v-c)) for v,r,c in zip(y,ref,candidate)]
        value=50*math.fsum(reductions)/math.fsum(y)
        max_error=max(max_error,abs(value-record['score_gain']));checked+=1
        for fold,gain in enumerate(record['fold_gains']):
            mask=a['folds']==fold
            fold_value=50*math.fsum(d for d,m in zip(reductions,mask) if m)/math.fsum(y[mask])
            max_error=max(max_error,abs(fold_value-gain));checked+=1
    for record in report['simulations']:
        a=arrays[record['seed']];y=a['y'];ref=a[record['reference']];cand=a[record['candidate']]
        path=out/f"draws-{record['seed']}-{record['scheme']}.npy"
        if sha(path)!=record['draws_sha256']:
            raise ValueError('Simulated sample identities changed')
        draws=np.load(path,allow_pickle=False)
        if draws.shape!=(spec['replicates'],spec['test_rows']):
            raise ValueError('Simulation size differs')
        if (np.diff(np.sort(draws,axis=1),axis=1)==0).any():
            raise ValueError('Duplicate row in simulated test')
        if record['scheme']=='test_spout_mix':
            spout=frame.spout_no.to_numpy()
            for s,q in report['test_spout_quotas'].items():
                if not ((spout[draws]==int(s)).sum(axis=1)==q).all():
                    raise ValueError('Stratified draw composition differs')
        values=[]
        for begin in range(0,len(draws),250):
            ids=draws[begin:begin+250];actual=y[ids];denom=actual.sum(axis=1)
            wmape_ref=np.abs(actual-ref[ids]).sum(axis=1)/denom
            wmape_candidate=np.abs(actual-cand[ids]).sum(axis=1)/denom
            values.extend(50*(wmape_ref-wmape_candidate))
        values=np.asarray(values)
        max_error=max(max_error,abs(float(values.mean())-record['mean']))
        if float(np.mean(values < 0))!=record['negative_fraction']:
            raise ValueError('Simulation negative frequency mismatch')
        for q,value in record['quantiles'].items():
            max_error=max(max_error,abs(float(np.quantile(values,float(q)))-value))
        checked+=len(values)
    if max_error>1e-10:
        raise ValueError(f'Independent arithmetic disagreement: {max_error}')
    payload=dict(G0='passed',report_sha256=sha(out/'report.json'),
                 execution_manifest_sha256=sha(out/'execution-manifest.json'),
                 auditor_sha256=sha(__file__),arithmetic_witnesses=checked,
                 max_arithmetic_difference=max_error,new_fits=0,packages=0,
                 G1='descriptive_only_not_platform_forecast')
    with (out/'independent-audit.json').open('x') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(payload))


if __name__=='__main__':
    main()
