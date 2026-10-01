"""Independent cold saved-state, same-split OOF and scalar score audit."""
import argparse
import json
import math
from pathlib import Path
import resource
import sys

import numpy as np
import yaml

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK/'src'))
from bf_tap_r2.component_regularization_audit import verify_saved
from bf_tap_r2.ema_dropout_development import ARMS, context
from bf_tap_r2.ema_evaluation_diagnostics import sha, write_new
from bf_tap_r2.ema_average_span import task_frames
from bf_tap_r2.ema_training_scale import assert_isolated
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_periodic import digest


def scalar_score(actual, iron, time):
    errors = [math.fsum(abs(float(row[j])-float(p)) for row,p in zip(actual,column)) /
        math.fsum(abs(float(row[j])) for row in actual) for j,column in enumerate((iron,time))]
    return 100-50*math.fsum(errors)


def main(output):
    from bf_tap_r2.candidate_tiers import classify_candidates
    out = Path(output); manifest, spec, frame, folds = context(out)
    report = json.loads((out/'report.json').read_text()); maximum = 0.; differences = []
    states = 0; old_states = 0; matched = {}; gains = {}; old_root = Path(spec['main_root'])/spec['old_development']

    def replay(directory, training, metadata, mechanisms):
        inner = np.asarray(group_safe_inner_folds(training, seed=42)['fold'])
        fitting = training.loc[inner != 0].reset_index(drop=True); calibration = training.loc[inner == 0]
        selector = verify_saved(directory/'selection.pt', fitting, fitting[['tap_time_len']].to_numpy(),
            'EMA', spec['training'], mechanisms, calibration)
        model = verify_saved(directory/'refit.pt', training, training[['tap_time_len']].to_numpy(),
            'EMA', spec['training'], mechanisms, expected_epoch=selector.saved['trace']['selected_epoch'])
        for phase,saved in [('selection',selector),('refit',model)]:
            if saved.saved['trace'] != metadata['model']['traces'][phase]: raise ValueError('Saved trace identity differs')
        return model

    def cold(model, query, expected):
        nonlocal maximum
        actual = model.predict(query)[:, 0]; np.testing.assert_array_equal(actual, expected)
        variants = [model.predict(query.iloc[::-1])[::-1,0],
            np.concatenate([model.predict(query.iloc[i:i+37])[:,0] for i in range(0,len(query),37)])]
        maximum = max(maximum, *[float(np.max(np.abs(v-actual))) for v in variants])
        if maximum > spec['cold_predict_atol']: raise ValueError('Original order/chunk tolerance failed')
        return actual

    for seed,fv in folds.items():
        path = out/f'oof-s{seed}.npz'
        if sha(path) != report['vector_sha256'][str(seed)]: raise ValueError('OOF identity changed')
        with np.load(path,allow_pickle=False) as z: a = {key:z[key].copy() for key in z.files}
        if len(a['query_ids']) != 2754 or len(set(a['query_ids'])) != 2754: raise ValueError('Complete unique OOF required')
        np.testing.assert_array_equal(a['query_ids'], frame.sample_id.to_numpy(str)); np.testing.assert_array_equal(a['folds'],fv)
        np.testing.assert_array_equal(a['actual'],frame[['tap_iron','tap_time_len']].to_numpy())
        np.testing.assert_array_equal(a['spouts'],frame.spout_no.to_numpy())
        for fold in range(5):
            held = fv == fold; training,query = task_frames(frame,fv,fold); assert_isolated(training,query)
            inner = np.asarray(group_safe_inner_folds(training,seed=42)['fold'])
            partition = dict(training=digest(training.sample_id.tolist()),query=digest(query.sample_id.tolist()),
                inner_fit=digest(training.loc[inner != 0,'sample_id'].tolist()),
                inner_validation=digest(training.loc[inner == 0,'sample_id'].tolist()))
            if manifest['plan'][f's{seed}-f{fold}'] != partition: raise ValueError('Frozen training partition differs')
            assert_isolated(training.loc[inner != 0], training.loc[inner == 0].drop(columns=['tap_iron','tap_time_len']))
            old = old_root/f'tap_time_len-EMA-s{seed}-f{fold}'
            old_meta = json.loads((old/'metadata.json').read_text())
            model = replay(old,training,old_meta,spec['mechanisms']); old_states += 2
            with np.load(old/'predictions.npz',allow_pickle=False) as values:
                np.testing.assert_array_equal(values['query_ids'],query.sample_id.to_numpy(str))
                old_ema = cold(model,query,values['prediction'][:,0])
            ref = old_root/f'reference-s{seed}-f{fold}'
            reference_meta = json.loads((ref/'metadata.json').read_text())
            if (reference_meta['fit_ids_digest'] != partition['training'] or reference_meta['query_ids_digest'] != partition['query']
                    or reference_meta['query_labels_received'] is not False): raise ValueError('Reference training/query binding differs')
            with np.load(ref/'predictions.npz',allow_pickle=False) as values:
                np.testing.assert_array_equal(values['query_ids'],query.sample_id.to_numpy(str))
                np.testing.assert_array_equal(a['iron'][held],.5*values['v36_iron']+.5*values['v12_iron'])
                np.testing.assert_array_equal(a['Q75'][held],values['tap_time_len']+.75*(old_ema-values['v7_time']))
            for arm in ARMS:
                directory = out/f'{arm}-s{seed}-f{fold}'; receipt = json.loads((directory/'complete.json').read_text())
                if receipt['manifest_sha256'] != sha(out/'manifest.json') or receipt['optimizer_runs'] != 2:
                    raise ValueError('Unit manifest/budget differs')
                for filename,expected in receipt['hashes'].items():
                    if sha(directory/filename) != expected: raise ValueError('Unit artifact changed')
                metadata = json.loads((directory/'metadata.json').read_text())
                mechanisms = dict(spec['mechanisms'],dropout_consistency_lambda=spec['dropout_consistency_lambda'][arm])
                if metadata['mechanisms'] != mechanisms or metadata['partitions'] != partition:
                    raise ValueError('Scientific mechanism or partition differs')
                if metadata['source_directory'] != str(out.resolve()) or metadata['trial'] != arm:
                    raise ValueError('Physical cache identity differs')
                if metadata['peak_rss_mib'] > spec['max_worker_rss_mib']: raise ValueError('Worker memory gate failed')
                model = replay(directory,training,metadata,mechanisms); states += 2
                for trace in metadata['model']['traces'].values():
                    if (trace['dropout_consistency_lambda'] != mechanisms['dropout_consistency_lambda']
                            or trace['training_forward_passes'] != 2*trace['updates']
                            or any(row['training_forward_passes'] != 2*row['updates'] for row in trace['history'])):
                        raise ValueError('Two-pass count or coefficient differs')
                with np.load(directory/'predictions.npz',allow_pickle=False) as values:
                    np.testing.assert_array_equal(values['query_ids'],query.sample_id.to_numpy(str))
                    ema = cold(model,query,values['ema'])
                    np.testing.assert_array_equal(values['old_ema'],old_ema); np.testing.assert_array_equal(values['q75'],a['Q75'][held])
                    np.testing.assert_array_equal(values['iron'],a['iron'][held])
                    prediction = values['q75']+.75*(ema-old_ema)
                    np.testing.assert_array_equal(prediction,values['candidate']); np.testing.assert_array_equal(prediction,a[arm][held])
                    if not np.isfinite(prediction).all() or (prediction<0).any(): raise ValueError('Invalid affine prediction')
        scores = {name:scalar_score(a['actual'],a['iron'],a[name]) for name in ('Q75',*ARMS)}
        record = report['records'][str(seed)]
        for name,value in scores.items(): differences.append(abs(value-record['scores'][name]))
        for arm in ARMS: differences.append(abs(scores[arm]-scores['Q75']-record['gains'][arm]))
        matched[str(seed)] = scores['CONSISTENCY']-scores['PAIR_CONTROL']; gains[str(seed)] = scores['CONSISTENCY']-scores['Q75']
        differences.append(abs(matched[str(seed)]-record['matched_consistency_gain']))
        for name in ('Q75',*ARMS):
            metric = report['metrics']['tap_time_len'][name][str(seed)]; y = a['actual'][:,1]; prediction = a[name]
            differences.append(abs(metric['wmape']-math.fsum(np.abs(y-prediction))/math.fsum(np.abs(y))))
            for kind,labels in [('by_fold',a['folds']),('by_spout',a['spouts'])]:
                for label,value in metric[kind].items():
                    mask = labels == int(label)
                    differences.append(abs(value-math.fsum(np.abs(y[mask]-prediction[mask]))/math.fsum(np.abs(y[mask]))))
    selected = 'CONSISTENCY' if all(value>0 for value in [*gains.values(),*matched.values()]) else None
    if report['selected_for_confirmation'] != selected or report['confirmation_eligible'] != (selected is not None):
        raise ValueError('Independent paired gate differs')
    tiers_spec = dict(split_seeds=[42,3407],folds=5,candidates={'tap_time_len':['CONSISTENCY']},
        tie_preference_by_target={'tap_time_len':['CONSISTENCY']},reference_by_target={'tap_time_len':'Q75'})
    tiers = classify_candidates(report['metrics'],tiers_spec,yaml.safe_load((Path(spec['main_root'])/'configs/candidate_tiers.yaml').read_text()))
    if tiers != report['tiers']: raise ValueError('Automatic candidate classification differs')
    ledger = [json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    expected_units = [(s,f,arm) for s in (42,3407) for f in range(5) for arm in ARMS]
    for event in ('unit_started','unit_completed'):
        if [(e['seed'],e['fold'],e['trial']) for e in ledger if e['event']==event] != expected_units:
            raise ValueError('Incomplete or duplicate unit ledger')
    for s,f,arm in expected_units:
        unit = [e for e in ledger if (e['seed'],e['fold'],e['trial'])==(s,f,arm)]
        if ([e['event'] for e in unit] != ['unit_started','optimizer_started','optimizer_completed','optimizer_started','optimizer_completed','unit_completed']
                or [e.get('phase') for e in unit[1:5]] != ['selection','selection','refit','refit']):
            raise ValueError('Unclosed native training phases')
    if (states!=40 or old_states!=20 or max(differences)>1e-10 or report['optimizer_runs']!=40
            or report['new_estimators']!=20 or report['new_saved_states']!=40 or report['formal_promotion'] is not False
            or any(report[k]!=0 for k in ('new_confirmation_seeds','full_data_fits','packages','desktop_writes','uploads'))):
        raise ValueError('Full scientific count/scalar/promotion scope differs')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>spec['max_worker_rss_mib']: raise ValueError('Audit memory gate failed')
    write_new(out/'independent-score.json',dict(status='passed',new_saved_states=states,old_saved_states=old_states,
        actual_optimizer_runs=40,maximum_scalar_difference=max(differences),full_batch_cold_difference=0,
        maximum_order_chunk_difference=maximum,peak_rss_mib=peak,selected_for_confirmation=selected,new_fits=0,
        protected_preliminary_target_reads=0,report_sha256=sha(out/'report.json'),manifest_sha256=sha(out/'manifest.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output',required=True); main(p.parse_args().output)
