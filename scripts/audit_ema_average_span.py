"""Independent saved-model, OOF and arithmetic audit; no trainer entrypoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    from bf_tap_r2.component_regularization_audit import verify_saved
    from bf_tap_r2.data import FEATURES, TARGETS
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    from bf_tap_r2.v5_library import load_v5_training_frame, fold_vector
    from bf_tap_r2.v5_spec import load_v5_spec
    from bf_tap_r2.v7_periodic import digest
    from bf_tap_r2.v49_run import check_runtime, verified_unit, unit_id
    from bf_tap_r2.candidate_tiers import classify_candidates

    out = Path(out); manifest = json.loads((out/'manifest.json').read_text()); spec = manifest['spec']
    root, workspace = Path(spec['main_root']), Path(manifest['workspace'])
    check_runtime(spec)
    for base, files in [(workspace, manifest['sources']), (root, spec['inputs']),
                        (root, manifest['original_private_dependencies'])]:
        for name, expected in files.items():
            if sha(base/name) != expected:
                raise ValueError('Frozen identity changed: '+name)
    frame = load_v5_training_frame(root); summary = json.loads((out/'summary.json').read_text())
    order = ['SHORT_SPAN', 'LONG_SPAN']; seeds = [42, 3407]
    if spec['candidate_order'] != order or spec['split_seeds'] != seeds:
        raise ValueError('Independent candidate/seed scope mismatch')
    maximum = 0.; cold_max = 0.; cells = 0; models = 0; controls = 0
    gains = {n: {} for n in order}; max_rss = 0.

    def close(a, b):
        nonlocal maximum, cells
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid independent arithmetic vectors')
        difference = float(np.max(np.abs(a-b))) if a.size else 0.
        maximum = max(maximum, difference); cells += a.size
        if difference > 1e-9:
            raise ValueError('Independent arithmetic mismatch: '+str(difference))

    def saved_pair(directory, training, mechanisms, metadata):
        inner = np.asarray(group_safe_inner_folds(training, seed=spec['training']['inner_seed'])['fold'])
        fitting, validation = training.loc[inner != 0].reset_index(drop=True), training.loc[inner == 0]
        selector = verify_saved(directory/'selection.pt', fitting, fitting[['tap_time_len']].to_numpy(),
            'EMA', spec['training'], mechanisms, validation)
        model = verify_saved(directory/'refit.pt', training, training[['tap_time_len']].to_numpy(),
            'EMA', spec['training'], mechanisms, expected_epoch=selector.saved['trace']['selected_epoch'])
        for phase, saved in [('selection', selector), ('refit', model)]:
            if saved.saved['trace'] != metadata['model']['traces'][phase]:
                raise ValueError('Metadata/saved trace mismatch')
        return model

    def cold_predictions(model, query, expected):
        nonlocal cold_max
        pred = model.predict(query)[:, 0]
        if not np.array_equal(pred, expected):
            raise ValueError('Full-batch cold prediction mismatch')
        variants = [model.predict(query.iloc[::-1])[::-1, 0],
            np.concatenate([model.predict(query.iloc[i:i+37])[:, 0] for i in range(0, len(query), 37)])]
        cold_max = max(cold_max, *[float(np.max(np.abs(v-pred))) for v in variants])
        if cold_max > spec['cold_predict_atol']:
            raise ValueError('Original cold order/chunk gate failed')
        return pred

    old_root = root/spec['old_development']; old_manifest = json.loads((old_root/'manifest.json').read_text())
    for seed in seeds:
        fv = fold_vector(root, frame, seed, load_v5_spec(root))
        if digest(fv.tolist()) != manifest['fold_digests'][str(seed)]:
            raise ValueError('Frozen full-seed fold identity mismatch')
        vectors = {n: np.full(len(frame), np.nan) for n in ['Q75', *order]}
        iron = np.full(len(frame), np.nan)
        for fold in range(5):
            mask = fv == fold
            training = frame.loc[~mask].reset_index(drop=True)
            query = frame.loc[mask, ['sample_id', 'spout_no', *FEATURES]].reset_index(drop=True)
            if set(training.sample_id) & set(query.sample_id):
                raise ValueError('Outer ID leakage')
            import pandas as pd
            group_keys = lambda f: set(pd.util.hash_pandas_object(f[list(FEATURES)], index=False))
            if group_keys(training) & group_keys(query):
                raise ValueError('Outer numerical duplicate leakage')
            inner = np.asarray(group_safe_inner_folds(training, seed=42)['fold'])
            expected_partition = dict(training=digest(training.sample_id.tolist()), query=digest(query.sample_id.tolist()),
                inner_fit=digest(training.loc[inner != 0, 'sample_id'].tolist()),
                inner_validation=digest(training.loc[inner == 0, 'sample_id'].tolist()),
                training_rows=len(training), query_rows=len(query))
            if manifest['plan'][f's{seed}-f{fold}'] != expected_partition:
                raise ValueError('Manifest training partition mismatch')
            oldkey = f'tap_time_len-EMA-s{seed}-f{fold}'; refkey = f'reference-s{seed}-f{fold}'
            for key in (oldkey, refkey):
                if not verified_unit(old_root/key, unit_id(old_manifest, key)):
                    raise ValueError('Original cache unit not closed')
            old_meta = json.loads((old_root/oldkey/'metadata.json').read_text())
            control = saved_pair(old_root/oldkey, training, spec['mechanisms'], old_meta); controls += 2
            with np.load(old_root/oldkey/'predictions.npz', allow_pickle=False) as stored:
                if stored['query_ids'].tolist() != query.sample_id.tolist():
                    raise ValueError('Original control query identity')
                old_ema = cold_predictions(control, query, stored['prediction'][:, 0])
            with np.load(old_root/refkey/'predictions.npz', allow_pickle=False) as saved:
                if saved['query_ids'].tolist() != query.sample_id.tolist():
                    raise ValueError('Reference query identity')
                b = .2*saved['v36_time']+.3*saved['n_time']+.5*saved['v7_time']
                parent = b+.75*(old_ema-saved['v7_time'])
                background = .5*saved['v36_iron']+.5*saved['v12_iron']
                close(b, saved['tap_time_len']); close(background, saved['tap_iron'])
            reference_meta = json.loads((old_root/refkey/'metadata.json').read_text())
            if (reference_meta['fit_ids_digest'] != expected_partition['training']
                    or reference_meta['query_ids_digest'] != expected_partition['query']
                    or reference_meta['query_labels_received'] is not False):
                raise ValueError('Original reference partition mismatch')
            vectors['Q75'][mask] = parent; iron[mask] = background
            for name in order:
                directory = out/f'{name}-s{seed}-f{fold}'
                complete = json.loads((directory/'complete.json').read_text())
                if complete['manifest_sha256'] != sha(out/'manifest.json') or complete['new_optimizer_runs'] != 2:
                    raise ValueError('Candidate unit budget/binding mismatch')
                for filename, expected in complete['hashes'].items():
                    if sha(directory/filename) != expected:
                        raise ValueError('Candidate unit artifact changed')
                meta = json.loads((directory/'metadata.json').read_text())
                mechanisms = dict(spec['mechanisms'], ema_beta=spec['candidate_beta'][name])
                model = saved_pair(directory, training, mechanisms, meta); models += 2
                if meta['beta'] != mechanisms['ema_beta'] or meta['partitions'] != expected_partition:
                    raise ValueError('Candidate beta or partition mismatch')
                max_rss = max(max_rss, meta['peak_rss_mib'])
                if max_rss > spec['max_worker_rss_mib']:
                    raise ValueError('Worker memory gate failed')
                with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
                    values = {k: saved[k].copy() for k in saved.files}
                if values['query_ids'].tolist() != query.sample_id.tolist():
                    raise ValueError('Candidate query order changed')
                ema = cold_predictions(model, query, values['ema'])
                close(values['q75'], parent); close(values['old_ema'], old_ema); close(values['iron_reference'], background)
                prediction = parent+.75*(ema-old_ema)
                close(values['candidate'], prediction)
                if not np.isfinite(prediction).all() or (prediction < 0).any():
                    raise ValueError('Invalid affine candidate')
                vectors[name][mask] = prediction
                train_query = training.drop(columns=list(TARGETS))
                train_mae = float(np.abs(training.tap_time_len.to_numpy()-model.predict(train_query)[:, 0]).mean())
                close(meta['refit_training_mae'], train_mae)
                half = math.log(.5)/math.log(mechanisms['ema_beta'])
                for phase, trace in meta['model']['traces'].items():
                    averaging = meta['averaging'][phase]; updates = math.ceil(trace['fit_rows']/spec['training']['batch_size'])
                    close(averaging['half_life_updates'], half); close(averaging['half_life_epochs'], half/updates)
                    if (averaging['updates_per_epoch'] != updates or averaging['total_updates'] != trace['updates']
                            or averaging['selected_epoch'] != trace['selected_epoch'] or averaging['stopped_epoch'] != trace['stopped_epoch']
                            or averaging['cap_hit'] != (trace['stopped_epoch'] == spec['training']['max_epochs'])):
                        raise ValueError('Averaging span trace mismatch')
                descriptive = summary['diagnostics'][str(seed)][f'{name}-f{fold}']
                raw_mae = math.fsum(np.abs(frame.loc[mask, 'tap_time_len'].to_numpy()-ema))/len(query)
                close(descriptive['outer_component_mae'], raw_mae)
                close(descriptive['refit_training_mae'], train_mae)
                close(descriptive['refit_train_outer_gap'], raw_mae-train_mae)
                close(descriptive['candidate_mae'], math.fsum(np.abs(frame.loc[mask, 'tap_time_len'].to_numpy()-prediction))/len(query))
                chosen = meta['model']['traces']['selection']['history'][meta['model']['selected_epoch']-1]
                close(descriptive['inner_selected_gap'], chosen['validation_mae']-chosen['training_eval_mae'])
                if descriptive['averaging'] != meta['averaging']:
                    raise ValueError('Summary averaging diagnostics mismatch')
        y = frame.tap_time_len.to_numpy(float); y_iron = frame.tap_iron.to_numpy(float)
        for name, prediction in vectors.items():
            if not np.isfinite(prediction).all() or not np.isfinite(iron).all():
                raise ValueError('Incomplete same-seed OOF')
            wmape = math.fsum(np.abs(y-prediction))/math.fsum(np.abs(y))
            metric = summary['metrics']['tap_time_len'][name][str(seed)]
            close(metric['wmape'], wmape)
            for fold in range(5):
                held = fv == fold
                close(metric['by_fold'][str(fold)], math.fsum(np.abs(y[held]-prediction[held]))/math.fsum(np.abs(y[held])))
            for spout in (1, 2):
                held = frame.spout_no.to_numpy() == spout
                close(metric['by_spout'][str(spout)], math.fsum(np.abs(y[held]-prediction[held]))/math.fsum(np.abs(y[held])))
            score = 100-50*(wmape+math.fsum(np.abs(y_iron-iron))/math.fsum(np.abs(y_iron)))
            close(summary['package_scores'][str(seed)][name], score)
        for name in order:
            value = 50*math.fsum(np.abs(y-vectors['Q75'])-np.abs(y-vectors[name]))/math.fsum(np.abs(y))
            close(summary['gains'][name][str(seed)], value); gains[name][str(seed)] = value
    eligible = [name for name in order if min(gains[name].values()) > 0]
    selected = None
    if eligible:
        means = {name: math.fsum(gains[name].values())/2 for name in eligible}; best = max(means.values())
        selected = next(name for name in order if name in means and means[name] >= best-1e-12)
    if summary['selected_for_confirmation'] != selected:
        raise ValueError('Independent confirmation selection mismatch')
    tier_spec = dict(split_seeds=seeds, folds=5, candidates={'tap_time_len': order},
        tie_preference_by_target={'tap_time_len': order}, reference_by_target={'tap_time_len': 'Q75'})
    tiers = classify_candidates(summary['metrics'], tier_spec, yaml.safe_load((root/'configs/candidate_tiers.yaml').read_text()))
    if tiers != summary['tiers']:
        raise ValueError('Candidate classification mismatch')
    events = [json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    expected = [(s, f, name) for s in seeds for f in range(5) for name in order]
    for event in ('unit_started', 'unit_completed'):
        if [(e['seed'], e['fold'], e['candidate']) for e in events if e['event'] == event] != expected:
            raise ValueError('Incomplete, duplicate or reordered units')
    for seed, fold, name in expected:
        ledger = [e for e in events if (e['seed'], e['fold'], e['candidate']) == (seed, fold, name)]
        if ([e['event'] for e in ledger] != ['unit_started', 'optimizer_started', 'optimizer_completed',
                'optimizer_started', 'optimizer_completed', 'unit_completed']
                or [e.get('phase') for e in ledger[1:5]] != ['selection', 'selection', 'refit', 'refit']):
            raise ValueError('Unclosed optimizer or unit ledger')
    if (summary['new_estimators'] != 20 or summary['new_optimizer_runs'] != 40
            or any(summary[key] != 0 for key in ('new_confirmation_seeds', 'full_data_fits', 'packages', 'desktop_writes', 'agent_uploads'))):
        raise ValueError('Unregistered scientific count')
    payload = dict(status='passed', new_saved_models=models, original_control_saved_models=controls,
        full_batch_cold_difference=0, maximum_order_chunk_difference=cold_max,
        independent_arithmetic_cells=cells, maximum_arithmetic_difference=maximum,
        maximum_worker_rss_mib=max_rss, actual_estimators=20, actual_optimizer_runs=40,
        outer_isolation_verified=True, selected_for_confirmation=selected,
        manifest_sha256=sha(out/'manifest.json'), summary_sha256=sha(out/'summary.json'),
        auditor_sha256=sha(__file__), new_fits=0, packages=0)
    with (out/'audit.json').open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
