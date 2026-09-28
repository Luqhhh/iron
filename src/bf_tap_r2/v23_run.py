"""V23 frozen complete-fold runner, preflight and independent inference audit."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, TARGETS
from .v5_library import fold_vector, load_v5_training_frame, load_column_reference
from .v5_spec import load_v5_spec
from .v5_resolution import package_score, wmape, paired_summary
from .v7_periodic import digest, file_hash, write_new
from .v23_histogram import HistogramRegressor, histogram_median, select_finalist

def worker_init():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)

def require_environment(spec):
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if os.environ.get(key) != '1':
            raise ValueError('Set '+key+'=1 before launch')
    actual = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if actual != spec['runtime_versions']:
        raise ValueError('Frozen runtime mismatch')
    return actual

def source_hashes(root):
    names = ['v23_histogram.py', 'v23_run.py', 'v7_periodic.py', 'v3_6_networks.py',
             'v3_4_bags.py', 'v17_run.py', 'v17_confirm.py', 'v5_library.py', 'v5_resolution.py']
    return {name: file_hash(root/'src/bf_tap_r2'/name) for name in names}

def data_digest(frame):
    return digest(pd.util.hash_pandas_object(frame, index=True).values.tolist())

def load_references(root, seeds):
    # Imports are lazy so probability and scoring logic remain usable without torch.
    from .v17_run import context
    frame = load_v5_training_frame(root)
    legacy = load_v5_spec(root)
    spec17 = yaml.safe_load((root/'configs/round2_v17/SPEC.yaml').read_text())
    folds = {s: fold_vector(root, frame, s, legacy) for s in seeds}
    if list(seeds) == [42, 3407]:
        _, _, a35, b0, hashes = context(root, spec17, seeds)
        column = load_column_reference(root, frame, legacy)
        v36 = {s: column.base_for('tap_time_len', s) for s in seeds}
    else:
        from .v17_confirm import confirmed_references
        from .v7_confirm import read_reference_fold
        from .v5_replicate import _identity
        all_seeds = [42, 3407, 7777, 12011]
        all_folds = {s: fold_vector(root, frame, s, legacy) for s in all_seeds}
        a35, b0, hashes = confirmed_references(root, frame, all_folds, spec17)
        v36 = {}
        meta = {'kind': 'v36', 'trial_id': 'v36-s1-N-0048', 'target': 'tap_time_len', 'line': 'N'}
        for s in seeds:
            vector = np.full(len(frame), np.nan)
            for fold in range(5):
                mask = folds[s] == fold
                path = root/f'local/runs/round2-v5-error-covariance/replication-r1/seed-{s}/fold-{fold}.npz'
                values, _ = read_reference_fold(path, _identity(frame, folds[s], s, fold, meta), mask)
                vector[mask] = values
                hashes[str(path.relative_to(root))] = file_hash(path)
            v36[s] = vector
    r23 = {}
    for seed in seeds:
        phase = 'development-r1' if seed in [42, 3407] else 'confirmation-r1'
        directory = root/'local/runs/round2-v17'/phase
        audit = directory/'audit-r1.json'
        if json.loads(audit.read_text())['status'] != 'passed':
            raise ValueError('Audited P_LL_T evidence required')
        ledger = directory/'fit_ledger.jsonl'
        events = {e['key']: e for e in map(json.loads, ledger.read_text().splitlines())}
        pll = np.full(len(frame), np.nan)
        v7 = np.full(len(frame), np.nan)
        for fold in range(5):
            mask = folds[seed] == fold
            key = f'P_LL_T-s{seed}-f{fold}'
            path = directory/(key+'.npy')
            event = events[key]
            if (event['event'] != 'complete' or file_hash(path) != event['prediction_sha256']
                    or event['metadata']['fit_ids_digest'] != digest(frame.loc[~mask, 'sample_id'].tolist())):
                raise ValueError('P_LL_T same-fold source identity failure')
            value = np.load(path, allow_pickle=False)
            if value.shape != (int(mask.sum()),) or not np.isfinite(value).all():
                raise ValueError('P_LL_T coverage failure')
            pll[mask] = value
            hashes[str(path.relative_to(root))] = file_hash(path)
            if seed in [42, 3407]:
                time_path = root/f'local/runs/round2-v7-periodic-networks/development-r1/tap_time_len-tabm_plr001-s{seed}-f{fold}.npy'
            else:
                time_path = root/f'local/runs/round2-v7-periodic-networks/confirmation-r1/seed-{seed}-fold-{fold}.npy'
            v7[mask] = np.load(time_path, allow_pickle=False)
            hashes[str(time_path.relative_to(root))] = file_hash(time_path)
        for path in [ledger, audit, directory/'manifest.json']:
            hashes[str(path.relative_to(root))] = file_hash(path)
        if not np.allclose(.5*a35[seed]['tap_time_len']+.5*v7, b0[seed]['tap_time_len'], rtol=0, atol=1e-12):
            raise ValueError('V7 raw endpoint identity failure')
        r23[seed] = {'tap_iron': b0[seed]['tap_iron'].copy(),
                     'tap_time_len': .40*v36[seed]+.25*v7+.35*pll}
        if any(not np.isfinite(v).all() for v in r23[seed].values()):
            raise ValueError('Incomplete R23 reference')
    return frame, folds, {s: b0[s] for s in seeds}, r23, hashes

def score_predictions(frame, folds, b0, refs, predictions, target):
    other = next(t for t in TARGETS if t != target)
    results = {}
    y = frame[target].to_numpy()
    for seed, fv in folds.items():
        blended = .8*refs[seed][target]+.2*predictions[seed]
        candidate = {target: blended, other: refs[seed][other]}
        ref_score = package_score(*[wmape(frame[t], refs[seed][t]) for t in TARGETS])
        score = package_score(*[wmape(frame[t], candidate[t]) for t in TARGETS])
        results[str(seed)] = {'reference_score': ref_score, 'candidate_score': score,
            'gain': score-ref_score, 'same_candidate_column_gain_vs_B0': 50*(wmape(y, b0[seed][target])-wmape(y, blended)),
            'raw_member_wmape': wmape(y, predictions[seed]), 'candidate_wmape': wmape(y, blended),
            'other_column_equal': bool(np.array_equal(candidate[other], refs[seed][other])),
            'folds_descriptive': [{'fold': int(f), 'gain': 50*(wmape(y[fv==f], refs[seed][target][fv==f])-wmape(y[fv==f], blended[fv==f]))}
                                  for f in sorted(np.unique(fv))]}
    return results

def preflight(root, output):
    spec = yaml.safe_load((root/'configs/round2_v23/SPEC.yaml').read_text())
    versions = require_environment(spec)
    worker_init()
    settings = yaml.safe_load((root/spec['training_source']).read_text())['training']
    rng = np.random.default_rng(23001)
    frame = pd.DataFrame(rng.normal(size=(1000, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'syn{i}' for i in range(len(frame))]
    frame['spout_no'] = rng.integers(1, 4, len(frame))
    y = 60+8*frame[FEATURES[0]].to_numpy()+3*np.sin(frame[FEATURES[1]])+rng.normal(scale=.2, size=len(frame))
    synthetic_settings = {**settings, 'width': 64, 'tabm_k': 4, 'embedding_dim': 8,
                          'n_frequencies': 8, 'dropout': 0., 'max_epochs': 100, 'patience': 20}
    model = HistogramRegressor('gaussian', synthetic_settings).fit(frame.iloc[:800].reset_index(drop=True), y[:800])
    query = frame.iloc[800:].reset_index(drop=True)
    pred = model.predict(query)
    constant = np.median(y[:800])
    learned = float(np.abs(pred-y[800:]).mean())
    baseline = float(np.abs(constant-y[800:]).mean())
    cold = pickle.loads(pickle.dumps(model)).predict(query)
    cold_difference = float(np.max(np.abs(pred-cold)))
    paired = [HistogramRegressor(a, settings) for a in ['hard', 'gaussian']]
    for m in paired:
        m._initialize(frame.iloc[:100], y[:100])
    import torch
    init_difference = max(float((a-b).abs().max()) for a,b in zip(paired[0].model_.parameters(), paired[1].model_.parameters()))
    official, folds, b0, refs, hashes = load_references(root, spec['split_seeds'])
    report = {'status': 'passed' if learned < baseline and cold_difference == 0 and init_difference == 0 else 'failed',
        'synthetic_mae': learned, 'constant_median_mae': baseline, 'cold_max_difference': cold_difference,
        'same_shape_initialization_difference': init_difference, 'versions': versions,
        'spec_sha256': file_hash(root/'configs/round2_v23/SPEC.yaml'), 'source_hashes': source_hashes(root),
        'data_digest': data_digest(official), 'fold_digests': {str(s): digest(fv.tolist()) for s, fv in folds.items()},
        'reference_hashes': hashes, 'reference_scores': {str(s): package_score(*[wmape(official[t], refs[s][t]) for t in TARGETS]) for s in folds},
        'official_candidate_fits': 0, 'synthetic_selected_epoch': model.metadata_['selected_epoch']}
    write_new(output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','reference_hashes']}, indent=2), flush=True)
    if report['status'] != 'passed':
        raise ValueError('V23 preflight failed; no official-data fits')

def fit_job(frame, fv, fold, target, arm, settings):
    training = frame.loc[fv != fold].reset_index(drop=True)
    query = frame.loc[fv == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = HistogramRegressor(arm, settings).fit(training, training[target].to_numpy())
    prediction = model.predict(query)
    import torch
    model.metadata_['threads'] = {'intra': torch.get_num_threads(), 'inter': torch.get_num_interop_threads()}
    model.optimizer_ = None
    model.model_.zero_grad(set_to_none=True)
    return prediction, model.metadata_, pickle.dumps(model)

def assemble(directory, frame, folds, target, arm):
    result = {}
    for seed, fv in folds.items():
        vector = np.full(len(frame), np.nan)
        for f in range(5):
            path = directory/f'{target}-{arm}-s{seed}-f{f}.npy'
            value = np.load(path, allow_pickle=False)
            if value.shape != (int((fv==f).sum()),) or not np.isfinite(value).all():
                raise ValueError('Complete prediction coverage required')
            vector[fv==f] = value
        if not np.isfinite(vector).all():
            raise ValueError('Incomplete assembled OOF')
        result[seed] = vector
    return result

def summarize(directory, frame, folds, b0, refs, targets):
    records = []
    for target in targets:
        for arm in ['hard', 'gaussian']:
            values = assemble(directory, frame, folds, target, arm)
            rows = score_predictions(frame, folds, b0, refs, values, target)
            records.append({'target': target, 'arm': arm, 'seed_results': rows,
                'seed_gains': {s:r['gain'] for s,r in rows.items()},
                'mean_gain': float(np.mean([r['gain'] for r in rows.values()]))})
    return records

def run(root, output, gate_path, development=None, workers=8):
    import multiprocessing
    spec_path = root/'configs/round2_v23/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = require_environment(spec)
    gate = json.loads(gate_path.read_text())
    if gate['status'] != 'passed' or gate['spec_sha256'] != file_hash(spec_path) or gate['source_hashes'] != source_hashes(root):
        raise ValueError('Frozen successful preflight required')
    if not 1 <= workers <= 8:
        raise ValueError('Workers must be1..8')
    allowed = root/'local/runs/round2-v23-histogram-target'
    if not output.is_relative_to(allowed):
        raise ValueError('Private V23 directory required')
    if development is None:
        targets, seeds = spec['targets'], spec['split_seeds']
    else:
        if not development.is_relative_to(allowed):
            raise ValueError('Private V23 development required')
        dev_summary = json.loads((development/'summary.json').read_text())
        dev_audit = json.loads((development/'audit-r1.json').read_text())
        dev_manifest = json.loads((development/'manifest.json').read_text())
        target = dev_summary['selected_for_confirmation']
        if dev_audit['status'] != 'passed' or not target or dev_manifest['source_hashes'] != source_hashes(root):
            raise ValueError('Audited earned finalist required')
        targets, seeds = [target], spec['confirmation_seeds']
    frame, folds, b0, refs, hashes = load_references(root, seeds)
    if data_digest(frame) != gate['data_digest']:
        raise ValueError('Snapshot changed')
    if development is None and (hashes != gate['reference_hashes'] or {str(s):digest(v.tolist()) for s,v in folds.items()} != gate['fold_digests']):
        raise ValueError('Development reference/fold identity changed')
    settings = yaml.safe_load((root/spec['training_source']).read_text())['training']
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {'spec_sha256': file_hash(spec_path), 'source_hashes': source_hashes(root), 'versions': versions,
                'data_digest': data_digest(frame), 'fold_digests': {str(s):digest(v.tolist()) for s,v in folds.items()},
                'reference_hashes': hashes, 'gate_sha256': file_hash(gate_path), 'workers': workers,
                'targets': targets, 'seeds': seeds, 'settings': settings}
    write_new(output/'manifest.json', identity)
    failed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=worker_init) as pool:
        jobs = {}
        for target in targets:
            for arm in spec['arms']:
                for seed, fv in folds.items():
                    for fold in spec['folds']:
                        jobs[pool.submit(fit_job, frame, fv, fold, target, arm, settings)] = f'{target}-{arm}-s{seed}-f{fold}'
        for future in as_completed(jobs):
            key = jobs[future]
            try:
                pred, meta, model = future.result()
                path = output/(key+'.npy')
                with path.open('xb') as stream:
                    np.save(stream, pred)
                model_path = output/(key+'.pkl')
                with model_path.open('xb') as stream:
                    stream.write(model)
                event = {'event': 'complete', 'key': key, 'metadata': meta,
                         'prediction_sha256': file_hash(path), 'model_sha256': file_hash(model_path)}
            except Exception as exc:
                failed += 1
                event = {'event': 'failed', 'key': key, 'error': repr(exc)}
            with (output/'fit_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in event.items() if k != 'metadata'}), flush=True)
    if failed:
        write_new(output/'failure.json', {'failed_fits': failed})
        raise RuntimeError('Failed fits retained; no classification')
    if source_hashes(root) != identity['source_hashes']:
        raise ValueError('Source changed during fits')
    records = summarize(output, frame, folds, b0, refs, targets)
    if development is None:
        result = {'status': 'development_complete', 'records': records, 'selected_for_confirmation': select_finalist(records)}
    else:
        combined = []
        dev_by_arm = {r['arm']:r for r in dev_summary['records'] if r['target'] == target}
        for row in records:
            all_rows = {**dev_by_arm[row['arm']]['seed_results'], **row['seed_results']}
            gains = [all_rows[str(s)]['gain'] for s in [42,3407,7777,12011]]
            stats = paired_summary(gains)
            local_gate = np.mean([all_rows[str(s)]['candidate_score'] for s in [42,3407]]) >= 96.25
            combined.append({'arm': row['arm'], 'seed_results': all_rows, 'paired_summary': stats,
                'local_working_gate_met': bool(local_gate), 'promoted': bool(row['arm']=='gaussian' and all(g>0 for g in gains) and stats['lcb95']>0 and local_gate)})
        result = {'status': 'confirmation_complete', 'records': combined, 'target': target}
    result.update({'fits': len(jobs), 'failed_fits': failed, 'wall_seconds': time.time()-started, 'packages': 0, 'uploads': 0})
    write_new(output/'summary.json', result)
    print(json.dumps(result), flush=True)
    return result

def audit(root, directory):
    worker_init()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['source_hashes'] != source_hashes(root) or manifest['spec_sha256'] != file_hash(root/'configs/round2_v23/SPEC.yaml'):
        raise ValueError('Audit identity changed')
    frame, folds, b0, refs, hashes = load_references(root, manifest['seeds'])
    if hashes != manifest['reference_hashes'] or data_digest(frame) != manifest['data_digest']:
        raise ValueError('Audit data/reference identity changed')
    events = list(map(json.loads, (directory/'fit_ledger.jsonl').read_text().splitlines()))
    expected = {f'{t}-{a}-s{s}-f{f}' for t in manifest['targets'] for a in ['hard','gaussian'] for s in manifest['seeds'] for f in range(5)}
    if len(events) != len(expected) or {e['key'] for e in events} != expected:
        raise ValueError('Audit ledger coverage failure')
    worst = 0.
    for event in events:
        key = event['key']
        target, arm, s, f = key.rsplit('-', 3)
        seed, fold = int(s[1:]), int(f[1:])
        mask = folds[seed] == fold
        path, model_path = directory/(key+'.npy'), directory/(key+'.pkl')
        if event['event'] != 'complete' or file_hash(path) != event['prediction_sha256'] or file_hash(model_path) != event['model_sha256']:
            raise ValueError('Audit file hash failure')
        meta = event['metadata']
        if meta['fit_ids_digest'] != digest(frame.loc[~mask, 'sample_id'].tolist()) or meta['threads'] != {'intra':1, 'inter':1}:
            raise ValueError('Audit train identity/threads failure')
        training = frame.loc[~mask].reset_index(drop=True)
        y = training[target].to_numpy()
        from .v3_4_bags import group_safe_inner_folds
        inner = group_safe_inner_folds(training, seed=42)['fold'] != 0
        for label, index in [('inner', inner), ('outer', np.ones(len(training), dtype=bool))]:
            subset = y[index]; spread = np.ptp(subset)
            support = [float(subset.min()-.05*spread), float(subset.max()+.05*spread)]
            if not np.allclose(support, meta[label+'_support'], rtol=0, atol=1e-12):
                raise ValueError('Audit train-only target support failure')
            expected_means = training.loc[index, list(FEATURES)].to_numpy(dtype=float).mean(0)
            if not np.allclose(expected_means, meta[label+'_feature_means'], rtol=0, atol=1e-10):
                raise ValueError('Audit train-only feature statistics failure')
        model = pickle.loads(model_path.read_bytes())
        query = frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
        delta = float(np.max(np.abs(model.predict(query)-np.load(path, allow_pickle=False))))
        worst = max(worst, delta)
        if delta != 0:
            raise ValueError('Cold inference mismatch')
    records = summarize(directory, frame, folds, b0, refs, manifest['targets'])
    summary = json.loads((directory/'summary.json').read_text())
    if manifest['seeds'] == [42,3407] and (records != summary['records'] or select_finalist(records) != summary['selected_for_confirmation']):
        raise ValueError('Audit selection mismatch')
    # Independent direct pooled arithmetic; does not call the scoring function.
    for row in records:
        target, arm = row['target'], row['arm']
        vectors = assemble(directory, frame, folds, target, arm)
        y = frame[target].to_numpy()
        for seed in manifest['seeds']:
            old = np.abs(y-refs[seed][target]).sum()
            new = np.abs(y-(.8*refs[seed][target]+.2*vectors[seed])).sum()
            gain = 50*(old-new)/np.abs(y).sum()
            if abs(gain-row['seed_gains'][str(seed)]) > 1e-11:
                raise ValueError('Independent gain arithmetic failure')
    report = {'status':'passed', 'fits':len(events), 'cold_inference_max_difference':worst,
              'summary_sha256':file_hash(directory/'summary.json'), 'packages':0, 'uploads':0}
    write_new(directory/'audit-r1.json', report)
    print(json.dumps(report), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--development', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root/args.output).resolve()
    if not output.is_relative_to(root/'local/runs/round2-v23-histogram-target'):
        parser.error('Private V23 output required')
    if args.preflight:
        output.parent.mkdir(parents=True, exist_ok=True)
        preflight(root, output)
    elif args.audit:
        audit(root, output)
    else:
        if args.gate is None:
            parser.error('--gate is required')
        run(root, output, (root/args.gate).resolve(), (root/args.development).resolve() if args.development else None, args.workers)

if __name__ == '__main__':
    main()
