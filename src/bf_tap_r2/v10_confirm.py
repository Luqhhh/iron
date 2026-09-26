"""Four-split confirmation of the frozen incremental V10 time candidate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .v3_4_bags import group_safe_inner_folds
from .v5_library import fold_vector, load_v5_training_frame
from .v5_replicate import _identity
from .v5_resolution import admit, nested_blend
from .v5_spec import load_v5_spec
from .v7_confirm import read_reference_fold
from .v7_periodic import digest, file_hash, references, write_new
from .v8_attention import v7_reference
from .v10_periodic_loss import evaluate_fold, select_confirmation


def selected_complete(summary, spec):
    if summary['status'] != 'development_complete':
        raise ValueError('Complete development required')
    records = summary['records']
    if len(records) != len(spec['recipes']) or {r['recipe'] for r in records} != set(spec['recipes']):
        raise ValueError('Incomplete or duplicate frozen pool')
    seeds = {str(s) for s in spec['split_seeds']}
    expected = {(s, f) for s in spec['split_seeds'] for f in range(spec['folds'])}
    for row in records:
        if row['target'] != 'tap_time_len':
            raise ValueError('Foreign target')
        for label in ['A35', 'Q20', 'V7_TIME']:
            comparison = row['comparisons'][label]
            if {str(s) for s in comparison['seed_gains']} != seeds:
                raise ValueError('Incomplete seed coverage')
            cells = comparison['cells']
            if len(cells) != len(expected) or {(c['seed'], c['fold']) for c in cells} != expected:
                raise ValueError('Incomplete or duplicate fold coverage')
    selected = select_confirmation(records, spec['tie_preference_by_target']['tap_time_len'])
    if selected is None or selected != summary['selected_confirmation']:
        raise ValueError('No verified incremental candidate')
    return selected


def four_seed_references(root, frame, folds, spec, confirmation):
    dev_base, dev_q20 = references(root, frame, spec)
    dev_v7, hashes = v7_reference(root, frame, folds, spec, dev_base)
    result = {label: {s: source[s]['tap_time_len'] for s in spec['split_seeds']}
              for label, source in [('A35', dev_base), ('Q20', dev_q20), ('V7_TIME', dev_v7)]}
    directory = root/confirmation['reference']['v7_run']
    manifest = json.loads((directory/'manifest.json').read_text())
    audit_path = directory/confirmation['reference']['v7_independent_audit']
    audit = json.loads(audit_path.read_text())
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if (manifest['data_digest'] != data_hash or audit['status'] != 'PASS'
            or manifest['selected']['target'] != 'tap_time_len'
            or manifest['selected']['recipe'] != spec['v7_reference']['recipe']):
        raise ValueError('V7 confirmation data or audit mismatch')
    hashes[str(audit_path.relative_to(root))] = file_hash(audit_path)
    hashes[str((directory/'manifest.json').relative_to(root))] = file_hash(directory/'manifest.json')
    events = [json.loads(line) for line in (directory/'fit_ledger.jsonl').read_text().splitlines()]
    expected = {(s, f) for s in confirmation['confirmation_seeds'] for f in range(5)}
    if (len(events) != len(expected) or any(e['event'] != 'complete' for e in events)
            or {(e['seed'], e['fold']) for e in events} != expected):
        raise ValueError('Incomplete V7 confirmation reference ledger')
    events = {(e['seed'], e['fold']): e for e in events}
    meta = {'kind': 'v36', 'trial_id': 'v36-s1-N-0048', 'target': 'tap_time_len', 'line': 'N'}
    for seed in confirmation['confirmation_seeds']:
        for label in result:
            result[label][seed] = np.full(len(frame), np.nan)
        for fold in range(5):
            mask = folds[seed] == fold
            path = root/confirmation['reference']['v5_time_cache']/f'seed-{seed}/fold-{fold}.npz'
            relative = str(path.relative_to(root))
            sha = file_hash(path)
            if sha != manifest['reference_cache_hashes'][relative]:
                raise ValueError('V5 reference changed since V7 confirmation')
            b36, old_n = read_reference_fold(path, _identity(frame, folds[seed], seed, fold, meta), mask)
            hashes[relative] = sha
            a35 = (1-spec['reference_time_alpha'])*b36 + spec['reference_time_alpha']*old_n
            q20 = (1-spec['secondary_reference_time_alpha'])*b36 + spec['secondary_reference_time_alpha']*old_n
            path = directory/f'seed-{seed}-fold-{fold}.npy'
            relative = str(path.relative_to(root)); sha = file_hash(path)
            if sha != audit['prediction_hashes'][relative]:
                raise ValueError('V7 confirmation prediction hash mismatch')
            values = np.load(path, allow_pickle=False)
            if values.shape != (int(mask.sum()),) or not np.isfinite(values).all():
                raise ValueError('Incomplete V7 confirmation predictions')
            training = frame.loc[~mask].reset_index(drop=True)
            metadata = events[(seed, fold)]['metadata']
            inner = group_safe_inner_folds(training, seed=spec['training']['inner_seed'])['fold'] != 0
            if (metadata['fit_ids_digest'] != digest(training.sample_id.tolist())
                    or metadata['inner_ids_digest'] != digest(training.loc[inner, 'sample_id'].tolist())):
                raise ValueError('V7 reference training identity mismatch')
            alpha = spec['v7_reference']['blend_weight']
            result['A35'][seed][mask] = a35
            result['Q20'][seed][mask] = q20
            result['V7_TIME'][seed][mask] = (1-alpha)*a35 + alpha*values
            hashes[relative] = sha
    if any(not np.isfinite(p).all() for source in result.values() for p in source.values()):
        raise ValueError('Incomplete four-seed reference coverage')
    return result, hashes


def run(root, output):
    root = Path(root).resolve()
    spec_path = root/'configs/round2_v10/SPEC.yaml'
    conf_path = root/'configs/round2_v10/CONFIRMATION.yaml'
    spec = yaml.safe_load(spec_path.read_text()); confirmation = yaml.safe_load(conf_path.read_text())
    dev = root/confirmation['development_run']
    summary = json.loads((dev/'summary.json').read_text())
    audit = json.loads((dev/'audit-r1.json').read_text())
    manifest = json.loads((dev/'manifest.json').read_text())
    selected = selected_complete(summary, spec)
    if (selected != confirmation['selected']['recipe'] or audit['status'] != 'passed'
            or audit['selected_confirmation'] != selected or audit['prediction_count'] != 20):
        raise ValueError('Frozen selection or development audit mismatch')
    if file_hash(spec_path) != manifest['spec_sha256']:
        raise ValueError('Development specification changed')
    for name, sha in manifest['dependency_code_hashes'].items():
        if file_hash(root/'src/bf_tap_r2'/name) != sha:
            raise ValueError('Development source changed')
    for key, sha in audit['hashes'].items():
        if file_hash(dev/(key+'.npy')) != sha:
            raise ValueError('Development prediction changed')
    for relative, sha in manifest['primary_reference_hashes'].items():
        if file_hash(root/relative) != sha:
            raise ValueError('Primary reference changed')
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != manifest['versions']:
        raise ValueError('Development runtime changed')
    frame = load_v5_training_frame(root)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if data_hash != manifest['data_digest']:
        raise ValueError('Development data changed')
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root))
             for s in spec['split_seeds'] + confirmation['confirmation_seeds']}
    for s in spec['split_seeds']:
        if digest(folds[s].tolist()) != manifest['fold_digests'][str(s)]:
            raise ValueError('Development fold changed')
    refs, hashes = four_seed_references(root, frame, folds, spec, confirmation)
    for relative, sha in manifest['v7_reference_hashes'].items():
        if hashes.get(relative) != sha:
            raise ValueError('V7 development reference changed')
    pred = {s: np.full(len(frame), np.nan) for s in folds}
    for s in spec['split_seeds']:
        for f in range(5):
            pred[s][folds[s] == f] = np.load(dev/f'tap_time_len-{selected}-s{s}-f{f}.npy')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v10-periodic-loss'):
        raise ValueError('Private V10 confirmation output required')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'manifest.json', {
        'selected': selected, 'spec_sha256': file_hash(spec_path), 'confirmation_sha256': file_hash(conf_path),
        'development_summary_sha256': file_hash(dev/'summary.json'), 'development_audit_sha256': file_hash(dev/'audit-r1.json'),
        'data_digest': data_hash, 'fold_digests': {s: digest(v.tolist()) for s, v in folds.items()},
        'versions': versions, 'reference_hashes': hashes,
        'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    failures = 0
    with ProcessPoolExecutor(max_workers=confirmation['budget']['workers']) as executor:
        jobs = {executor.submit(evaluate_fold, frame, folds[s], spec['recipes'][selected], spec['training'], f): (s, f)
                for s in confirmation['confirmation_seeds'] for f in range(5)}
        for future in as_completed(jobs):
            s, f = jobs[future]
            try:
                values, metadata = future.result()
                path = out/f'seed-{s}-fold-{f}.npy'
                with path.open('xb') as stream:
                    np.save(stream, values)
                pred[s][folds[s] == f] = values
                event = {'event': 'complete', 'seed': s, 'fold': f, 'metadata': metadata, 'prediction_sha256': file_hash(path)}
            except Exception as exc:
                failures += 1
                event = {'event': 'failed', 'seed': s, 'fold': f, 'error': repr(exc)}
            with (out/'fit_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            print(json.dumps({k: v for k, v in event.items() if k != 'metadata'}), flush=True)
    if failures or any(not np.isfinite(p).all() for p in pred.values()):
        raise RuntimeError('Incomplete confirmation; all failure evidence retained')
    comparisons = {label: nested_blend(frame.tap_time_len.values, folds, ref, pred, spec['blend_grid']) for label, ref in refs.items()}
    decisions = {label: admit(comparisons[label], min_seeds=4, positive_cells_min=8,
                              positive_cells_total=10, enforce_fold_criteria=False) for label in ['A35', 'V7_TIME']}
    row = next(r for r in summary['records'] if r['recipe'] == selected)
    local_score = float(np.mean(list(row['development_package_scores'].values())))
    result = {'candidate': selected, 'target': 'tap_time_len', 'comparisons': comparisons, 'decisions': decisions,
               'development_package_score': local_score,
               'local_working_gate_met': local_score >= spec['promotion']['local_working_gate'],
               'packages': 0, 'uploads': 0, 'release_authorized': False}
    write_new(out/'summary.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if os.environ.get(name) != '1':
            parser.error(f'Set {name}=1')
    run(Path.cwd(), args.output)


if __name__ == '__main__':
    main()
