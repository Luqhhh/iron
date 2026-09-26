"""Confirm one frozen V11 quantile candidate against same-fold references."""
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

from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import admit, nested_blend
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, write_new
from .v10_confirm import four_seed_references as time_references
from .v11_quantile import choose_confirmation, evaluate_fold, load_references


def selected_complete(summary, spec):
    if summary['status'] != 'development_complete':
        raise ValueError('Complete development required')
    rows = summary['records']
    expected = {(t, r) for t in spec['targets'] for r in spec['recipes']}
    if len(rows) != len(expected) or {(r['target'], r['recipe']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicate frozen pool')
    cells = {(s, f) for s in spec['split_seeds'] for f in range(spec['folds'])}
    for row in rows:
        for label in ['A35', 'Q20', 'CURRENT']:
            comparison = row['comparisons'][label]
            if {str(s) for s in comparison['seed_gains']} != {str(s) for s in spec['split_seeds']}:
                raise ValueError('Incomplete seed coverage')
            actual = comparison['cells']
            if len(actual) != len(cells) or {(c['seed'], c['fold']) for c in actual} != cells:
                raise ValueError('Incomplete or duplicate fold coverage')
    selected = choose_confirmation(rows, spec)
    if selected is None or selected != summary['selected_confirmation']:
        raise ValueError('No verified incremental candidate')
    return selected


def four_seed_references(root, frame, folds, spec, confirmation):
    target = confirmation['selected']['target']
    development_folds = {s: folds[s] for s in spec['split_seeds']}
    base, q20, current, _, hashes = load_references(root, frame, development_folds, spec)
    result = {label: {s: source[s][target] for s in development_folds}
              for label, source in [('A35', base), ('Q20', q20), ('CURRENT', current)]}
    if target != 'tap_time_len':
        raise ValueError('Only the frozen time finalist is authorized')
    verified, h = time_references(root, frame, folds, spec, confirmation)
    for label, old in [('A35', 'A35'), ('Q20', 'Q20'), ('CURRENT', 'V7_TIME')]:
        for seed in development_folds:
            if not np.array_equal(result[label][seed], verified[old][seed]):
                raise ValueError('Development time reference changed')
        result[label] = verified[old]
    hashes.update(h)
    if any(not np.isfinite(p).all() for source in result.values() for p in source.values()):
        raise ValueError('Incomplete reference coverage')
    return result, hashes


def run(root, output):
    root = Path(root).resolve()
    spec_path = root/'configs/round2_v11/SPEC.yaml'
    confirmation_path = root/'configs/round2_v11/CONFIRMATION.yaml'
    spec = yaml.safe_load(spec_path.read_text()); confirmation = yaml.safe_load(confirmation_path.read_text())
    dev = root/confirmation['development_run']
    summary = json.loads((dev/'summary.json').read_text())
    audit = json.loads((dev/'audit-r1.json').read_text()); manifest = json.loads((dev/'manifest.json').read_text())
    selected = selected_complete(summary, spec)
    if selected != confirmation['selected'] or audit['status'] != 'passed' or audit['selected_confirmation'] != selected or audit['prediction_count'] != 80:
        raise ValueError('Frozen selection or audit mismatch')
    if file_hash(spec_path) != manifest['spec_sha256']:
        raise ValueError('Development specification changed')
    for name, sha in manifest['source_hashes'].items():
        if file_hash(root/'src/bf_tap_r2'/name) != sha:
            raise ValueError('Development source changed')
    for key, sha in audit['hashes'].items():
        if file_hash(dev/(key+'.npy')) != sha:
            raise ValueError('Development prediction changed')
    for relative, sha in manifest['reference_hashes'].items():
        if file_hash(root/relative) != sha:
            raise ValueError('Development reference changed')
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != manifest['versions']:
        raise ValueError('Runtime changed')
    frame = load_v5_training_frame(root)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if data_hash != manifest['data_digest']:
        raise ValueError('Development data changed')
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec['split_seeds']+confirmation['confirmation_seeds']}
    for s in spec['split_seeds']:
        if digest(folds[s].tolist()) != manifest['fold_digests'][str(s)]:
            raise ValueError('Development fold changed')
    refs, hashes = four_seed_references(root, frame, folds, spec, confirmation)
    if any(hashes.get(relative) != sha for relative, sha in manifest['reference_hashes'].items()):
        raise ValueError('Development reference identity changed')
    pred = {s: np.full(len(frame), np.nan) for s in folds}
    target, recipe = selected['target'], selected['recipe']
    for s in spec['split_seeds']:
        for f in range(5):
            pred[s][folds[s] == f] = np.load(dev/f'{target}-{recipe}-s{s}-f{f}.npy')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v11-quantile-representation'):
        raise ValueError('Private V11 output required')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'manifest.json', {'selected': selected, 'spec_sha256': file_hash(spec_path),
        'confirmation_sha256': file_hash(confirmation_path), 'development_summary_sha256': file_hash(dev/'summary.json'),
        'development_audit_sha256': file_hash(dev/'audit-r1.json'), 'data_digest': data_hash,
        'fold_digests': {s: digest(v.tolist()) for s, v in folds.items()}, 'versions': versions,
        'reference_hashes': hashes, 'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    failures = 0
    with ProcessPoolExecutor(max_workers=confirmation['budget']['workers']) as executor:
        jobs = {executor.submit(evaluate_fold, frame, folds[s], target, spec['recipes'][recipe], spec['training'], spec['quantile_policy'], f): (s, f)
                for s in confirmation['confirmation_seeds'] for f in range(5)}
        for future in as_completed(jobs):
            s, f = jobs[future]
            try:
                values, metadata = future.result(); path = out/f'seed-{s}-fold-{f}.npy'
                with path.open('xb') as stream:
                    np.save(stream, values)
                pred[s][folds[s] == f] = values
                event = {'event': 'complete', 'seed': s, 'fold': f, 'metadata': metadata, 'prediction_sha256': file_hash(path)}
            except Exception as exc:
                failures += 1; event = {'event': 'failed', 'seed': s, 'fold': f, 'error': repr(exc)}
            with (out/'fit_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            print(json.dumps({k: v for k, v in event.items() if k != 'metadata'}), flush=True)
    if failures or any(not np.isfinite(p).all() for p in pred.values()):
        raise RuntimeError('Incomplete confirmation; failed evidence retained')
    comparisons = {label: nested_blend(frame[target].values, folds, ref, pred, spec['blend_grid']) for label, ref in refs.items()}
    decisions = {label: admit(comparisons[label], min_seeds=4, positive_cells_min=8, positive_cells_total=10,
                              enforce_fold_criteria=False) for label in ['A35', 'CURRENT']}
    row = next(r for r in summary['records'] if r['target'] == target and r['recipe'] == recipe)
    score = float(np.mean(list(row['development_package_scores'].values())))
    result = {'status': 'confirmation_complete', 'selected': selected, 'comparisons': comparisons, 'decisions': decisions,
              'development_package_score': score, 'local_working_gate_met': score >= spec['promotion']['local_working_gate'],
              'packages': 0, 'uploads': 0, 'release_authorized': False}
    write_new(out/'summary.json', result); print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if os.environ.get(name) != '1':
            parser.error(f'Set {name}=1')
    run(Path.cwd(), args.output)


if __name__ == '__main__':
    main()
