"""Confirm one frozen V12 joint periodic iron candidate against same-fold references."""
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
from .v9_confirm import _setup as v9_setup, _reference_events, read_reference
from .v11_confirm import selected_complete
from .v11_quantile import load_references
from .v12_joint import evaluate_fold


def iron_column(values, rows):
    values=np.asarray(values,dtype=float)
    if values.shape!=(rows,2) or not np.isfinite(values).all():
        raise ValueError('Complete joint iron/time prediction matrix required')
    return values[:,0]


def four_seed_references(root, frame, folds, spec, confirmation):
    if confirmation['selected'] != {'target':'tap_iron','recipe':'joint_plr001'}:
        raise ValueError('Only frozen V12 iron finalist is allowed')
    dev_folds={s:folds[s] for s in spec['split_seeds']}
    base,q20,current,_,hashes=load_references(root,frame,dev_folds,spec)
    result={label:{s:source[s]['tap_iron'] for s in dev_folds}
            for label,source in [('A35',base),('Q20',q20),('CURRENT',current)]}
    # The previous iron confirmation already audited the V8 baseline recipe.
    v9spec,v9config,selected,old_frame,old_folds,dev,baseline,_=v9_setup(root)
    if not frame.equals(old_frame) or any(not np.array_equal(fv,old_folds[s]) for s,fv in folds.items()):
        raise ValueError('V9 reference data/folds changed')
    directory=root/confirmation['reference']['v9_run']
    manifest=json.loads((directory/'manifest.json').read_text())
    audit_path=directory/confirmation['reference']['v9_independent_audit']
    audit=json.loads(audit_path.read_text())
    if audit['status']!='passed' or audit['candidate_prediction_files']!=20 or manifest['selected']!=v9config['selected']:
        raise ValueError('Complete audited V9 reference required')
    for name,sha in manifest['source_hashes'].items():
        if file_hash(root/name)!=sha: raise ValueError('V9 reference source changed')
    if manifest['parent_spec_sha256']!=file_hash(root/'configs/round2_v9/SPEC.yaml') or manifest['confirmation_spec_sha256']!=file_hash(root/'configs/round2_v9/CONFIRMATION.yaml'):
        raise ValueError('V9 reference specification changed')
    if manifest['development_summary_sha256']!=file_hash(dev/'summary.json') or manifest['reference_manifest_sha256']!=file_hash(baseline/'manifest.json'):
        raise ValueError('V9 parent evidence changed')
    for s,fv in folds.items():
        if manifest['fold_digests'][str(s)]!=digest(fv.tolist()): raise ValueError('V9 confirmation fold changed')
    for path,sha in audit['candidate_hashes'].items():
        if file_hash(root/path)!=sha: raise ValueError('V9 audited prediction changed')
        hashes[path]=sha
    events=_reference_events(baseline,v9config)
    if events is None: raise ValueError('V8 reference incomplete')
    rows=[json.loads(line) for line in (directory/'fit_ledger.jsonl').read_text().splitlines()]
    expected={(s,f) for s in confirmation['confirmation_seeds'] for f in range(5)}
    if len(rows)!=10 or any(r['event']!='complete' for r in rows) or {(r['seed'],r['fold']) for r in rows}!=expected:
        raise ValueError('V9 reference coverage incomplete')
    bycell={(r['seed'],r['fold']):r for r in rows}
    for s in confirmation['confirmation_seeds']:
        fv=folds[s]; iron=np.full(len(frame),np.nan); member=iron.copy()
        for f in range(5):
            mask=fv==f; path=directory/f'seed-{s}-fold-{f}.npy'; event=bycell[s,f]
            if file_hash(path)!=event['prediction_sha256']: raise ValueError('V9 prediction changed')
            if event['metadata']['fit_ids_digest']!=digest(frame.loc[~mask,'sample_id'].tolist()):
                raise ValueError('V9 training row identity changed')
            values=np.load(path,allow_pickle=False)
            if values.shape!=(int(mask.sum()),) or not np.isfinite(values).all(): raise ValueError('V9 prediction shape mismatch')
            member[mask]=values; ref_path=baseline/f'seed-{s}-fold-{f}.npz'
            iron[mask]=read_reference(ref_path,events[s,f],np.flatnonzero(mask),len(frame))
            hashes[str(ref_path.relative_to(root))]=file_hash(ref_path)
        result['A35'][s]=iron; result['Q20'][s]=iron.copy()
        a=spec['v9_reference']['blend_weight']; result['CURRENT'][s]=(1-a)*iron+a*member
    for directory_,names in [(directory,['manifest.json','fit_ledger.jsonl',confirmation['reference']['v9_independent_audit']]),
                             (baseline,['manifest.json','fit_ledger.jsonl','control.json','control-baseline.npz'])]:
        for name in names:
            path=directory_/name; hashes[str(path.relative_to(root))]=file_hash(path)
    if any(not np.isfinite(p).all() for ref in result.values() for p in ref.values()):
        raise ValueError('Incomplete four-seed references')
    return result,hashes


def run(root, output):
    root = Path(root).resolve()
    spec_path = root/'configs/round2_v12/SPEC.yaml'
    confirmation_path = root/'configs/round2_v12/CONFIRMATION.yaml'
    spec = yaml.safe_load(spec_path.read_text()); confirmation = yaml.safe_load(confirmation_path.read_text())
    dev = root/confirmation['development_run']
    summary = json.loads((dev/'summary.json').read_text())
    audit = json.loads((dev/'audit-r1.json').read_text()); manifest = json.loads((dev/'manifest.json').read_text())
    selected = selected_complete(summary, spec)
    if selected != confirmation['selected'] or audit['status'] != 'passed' or audit['selected_confirmation'] != selected or audit['prediction_count'] != 20:
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
            pred[s][folds[s] == f] = iron_column(np.load(dev/f'joint-{recipe}-s{s}-f{f}.npy',allow_pickle=False), int((folds[s]==f).sum()))
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v12-joint-tabm'):
        raise ValueError('Private V12 output required')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'manifest.json', {'selected': selected, 'spec_sha256': file_hash(spec_path),
        'confirmation_sha256': file_hash(confirmation_path), 'development_summary_sha256': file_hash(dev/'summary.json'),
        'development_audit_sha256': file_hash(dev/'audit-r1.json'), 'data_digest': data_hash,
        'fold_digests': {s: digest(v.tolist()) for s, v in folds.items()}, 'versions': versions,
        'reference_hashes': hashes, 'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    failures = 0
    with ProcessPoolExecutor(max_workers=confirmation['budget']['workers']) as executor:
        jobs = {executor.submit(evaluate_fold, frame, folds[s], spec['recipes'][recipe], spec['training'], f): (s, f)
                for s in confirmation['confirmation_seeds'] for f in range(5)}
        for future in as_completed(jobs):
            s, f = jobs[future]
            try:
                values, metadata = future.result(); path = out/f'seed-{s}-fold-{f}.npy'
                with path.open('xb') as stream:
                    np.save(stream, values)
                pred[s][folds[s] == f] = iron_column(values,int((folds[s]==f).sum()))
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
