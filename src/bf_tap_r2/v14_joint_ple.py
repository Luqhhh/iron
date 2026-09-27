"""Joint supervision with repaired PLE-B: separate frozen V14 experiment."""
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

from .candidate_tiers import classify_candidates
from .data import FEATURES, TARGETS
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import nested_blend, package_score, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, score_detail, write_new
from .v11_quantile import load_references as legacy_references, load_oof, choose_confirmation
from .v12_joint import JointRegressor
import torch
from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings
from tabm import TabM


class JointPLERegressor(JointRegressor):
    def __init__(self, recipe, settings, policy):
        super().__init__({**recipe, 'backbone': 'tabm', 'frequency': .01}, settings)
        self.policy = dict(policy)

    def _initialize(self, frame, y):
        super()._initialize(frame, y)
        if self.recipe.get('control'):
            return
        torch.manual_seed(self.settings['random_seed'])
        numeric, _ = self.preprocessor_.transform_tabm(frame)
        bins = compute_bins(torch.as_tensor(numeric), n_bins=self.policy['n_bins'])
        embedding = PiecewiseLinearEmbeddings(bins, d_embedding=self.policy['d_embedding'],
            activation=self.recipe['activation'], version=self.policy['version'])
        self.model_ = TabM.make(n_num_features=len(FEATURES),
            cat_cardinalities=[self.preprocessor_.n_spout_categories_], d_out=y.shape[1],
            k=self.settings['tabm_k'], n_blocks=self.settings['blocks'],
            d_block=self.settings['width'], dropout=self.settings['dropout'], num_embeddings=embedding)
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
            lr=self.settings['learning_rate'], weight_decay=self.settings['weight_decay'])
        self.bin_trace_.append({'rows': len(frame), 'ids_digest': digest(frame.sample_id.tolist()),
            'standardized_input_sha256': hashlib.sha256(numeric.tobytes()).hexdigest(),
            'bins': [b.tolist() for b in bins]})

    def fit(self, frame, y):
        self.bin_trace_ = []
        super().fit(frame, y)
        self.metadata_['bin_trace'] = self.bin_trace_
        self.metadata_['activation'] = self.recipe.get('activation')
        if not self.recipe.get('control'):
            weight = self.model_.num_module.linear.weight.detach()
            self.metadata_['nonlinear_weights_nonzero'] = int(torch.count_nonzero(weight))
            self.metadata_['nonlinear_weight_norm'] = float(weight.norm())
        return self


def load_joint_cache(root, directory, frame, folds, recipe):
    directory = root/directory
    manifest = json.loads((directory/'manifest.json').read_text())
    audit = json.loads((directory/'audit-r1.json').read_text())
    if audit['status'] != 'passed':
        raise ValueError('Audited joint reference required')
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if data_hash != manifest['data_digest']:
        raise ValueError('Joint reference data identity mismatch')
    events = [json.loads(line) for line in (directory/'fit_ledger.jsonl').read_text().splitlines()]
    completed = [e for e in events if e['event'] == 'complete']
    bykey = {e['key']: e for e in completed}
    if len(bykey) != len(completed):
        raise ValueError('Duplicate joint reference fit')
    hashes = {str((directory/name).relative_to(root)): file_hash(directory/name)
              for name in ['manifest.json', 'audit-r1.json', 'fit_ledger.jsonl']}
    result = {}
    for seed, fv in folds.items():
        if digest(fv.tolist()) != manifest['fold_digests'][str(seed)]:
            raise ValueError('Joint reference fold mismatch')
        values = np.full((len(frame), 2), np.nan)
        for fold in range(5):
            key = f'joint-{recipe}-s{seed}-f{fold}'; path = directory/(key+'.npy')
            event = bykey[key]; sha = file_hash(path); mask = fv == fold
            if sha != event['prediction_sha256'] or sha != audit['hashes'][key]:
                raise ValueError('Joint reference prediction hash mismatch')
            if event['metadata']['fit_ids_digest'] != digest(frame.loc[~mask, 'sample_id'].tolist()):
                raise ValueError('Joint reference training rows mismatch')
            p = np.load(path, allow_pickle=False)
            if p.shape != (int(mask.sum()), 2) or not np.isfinite(p).all():
                raise ValueError('Joint reference prediction coverage mismatch')
            values[mask] = p; hashes[str(path.relative_to(root))] = sha
        if not np.isfinite(values).all():
            raise ValueError('Incomplete joint reference')
        result[seed] = values
    return result, hashes


def load_references(root, frame, folds, spec):
    base, q20, current, _, hashes = legacy_references(root, frame, folds, spec)
    joint, h = load_joint_cache(root, spec['v12_reference']['prediction_directory'], frame, folds,
                              spec['v12_reference']['recipe'])
    hashes.update(h)
    alpha = spec['v12_reference']['blend_weight']
    for seed in folds:
        current[seed]['tap_iron'] = (1-alpha)*base[seed]['tap_iron']+alpha*joint[seed][:, 0]
    cfg = spec['v13_control']; directory = root/cfg['prediction_directory']
    path = directory/cfg['audit']; audit = json.loads(path.read_text())
    if audit['status'] != 'passed' or audit['prediction_count'] != 40:
        raise ValueError('Complete audited V13 control required')
    hashes[str(path.relative_to(root))] = file_hash(path)
    controls = {target: {} for target in TARGETS}
    for target in TARGETS:
        for recipe, single in cfg['recipe_mapping'].items():
            values, h = load_oof(root, cfg['prediction_directory'], frame, folds, target, single)
            for seed in folds:
                for fold in range(5):
                    key = f'{target}-{single}-s{seed}-f{fold}'
                    if file_hash(directory/(key+'.npy')) != audit['hashes'][key]:
                        raise ValueError('V13 control audit hash mismatch')
            controls[target][recipe] = values; hashes.update(h)
    return base, q20, current, controls, hashes, joint


def evaluate_fold(frame, folds, recipe, settings, policy, fold):
    training = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = JointPLERegressor(recipe, settings, policy).fit(training, training[list(TARGETS)].to_numpy())
    return model.predict(query), model.metadata_


def run(root, output):
    root = Path(root).resolve(); spec_path = root/'configs/round2_v14/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != spec['runtime_versions']:
        raise ValueError('V14 runtime mismatch')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v14-joint-ple'):
        raise ValueError('Private V14 output required')
    out.mkdir(parents=True, exist_ok=False)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec['split_seeds']}
    base, q20, current, controls, hashes, joint = load_references(root, frame, folds, spec)
    write_new(out/'manifest.json', {'spec_sha256': file_hash(spec_path), 'versions': versions,
        'data_digest': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'fold_digests': {s: digest(f.tolist()) for s, f in folds.items()}, 'reference_hashes': hashes,
        'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    c = spec['control']; seed, fold = c['split_seed'], c['fold']
    pred, metadata = evaluate_fold(frame, folds[seed],
        {'control': True}, spec['training'], spec['ple_policy'], fold)
    expected = joint[seed][folds[seed] == fold]
    with (out/'control.npy').open('xb') as stream:
        np.save(stream, pred)
    diff = float(np.max(np.abs(pred-expected)))
    write_new(out/'control.json', {'metadata': metadata, 'sha256': file_hash(out/'control.npy'), 'max_absolute_difference': diff})
    if not np.array_equal(pred, expected):
        raise ValueError('Standard-coordinate control failed; candidates not fitted')
    print(json.dumps({'control': 'passed', 'max_absolute_difference': diff}), flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=spec['budget']['workers']) as executor:
        jobs = {executor.submit(evaluate_fold, frame, folds[s], recipe, spec['training'], spec['ple_policy'], f): (name, s, f)
                for name, recipe in spec['recipes'].items() for s in folds for f in range(5)}
        for future in as_completed(jobs):
            name, seed, fold = jobs[future]; key = f'joint-{name}-s{seed}-f{fold}'
            try:
                p, meta = future.result(); path = out/(key+'.npy')
                with path.open('xb') as stream:
                    np.save(stream, p)
                event = {'event': 'complete', 'key': key, 'metadata': meta, 'prediction_sha256': file_hash(path)}
            except Exception as exc:
                failures += 1; event = {'event': 'failed', 'key': key, 'error': repr(exc)}
            with (out/'fit_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            print(json.dumps({k: v for k, v in event.items() if k != 'metadata'}), flush=True)
    if failures:
        raise RuntimeError(f'V14 failed fits retained: {failures}')
    records, metrics = [], {}
    for target in spec['targets']:
        y = frame[target].values
        metrics[target] = {spec['reference_by_target'][target]: {str(s): score_detail(y, current[s][target], folds[s], frame.spout_no.values) for s in folds}}
        for name, recipe in spec['recipes'].items():
            pred = {s: np.full(len(frame), np.nan) for s in folds}
            for s in folds:
                for f in range(5):
                    pred[s][folds[s] == f] = np.load(out/f'joint-{name}-s{s}-f{f}.npy', allow_pickle=False)[:, list(TARGETS).index(target)]
                if not np.isfinite(pred[s]).all():
                    raise ValueError('Incomplete V14 OOF')
            comparisons = {label: nested_blend(y, folds, {s: ref[s][target] for s in folds}, pred, spec['blend_grid'])
                           for label, ref in [('A35', base), ('Q20', q20), ('CURRENT', current)]}
            metrics[target][name], scores = {}, {}
            other = next(t for t in TARGETS if t != target)
            for s in folds:
                alpha = comparisons['CURRENT']['alphas'][s]
                blend = (1-alpha)*current[s][target]+alpha*pred[s]
                metrics[target][name][str(s)] = score_detail(y, blend, folds[s], frame.spout_no.values)
                # The other target remains A35, never an automatic joint blend.
                scores[s] = package_score(wmape(y, blend), wmape(frame[other].values, base[s][other]))
            control = name
            records.append({'target': target, 'recipe': name, 'comparisons': comparisons,
                'single_wmape': {s: wmape(y, pred[s]) for s in folds},
                'standard_control_wmape': {s: wmape(y, controls[target][control][s]) for s in folds},
                'development_package_scores': scores})
    result = {'status': 'development_complete', 'records': records,
        'candidate_tiers': classify_candidates(metrics, spec, yaml.safe_load((root/'configs/candidate_tiers.yaml').read_text())),
        'selected_confirmation': choose_confirmation(records, spec), 'packages': 0, 'uploads': 0, 'release_authorized': False}
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
