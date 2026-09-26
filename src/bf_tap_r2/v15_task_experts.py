"""Frozen dense task-specific expert routing, trained jointly from scratch."""
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
from .v11_quantile import load_references as legacy_references
from .v12_joint import JointRegressor, make_network
from .v14_joint_ple import load_joint_cache
from .v7_periodic import references
import torch


class TaskExpertNetwork(torch.nn.Module):
    def __init__(self, settings, policy, n_categories, learned_gates):
        super().__init__()
        self.n_categories = n_categories
        self.n_experts = policy['n_experts']
        self.learned_gates = learned_gates
        expert_settings = {**settings, 'width': policy['width']}
        recipe = {'backbone': 'tabm', 'frequency': policy['frequency']}
        self.experts = torch.nn.ModuleList([
            make_network(recipe, expert_settings, n_categories, policy['latent_dim'])
            for _ in range(self.n_experts)])
        self.heads = torch.nn.ModuleList([torch.nn.Linear(policy['latent_dim'], 1) for _ in TARGETS])
        self.gate = torch.nn.Linear(len(FEATURES)+n_categories, len(TARGETS)*self.n_experts)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)
        self.gate.requires_grad_(learned_gates)

    def gate_weights(self, x_num, x_cat):
        categorical = torch.nn.functional.one_hot(x_cat[:, 0], self.n_categories).to(x_num.dtype)
        logits = self.gate(torch.cat([x_num, categorical], dim=1))
        return logits.reshape(len(x_num), len(TARGETS), self.n_experts).softmax(-1)

    def forward(self, x_num, x_cat):
        experts = torch.stack([model(x_num, x_cat) for model in self.experts], dim=2)
        mixed = torch.einsum('nkel,nte->nktl', experts, self.gate_weights(x_num, x_cat))
        return torch.stack([head(mixed[:, :, t, :]).squeeze(-1)
                            for t, head in enumerate(self.heads)], dim=-1)


class TaskExpertRegressor(JointRegressor):
    def __init__(self, recipe, settings, policy):
        super().__init__({**recipe, 'backbone': 'tabm', 'frequency': .01}, settings)
        self.policy = dict(policy)

    def _initialize(self, frame, y):
        super()._initialize(frame, y)
        if self.recipe.get('control'):
            return
        torch.manual_seed(self.settings['random_seed'])
        self.model_ = TaskExpertNetwork(self.settings, self.policy,
            self.preprocessor_.n_spout_categories_, self.recipe['learned_gates'])
        self.optimizer_ = torch.optim.AdamW((p for p in self.model_.parameters() if p.requires_grad),
            lr=self.settings['learning_rate'], weight_decay=self.settings['weight_decay'])

    def fit(self, frame, y):
        if np.asarray(y).shape != (len(frame), 2):
            raise ValueError('Exactly two ordered training targets required')
        super().fit(frame, y)
        if not self.recipe.get('control'):
            x, cat = self._inputs(frame)
            with torch.no_grad():
                weights = self.model_.gate_weights(x, cat)
                self.metadata_['gate_diagnostics'] = {
                    'learned': self.recipe['learned_gates'],
                    'parameter_norm': float(self.model_.gate.weight.norm()+self.model_.gate.bias.norm()),
                    'mean_weights': weights.mean(0).tolist(),
                    'min_weights': weights.amin(0).tolist(),
                    'max_weights': weights.amax(0).tolist(),
                    'mean_entropy': (-(weights*weights.clamp_min(1e-12).log()).sum(-1)).mean(0).tolist(),
                    'task_mean_absolute_weight_difference': float((weights[:, 0]-weights[:, 1]).abs().mean())}
            self.metadata_['parameter_count'] = sum(p.numel() for p in self.model_.parameters())
            self.metadata_['trainable_parameter_count'] = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        return self


def load_references(root, frame, folds, spec):
    base, q20, current, _, hashes = legacy_references(root, frame, folds, spec)
    joint, h = load_joint_cache(root, spec['v12_reference']['prediction_directory'], frame,
                              folds, spec['v12_reference']['recipe'])
    hashes.update(h)
    for seed in folds:
        a = spec['v12_reference']['blend_weight']
        current[seed]['tap_iron'] = (1-a)*base[seed]['tap_iron']+a*joint[seed][:, 0]
    platform, _ = references(root, frame, {**spec, 'reference_time_alpha': spec['platform_reference_time_alpha']})
    return platform, base, q20, current, joint, hashes


def choose_confirmation(records, spec):
    expected = {(target, recipe) for target in spec['targets'] for recipe in spec['recipes']}
    if len(records) != len(expected) or {(r['target'], r['recipe']) for r in records} != expected:
        raise ValueError('Complete frozen candidate pool required')
    cells = {(s, f) for s in spec['split_seeds'] for f in range(spec['folds'])}
    for row in records:
        for label in ['A60', 'A35', 'Q20', 'CURRENT']:
            c = row['comparisons'][label]
            if {str(s) for s in c['seed_gains']} != {str(s) for s in spec['split_seeds']}:
                raise ValueError('Complete split seeds required')
            if len(c['cells']) != len(cells) or {(x['seed'], x['fold']) for x in c['cells']} != cells:
                raise ValueError('Complete fold coverage required')
    eligible = [r for r in records if all(g > 0 for label in ['A60', 'CURRENT']
                                         for g in r['comparisons'][label]['seed_gains'].values())]
    if not eligible:
        return None
    best = min(eligible, key=lambda r: (-r['comparisons']['CURRENT']['seed_summary']['mean'],
        spec['tie_preference_by_target'][r['target']].index(r['recipe']), r['target']))
    return {key: best[key] for key in ['target', 'recipe']}


def evaluate_fold(frame, folds, recipe, settings, policy, fold):
    training = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = TaskExpertRegressor(recipe, settings, policy).fit(training, training[list(TARGETS)].to_numpy())
    return model.predict(query), model.metadata_


def run(root, output):
    root = Path(root).resolve(); spec_path = root/'configs/round2_v15/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != spec['runtime_versions']:
        raise ValueError('V15 runtime mismatch')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v15-task-experts'):
        raise ValueError('Private V15 output required')
    out.mkdir(parents=True, exist_ok=False)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec['split_seeds']}
    platform, base, q20, current, joint, hashes = load_references(root, frame, folds, spec)
    write_new(out/'manifest.json', {'spec_sha256': file_hash(spec_path), 'versions': versions,
        'data_digest': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'fold_digests': {s: digest(f.tolist()) for s, f in folds.items()}, 'reference_hashes': hashes,
        'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    c = spec['control']; seed, fold = c['split_seed'], c['fold']
    pred, metadata = evaluate_fold(frame, folds[seed],
        {'control': True}, spec['training'], spec['expert_policy'], fold)
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
        jobs = {executor.submit(evaluate_fold, frame, folds[s], recipe, spec['training'], spec['expert_policy'], f): (name, s, f)
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
        raise RuntimeError(f'V15 failed fits retained: {failures}')
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
                    raise ValueError('Incomplete V15 OOF')
            comparisons = {label: nested_blend(y, folds, {s: ref[s][target] for s in folds}, pred, spec['blend_grid'])
                           for label, ref in [('A60', platform), ('A35', base), ('Q20', q20), ('CURRENT', current)]}
            metrics[target][name], scores = {}, {}
            other = next(t for t in TARGETS if t != target)
            for s in folds:
                alpha = comparisons['CURRENT']['alphas'][s]
                blend = (1-alpha)*current[s][target]+alpha*pred[s]
                metrics[target][name][str(s)] = score_detail(y, blend, folds[s], frame.spout_no.values)
                # The other target remains A60, never an automatic joint blend.
                scores[s] = package_score(wmape(y, blend), wmape(frame[other].values, platform[s][other]))
            records.append({'target': target, 'recipe': name, 'comparisons': comparisons,
                'single_wmape': {s: wmape(y, pred[s]) for s in folds},
                'joint_periodic_control_wmape': {s: wmape(y, joint[s][:, list(TARGETS).index(target)]) for s in folds},
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
