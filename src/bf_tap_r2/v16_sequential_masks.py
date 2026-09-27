"""Frozen sequential sparse feature masks with joint standardized targets."""
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
from .v12_joint import JointRegressor, joint_loss
from .v15_task_experts import load_references as previous_references
from .v14_joint_ple import load_joint_cache
from .v7_periodic import references
import torch


class UniformSelector(torch.nn.Module):
    def forward(self, logits):
        return torch.full_like(logits, 1 / logits.shape[-1])


class SequentialMaskNetwork(torch.nn.Module):
    def __init__(self, policy, n_categories, learned_masks):
        super().__init__()
        from pytorch_tabnet.tab_network import TabNetNoEmbeddings
        self.n_categories = n_categories
        self.learned_masks = learned_masks
        arguments = {k: policy[k] for k in ['n_d', 'n_a', 'n_steps', 'gamma',
            'n_shared', 'n_independent', 'virtual_batch_size', 'momentum', 'epsilon', 'mask_type']}
        self.network = TabNetNoEmbeddings(len(FEATURES)+n_categories, 2, **arguments)
        if not learned_masks:
            for attention in self.network.encoder.att_transformers:
                attention.selector = UniformSelector()

    def joined(self, x_num, x_cat):
        one_hot = torch.nn.functional.one_hot(x_cat[:, 0], self.n_categories).to(x_num.dtype)
        return torch.cat([x_num, one_hot], dim=1)

    def forward(self, x_num, x_cat):
        output, negative_entropy = self.network(self.joined(x_num, x_cat))
        self.entropy_penalty = -negative_entropy
        return output.unsqueeze(1)

    def masks(self, x_num, x_cat):
        _, masks = self.network.forward_masks(self.joined(x_num, x_cat))
        return torch.stack([masks[i] for i in sorted(masks)])


def training_batches(order, batch_size):
    if len(order) < 2 or batch_size < 2:
        raise ValueError('Batch normalization requires at least two training rows')
    batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
    if len(batches[-1]) == 1:
        batches[-2:] = [np.concatenate(batches[-2:])]
    return batches


class SequentialMaskRegressor(JointRegressor):
    def __init__(self, recipe, settings, policy):
        super().__init__({**recipe, 'backbone': 'tabm', 'frequency': .01}, settings)
        self.policy = dict(policy)

    def _initialize(self, frame, y):
        if self.recipe.get('control'):
            return super()._initialize(frame, y)
        torch.set_num_threads(1)
        torch.manual_seed(self.settings['random_seed'])
        self.preprocessor_ = NumericPreprocessor(structure='raw_tabm').fit(frame)
        self.mean_, self.std_ = np.mean(y, axis=0), np.std(y, axis=0)
        if not (self.std_ > 0).all():
            raise ValueError('Constant target')
        self.model_ = SequentialMaskNetwork(self.policy, self.preprocessor_.n_spout_categories_,
                                            self.recipe['learned_masks'])
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(),
            lr=self.settings['learning_rate'], weight_decay=self.settings['weight_decay'])
        self.scheduler_ = torch.optim.lr_scheduler.StepLR(self.optimizer_,
            step_size=self.settings['scheduler_step_size'], gamma=self.settings['scheduler_gamma'])

    def _train(self, frame, y, epochs, validation=None):
        if self.recipe.get('control'):
            return super()._train(frame, y, epochs, validation)
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings['random_seed'])
        best, best_epoch, stale = float('inf'), 0, 0
        if validation is not None:
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1]-self.mean_)/self.std_, dtype=torch.float32)
        for epoch in range(1, epochs+1):
            self.model_.train()
            for idx in training_batches(rng.permutation(len(frame)), self.settings['batch_size']):
                self.optimizer_.zero_grad(set_to_none=True)
                pred = self.model_(x[idx], cat[idx])
                loss = joint_loss(pred, target[idx]) + self.policy['lambda_sparse']*self.model_.entropy_penalty
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite TabNet training loss')
                loss.backward()
                self.optimizer_.step()
            self.scheduler_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    value = float((self.model_(vx, vc).mean(1)-vy).abs().mean())
                if value < best-self.settings['min_delta']:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
                if stale >= self.settings['patience']:
                    break
        if validation is not None:
            self.selection_stopped_epoch_ = epoch
        return best_epoch if validation is not None else epochs

    def fit(self, frame, y):
        if np.asarray(y).shape != (len(frame), 2):
            raise ValueError('Exactly two ordered training targets required')
        super().fit(frame, y)
        if not self.recipe.get('control'):
            x, cat = self._inputs(frame)
            with torch.no_grad():
                masks = self.model_.masks(x, cat)
                self.metadata_['mask_diagnostics'] = {
                    'learned': self.recipe['learned_masks'],
                    'shape': list(masks.shape),
                    'minimum': float(masks.min()), 'maximum': float(masks.max()),
                    'simplex_max_error': float((masks.sum(-1)-1).abs().max()),
                    'nonzero_fraction_by_step': (masks>0).float().mean((1, 2)).tolist(),
                    'mean_entropy_by_step': (-(masks*masks.clamp_min(1e-15).log()).sum(-1)).mean(1).tolist(),
                    'maximum_row_std': float(masks.std(1, unbiased=False).max()),
                    'attention_gradient_norm': sum(float(p.grad.norm()) for a in self.model_.network.encoder.att_transformers for p in a.parameters() if p.grad is not None)}
            self.metadata_['parameter_count'] = sum(p.numel() for p in self.model_.parameters())
        return self


def implementation_hashes():
    import importlib
    return {name: file_hash(importlib.import_module(name).__file__)
            for name in ['pytorch_tabnet.tab_network', 'pytorch_tabnet.sparsemax']}


def load_references(root, frame, folds, spec):
    a60, base, q20, current, joint, hashes = previous_references(root, frame, folds, spec)
    platform = {s: {'tap_iron': current[s]['tap_iron'].copy(),
                    'tap_time_len': base[s]['tap_time_len'].copy()} for s in folds}
    return platform, a60, base, q20, current, joint, hashes


def choose_confirmation(records, spec):
    expected = {(target, recipe) for target in spec['targets'] for recipe in spec['recipes']}
    if len(records) != len(expected) or {(r['target'], r['recipe']) for r in records} != expected:
        raise ValueError('Complete frozen candidate pool required')
    cells = {(s, f) for s in spec['split_seeds'] for f in range(spec['folds'])}
    for row in records:
        for label in ['V12_PLATFORM', 'A60', 'A35', 'Q20', 'CURRENT']:
            c = row['comparisons'][label]
            if {str(s) for s in c['seed_gains']} != {str(s) for s in spec['split_seeds']}:
                raise ValueError('Complete split seeds required')
            if len(c['cells']) != len(cells) or {(x['seed'], x['fold']) for x in c['cells']} != cells:
                raise ValueError('Complete fold coverage required')
    eligible = [r for r in records if all(g > 0 for label in ['V12_PLATFORM', 'CURRENT']
                                         for g in r['comparisons'][label]['seed_gains'].values())]
    if not eligible:
        return None
    best = min(eligible, key=lambda r: (-r['comparisons']['CURRENT']['seed_summary']['mean'],
        spec['tie_preference_by_target'][r['target']].index(r['recipe']), r['target']))
    return {key: best[key] for key in ['target', 'recipe']}


def evaluate_fold(frame, folds, recipe, settings, policy, fold):
    training = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = SequentialMaskRegressor(recipe, settings, policy).fit(training, training[list(TARGETS)].to_numpy())
    return model.predict(query), model.metadata_


def run(root, output):
    root = Path(root).resolve(); spec_path = root/'configs/round2_v16/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != spec['runtime_versions']:
        raise ValueError('V16 runtime mismatch')
    if implementation_hashes() != spec['implementation_hashes']:
        raise ValueError('TabNet implementation hash mismatch')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v16-sequential-masks'):
        raise ValueError('Private V16 output required')
    out.mkdir(parents=True, exist_ok=False)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec['split_seeds']}
    platform, a60, base, q20, current, joint, hashes = load_references(root, frame, folds, spec)
    write_new(out/'manifest.json', {'spec_sha256': file_hash(spec_path), 'versions': versions,
        'implementation_hashes': implementation_hashes(),
        'data_digest': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'fold_digests': {s: digest(f.tolist()) for s, f in folds.items()}, 'reference_hashes': hashes,
        'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    c = spec['control']; seed, fold = c['split_seed'], c['fold']
    pred, metadata = evaluate_fold(frame, folds[seed],
        {'control': True}, spec['control_training'], spec['tabnet_policy'], fold)
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
        jobs = {executor.submit(evaluate_fold, frame, folds[s], recipe, spec['training'], spec['tabnet_policy'], f): (name, s, f)
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
        raise RuntimeError(f'V16 failed fits retained: {failures}')
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
                    raise ValueError('Incomplete V16 OOF')
            comparisons = {label: nested_blend(y, folds, {s: ref[s][target] for s in folds}, pred, spec['blend_grid'])
                           for label, ref in [('V12_PLATFORM', platform), ('A60', a60), ('A35', base), ('Q20', q20), ('CURRENT', current)]}
            metrics[target][name], scores = {}, {}
            other = next(t for t in TARGETS if t != target)
            for s in folds:
                alpha = comparisons['CURRENT']['alphas'][s]
                blend = (1-alpha)*current[s][target]+alpha*pred[s]
                metrics[target][name][str(s)] = score_detail(y, blend, folds[s], frame.spout_no.values)
                # The other target remains the released V12 platform recipe, never an automatic joint blend.
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
