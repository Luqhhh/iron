"""Frozen, from-scratch loss controls on V7's periodic representation."""
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
import torch
import yaml

from .candidate_tiers import classify_candidates
from .data import TARGETS
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import nested_blend, package_score, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import PeriodicRegressor, digest, file_hash, references, score_detail, write_new
from .v8_attention import v7_reference


def training_loss(prediction, target, recipe):
    if prediction.ndim != 2 or target.shape != (len(prediction),):
        raise ValueError("Expected per-head prediction and one-dimensional target")
    residual = prediction - target[:, None]
    if recipe['loss'] == 'mse':
        return residual.square().mean()
    if recipe['loss'] == 'mae':
        return residual.abs().mean()
    if recipe['loss'] == 'smooth_l1':
        beta = float(recipe['beta'])
        if not beta > 0 or not np.isfinite(beta):
            raise ValueError('SmoothL1 beta must be positive and finite')
        absolute = residual.abs()
        return torch.where(absolute < beta, residual.square() / (2 * beta),
                           absolute - beta / 2).mean()
    raise ValueError('Unregistered loss')


class LossRegressor(PeriodicRegressor):
    def _train(self, frame, y, epochs, validation=None):
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y - self.mean_) / self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings['random_seed'])
        best, best_epoch, stale = float('inf'), 0, 0
        if validation is not None:
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1] - self.mean_) / self.std_, dtype=torch.float32)
        for epoch in range(1, epochs + 1):
            self.model_.train()
            order = rng.permutation(len(frame))
            for start in range(0, len(order), self.settings['batch_size']):
                idx = order[start:start + self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                pred = self.model_(x[idx], cat[idx])[:, :, 0]
                loss = training_loss(pred, target[idx], self.recipe)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                self.optimizer_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    pred = self.model_(vx, vc).mean(1).flatten()
                    value = float((pred - vy).abs().mean())
                if value < best - self.settings['min_delta']:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
                if stale >= self.settings['patience']:
                    break
        if validation is not None:
            self.selection_stopped_epoch_ = epoch
        return best_epoch if validation is not None else epochs

    def fit(self, frame, y):
        super().fit(frame, y)
        self.metadata_.update(loss=self.recipe['loss'], beta=self.recipe.get('beta'),
                              selection_stopped_epoch=self.selection_stopped_epoch_,
                              budget_limited=self.selection_stopped_epoch_ >= self.settings['max_epochs'])
        return self


def evaluate_fold(frame, folds, recipe, settings, fold):
    training = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = LossRegressor(recipe, settings).fit(training, training.tap_time_len.to_numpy())
    return model.predict(query), model.metadata_


def select_confirmation(records, order):
    eligible = [r for r in records if all(
        g > 0 for label in ('A35', 'V7_TIME') for g in r['comparisons'][label]['seed_gains'].values())]
    return min(eligible, key=lambda r: (-r['comparisons']['V7_TIME']['seed_summary']['mean'],
                                      order.index(r['recipe'])))['recipe'] if eligible else None


def run(root, output):
    root = Path(root).resolve()
    spec_path = root/'configs/round2_v10/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != spec['runtime_versions']:
        raise ValueError('V10 runtime mismatch')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v10-periodic-loss'):
        raise ValueError('Private V10 output required')
    out.mkdir(parents=True, exist_ok=False)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec['split_seeds']}
    audit_path = root/spec['primary_reference_audit']
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed':
        raise ValueError('Verified primary references required')
    for relative, sha in audit['file_sha256'].items():
        if file_hash(root/relative) != sha:
            raise ValueError('Historical primary reference changed')
    base, q20 = references(root, frame, spec)
    v7, hashes = v7_reference(root, frame, folds, spec, base)
    write_new(out/'manifest.json', {
        'spec_sha256': file_hash(spec_path), 'code_sha256': file_hash(__file__), 'versions': versions,
        'data_digest': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'fold_digests': {s: digest(f.tolist()) for s, f in folds.items()},
        'primary_reference_hashes': audit['file_sha256'], 'primary_audit_sha256': file_hash(audit_path),
        'v7_reference_hashes': hashes,
        'dependency_code_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    control = spec['control']
    seed, fold = control['split_seed'], control['fold']
    pred, metadata = evaluate_fold(frame, folds[seed], {'backbone': 'tabm', 'frequency': 0.01, 'loss': 'mse'},
                                   spec['training'], fold)
    expected_path = root/spec['v7_reference']['prediction_directory']/f'tap_time_len-tabm_plr001-s{seed}-f{fold}.npy'
    expected = np.load(expected_path)
    diff = float(np.max(np.abs(pred-expected)))
    with (out/'control.npy').open('xb') as stream:
        np.save(stream, pred)
    write_new(out/'control.json', {'max_absolute_difference': diff, 'metadata': metadata,
                                   'sha256': file_hash(out/'control.npy'), 'expected_sha256': file_hash(expected_path)})
    if not np.array_equal(pred, expected):
        raise ValueError('MSE control failed; no candidate fits permitted')
    print(json.dumps({'control': 'passed', 'max_absolute_difference': diff}), flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=spec['budget']['workers']) as executor:
        jobs = {executor.submit(evaluate_fold, frame, folds[s], recipe, spec['training'], f): (name, s, f)
                for name, recipe in spec['recipes'].items() for s in folds for f in range(5)}
        for future in as_completed(jobs):
            name, seed, fold = jobs[future]
            key = f'tap_time_len-{name}-s{seed}-f{fold}'
            try:
                pred, metadata = future.result()
                path = out/(key+'.npy')
                with path.open('xb') as stream:
                    np.save(stream, pred)
                event = {'event': 'complete', 'key': key, 'metadata': metadata, 'prediction_sha256': file_hash(path)}
            except Exception as exc:
                failures += 1
                event = {'event': 'failed', 'key': key, 'error': repr(exc)}
            with (out/'fit_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            print(json.dumps({k: v for k, v in event.items() if k != 'metadata'}), flush=True)
    if failures:
        raise RuntimeError(f'V10 failures retained: {failures}')
    records, y, target = [], frame.tap_time_len.to_numpy(), 'tap_time_len'
    metrics = {target: {'V7_TIME': {str(s): score_detail(y, v7[s][target], folds[s], frame.spout_no.values) for s in folds}}}
    raw_mse = {s: (v7[s][target]-(1-spec['v7_reference']['blend_weight'])*base[s][target])/spec['v7_reference']['blend_weight'] for s in folds}
    for name in spec['recipes']:
        pred = {s: np.full(len(frame), np.nan) for s in folds}
        for s in folds:
            for f in range(5):
                pred[s][folds[s] == f] = np.load(out/f'tap_time_len-{name}-s{s}-f{f}.npy')
            if not np.isfinite(pred[s]).all():
                raise ValueError('Incomplete V10 coverage')
        comparisons = {label: nested_blend(y, folds, {s: ref[s][target] for s in folds}, pred, spec['blend_grid'])
                       for label, ref in [('A35', base), ('Q20', q20), ('V7_TIME', v7)]}
        package_scores = {}
        metrics[target][name] = {}
        for s in folds:
            alpha = comparisons['V7_TIME']['alphas'][s]
            blend = (1-alpha)*v7[s][target]+alpha*pred[s]
            metrics[target][name][str(s)] = score_detail(y, blend, folds[s], frame.spout_no.values)
            package_scores[s] = package_score(wmape(frame.tap_iron.values, base[s]['tap_iron']), wmape(y, blend))
        records.append({'target': target, 'recipe': name, 'comparisons': comparisons,
                        'single_wmape': {s: wmape(y, pred[s]) for s in folds},
                        'raw_mse_single_wmape': {s: wmape(y, raw_mse[s]) for s in folds},
                        'development_package_scores': package_scores})
    summary = {'status': 'development_complete', 'records': records,
               'candidate_tiers': classify_candidates(metrics, spec, yaml.safe_load((root/'configs/candidate_tiers.yaml').read_text())),
               'selected_confirmation': select_confirmation(records, spec['tie_preference_by_target'][target]),
               'packages': 0, 'uploads': 0, 'release_authorized': False}
    write_new(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


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
