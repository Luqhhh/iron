"""From-scratch joint-output TabM with complete-coverage isolated-column scoring."""
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
from .v11_quantile import load_references, choose_confirmation


def joint_loss(prediction, target):
    if prediction.ndim != 3 or target.shape != (len(prediction), prediction.shape[2]):
        raise ValueError('Expected row/head/output predictions and row/output targets')
    return (prediction-target[:, None, :]).square().mean()


def make_network(recipe, settings, n_categories, n_outputs):
    import torch
    from rtdl_num_embeddings import PeriodicEmbeddings

    embedding = None
    if recipe["frequency"] is not None:
        embedding = PeriodicEmbeddings(
            len(FEATURES), d_embedding=settings["embedding_dim"],
            n_frequencies=settings["n_frequencies"],
            frequency_init_scale=recipe["frequency"], lite=settings["lite"],
        )
    if recipe["backbone"] == "tabm":
        from tabm import TabM
        return TabM.make(
            n_num_features=len(FEATURES), cat_cardinalities=[n_categories],
            d_out=n_outputs, k=settings["tabm_k"], n_blocks=settings["blocks"],
            d_block=settings["width"], dropout=settings["dropout"],
            num_embeddings=embedding,
        )

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = embedding
            width = len(FEATURES) * (settings["embedding_dim"] if embedding else 1) + n_categories
            layers = []
            for _ in range(settings["blocks"]):
                layers += [torch.nn.Linear(width, settings["width"]), torch.nn.ReLU(),
                           torch.nn.Dropout(settings["dropout"])]
                width = settings["width"]
            layers.append(torch.nn.Linear(width, n_outputs))
            self.layers = torch.nn.Sequential(*layers)

        def forward(self, x_num, x_cat):
            numeric = x_num if self.embedding is None else self.embedding(x_num).flatten(1)
            categorical = torch.nn.functional.one_hot(x_cat[:, 0], n_categories).float()
            return self.layers(torch.cat([numeric, categorical], dim=1)).unsqueeze(1)

    return MLP()


class JointRegressor:
    def __init__(self, recipe, settings):
        self.recipe, self.settings = dict(recipe), dict(settings)

    def _initialize(self, frame, y):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_tabm").fit(frame)
        self.mean_, self.std_ = np.mean(y, axis=0), np.std(y, axis=0)
        if not (self.std_ > 0).all():
            raise ValueError("Constant target")
        self.model_ = make_network(self.recipe, self.settings, self.preprocessor_.n_spout_categories_, y.shape[1])
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
                                           lr=self.settings["learning_rate"],
                                           weight_decay=self.settings["weight_decay"])

    def _inputs(self, frame):
        import torch
        numeric, cat = self.preprocessor_.transform_tabm(frame)
        return torch.as_tensor(numeric), torch.as_tensor(cat, dtype=torch.long)

    def _train(self, frame, y, epochs, validation=None):
        import torch
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y - self.mean_) / self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        best, best_epoch, stale = float("inf"), 0, 0
        if validation is not None:
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1] - self.mean_) / self.std_, dtype=torch.float32)
        for epoch in range(1, epochs + 1):
            self.model_.train()
            order = rng.permutation(len(frame))
            for start in range(0, len(order), self.settings["batch_size"]):
                idx = order[start:start + self.settings["batch_size"]]
                self.optimizer_.zero_grad(set_to_none=True)
                pred = self.model_(x[idx], cat[idx])
                loss = joint_loss(pred, target[idx])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                self.optimizer_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    pred = self.model_(vx, vc).mean(1)
                    value = float((pred - vy).abs().mean())
                if value < best - self.settings["min_delta"]:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
                if stale >= self.settings["patience"]:
                    break
        if validation is not None:
            self.selection_stopped_epoch_ = epoch
        return best_epoch if validation is not None else epochs

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or y.shape[0] != len(frame) or y.shape[1] not in (1, 2) or not np.isfinite(y).all():
            raise ValueError("Invalid training targets")
        folds = group_safe_inner_folds(frame, seed=self.settings["inner_seed"])["fold"]
        mask = folds != 0
        inner = frame.loc[mask].reset_index(drop=True)
        self._initialize(inner, y[mask])
        inner_means = self.preprocessor_.means_.copy()
        inner_target_mean, inner_target_std = self.mean_.copy(), self.std_.copy()
        epoch = self._train(inner, y[mask], self.settings["max_epochs"],
                            (frame.loc[~mask], y[~mask]))
        if epoch < 1:
            raise ValueError("No finite inner-validation epoch")
        self._initialize(frame, y)
        self._train(frame, y, epoch)
        self.model_.eval()
        self.metadata_ = {"selected_epoch": epoch,
                          "selection_stopped_epoch": self.selection_stopped_epoch_,
                          "budget_limited": self.selection_stopped_epoch_ >= self.settings["max_epochs"],
                          "n_outputs": y.shape[1],
                          "inner_target_mean": inner_target_mean.tolist(),
                          "inner_target_std": inner_target_std.tolist(),
                          "outer_target_mean": self.mean_.tolist(),
                          "outer_target_std": self.std_.tolist(), "fit_rows": len(frame),
                          "inner_fit_rows": int(mask.sum()), "optimizer_runs": 2,
                          "inner_feature_means": inner_means.tolist(),
                          "outer_feature_means": self.preprocessor_.means_.tolist(),
                          "fit_ids_digest": digest(frame.sample_id.tolist()),
                          "inner_ids_digest": digest(inner.sample_id.tolist())}
        return self

    def predict(self, frame):
        import torch
        self.model_.eval()
        x, cat = self._inputs(frame)
        with torch.no_grad():
            pred = self.model_(x, cat).mean(1).numpy().astype(float)
        result = pred * self.std_ + self.mean_
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prediction")
        return result


def evaluate_fold(frame, folds, recipe, settings, fold, targets=TARGETS):
    training = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = JointRegressor(recipe, settings).fit(training, training[list(targets)].to_numpy())
    return model.predict(query), model.metadata_


def run(root, output):
    root = Path(root).resolve(); spec_path = root/'configs/round2_v12/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != spec['runtime_versions']:
        raise ValueError('V12 runtime mismatch')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v12-joint-tabm'):
        raise ValueError('Private V12 output required')
    out.mkdir(parents=True, exist_ok=False)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec['split_seeds']}
    base, q20, current, controls, hashes = load_references(root, frame, folds, spec)
    write_new(out/'manifest.json', {'spec_sha256': file_hash(spec_path), 'versions': versions,
        'data_digest': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'fold_digests': {s: digest(f.tolist()) for s, f in folds.items()}, 'reference_hashes': hashes,
        'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    c = spec['control']; seed, fold = c['split_seed'], c['fold']
    pred, metadata = evaluate_fold(frame, folds[seed],
        {'backbone': 'tabm', 'frequency': .01}, spec['training'], fold, (c['target'],))
    pred = pred[:, 0]
    expected = controls[c['target']][c['recipe']][seed][folds[seed] == fold]
    with (out/'control.npy').open('xb') as stream:
        np.save(stream, pred)
    diff = float(np.max(np.abs(pred-expected)))
    write_new(out/'control.json', {'metadata': metadata, 'sha256': file_hash(out/'control.npy'), 'max_absolute_difference': diff})
    if not np.array_equal(pred, expected):
        raise ValueError('Standard-coordinate control failed; candidates not fitted')
    print(json.dumps({'control': 'passed', 'max_absolute_difference': diff}), flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=spec['budget']['workers']) as executor:
        jobs = {executor.submit(evaluate_fold, frame, folds[s], recipe, spec['training'], f): (name, s, f)
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
        raise RuntimeError(f'V12 failed fits retained: {failures}')
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
                    raise ValueError('Incomplete V12 OOF')
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
            control = 'tabm_raw' if recipe['frequency'] is None else 'tabm_plr001'
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
