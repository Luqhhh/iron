"""V7 from-scratch periodic networks; no release or external model downloads."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, TARGETS
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_column_reference, load_v5_training_frame
from .v5_resolution import nested_blend, wmape
from .v5_spec import load_v5_spec


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def make_network(recipe, settings, n_categories):
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
            d_out=1, k=settings["tabm_k"], n_blocks=settings["blocks"],
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
            layers.append(torch.nn.Linear(width, 1))
            self.layers = torch.nn.Sequential(*layers)

        def forward(self, x_num, x_cat):
            numeric = x_num if self.embedding is None else self.embedding(x_num).flatten(1)
            categorical = torch.nn.functional.one_hot(x_cat[:, 0], n_categories).float()
            return self.layers(torch.cat([numeric, categorical], dim=1)).unsqueeze(1)

    return MLP()


class PeriodicRegressor:
    def __init__(self, recipe, settings):
        self.recipe, self.settings = dict(recipe), dict(settings)

    def _initialize(self, frame, y):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_tabm").fit(frame)
        self.mean_, self.std_ = float(np.mean(y)), float(np.std(y))
        if not self.std_ > 0:
            raise ValueError("Constant target")
        self.model_ = make_network(self.recipe, self.settings, self.preprocessor_.n_spout_categories_)
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
                pred = self.model_(x[idx], cat[idx])[:, :, 0]
                loss = ((pred - target[idx, None]) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                self.optimizer_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    pred = self.model_(vx, vc).mean(1).flatten()
                    value = float((pred - vy).abs().mean())
                if value < best - self.settings["min_delta"]:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
                if stale >= self.settings["patience"]:
                    break
        return best_epoch if validation is not None else epochs

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all():
            raise ValueError("Invalid training targets")
        folds = group_safe_inner_folds(frame, seed=self.settings["inner_seed"])["fold"]
        mask = folds != 0
        inner = frame.loc[mask].reset_index(drop=True)
        self._initialize(inner, y[mask])
        inner_means = self.preprocessor_.means_.copy()
        epoch = self._train(inner, y[mask], self.settings["max_epochs"],
                            (frame.loc[~mask], y[~mask]))
        if epoch < 1:
            raise ValueError("No finite inner-validation epoch")
        self._initialize(frame, y)
        self._train(frame, y, epoch)
        self.model_.eval()
        self.metadata_ = {"selected_epoch": epoch, "fit_rows": len(frame),
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
            pred = self.model_(x, cat).mean(1).flatten().numpy().astype(float)
        result = pred * self.std_ + self.mean_
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prediction")
        return result


def evaluate_fold(frame, folds, target, recipe, settings, fold):
    mask = folds == fold
    model = PeriodicRegressor(recipe, settings).fit(
        frame.loc[~mask].reset_index(drop=True), frame.loc[~mask, target].to_numpy())
    prediction = model.predict(frame.loc[mask])
    return prediction, model.metadata_


def _fit_job(payload):
    frame, folds, target, recipe, settings, fold = payload
    return evaluate_fold(frame, folds, target, recipe, settings, fold)


def references(root, train, spec):
    legacy = load_v5_spec(root)
    ref = load_column_reference(root, train, legacy)
    base, secondary = {}, {}
    for seed in spec["split_seeds"]:
        n = np.load(root / "local/runs/round2-v5-error-covariance/time-n-family-r1" /
                    f"seed-{seed}/pred-v36-s1-N-0048.npy")
        if n.shape != (len(train),) or not np.isfinite(n).all():
            raise ValueError("Incomplete N-0048 reference")
        base[seed], secondary[seed] = {}, {}
        for target in TARGETS:
            v36 = ref.base_for(target, seed)
            a = spec["reference_time_alpha"] if target == "tap_time_len" else 0
            b = spec["secondary_reference_time_alpha"] if target == "tap_time_len" else 0
            base[seed][target] = (1 - a) * v36 + a * n
            secondary[seed][target] = (1 - b) * v36 + b * n
    return base, secondary


def score_detail(y, prediction, folds, spout):
    return {"wmape": wmape(y, prediction),
            "by_fold": {str(f): wmape(y[folds == f], prediction[folds == f]) for f in range(5)},
            "by_spout": {str(s): wmape(y[spout == s], prediction[spout == s]) for s in np.unique(spout)}}


def summarize(root, out, train, spec, folds):
    from .candidate_tiers import classify_candidates
    base, secondary = references(root, train, spec)
    rows, metrics = [], {}
    for target in TARGETS:
        y = train[target].to_numpy()
        refs = {s: base[s][target] for s in spec["split_seeds"]}
        q20 = {s: secondary[s][target] for s in spec["split_seeds"]}
        metrics[target] = {spec["reference_by_target"][target]: {
            str(s): score_detail(y, refs[s], folds[s], train.spout_no.to_numpy()) for s in refs}}
        for name in spec["candidates"][target]:
            pred = {s: np.full(len(train), np.nan) for s in refs}
            for s in refs:
                for f in range(5):
                    key = f"{target}-{name}-s{s}-f{f}"
                    pred[s][folds[s] == f] = np.load(out / f"{key}.npy")
                if not np.isfinite(pred[s]).all():
                    raise ValueError("Incomplete development coverage")
            result = nested_blend(y, folds, refs, pred, spec["blend_grid"])
            blend = {s: (1 - result["alphas"][s]) * refs[s] + result["alphas"][s] * pred[s] for s in refs}
            metrics[target][name] = {str(s): score_detail(y, blend[s], folds[s], train.spout_no.to_numpy()) for s in refs}
            rows.append({"target": target, "recipe": name, "a35_nested": result,
                         "q20_nested": nested_blend(y, folds, q20, pred, spec["blend_grid"]),
                         "single_wmape": {s: wmape(y, pred[s]) for s in refs},
                         "confirmation_eligible": all(g > 0 for g in result["seed_gains"].values())})
    policy = yaml.safe_load((root / "configs/candidate_tiers.yaml").read_text())
    tiers = classify_candidates(metrics, spec, policy)
    summary = {"status": "development_complete", "rows": rows, "candidate_tiers": tiers,
               "reference_role": "recorded_development_replay_not_fresh_confirmation",
               "promotion": False, "packages": 0, "uploads": 0}
    write_new(out / "summary.json", summary)
    return summary


def run(root, output, workers):
    root = Path(root).resolve()
    spec_path = root / "configs/round2_v7/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec["runtime_versions"]}
    if versions != spec["runtime_versions"]:
        raise ValueError(f"V7 frozen runtime mismatch: {versions}")
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v7-periodic-networks"):
        raise ValueError("V7 outputs must remain private")
    out.mkdir(parents=True, exist_ok=False)
    train = load_v5_training_frame(root)
    legacy = load_v5_spec(root)
    folds = {s: fold_vector(root, train, s, legacy) for s in spec["split_seeds"]}
    base, secondary = references(root, train, spec)
    identity = {"spec_sha256": file_hash(spec_path),
                "code_sha256": file_hash(__file__),
                "data_digest": hashlib.sha256(pd.util.hash_pandas_object(train, index=True).values.tobytes()).hexdigest(),
                "fold_digests": {s: digest(folds[s].tolist()) for s in folds},
                "reference_digests": {s: {t: digest(base[s][t].tolist()) for t in TARGETS} for s in base},
                "versions": versions,
                "dependency_code_hashes": {p.name: file_hash(p) for p in
                                          sorted((root / "src/bf_tap_r2").glob("*.py"))},
                "workers": workers, "start_time": time.time()}
    write_new(out / "manifest.json", identity)
    ledger = out / "fit_ledger.jsonl"
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {}
        for target in TARGETS:
            for name, recipe in spec["recipes"].items():
                for seed in spec["split_seeds"]:
                    for fold in range(5):
                        key = f"{target}-{name}-s{seed}-f{fold}"
                        future = executor.submit(_fit_job, (train, folds[seed], target, recipe, spec["training"], fold))
                        jobs[future] = key
        for future in as_completed(jobs):
            key = jobs[future]
            try:
                prediction, metadata = future.result()
                with (out / f"{key}.npy").open("xb") as stream:
                    np.save(stream, prediction)
                event = {"event": "complete", "key": key, "metadata": metadata,
                         "prediction_sha256": file_hash(out / f"{key}.npy")}
            except Exception as exc:
                failures += 1
                event = {"event": "failed", "key": key, "error": repr(exc)}
            with ledger.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in event.items() if k != "metadata"}), flush=True)
    if failures:
        write_new(out / "failure.json", {"failed_fits": failures})
        raise RuntimeError(f"{failures} failed fits retained; no classification")
    result = summarize(root, out, train, spec, folds)
    for row in result["rows"]:
        print(json.dumps({"target": row["target"], "recipe": row["recipe"],
                          "gains": row["a35_nested"]["seed_gains"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1 before launch")
    run(Path.cwd(), args.output, args.workers)


if __name__ == "__main__":
    main()
