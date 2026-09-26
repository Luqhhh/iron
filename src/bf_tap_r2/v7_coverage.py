"""Exact N-0048 replay and full-training-fold refit; no packaging."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

from .data import FEATURES
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor, V36NetworkRegressor, _make_tabm, _set_seed
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import paired_cells, seed_gains, seed_level_summary, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, references, score_detail, write_new


class CoverageRegressor(V36NetworkRegressor):
    """Original optimizer loop with externally frozen legal epoch count."""

    def fit_fixed(self, frame, y, *, epochs, all_rows):
        import torch
        if self.structure != "raw_tabm" or self.params["optimizer"] != "adam":
            raise ValueError("Coverage control only supports frozen raw-TabM/Adam")
        if not 1 <= epochs <= self.params["max_epochs"]:
            raise ValueError("Epoch count outside frozen training budget")
        torch.set_num_threads(1)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all():
            raise ValueError("Invalid target")
        self.target_mean_, self.target_std_ = float(y.mean()), float(y.std())
        if self.target_std_ <= 0:
            raise ValueError("Constant target")
        _set_seed(self.params["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure=self.structure,
            n_bins=self.params["n_bins"], d_embedding=self.params["d_embedding"]).fit(frame)
        self.model_ = _make_tabm(self.preprocessor_, self.params)
        inner = group_safe_inner_folds(frame, seed=self.params["inner_validation_seed"],
                                      n_splits=self.params["inner_validation_folds"])["fold"]
        mask = np.ones(len(frame), dtype=bool) if all_rows else inner != 0
        numeric, cat = self.preprocessor_.transform_tabm(frame.loc[mask])
        x = torch.as_tensor(numeric, dtype=torch.float32)
        c = torch.as_tensor(cat, dtype=torch.long)
        z = torch.as_tensor(((y - self.target_mean_) / self.target_std_)[mask], dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.params["learning_rate"],
                                     weight_decay=self.params["weight_decay"])
        rng = np.random.default_rng(self.params["random_seed"])
        steps = 0
        for _ in range(epochs):
            self.model_.train()
            order = rng.permutation(int(mask.sum()))
            for start in range(0, len(order), self.params["batch_size"]):
                idx = order[start:start + self.params["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                prediction = self._predict_raw(self.model_, self.structure, x[idx], c[idx], train_mode=True)
                loss = self._loss_for_tabm(prediction, z[idx])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                steps += 1
        self.model_.eval()
        self.fit_meta_ = {"epochs": int(epochs), "gradient_rows": int(mask.sum()),
                          "provided_rows": len(frame), "steps": steps,
                          "gradient_ids_digest": digest(frame.loc[mask, "sample_id"].tolist()),
                          "all_rows": bool(all_rows), "parameters": deepcopy(self.params)}
        return self


def source_records(root, train, folds, spec):
    from .v3_6_run import _code_version, _hash_frame, _hash_folds
    from .v3_6_sampler import sample_v36
    trial = next(t for t in sample_v36(root) if t["trial_id"] == spec["source_trial"])
    sources = {}
    for seed in spec["split_seeds"]:
        path = root / spec["source_directory"] / f"seed-{seed}/fit_ledger.jsonl"
        matches = [json.loads(line) for line in path.read_text().splitlines()
                   if json.loads(line).get("trial_id") == spec["source_trial"]
                   and json.loads(line).get("event") == "complete"]
        if len(matches) != 1:
            raise ValueError("Source ledger must contain exactly one matching completion")
        record = matches[0]
        identity = record["identity"]
        if identity["data_hash"] != _hash_frame(train) or identity["fold_hash"] != _hash_folds(folds[seed], tuple(range(5))):
            raise ValueError("Source data/fold identity mismatch")
        if identity["code_version"].split(":")[-1] != _code_version(root).split(":")[-1]:
            raise ValueError("Frozen source implementation changed")
        if identity["fold_seed"] != seed or identity["fold_ids"] != list(range(5)):
            raise ValueError("Source fold order mismatch")
        metas = {m["outer_fold"]: m for m in record["fit_meta"]}
        if set(metas) != set(range(5)):
            raise ValueError("Incomplete epoch metadata")
        for fold, meta in metas.items():
            fit = train.loc[folds[seed] != fold]
            numeric = fit[list(FEATURES)].to_numpy(dtype=float)
            if meta["parameters"] != trial["parameters"] or meta["best_epoch"] != spec["epochs"][seed][fold]:
                raise ValueError("Source recipe/epoch mismatch")
            np.testing.assert_allclose(meta["preprocessing"]["means"], numeric.mean(0), rtol=0, atol=1e-12)
            np.testing.assert_allclose(meta["preprocessing"]["stds"], numeric.std(0), rtol=0, atol=1e-12)
            np.testing.assert_allclose([meta["target_mean"], meta["target_std"]],
                [fit[spec["target"]].mean(), fit[spec["target"]].std(ddof=0)], rtol=0, atol=1e-12)
        sources[seed] = {"ledger_sha256": file_hash(path), "identity": identity}
    return trial, sources


def _job(train, folds, trial, fold, epochs, all_rows):
    target = trial["target"]
    training = train.loc[folds != fold].reset_index(drop=True)
    model = CoverageRegressor(trial).fit_fixed(training, training[target].values,
                                              epochs=epochs, all_rows=all_rows)
    return model.predict(train.loc[folds == fold]), model.fit_meta_


def run(root, output):
    root = Path(root).resolve()
    spec_path = root / "configs/round2_v7/COVERAGE_SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    runtime = yaml.safe_load((root / "configs/round2_v7/SPEC.yaml").read_text())["runtime_versions"]
    import importlib.metadata
    versions = {k: importlib.metadata.version(k) for k in runtime}
    if versions != runtime:
        raise ValueError("Frozen neural runtime mismatch")
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v7-coverage"):
        raise ValueError("Coverage output must remain private")
    out.mkdir(parents=True, exist_ok=False)
    train = load_v5_training_frame(root)
    folds = {s: fold_vector(root, train, s, load_v5_spec(root)) for s in spec["split_seeds"]}
    trial, sources = source_records(root, train, folds, spec)
    write_new(out / "manifest.json", {"spec_sha256": file_hash(spec_path), "code_sha256": file_hash(__file__),
              "versions": versions, "sources": sources, "trial": trial, "start_time": time.time()})
    seed, fold = spec["control"]["seed"], spec["control"]["fold"]
    control, meta = _job(train, folds[seed], trial, fold, spec["epochs"][seed][fold], False)
    old = np.load(root / spec["source_directory"] / f"seed-{seed}/pred-{spec['source_trial']}.npy")
    delta = float(np.max(np.abs(control - old[folds[seed] == fold])))
    write_new(out / "control.json", {"seed": seed, "fold": fold, "max_abs_diff": delta, "metadata": meta})
    np.save(out / "control-prediction.npy", control)
    print(json.dumps({"control_max_abs_diff": delta}), flush=True)
    if delta > spec["control"]["prediction_max_abs_tolerance"]:
        raise ValueError("Real-data control did not reproduce; no candidate fits allowed")
    failures = 0
    with ProcessPoolExecutor(max_workers=spec["budget"]["workers"]) as executor:
        jobs = {executor.submit(_job, train, folds[s], trial, f, spec["epochs"][s][f], True): (s, f)
                for s in folds for f in range(5)}
        for future in as_completed(jobs):
            s, f = jobs[future]
            try:
                pred, meta = future.result()
                with (out / f"seed-{s}-fold-{f}.npy").open("xb") as handle:
                    np.save(handle, pred)
                event = {"event": "complete", "seed": s, "fold": f, "metadata": meta}
            except Exception as exc:
                failures += 1
                event = {"event": "failed", "seed": s, "fold": f, "error": repr(exc)}
            with (out / "fit_ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
    if failures:
        raise RuntimeError(f"Failed fits retained: {failures}")
    predictions = {s: np.empty(len(train)) for s in folds}
    original = {}
    for s in folds:
        for f in range(5):
            predictions[s][folds[s] == f] = np.load(out / f"seed-{s}-fold-{f}.npy")
        original[s] = np.load(root / spec["source_directory"] / f"seed-{s}/pred-{spec['source_trial']}.npy")
    refspec = {"split_seeds": list(folds), "reference_time_alpha": spec["endpoint_alpha"],
               "secondary_reference_time_alpha": spec["secondary_endpoint_alpha"]}
    base, secondary = references(root, train, refspec)
    target = spec["target"]
    y = train[target].values
    metrics = {target: {"A35_TIME": {}, "N0048_FULL_REFIT_A35": {}}}
    results = {}
    for name, alpha, reference in [("A35", spec["endpoint_alpha"], base),
                                    ("Q20", spec["secondary_endpoint_alpha"], secondary)]:
        before = {s: reference[s][target] for s in folds}
        after = {s: before[s] + alpha * (predictions[s] - original[s]) for s in folds}
        cells = paired_cells(y, folds, before, after)
        gains = seed_gains(cells)
        results[name] = {"cells": [c.as_dict() for c in cells], "seed_gains": gains,
                         "seed_summary": seed_level_summary(gains),
                         "pooled_seed_gains": {s: 50 * (wmape(y, before[s]) - wmape(y, after[s])) for s in folds}}
        if name == "A35":
            for s in folds:
                metrics[target]["A35_TIME"][str(s)] = score_detail(y, before[s], folds[s], train.spout_no.values)
                metrics[target]["N0048_FULL_REFIT_A35"][str(s)] = score_detail(y, after[s], folds[s], train.spout_no.values)
    from .candidate_tiers import classify_candidates
    tiers = classify_candidates(metrics, spec, yaml.safe_load((root / "configs/candidate_tiers.yaml").read_text()))
    write_new(out / "summary.json", {"results": results, "candidate_tiers": tiers,
              "confirmation_eligible": all(v > 0 for v in results["A35"]["pooled_seed_gains"].values()),
              "promotion": False, "packages": 0, "uploads": 0})
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1")
    run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
