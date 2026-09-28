"""V20: masked feature reconstruction as an auxiliary training task (time target only).

Preregistration: docs/round2_v20/PREREGISTRATION.md; spec: configs/round2_v20/SPEC.yaml.

The control arm and the candidate arm run through THIS module and differ only in
``mask_rate``/``auxiliary_weight``. The masked forward draws from a dedicated
``torch.Generator``, so the main RNG stream (initialisation, dropout, batch order) is
identical in both arms -- that is what makes the control arm an identity replay of the
cached V7 ``tabm_plr001`` predictions (gate 0b).

The frozen ``v7_periodic`` module is imported, never modified: its network factory is
the control architecture, and its ``code_sha256`` stays a valid frozen identity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from .v7_periodic import digest, file_hash, make_network, write_new

TARGET = "tap_time_len"
TIME_ALPHA = 0.35          # V36 time -> A35 time, as in V7's reference
N0048 = "local/runs/round2-v5-error-covariance/time-n-family-r1/seed-{seed}/pred-v36-s1-N-0048.npy"
V12_IRON = "local/runs/round2-v12-joint-tabm/development-r1/joint-joint_plr001-s{seed}-f{fold}.npy"
V7_TIME = "local/runs/round2-v7-periodic-networks/development-r1/tap_time_len-tabm_plr001-s{seed}-f{fold}.npy"


def make_network_with_decoder(recipe, settings, n_categories, n_features):
    """TabM plus a reconstruction decoder reading the shared backbone.

    TabM only: the decoder must read the shared backbone output, and the MLP branch of
    ``make_network`` has no separable backbone. The V20 recipe is fixed to ``tabm_plr001``.
    """
    import torch
    from tabm import LinearEnsemble

    if recipe["backbone"] != "tabm":
        raise ValueError("V20 requires the tabm backbone")
    base = make_network(recipe, settings, n_categories)
    # Building the decoder consumes global RNG draws (its parameters are initialised).
    # Left alone, that shifts the dropout stream for the whole run relative to the cached
    # V7 fit, and the control arm no longer replays it (gate 0b). Save and restore so the
    # global stream resumes exactly where the base network left it: the decoder is the
    # only difference between the two arms, and nothing else moves.
    rng_state = torch.get_rng_state()

    class MaskedReconNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.decoder = LinearEnsemble(settings["width"], n_features, k=settings["tabm_k"])

        def _hidden(self, x_num, x_cat):
            stem = []
            num_module = getattr(self.base, "num_module", None)
            if x_num is not None:
                stem.append(x_num if num_module is None else num_module(x_num))
            if x_cat is not None:
                stem.append(self.base.cat_module(x_cat))
            x = torch.column_stack([part.flatten(1, -1) for part in stem])
            return self.base.backbone(self.base.ensemble_view(x))

        def forward(self, x_num, x_cat):
            return self.base.output(self._hidden(x_num, x_cat))

        def reconstruct(self, x_num, x_cat):
            return self.decoder(self._hidden(x_num, x_cat))

    network = MaskedReconNet()
    torch.set_rng_state(rng_state)
    return network


class MaskedReconRegressor:
    """Same fitting contract as v7_periodic.PeriodicRegressor, plus the auxiliary task."""

    def __init__(self, recipe, settings, mask_rate, auxiliary_weight):
        self.recipe, self.settings = dict(recipe), dict(settings)
        self.mask_rate, self.auxiliary_weight = float(mask_rate), float(auxiliary_weight)

    def _initialize(self, frame, y):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_tabm").fit(frame)
        self.mean_, self.std_ = float(np.mean(y)), float(np.std(y))
        if not self.std_ > 0:
            raise ValueError("Constant target")
        # Base network is built (and thus initialised) before the decoder, so its
        # parameters are drawn from the same stream in both arms.
        self.model_ = make_network_with_decoder(
            self.recipe, self.settings, self.preprocessor_.n_spout_categories_, len(FEATURES))
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
                                            lr=self.settings["learning_rate"],
                                            weight_decay=self.settings["weight_decay"])
        self.mask_generator_ = torch.Generator().manual_seed(self.settings["random_seed"] + 1)

    def _inputs(self, frame):
        import torch
        numeric, cat = self.preprocessor_.transform_tabm(frame)
        return torch.as_tensor(numeric), torch.as_tensor(cat, dtype=torch.long)

    def _auxiliary_loss(self, x, cat):
        import torch
        mask = torch.rand(x.shape, generator=self.mask_generator_) < self.mask_rate
        if not bool(mask.any()):
            return None
        masked = x.clone()
        masked[mask] = 0.0
        reconstruction = self.model_.reconstruct(masked, cat)
        wide = mask[:, None, :].expand_as(reconstruction)
        residual = reconstruction[wide] - x[:, None, :].expand_as(reconstruction)[wide]
        return (residual ** 2).mean()

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
                if self.auxiliary_weight > 0 and self.mask_rate > 0:
                    auxiliary = self._auxiliary_loss(x[idx], cat[idx])
                    if auxiliary is not None:
                        loss = loss + self.auxiliary_weight * auxiliary
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
                          "mask_rate": self.mask_rate, "auxiliary_weight": self.auxiliary_weight,
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


def evaluate_fold(frame, folds, recipe, settings, fold, mask_rate, auxiliary_weight):
    mask = folds == fold
    model = MaskedReconRegressor(recipe, settings, mask_rate, auxiliary_weight).fit(
        frame.loc[~mask].reset_index(drop=True), frame.loc[~mask, TARGET].to_numpy())
    return model.predict(frame.loc[mask]), model.metadata_


def _fit_job(payload):
    return evaluate_fold(*payload)


def stitch(frame, folds, arrays):
    """Per-fold files hold fold f's rows in ascending train-frame row index order.

    ``folds-*.csv`` is a sample_id lookup, not the row order -- reading it literally is
    the documented way to manufacture a fake 0.23 WMAPE.
    """
    out = np.full(len(frame), np.nan)
    position = {s: i for i, s in enumerate(frame["sample_id"])}
    for fold, array in enumerate(arrays):
        rows = folds == fold
        if array.shape[0] != int(rows.sum()):
            raise ValueError("Fold coverage mismatch")
        out[np.sort(np.array([position[s] for s in frame["sample_id"][rows]]))] = array
    if not np.isfinite(out).all():
        raise ValueError("Incomplete coverage")
    return out


def reassemble(root, frame, folds, pattern, seed, column=None):
    arrays = []
    for fold in range(5):
        array = np.load(root / pattern.format(seed=seed, fold=fold))
        arrays.append(array if column is None else array[:, column])
    return stitch(frame, folds, arrays)


def reassemble_run(out, frame, folds, seed, arm):
    return stitch(frame, folds, [np.load(out / f"{arm}-s{seed}-f{fold}.npy") for fold in range(5)])


def wm(prediction, truth):
    return float(np.abs(truth - prediction).sum() / np.abs(truth).sum())


def score(iron_wmape, time_wmape):
    return 100 - 50 * (iron_wmape + time_wmape)


def build_incumbent(root, frame, folds, seed):
    """V12 iron column + V7 time column -- the delivered incumbent (R2)."""
    legacy = load_v5_spec(root)
    reference = load_column_reference(root, frame, legacy)
    n0048 = np.load(root / N0048.format(seed=seed))
    a35_time = (1 - TIME_ALPHA) * reference.base_for(TARGET, seed) + TIME_ALPHA * n0048
    v12 = reassemble(root, frame, folds, V12_IRON, seed, column=0)
    v7 = reassemble(root, frame, folds, V7_TIME, seed)
    iron = 0.5 * reference.base_for("tap_iron", seed) + 0.5 * v12
    return iron, 0.5 * a35_time + 0.5 * v7, a35_time, v7


def synthetic_check():
    """Gate 0a: can the decoder learn inter-feature structure at all?

    Data are 21 features driven by a rank-3 latent plus small noise, so a masked entry is
    predictable from the others. The bar is the per-feature constant (column-mean)
    predictor, evaluated on the SAME held-out masked entries.
    """
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    settings = {"random_seed": 0, "inner_seed": 0, "width": 64, "blocks": 2, "tabm_k": 8,
                "dropout": 0.0, "embedding_dim": 8, "n_frequencies": 8, "lite": True,
                "loss": "mse", "optimizer": "adamw", "learning_rate": 0.001,
                "weight_decay": 0.0001, "batch_size": 128, "max_epochs": 60, "patience": 25,
                "min_delta": 1e-5}
    generator = np.random.default_rng(0)
    latent = generator.normal(size=(1536, 3))
    numeric = latent @ generator.normal(size=(3, 21)) + 0.05 * generator.normal(size=(1536, 21))
    train = torch.as_tensor(numeric[:1024], dtype=torch.float32)
    test = torch.as_tensor(numeric[1024:], dtype=torch.float32)
    cat_train = torch.zeros(len(train), 1, dtype=torch.long)
    cat_test = torch.zeros(len(test), 1, dtype=torch.long)

    network = make_network_with_decoder({"backbone": "tabm", "frequency": 0.01}, settings,
                                        n_categories=2, n_features=21)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    mask_generator = torch.Generator().manual_seed(1)
    final = None
    for _ in range(300):
        mask = torch.rand(train.shape, generator=mask_generator) < 0.15
        masked = train.clone()
        masked[mask] = 0.0
        reconstruction = network.reconstruct(masked, cat_train)
        wide = mask[:, None, :].expand_as(reconstruction)
        residual = reconstruction[wide] - train[:, None, :].expand_as(reconstruction)[wide]
        loss = (residual ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())

    mask = torch.rand(test.shape, generator=mask_generator) < 0.15
    masked = test.clone()
    masked[mask] = 0.0
    with torch.no_grad():
        reconstruction = network.reconstruct(masked, cat_test)
    wide = mask[:, None, :].expand_as(reconstruction)
    learned = float(((reconstruction[wide] - test[:, None, :].expand_as(reconstruction)[wide]) ** 2).mean())
    rows, columns = mask.nonzero(as_tuple=True)
    constant = float(((test[rows, columns] - train[:, columns].mean(0)) ** 2).mean())
    report = {"gate_0a": "synthetic_learnability", "learned_mse": learned,
              "constant_baseline_mse": constant, "ratio": learned / constant,
              "last_train_loss": final, "passed": learned < constant}
    print(json.dumps(report, indent=2))
    return learned < constant


def run(root, output, workers):
    root = Path(root).resolve()
    spec_path = root / "configs/round2_v20/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec["runtime_versions"]}
    if versions != spec["runtime_versions"]:
        raise ValueError(f"V20 frozen runtime mismatch: {versions}")
    v7_spec = yaml.safe_load((root / spec["training_source"]).read_text())
    recipe = v7_spec["recipes"][spec["training_source_recipe"]]
    settings = v7_spec["training"]

    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v20-masked-recon"):
        raise ValueError("V20 outputs must remain private")
    out.mkdir(parents=True, exist_ok=False)
    frame = load_v5_training_frame(root)
    legacy = load_v5_spec(root)
    folds = {s: fold_vector(root, frame, s, legacy) for s in spec["split_seeds"]}
    y = frame[TARGET].to_numpy()

    identity = {"spec_sha256": file_hash(spec_path), "code_sha256": file_hash(__file__),
                "v7_spec_sha256": file_hash(root / spec["training_source"]),
                "v7_code_sha256": file_hash(Path(__file__).with_name("v7_periodic.py")),
                "recipe": recipe, "arms": spec["arms"],
                "fold_digests": {s: digest(folds[s].tolist()) for s in folds},
                "versions": versions, "workers": workers, "start_time": time.time()}
    write_new(out / "manifest.json", identity)
    ledger = out / "fit_ledger.jsonl"
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {}
        for arm, arm_spec in spec["arms"].items():
            for seed in spec["split_seeds"]:
                for fold in range(5):
                    key = f"{arm}-s{seed}-f{fold}"
                    jobs[executor.submit(_fit_job, (frame, folds[seed], recipe, settings, fold,
                                                    arm_spec["mask_rate"],
                                                    arm_spec["auxiliary_weight"]))] = key
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

    summary = {"status": "development_complete", "arms": {}, "gates": {}}
    per_arm = {arm: {seed: reassemble_run(out, frame, folds[seed], seed, arm)
                     for seed in spec["split_seeds"]} for arm in spec["arms"]}

    gate0b = {}
    for seed in spec["split_seeds"]:
        cached = reassemble(root, frame, folds[seed], V7_TIME, seed)
        delta = float(np.abs(per_arm["control"][seed] - cached).max())
        gate0b[str(seed)] = {"max_abs_delta_vs_cached_v7": delta, "passed": delta == 0.0}
    summary["gates"]["gate_0b_control_replay_identity"] = gate0b

    rows = []
    for seed in spec["split_seeds"]:
        iron, incumbent_time, a35_time, _ = build_incumbent(root, frame, folds[seed], seed)
        incumbent = score(wm(iron, frame["tap_iron"].to_numpy()), wm(incumbent_time, y))
        row = {"seed": seed, "incumbent_local": incumbent}
        for arm in spec["arms"]:
            delivered = 0.5 * a35_time + 0.5 * per_arm[arm][seed]
            row[arm] = score(wm(iron, frame["tap_iron"].to_numpy()), wm(delivered, y)) - incumbent
            row[f"{arm}_single_wmape"] = wm(per_arm[arm][seed], y)
        rows.append(row)
    summary["rows"] = rows
    summary["arms"] = {arm: {"two_seed_mean_increment": float(np.mean([r[arm] for r in rows]))}
                       for arm in spec["arms"]}
    summary["gates"]["gate_1_two_complete_development_seeds_positive"] = all(
        r["candidate"] > 0 for r in rows)
    summary["gates"]["gate_2_increment_over_incumbent_0.01"] = bool(
        np.mean([r["candidate"] for r in rows]) >= 0.01)
    summary["reference_role"] = "local_only_not_a_platform_forecast"
    summary["promotion"] = False
    summary["packages"] = 0
    summary["uploads"] = 0
    write_new(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        raise SystemExit(0 if synthetic_check() else 1)
    if args.output is None:
        parser.error("--output is required unless --synthetic")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1 before launch")
    run(Path.cwd(), args.output, args.workers)


if __name__ == "__main__":
    main()
