"""Joint TabM training with an explicit learning-rate schedule and loss choice.

The incumbent protocol (inner-fold epoch selection followed by a fresh refit at
the selected epoch) is preserved.  Only the per-epoch learning-rate multiplier
and the training loss differ.  The constant-schedule MSE configuration reuses
the parent implementation unchanged so it reproduces recorded predictions
bit-exactly.
"""
from __future__ import annotations

import math

import numpy as np

from .v12_joint import JointRegressor, joint_loss

SCHEDULES = ("constant", "cosine")
LOSSES = ("mse", "mae")


class ScheduledJointRegressor(JointRegressor):
    """JointRegressor with a cosine schedule, optional warmup and L1 loss."""

    def _lr_factor(self, epoch: int, total: int) -> float:
        schedule = self.settings.get("lr_schedule", "constant")
        if schedule == "constant":
            return 1.0
        if schedule != "cosine":
            raise ValueError(f"Unsupported schedule: {schedule}")
        warmup = int(self.settings.get("warmup_epochs", 0))
        if warmup and epoch <= warmup:
            return epoch / (warmup + 1)
        progress = (epoch - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _train(self, frame, y, epochs, validation=None):
        if (self.settings.get("lr_schedule", "constant") == "constant"
                and self.settings.get("loss", "mse") == "mse"):
            return super()._train(frame, y, epochs, validation)
        import torch

        loss_name = self.settings.get("loss", "mse")
        if loss_name not in LOSSES:
            raise ValueError(f"Unsupported loss: {loss_name}")
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y - self.mean_) / self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        base_lr = float(self.settings["learning_rate"])
        best, best_epoch, stale = float("inf"), 0, 0
        if validation is not None:
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1] - self.mean_) / self.std_, dtype=torch.float32)
        for epoch in range(1, epochs + 1):
            factor = self._lr_factor(epoch, epochs)
            for group in self.optimizer_.param_groups:
                group["lr"] = base_lr * factor
            self.model_.train()
            order = rng.permutation(len(frame))
            for start in range(0, len(order), self.settings["batch_size"]):
                idx = order[start:start + self.settings["batch_size"]]
                self.optimizer_.zero_grad(set_to_none=True)
                pred = self.model_(x[idx], cat[idx])
                if loss_name == "mse":
                    loss = joint_loss(pred, target[idx])
                else:
                    loss = (pred - target[idx][:, None, :]).abs().mean()
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


CONFIGS = {
    "CONTROL": {},
    "COS": {"lr_schedule": "cosine"},
    "COS_LR3": {"lr_schedule": "cosine", "learning_rate": 0.003},
    "COS_LR03": {"lr_schedule": "cosine", "learning_rate": 0.0003},
    "COS_WARM": {"lr_schedule": "cosine", "warmup_epochs": 10},
    "COS_DROP0": {"lr_schedule": "cosine", "dropout": 0.0},
    "COS_DROP2": {"lr_schedule": "cosine", "dropout": 0.2},
    "COS_WD0": {"lr_schedule": "cosine", "weight_decay": 0.0},
    "COS_LONG": {"lr_schedule": "cosine", "max_epochs": 600},
    "MAE": {"loss": "mae"},
    "COS_MAE": {"lr_schedule": "cosine", "loss": "mae"},
}
TARGETS = ("tap_iron", "tap_time_len")
RECORDED = "local/runs/round2-v12-joint-tabm/development-r1"


def _wmape(y, p):
    import numpy as np
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def _run_unit(root, frame, folds, settings, recipe, seed, fold, configs):
    import numpy as np
    from pathlib import Path

    mask = folds[seed] == fold
    training = frame.loc[~mask].reset_index(drop=True)
    query = frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
    records = {}
    for name in configs:
        unit = dict(settings)
        unit.update(CONFIGS[name])
        model = ScheduledJointRegressor(recipe, unit).fit(
            training, training[list(TARGETS)].to_numpy())
        prediction = model.predict(query)
        record = {"config": name, "seed": seed, "fold": fold,
                  "selected_epoch": int(model.metadata_["selected_epoch"]),
                  "rows": int(len(training)),
                  "wmape": {t: _wmape(frame[t].to_numpy()[mask], prediction[:, i])
                            for i, t in enumerate(TARGETS)}}
        if name == "CONTROL":
            reference = np.load(Path(root) / RECORDED / f"joint-joint_plr001-s{seed}-f{fold}.npy")
            record["control_max_abs_diff"] = float(np.max(np.abs(prediction - reference)))
        records[name] = record
    return seed, fold, records


def run_stage1(root, output, configs=None, units=((42, 0), (42, 1)), workers=8):
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    import yaml

    from .v5_library import fold_vector, load_v5_training_frame
    from .v5_spec import load_v5_spec

    root = Path(root).resolve()
    out = (root / output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = list(configs or CONFIGS)
    frame = load_v5_training_frame(root)
    spec = yaml.safe_load((root / "configs/round2_v12/SPEC.yaml").read_text())
    settings = dict(spec["training"])
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root))
             for seed, _ in units}
    recipe = {"backbone": "tabm", "frequency": 0.01}
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(_run_unit, root, frame, folds, settings, recipe, seed, fold, configs):
                (seed, fold) for seed, fold in units}
        for future in as_completed(jobs):
            seed, fold = jobs[future]
            key = f"s{seed}-f{fold}"
            try:
                _, _, records = future.result()
                results[key] = records
                print(json.dumps({"unit": key, "wmape": {n: r["wmape"] for n, r in records.items()}}),
                      flush=True)
            except Exception as exc:
                failures.append({"unit": key, "error": repr(exc)})
                print(json.dumps({"unit": key, "error": repr(exc)}), flush=True)
    payload = {"stage": "stage1", "units": results, "failures": failures,
               "configs": {n: CONFIGS[n] for n in configs}}
    (out / "stage1.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    if failures:
        return 1
    control = max(r["control_max_abs_diff"] for unit in results.values()
                  for r in unit.values() if "control_max_abs_diff" in r)
    print(json.dumps({"control_max_abs_diff": control,
                      "control": "passed" if control <= 1e-9 else "FAILED"}))
    import numpy as np
    for target in TARGETS:
        base = float(np.mean([results[u]["CONTROL"]["wmape"][target] for u in results]))
        print(f"== {target} control {base:.6f}")
        for name in configs:
            values = np.array([results[u][name]["wmape"][target] for u in sorted(results)])
            better = values < np.array([results[u]["CONTROL"]["wmape"][target] for u in sorted(results)])
            print(f"   {name:10s} mean {values.mean():.6f} delta {values.mean() - base:+.6f} "
                  f"both folds better: {bool(better.all())}")
    return 0


def run_stage2(root, output, configs, units=None, workers=8):
    """Confirm selected configurations on complete two-seed five-fold coverage."""
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    import yaml

    from .v5_library import fold_vector, load_v5_training_frame
    from .v5_spec import load_v5_spec

    root = Path(root).resolve()
    out = (root / output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = list(configs)
    for name in configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config: {name}")
    frame = load_v5_training_frame(root)
    spec = yaml.safe_load((root / "configs/round2_v12/SPEC.yaml").read_text())
    settings = dict(spec["training"])
    units = tuple(units or [(seed, fold) for seed in (42, 3407) for fold in range(5)])
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed, _ in units}
    recipe = {"backbone": "tabm", "frequency": 0.01}
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(_run_unit, root, frame, folds, settings, recipe, seed, fold,
                                list(configs)): (seed, fold) for seed, fold in units}
        for future in as_completed(jobs):
            seed, fold = jobs[future]
            key = f"s{seed}-f{fold}"
            try:
                _, _, records = future.result()
                results[key] = records
                with (out / f"{key}.json").open("x", encoding="utf-8") as stream:
                    json.dump(records, stream, indent=1)
                print(json.dumps({"unit": key, "wmape": {n: r["wmape"] for n, r in records.items()}}),
                      flush=True)
            except Exception as exc:
                failures.append({"unit": key, "error": repr(exc)})
                print(json.dumps({"unit": key, "error": repr(exc)}), flush=True)
    payload = {"stage": "stage2", "units": results, "failures": failures,
               "configs": {n: CONFIGS[n] for n in configs}}
    (out / "stage2.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    if failures:
        return 1
    # Stage 2 compares against the recorded V12 joint_plr001 folds, so the
    # baseline column is read from disk instead of re-fitting CONTROL.
    baseline = {}
    for seed, fold in units:
        path = root / RECORDED / f"joint-joint_plr001-s{seed}-f{fold}.npy"
        baseline[f"s{seed}-f{fold}"] = np.load(path)
    truth = {seed: frame[list(TARGETS)].to_numpy(float) for seed, _ in units}
    for name in configs:
        print(f"== {name}")
        for index, target in enumerate(TARGETS):
            gains, base_w, cand_w = [], [], []
            for seed, fold in sorted(units):
                unit = f"s{seed}-f{fold}"
                mask = folds[seed] == fold
                y = truth[seed][mask, index]
                base = _wmape(y, baseline[unit][:, index])
                cand = results[unit][name]["wmape"][target]
                gains.append(base - cand)
                base_w.append(base)
                cand_w.append(cand)
            gains = np.array(gains)
            print(f"   {target:13s} mean pooled-baseline {np.mean(base_w):.6f} "
                  f"candidate {np.mean(cand_w):.6f} mean delta {gains.mean():+.6f} "
                  f"positive folds {int((gains > 0).sum())}/{len(gains)} min {gains.min():+.6f}")
    return 0
