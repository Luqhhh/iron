"""Frozen training-seed averaging and deployment-aware epoch selection.

Outer splits, inner partitions and training RNGs are separate identities.
Nothing here selects a seed, fits a blend weight, or builds a submission.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .component_regularization import ComponentRegressor, clone_state
from .component_regularization_run import RECIPE, outputs
from .data import FEATURES, TARGETS
from .v12_joint import joint_loss
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest, write_new
from .v49_run import append_event

ARMS = ("E-NATIVE", "E-TARGET", "E-COMPOSE")


class AccountedRegressor(ComponentRegressor):
    def _train(self, frame, y, epochs, validation=None):
        phase = "selector" if validation is not None else "refit"
        path = self.directory/"fit-events.jsonl"
        append_event(path, {"event": f"{phase}_start", "fit_ids_digest": digest(frame.sample_id.tolist()),
                            "training_seed": self.settings["random_seed"], "inner_seed": self.settings["inner_seed"],
                            "requested_epochs": epochs})
        try:
            result = super()._train(frame, y, epochs, validation)
        except BaseException as exc:
            append_event(path, {"event": f"{phase}_failed", "error": repr(exc)})
            raise
        append_event(path, {"event": f"{phase}_complete", "selected_epoch": result,
                            "updates": self.traces["selection" if validation is not None else "refit"]["updates"]})
        return result


def clean_query(frame):
    return frame.loc[:, ["sample_id", "spout_no", *FEATURES]].copy()


def native_parts(training, inner_seed=42):
    if training.sample_id.duplicated().any():
        raise ValueError("Duplicate training IDs")
    folds = group_safe_inner_folds(training, seed=inner_seed)["fold"]
    mask = folds != 0
    return mask, training.loc[mask].reset_index(drop=True), training.loc[~mask].reset_index(drop=True)


def verify_disjoint(fitting, calibration, query=None):
    groups = [set(f.sample_id) for f in (fitting, calibration)]
    if query is not None:
        if any(t in query for t in TARGETS):
            raise ValueError("Query contains labels")
        groups.append(set(query.sample_id))
    for i, a in enumerate(groups):
        if any(a & b for b in groups[i+1:]):
            raise ValueError("Fitting/calibration/query overlap")


def select_epochs(history, settings):
    if not history or [r["epoch"] for r in history] != list(range(1, len(history)+1)):
        raise ValueError("Incomplete selector history")
    best, native, stale = math.inf, 0, 0
    for row in history:
        if stale >= settings["patience"]:
            raise ValueError("Epochs beyond native patience")
        for key in ("validation_mae", "target_mae", "compose_mae"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Invalid selector metric")
        if row["validation_mae"] < best-settings["min_delta"]:
            best, native, stale = row["validation_mae"], row["epoch"], 0
        else:
            stale += 1
    if len(history) > settings["max_epochs"] or (len(history) < settings["max_epochs"] and stale != settings["patience"]):
        raise ValueError("Premature native stopping")
    return {"E-NATIVE": native,
            "E-TARGET": min(history, key=lambda r: (r["target_mae"], r["epoch"]))["epoch"],
            "E-COMPOSE": min(history, key=lambda r: (r["compose_mae"], r["epoch"]))["epoch"]}


class EpochSelector(ComponentRegressor):
    """One unchanged native trajectory; three observers and a fixed epoch set."""
    def __init__(self, settings, target, directory):
        super().__init__(RECIPE, settings, "BASE", {}, directory)
        if target not in TARGETS:
            raise ValueError("Invalid target")
        self.target = target

    def select(self, training, background, query=None):
        y = training[outputs(self.target)].to_numpy()
        mask, fitting, calibration = native_parts(training, self.settings["inner_seed"])
        verify_disjoint(fitting, calibration, query)
        background = np.asarray(background, float)
        if background.shape != (len(calibration),) or not np.isfinite(background).all():
            raise ValueError("Invalid calibration background")
        # Preserve native NumPy reduction layout, including the corrective audit.
        self._initialize(fitting, y[mask])
        x, cat = self._inputs(fitting)
        vx, vc = self._inputs(clean_query(calibration))
        yt = torch.as_tensor((y[mask]-self.mean_)/self.std_, dtype=torch.float32)
        vy = torch.as_tensor((y[~mask]-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        best = {arm: math.inf for arm in ARMS}
        chosen, states, history, predictions = {}, {}, [], []
        stale = 0
        started = time.monotonic()
        for epoch in range(1, self.settings["max_epochs"]+1):
            self.model_.train()
            order = rng.permutation(len(fitting))
            losses = []
            for start in range(0, len(order), self.settings["batch_size"]):
                idx = order[start:start+self.settings["batch_size"]]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = joint_loss(self.model_(x[idx], cat[idx]), yt[idx])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite native loss")
                loss.backward()
                self.optimizer_.step()
                losses.append(float(loss.detach()))
            self.model_.eval()
            with torch.no_grad():
                pred = self.model_(vx, vc).mean(1)
                native = float((pred-vy).abs().mean())
            standardized = pred.numpy().copy()
            raw = standardized.astype(float)*self.std_+self.mean_
            actual = y[~mask, 0]
            metrics = {"E-NATIVE": native,
                       "E-TARGET": float(np.abs(actual-raw[:, 0]).mean()),
                       "E-COMPOSE": float(np.abs(actual-background-.5*raw[:, 0]).mean())}
            improved_native = native < best["E-NATIVE"]-self.settings["min_delta"]
            stale = 0 if improved_native else stale+1
            for arm, value in metrics.items():
                if (improved_native if arm == "E-NATIVE" else value < best[arm]):
                    best[arm], chosen[arm], states[arm] = value, epoch, clone_state(self.model_)
            history.append({"epoch": epoch, "updates": len(losses),
                            "gradient_evaluations": len(losses), "validation_mae": native,
                            "target_mae": metrics["E-TARGET"], "compose_mae": metrics["E-COMPOSE"],
                            "mean_batch_training_loss": float(np.mean(losses))})
            predictions.append(standardized)
            if stale >= self.settings["patience"]:
                break
        if chosen != select_epochs(history, self.settings):
            raise ValueError("Observer selection mismatch")
        trace = {"history": history, "stopped_epoch": len(history),
                 "fit_ids_digest": digest(fitting.sample_id.tolist()), "fit_rows": len(fitting),
                 "updates": sum(r["updates"] for r in history),
                 "gradient_evaluations": sum(r["gradient_evaluations"] for r in history),
                 "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        for arm in ARMS:
            self.model_.load_state_dict(states[arm])
            self.save(self.directory/f"selection-{arm}.pt", {**trace, "selected_epoch": chosen[arm]})
        with (self.directory/"epoch-predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, standardized=np.stack(predictions),
                                calibration_ids=calibration.sample_id.to_numpy(dtype=str))
        self.selection_ = {"epochs": chosen, "history": history, "visited_epochs": len(history),
                           "fit_ids_digest": digest(fitting.sample_id.tolist()),
                           "calibration_ids_digest": digest(calibration.sample_id.tolist()),
                           "inner_fold_hash": digest(mask.tolist()),
                           "seconds": time.monotonic()-started,
                           "candidate_set": "native_visited_epochs_only",
                           "native_and_target_epoch_equal": chosen["E-NATIVE"] == chosen["E-TARGET"]}
        write_new(self.directory/"selection.json", self.selection_)
        return self.selection_


def fresh_refit(training, target, settings, epoch, directory):
    if not 1 <= epoch <= settings["max_epochs"]:
        raise ValueError("Refit epoch outside frozen budget")
    model = AccountedRegressor(RECIPE, settings, "BASE", {}, directory)
    y = training[outputs(target)].to_numpy()
    model._initialize(training, y)
    model._train(training, y, epoch)
    model.model_.eval()
    return model


@contextmanager
def solver_accounting(path, scope):
    """Observe solver fit calls without altering their data or mathematics.

Forked factory workers inherit the wrappers and append their own fit events.
Pipeline counts are separate from these leaf solver and optimizer calls.
"""
    names = [("catboost", "CatBoostRegressor", "fit"),
             ("lightgbm", "LGBMRegressor", "fit"),
             ("sklearn.neural_network", "MLPRegressor", "fit"),
             ("interpret.glassbox", "ExplainableBoostingRegressor", "fit"),
             ("bf_tap_r2.v3_6_networks", "V36NetworkRegressor", "fit")]
    originals = []
    def wrap(original, solver):
        def observed(self, *args, **kwargs):
            start = time.monotonic()
            append_event(path, {"event": "solver_start", "scope": scope, "solver": solver,
                                "rows": len(args[0]) if args else None})
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                append_event(path, {"event": "solver_failed", "scope": scope, "solver": solver, "error": repr(exc)})
                raise
            append_event(path, {"event": "solver_complete", "scope": scope,
                                "solver": solver, "seconds": time.monotonic()-start})
            return result
        return observed
    try:
        for module, name, method in names:
            cls = getattr(importlib.import_module(module), name)
            original = getattr(cls, method)
            originals.append((cls, method, original))
            setattr(cls, method, wrap(original, f"{module}.{name}"))
        yield
    finally:
        for cls, method, original in originals:
            setattr(cls, method, original)


def fit_background(root, fitting, calibration, spec, directory):
    """Only A can fit the frozen V36/N background used to select on C."""
    from .v4_1_reference import V36FixedRecipeFactory
    from .v5_replicate import load_candidate
    from .v5_spec import load_v5_spec
    verify_disjoint(fitting, calibration)
    query = clean_query(calibration)
    factory = V36FixedRecipeFactory(root, workers=spec["budget"]["reference_workers"])
    pipelines = (len(factory.a_factory._recovery.BASE_NAMES)
                 + sum(map(len, factory.a_factory._expert_ids.values()))
                 + sum(map(len, factory.selected_experts.values())) + 1)
    if pipelines != spec["budget"]["inner_reference_pipeline_fits_per_factory"]:
        raise ValueError("Frozen background pipeline count changed")
    started = time.monotonic()
    with solver_accounting(directory/"solver-fits.jsonl", "inner_reference_A_only"):
        bundle = factory.fit_predict(fitting, query)
        n_fit, n_metadata = load_candidate(root, load_v5_spec(root), "v36", spec["reference"]["n_trial"])
        n = n_fit(fitting, query, "tap_time_len")
    values = {"v36_iron": bundle["b36"]["tap_iron"], "v36_time": bundle["b36"]["tap_time_len"], "n_time": n}
    values.update(tap_iron=.5*values["v36_iron"], tap_time_len=.2*values["v36_time"]+.3*values["n_time"])
    if any(np.shape(v) != (len(calibration),) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Invalid inner background")
    return values, {"fit_ids_digest": digest(fitting.sample_id.tolist()),
                    "query_ids_digest": digest(query.sample_id.tolist()),
                    "training_rows": len(fitting), "query_rows": len(query),
                    "factory_calls": 1, "component_pipeline_fits": pipelines,
                    "query_labels_received": False, "seconds": time.monotonic()-started,
                    "peak_rss_mib": max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)/1024,
                    "v36": bundle["meta"], "n": n_metadata}
