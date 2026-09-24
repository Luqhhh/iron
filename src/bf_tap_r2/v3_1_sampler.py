"""Directed S1 sampler for Round2 V3.1.

Centers are recovered from the private V3 first-batch trial ledger by exact
trial_id.  The sampler itself is label-free and only produces trial dictionaries;
identity hashes are assigned by the runner.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .v3_local_search import (
    _choice,
    _legal_kernel,
    _legal_lightgbm,
    _legal_mlp,
    _legal_xgboost,
)
from .v3_run import read_complete_records, trial_details

TIME_CENTERS = {
    "v3-catboost-tap_time_len-0021": (64, "time_core"),
    "v3-catboost-tap_time_len-0117": (32, "time_alt"),
    "v3-catboost-tap_time_len-0072": (32, "time_alt"),
}
IRON_CENTERS = {
    "v3-catboost-tap_iron-0107": (16, "iron_core"),
    "v3-catboost-tap_iron-0043": (16, "iron_core"),
    "v3-catboost-tap_iron-0075": (16, "iron_core"),
}


def load_centers(root: Path) -> dict[str, dict]:
    records = read_complete_records(Path(root) / "local/runs/round2-v3-local-search/coarse-catboost-r1/fit_ledger.jsonl")
    by_id = {r["trial_id"]: trial_details(r) for r in records}
    wanted = set(TIME_CENTERS) | set(IRON_CENTERS)
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(f"Missing V3.1 centers: {sorted(missing)}")
    return {key: by_id[key] for key in wanted}


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _set_bootstrap(params: dict, bootstrap: str, rng: np.random.Generator) -> None:
    old = params.get("bootstrap_type")
    if old in {"MVS", "Bernoulli"}:
        params.pop("subsample", None)
    if old == "Bayesian":
        params.pop("bagging_temperature", None)
    params["bootstrap_type"] = bootstrap
    if bootstrap == "MVS":
        params["subsample"] = float(rng.uniform(0.6, 1.0))
    elif bootstrap == "Bernoulli":
        params["subsample"] = float(rng.uniform(0.6, 1.0))
    elif bootstrap == "Bayesian":
        params["bagging_temperature"] = float(rng.uniform(0.0, 2.0))
    elif bootstrap == "No":
        pass
    else:
        raise ValueError(bootstrap)


def mutate_catboost(parent: Mapping[str, Any], rng: np.random.Generator, *, wide: bool = False) -> dict:
    trial = deepcopy(dict(parent))
    params = dict(trial["parameters"])
    grow = params.get("grow_policy", "SymmetricTree")
    # Mostly inherit the discrete structure.
    switch = rng.uniform()
    if wide or switch < 0.10:
        grow = _choice(rng, ("SymmetricTree", "SymmetricTree", "SymmetricTree", "Depthwise", "Lossguide"))
    if grow == "SymmetricTree":
        params["grow_policy"] = grow
        params["depth"] = int(_clip(int(params.get("depth", 6)) + int(rng.choice([-1, 0, 0, 1])), 3, 8))
        params.pop("max_leaves", None)
        params.pop("min_data_in_leaf", None)
    elif grow == "Depthwise":
        params["grow_policy"] = grow
        params["depth"] = int(_clip(int(params.get("depth", 6)) + int(rng.choice([-1, 0, 1])), 3, 8))
        params["min_data_in_leaf"] = int(_choice(rng, (10, 20, 50, 100)))
        params.pop("max_leaves", None)
    else:
        params["grow_policy"] = "Lossguide"
        params["max_leaves"] = int(_choice(rng, (15, 31, 63, 127)))
        params["min_data_in_leaf"] = int(_choice(rng, (10, 20, 50, 100)))
        params.pop("depth", None)
    params["learning_rate"] = _clip(float(params.get("learning_rate", 0.03)) * (10 ** float(rng.uniform(-0.30, 0.20))), 0.005, 0.20)
    params["l2_leaf_reg"] = _clip(float(params.get("l2_leaf_reg", 10.0)) * (10 ** float(rng.uniform(-0.50, 0.50))), 0.05, 300.0)
    params["random_strength"] = _clip(float(params.get("random_strength", 1.0)) * (10 ** float(rng.uniform(-0.30, 0.30))), 0.005, 10.0)
    params["iterations"] = int(_choice(rng, (1500, 2000, 3000, 4000, 6000)))
    params["random_seed"] = int(_choice(rng, (42, 2026, 2027)))
    if params.get("boosting_type") == "Ordered" and params["grow_policy"] != "SymmetricTree":
        params["boosting_type"] = "Plain"
    if rng.uniform() < 0.15:
        params["boosting_type"] = _choice(rng, ("Plain", "Plain", "Ordered")) if params["grow_policy"] == "SymmetricTree" else "Plain"
    if rng.uniform() < 0.20:
        _set_bootstrap(params, _choice(rng, ("MVS", "Bayesian", "Bernoulli", "No")), rng)
    else:
        # Keep the parent bootstrap but trim illegal leftovers.
        _set_bootstrap(params, str(params.get("bootstrap_type", "MVS")), rng)
    params["thread_count"] = 1
    params["allow_writing_files"] = False
    params["verbose"] = False
    params["cat_features"] = ["spout_no"]
    trial["parameters"] = params
    return trial


_EXPR_RECIPES = (
    ("raw", "log1p", "RMSE"),
    ("raw", "log1p", "MAE"),
    ("raw", "mean", "RMSE"),
    ("raw", "mean_std", "RMSE"),
    ("raw", "identity", "RMSE"),
    ("raw", "identity", "MAE"),
    ("four", "log1p", "RMSE"),
    ("four", "log1p", "MAE"),
    ("four", "mean", "RMSE"),
    ("four", "mean_std", "RMSE"),
    ("four", "identity", "RMSE"),
    ("four", "mean", "Huber:delta=0.05"),
)


def expression_trials(backbones: list[Mapping[str, Any]], rng: np.random.Generator) -> list[dict]:
    out = []
    for parent in backbones:
        for feature_set, transform, loss in _EXPR_RECIPES:
            trial = mutate_catboost(parent, rng)
            trial["feature_set"] = feature_set
            trial["target_transform"] = transform
            trial["parameters"]["loss_function"] = loss
            out.append(trial)
    return out


def smooth_residual_trials(target: str, rng: np.random.Generator) -> list[dict]:
    out = []
    for idx in range(8):
        transform = _choice(rng, ("identity", "mean", "mean_std", "log1p"))
        out.append({
            "family": "spline", "target": target,
            "feature_set": _choice(rng, ("raw", "raw", "four")),
            "target_transform": transform,
            "parameters": {"n_knots": int(_choice(rng, (4, 5, 6, 8))), "degree": int(_choice(rng, (2, 3))),
                           "alpha": float(10 ** rng.uniform(-1.0, 2.0))},
        })
    for idx in range(8):
        out.append({
            "family": "poly", "target": target,
            "feature_set": _choice(rng, ("raw", "raw", "four")),
            "target_transform": _choice(rng, ("identity", "mean", "mean_std", "log1p")),
            "parameters": {"degree": 2, "interaction_only": bool(rng.uniform() < 0.25),
                           "alpha": float(10 ** rng.uniform(-1.0, 2.0))},
        })
    for idx in range(8):
        base = {
            "task_type": "CPU", "loss_function": "RMSE", "depth": int(_choice(rng, (3, 4))),
            "iterations": int(_choice(rng, (1000, 1500, 2000))), "learning_rate": float(rng.uniform(0.02, 0.06)),
            "l2_leaf_reg": float(10 ** rng.uniform(0.0, 1.7)), "random_strength": float(rng.uniform(0.0, 1.0)),
            "random_seed": int(_choice(rng, (42, 2026, 2027))), "thread_count": 1,
            "cat_features": ["spout_no"], "allow_writing_files": False, "verbose": False,
            "bootstrap_type": "MVS", "subsample": float(rng.uniform(0.7, 1.0)),
        }
        residual = {
            "task_type": "CPU", "loss_function": "RMSE", "depth": int(_choice(rng, (2, 3))),
            "iterations": int(_choice(rng, (500, 1000, 1500))), "learning_rate": float(rng.uniform(0.02, 0.08)),
            "l2_leaf_reg": float(10 ** rng.uniform(0.5, 2.0)), "random_strength": float(rng.uniform(0.0, 1.0)),
            "random_seed": int(_choice(rng, (42, 2026, 2027))), "thread_count": 1,
            "cat_features": ["spout_no"], "allow_writing_files": False, "verbose": False,
            "bootstrap_type": "MVS", "subsample": float(rng.uniform(0.7, 1.0)),
        }
        out.append({
            "family": "residual", "target": target, "feature_set": "raw", "target_transform": "identity",
            "parameters": {"base": base, "residual": residual, "alpha": float(_choice(rng, (0.25, 0.5, 1.0))), "inner_seed": 777},
        })
    return out


def far_trials(target: str, rng: np.random.Generator, v3spec: Mapping[str, Any]) -> list[dict]:
    out = []
    for family, sampler, space_key in (
        ("lightgbm", _legal_lightgbm, "lightgbm_space"),
        ("xgboost", _legal_xgboost, "xgboost_space"),
        ("mlp", _legal_mlp, "mlp_kernel_space"),
        ("kernel", _legal_kernel, "mlp_kernel_space"),
    ):
        for _ in range(4):
            feature_set = _choice(rng, v3spec[space_key].get("feature_sets", ("raw", "four")))
            transform = _choice(rng, v3spec[space_key].get("target_transforms", ("mean", "mean_std", "identity", "log1p")))
            out.append({"family": family, "target": target, "feature_set": feature_set,
                        "target_transform": transform, "parameters": sampler(rng, v3spec[space_key])})
    return out


def sample_s1(root: Path | str, seed: int = 20260923,
              centers: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict]:
    """The frozen V3.1 S1 schedule.

    ``centers`` defaults to ``load_centers``, which reads a run ledger.  It is
    injectable so the schedule can be replayed when that ledger is unavailable
    but its contents are still reproducible; see ``next_phase_v31_ref``.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = load_centers(root)
    # Reuse the V3.1 config only for far-family spaces.
    import yaml
    v3spec = yaml.safe_load((root / "configs/round2_v3/experiment.yaml").read_text(encoding="utf-8"))
    trials: list[dict] = []
    for tid, (count, line) in TIME_CENTERS.items():
        parent = centers[tid]
        for i in range(count):
            trial = mutate_catboost(parent, rng, wide=(line == "time_alt"))
            trial["target"] = "tap_time_len"
            trial["family"] = "catboost"
            trial["line"] = line
            trial["trial_id"] = f"v31-s1-time-{tid.split('-')[-1]}-{i:04d}"
            trials.append(trial)
    for tid, (count, line) in IRON_CENTERS.items():
        parent = centers[tid]
        for i in range(count):
            trial = mutate_catboost(parent, rng, wide=False)
            trial["target"] = "tap_iron"
            trial["family"] = "catboost"
            trial["line"] = line
            trial["trial_id"] = f"v31-s1-iron-{tid.split('-')[-1]}-{i:04d}"
            trials.append(trial)
    # Wide iron 16
    iron_parents = [centers[tid] for tid in IRON_CENTERS]
    for i in range(16):
        parent = iron_parents[i % len(iron_parents)]
        trial = mutate_catboost(parent, rng, wide=True)
        trial["target"] = "tap_iron"; trial["family"] = "catboost"; trial["line"] = "iron_wide"
        trial["trial_id"] = f"v31-s1-iron-wide-{i:04d}"
        trials.append(trial)
    # Expression 48
    time_backbones = [centers["v3-catboost-tap_time_len-0021"], centers["v3-catboost-tap_time_len-0072"]]
    iron_backbones = [centers["v3-catboost-tap_iron-0075"], centers["v3-catboost-tap_iron-0107"]]
    for i, trial in enumerate(expression_trials(time_backbones, rng)):
        trial.update({"target": "tap_time_len", "family": "catboost", "line": "expression", "trial_id": f"v31-s1-expr-time-{i:04d}"})
        trials.append(trial)
    for i, trial in enumerate(expression_trials(iron_backbones, rng)):
        trial.update({"target": "tap_iron", "family": "catboost", "line": "expression", "trial_id": f"v31-s1-expr-iron-{i:04d}"})
        trials.append(trial)
    # Smooth/residual 48
    for target in ("tap_iron", "tap_time_len"):
        for i, trial in enumerate(smooth_residual_trials(target, rng)):
            trial["line"] = "smooth_residual"
            trial["trial_id"] = f"v31-s1-smooth-{target}-{i:04d}"
            trials.append(trial)
    # Far 32
    for target in ("tap_iron", "tap_time_len"):
        for i, trial in enumerate(far_trials(target, rng, v3spec)):
            trial["line"] = "far"
            trial["trial_id"] = f"v31-s1-far-{target}-{i:04d}"
            trials.append(trial)
    counts = {}
    for t in trials:
        counts[t["target"]] = counts.get(t["target"], 0) + 1
    if counts != {"tap_iron": 128, "tap_time_len": 192}:
        raise AssertionError(f"V3.1 sampler target counts mismatch: {counts}")
    if len(trials) != 320:
        raise AssertionError(f"V3.1 sampler produced {len(trials)} trials")
    return trials
