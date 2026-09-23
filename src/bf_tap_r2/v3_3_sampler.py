"""V3.3 structural-search sampler: 64 feature packs, 64 EBM, 48 full residual."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .v3_run import read_complete_records

FEATURE_PARENTS = {
    "tap_iron": ["v31-s1-expr-iron-0018", "v32-s1-iron_log_expression-0019"],
    "tap_time_len": ["v31-s1-time-0021-0050", "v32-s1-time_neighborhood-0126"],
}
FEATURE_PACKS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]
L2_MULTIPLIERS = [1.0, 3.0]
RESIDUAL_PARENTS = FEATURE_PARENTS


def _trials(root: Path) -> dict[str, dict]:
    v31 = read_complete_records(root / "local/runs/round2-v3.1-directed-search/s1-coarse-r1/fit_ledger.jsonl")
    v32 = read_complete_records(root / "local/runs/round2-v3.2-ensemble-and-target-search/coarse-r1/fit_ledger.jsonl")
    return {r["trial_id"]: r["trial"] for r in [*v31, *v32]}


def feature_trials(trials: dict[str, dict]) -> list[dict]:
    out = []
    for target, parents in FEATURE_PARENTS.items():
        for parent_id in parents:
            parent = trials[parent_id]
            for pack in FEATURE_PACKS:
                for multiplier in L2_MULTIPLIERS:
                    out.append({
                        "kind": "pack_catboost", "target": target,
                        "feature_pack": pack, "l2_multiplier": float(multiplier),
                        "target_transform": parent["target_transform"],
                        "parameters": deepcopy(parent["parameters"]),
                        "line": "feature_structure",
                        "parent_id": parent_id,
                    })
    return out


def ebm_trials() -> list[dict]:
    out = []
    for target in ("tap_iron", "tap_time_len"):
        for max_bins in (64, 128):
            for min_samples_leaf in (10, 30):
                for interactions in (0, 3, 6, 10):
                    for transform in ("mean", "log1p"):
                        out.append({
                            "kind": "ebm", "target": target, "target_transform": transform,
                            "parameters": {
                                "max_bins": int(max_bins), "min_samples_leaf": int(min_samples_leaf),
                                "interactions": int(interactions), "max_interaction_bins": 32,
                                "max_leaves": 2, "objective": "rmse", "learning_rate": 0.03,
                                "outer_bags": 4, "inner_bags": 0, "max_rounds": 6000,
                                "early_stopping_rounds": 100, "random_state": 42, "n_jobs": 1,
                            },
                            "line": "ebm",
                        })
    return out


def _residual_correctors() -> list[dict]:
    correctors = []
    for alpha in (1.0, 10.0, 100.0):
        correctors.append({"kind": "ridge", "params": {"alpha": float(alpha)},
                           "name": f"ridge_{int(alpha)}"})
    for l2 in (10.0, 30.0, 100.0):
        correctors.append({
            "kind": "catboost",
            "name": f"cat_l2_{int(l2)}",
            "params": {
                "task_type": "CPU", "loss_function": "RMSE", "depth": 2, "iterations": 600,
                "learning_rate": 0.03, "l2_leaf_reg": float(l2), "random_strength": 0.3,
                "random_seed": 42, "thread_count": 1, "cat_features": ["spout_no"],
                "allow_writing_files": False, "verbose": False, "bootstrap_type": "MVS", "subsample": 0.8,
            },
        })
    return correctors


def residual_trials(trials: dict[str, dict]) -> list[dict]:
    out = []
    for target, parents in RESIDUAL_PARENTS.items():
        for parent_id in parents:
            parent = trials[parent_id]
            for coordinate in ("original_unit", "log_unit"):
                for corrector in _residual_correctors():
                    out.append({
                        "kind": "full_residual", "target": target,
                        "line": "full_recipe_residual", "coordinate": coordinate, "corrector": corrector["name"],
                        "parameters": {
                            "base_trial": deepcopy(parent),
                            "residual": {"kind": corrector["kind"], "params": deepcopy(corrector["params"])},
                            "alpha": 0.25, "coordinate": coordinate, "inner_seed": 777, "inner_splits": 5,
                        },
                    })
    return out


def sample_v33(root: Path | str) -> list[dict]:
    root = Path(root)
    trials = _trials(root)
    out = []
    out.extend(feature_trials(trials))
    out.extend(ebm_trials())
    out.extend(residual_trials(trials))
    counts = {"feature_structure": 64, "ebm": 64, "full_recipe_residual": 48}
    actual = {}
    for trial in out:
        actual[trial["line"]] = actual.get(trial["line"], 0) + 1
    if actual != counts:
        raise AssertionError(f"V3.3 sampler line counts mismatch: {actual}")
    if len(out) != 176:
        raise AssertionError(f"V3.3 sampler produced {len(out)} trials")
    for i, trial in enumerate(out):
        trial["trial_id"] = f"v33-s1-{trial['line']}-{i:04d}"
    return out
