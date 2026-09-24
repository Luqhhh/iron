"""Batch-1 members for the round2 next-phase search.

Batch 1 attacks the time side, where V3.3 recorded only a single EBM center and
V3.4 was therefore forced to mark 8 of its 16 time residual slots
``not_applicable``.  This module supplies time-side EBM experts as harness
members so the nested driver can measure whether they add anything to the
anchor.

The EBM recipe is the frozen one from ``configs/round2_v3_4/search.yaml``; only
the grid point varies.  ``V34EBMRegressor`` already implements the
group-safe-bags-v1 protocol and the ``log1p`` inverse transform, so it is
reused rather than reimplemented.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .next_phase_nested import C2_PARAMS, CatBoostMember, Member

EBM_BASE: dict[str, Any] = {
    "max_leaves": 2,
    "objective": "rmse",
    "learning_rate": 0.03,
    "outer_bags": 4,
    "inner_bags": 0,
    "max_rounds": 6000,
    "early_stopping_rounds": 100,
    "random_state": 42,
    "n_jobs": 1,
}

TIME_TARGET = "tap_time_len"

# Grid points spanning the frozen boundary grid.  Kept small for the first
# screen: every member costs an inner OOF plus a refit per outer fold.
TIME_EBM_CENTERS: tuple[dict[str, Any], ...] = (
    {
        "name": "time_ebm_b128_l30_i20_mb32",
        "max_bins": 128,
        "min_samples_leaf": 30,
        "interactions": 20,
        "max_interaction_bins": 32,
    },
    {
        "name": "time_ebm_b256_l60_i40_mb64",
        "max_bins": 256,
        "min_samples_leaf": 60,
        "interactions": 40,
        "max_interaction_bins": 64,
    },
)


class EBMMember(Member):
    """One EBM boundary grid point, fit inside the current training part.

    ``inner_splits`` seeds the EBM's own group-safe bags.  That split happens
    inside whatever frame the harness hands over, which is already the fold's
    training part, so it never touches validation labels.
    """

    def __init__(
        self,
        name: str,
        target: str,
        target_transform: str = "log1p",
        parameters: dict[str, Any] | None = None,
        inner_splits: int = 5,
        bag_seed: int = 42,
    ):
        self.name = name
        self.targets = (target,)
        self.target = target
        self.target_transform = target_transform
        self.parameters = dict(EBM_BASE if parameters is None else parameters)
        self.inner_splits = int(inner_splits)
        self.bag_seed = int(bag_seed)

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        from .v3_4_models import V34EBMRegressor

        if target != self.target:
            raise ValueError(f"{self.name} is a {self.target} expert, asked for {target}")
        model = V34EBMRegressor(
            {
                "kind": "ebm_boundary",
                "target": self.target,
                "target_transform": self.target_transform,
                "parameters": self.parameters,
                "protocol": {"inner_splits": self.inner_splits, "bag_seed": self.bag_seed},
            }
        )
        model.fit(train, train[target].to_numpy(dtype=float))
        return np.asarray(model.predict(valid), dtype=float)


def time_ebm_members(centers: Sequence[dict[str, Any]] = TIME_EBM_CENTERS) -> list[Member]:
    members: list[Member] = []
    for centre in centers:
        parameters = {**EBM_BASE}
        for key in ("max_bins", "min_samples_leaf", "interactions", "max_interaction_bins"):
            parameters[key] = centre[key]
        members.append(
            EBMMember(
                str(centre["name"]),
                target=TIME_TARGET,
                target_transform=str(centre.get("target_transform", "log1p")),
                parameters=parameters,
            )
        )
    return members


def r0_plus_time_ebm() -> list[Member]:
    """The anchor plus the batch-1 time-side experts."""
    return [CatBoostMember("c2_raw"), *time_ebm_members()]


# The V3.1 parent recipes the frozen shrink grid points at
# (v31-s1-time-0021-0050, v32-s1-time_neighborhood-0126) are search outputs
# that only exist in the missing run cache.  The C2 recipe stands in for them
# and is documented as a stand-in: this screen asks whether a per-spout
# correction adds anything at all, not whether one particular parent wins.
SHRINK_PARENT_PARAMS: dict[str, Any] = {**C2_PARAMS, "cat_features": ["spout_no"]}

SHRINK_GRID: tuple[dict[str, Any], ...] = (
    {"name": "shrink_m1.0_b0.25", "local_l2_multiplier": 1.0, "beta": 0.25},
    {"name": "shrink_m3.0_b0.50", "local_l2_multiplier": 3.0, "beta": 0.50},
)


def catboost_trial(
    target: str,
    feature_set: str = "raw",
    target_transform: str = "identity",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A V3.1/V3 trial dict, which is what the shrink wrapper consumes."""
    return {
        "family": "catboost",
        "feature_set": feature_set,
        "target_transform": target_transform,
        "parameters": dict(SHRINK_PARENT_PARAMS if parameters is None else parameters),
        "target": target,
    }


class ShrinkSpoutMember(Member):
    """Global parent plus a per-spout correction, shrunk by beta.

    Reuses ``ShrunkSpoutRegressor`` so the frozen combination rule, the
    ``min_spout_samples`` fallback and the inverse-transform ordering are the
    recorded ones rather than a reimplementation.
    """

    def __init__(
        self,
        name: str,
        target: str,
        beta: float,
        local_l2_multiplier: float,
        parent: dict[str, Any] | None = None,
        min_spout_samples: int = 200,
    ):
        self.name = name
        self.targets = (target,)
        self.target = target
        self.beta = float(beta)
        self.local_l2_multiplier = float(local_l2_multiplier)
        self.parent = catboost_trial(target) if parent is None else parent
        self.min_spout_samples = int(min_spout_samples)

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        from .v3_4_models import ShrunkSpoutRegressor

        if target != self.target:
            raise ValueError(f"{self.name} is a {self.target} expert, asked for {target}")
        model = ShrunkSpoutRegressor(
            {
                "kind": "global_spout_shrink",
                "target": self.target,
                "parameters": {
                    "parent_trial": self.parent,
                    "local_l2_multiplier": self.local_l2_multiplier,
                    "beta": self.beta,
                    "min_spout_samples": self.min_spout_samples,
                    "local_include_spout": True,
                },
            }
        )
        model.fit(train, train[target].to_numpy(dtype=float))
        return np.asarray(model.predict(valid), dtype=float)


def spout_shrink_members(
    target: str = TIME_TARGET,
    grid: Sequence[dict[str, Any]] = SHRINK_GRID,
) -> list[Member]:
    members: list[Member] = []
    for point in grid:
        members.append(
            ShrinkSpoutMember(
                str(point["name"]),
                target=target,
                beta=float(point["beta"]),
                local_l2_multiplier=float(point["local_l2_multiplier"]),
            )
        )
    return members


def r0_plus_shrink() -> list[Member]:
    """The anchor plus the batch-1 per-spout shrink candidates."""
    return [CatBoostMember("c2_raw"), *spout_shrink_members()]
