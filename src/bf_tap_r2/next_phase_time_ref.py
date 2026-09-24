"""Time-side reference reconstruction, the traceable half of L0.

``configs/round2_v3_1/search.yaml`` pins L0's frozen recipe with exact weights,
but the run that produced those members is not on this machine.  Six of the
seven members are still reconstructible from checked-in recipes; the seventh,
``AJM1``, exists only in prose and carries the largest iron weight, so this
module rebuilds **the time side only**, where every member traces to a config.

Members, and where each recipe comes from:

``v3-catboost-tap_time_len-0021``
    ``configs/round2_v3/experiment.yaml`` via ``sample_trials``.  The sampler is
    seeded random rather than a grid, so the schedule is reproducible only while
    the seed and the search space are unchanged.
``AORD``
    ``configs/round2_v2_5/experiment.yaml``: the mean of B3 and ORD.
``B3_C2_SEED_ENSEMBLE``
    ``configs/round2_v2_1/experiment.yaml``: the mean of C2 at seeds 42, 2026
    and 2027, which ``v2_refinement.new_model`` shows is how the seed is bound.
``T1_B3_PREFIX1000``
    The mean of the same three C2 members truncated to their first 1000 trees,
    matching ``v2_checkpoint.prefix_predict`` with ``target_scale_`` of 1.0.

Rebuilding the time side does **not** establish identity with V34_A.  It gives a
time-only reference that can be compared on the same folds, which is what the
asymmetric platform plan needs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .models import inputs
from .next_phase_nested import Member
from .v3_local_search import TrialRegressor, sample_trials

TIME_TARGET = "tap_time_len"

# configs/round2_v0_1/models.yaml, C2 route plus the execution block.
C2_SPEC: dict[str, Any] = {
    "loss_function": "RMSE",
    "depth": 6,
    "iterations": 1500,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
}
C2_THREAD_COUNT = 4

# configs/round2_v2_1/experiment.yaml
B3_SEEDS: tuple[int, ...] = (42, 2026, 2027)

# configs/round2_v2_5/experiment.yaml, route ORD over that file's common block.
ORD_SPEC: dict[str, Any] = {
    "loss_function": "RMSE",
    "depth": 6,
    "iterations": 1500,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
}

# configs/round2_v3_1/search.yaml, l0_v3_frozen_recipe.time_members, full precision.
L0_TIME_RECIPE: dict[str, float] = {
    "v3-catboost-tap_time_len-0021": 0.6688582528038586,
    "AORD": 0.17410974011267225,
    "T1_B3_PREFIX1000": 0.15703200708346923,
}

PREFIX_TREES = 1000
V3_CONFIG = Path("configs/round2_v3/experiment.yaml")


def _catboost(spec: dict[str, Any], seed: int, frame: pd.DataFrame, y: np.ndarray):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        **spec,
        random_seed=int(seed),
        thread_count=C2_THREAD_COUNT,
        cat_features=["spout_no"],
        verbose=False,
        allow_writing_files=False,
        use_best_model=False,
    )
    model.fit(inputs(frame, True), np.asarray(y, dtype=float))
    return model


def _as_time(expert_target: str, target: str, name: str) -> None:
    if target != expert_target:
        raise ValueError(f"{name} is a {expert_target} expert, asked for {target}")


class C2SeedMember(Member):
    """C2 at one frozen seed."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.name = f"c2_seed{self.seed}"
        self.targets = (TIME_TARGET,)

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        _as_time(TIME_TARGET, target, self.name)
        model = _catboost(C2_SPEC, self.seed, train, train[target].to_numpy(dtype=float))
        return np.asarray(model.predict(inputs(valid, True)), dtype=float)


class ORDMember(Member):
    """v2.5's Ordered-boosting, Bayesian-bootstrap time model."""

    def __init__(self, name: str = "ord_ordered_bayes"):
        self.name = name
        self.targets = (TIME_TARGET,)

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        _as_time(TIME_TARGET, target, self.name)
        model = _catboost(ORD_SPEC, 42, train, train[target].to_numpy(dtype=float))
        return np.asarray(model.predict(inputs(valid, True)), dtype=float)


class B3Member(Member):
    """The mean of C2 at seeds 42, 2026 and 2027."""

    def __init__(self, name: str = "b3_c2_seed_ensemble"):
        self.name = name
        self.targets = (TIME_TARGET,)

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        _as_time(TIME_TARGET, target, self.name)
        y = train[target].to_numpy(dtype=float)
        columns = [
            np.asarray(_catboost(C2_SPEC, seed, train, y).predict(inputs(valid, True)), dtype=float)
            for seed in B3_SEEDS
        ]
        return np.mean(np.stack(columns), axis=0)


class AORDMember(Member):
    """AORD = the mean of B3 and ORD, per ``combine_same_fold``."""

    def __init__(self, name: str = "aord"):
        self.name = name
        self.targets = (TIME_TARGET,)
        self.b3 = B3Member()
        self.ord = ORDMember()

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        _as_time(TIME_TARGET, target, self.name)
        return np.mean(
            np.stack(
                [
                    self.b3.fit_predict(train, valid, target),
                    self.ord.fit_predict(train, valid, target),
                ]
            ),
            axis=0,
        )


class T1PrefixMember(Member):
    """The mean of the three B3 members truncated to 1000 trees."""

    def __init__(self, name: str = "t1_b3_prefix1000", ntree_end: int = PREFIX_TREES):
        self.name = name
        self.targets = (TIME_TARGET,)
        self.ntree_end = int(ntree_end)
        # An impossible prefix is a configuration error, so it fails at
        # construction rather than after the caller has started fitting.
        if not 0 < self.ntree_end <= C2_SPEC["iterations"]:
            raise ValueError("Frozen prefix rule is invalid for this recipe")

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        _as_time(TIME_TARGET, target, self.name)
        y = train[target].to_numpy(dtype=float)
        x_valid = inputs(valid, True)
        columns = []
        for seed in B3_SEEDS:
            model = _catboost(C2_SPEC, seed, train, y)
            raw = model.predict(x_valid, ntree_start=0, ntree_end=self.ntree_end)
            columns.append(np.asarray(raw, dtype=float))
        return np.mean(np.stack(columns), axis=0)


class V3TrialMember(Member):
    """One frozen V3 first-batch trial."""

    def __init__(self, trial: dict[str, Any], name: str | None = None):
        self.trial = dict(trial)
        self.name = name or str(trial["trial_id"])
        self.targets = (str(trial["target"]),)

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        if target != self.targets[0]:
            raise ValueError(f"{self.name} is a {self.targets[0]} expert, asked for {target}")
        model = TrialRegressor(self.trial)
        model.fit(train, train[target].to_numpy(dtype=float))
        return np.asarray(model.predict(valid), dtype=float)


class WeightedMember(Member):
    """A frozen-weight arithmetic mean over member predictions.

    L0 is a weighted mean in original units, and its time weights sum to one.
    This exists so a fixed historical recipe can be evaluated on the same folds
    as everything else without re-learning anything.
    """

    def __init__(self, name: str, members: Sequence[Member], weights: Sequence[float]):
        if len(members) != len(weights) or not members:
            raise ValueError("WeightedMember needs one weight per member")
        if abs(float(np.sum(weights)) - 1.0) > 1e-9:
            raise ValueError("WeightedMember weights must sum to one")
        if any(float(w) < 0 for w in weights):
            raise ValueError("WeightedMember weights must be nonnegative")
        self.name = name
        self.members = list(members)
        self.weights = [float(w) for w in weights]
        targets = {member.targets for member in self.members}
        if len(targets) != 1:
            raise ValueError("WeightedMember members must share one target")
        self.targets = targets.pop()

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        columns = [
            np.asarray(member.fit_predict(train, valid, target), dtype=float)
            for member in self.members
        ]
        return np.average(np.stack(columns), axis=0, weights=self.weights)


def v3_time_0021(root: Path) -> dict[str, Any]:
    """The frozen V3 trial that carries 0.669 of L0's time weight."""
    spec = yaml.safe_load((root / V3_CONFIG).read_text(encoding="utf-8"))
    wanted = "v3-catboost-tap_time_len-0021"
    for trial in sample_trials(spec):
        if trial["trial_id"] == wanted:
            return trial
    raise ValueError(f"{wanted} is not in the frozen V3 schedule")


def l0_time_pairs(root: Path) -> list[tuple[Member, float]]:
    """L0's time members paired with their frozen full-precision weights."""
    return [
        (V3TrialMember(v3_time_0021(root)), L0_TIME_RECIPE["v3-catboost-tap_time_len-0021"]),
        (AORDMember(), L0_TIME_RECIPE["AORD"]),
        (T1PrefixMember(), L0_TIME_RECIPE["T1_B3_PREFIX1000"]),
    ]


def l0_time_components(root: Path) -> list[Member]:
    """L0's three time members, unweighted, for pool comparison."""
    return [member for member, _ in l0_time_pairs(root)]


def l0_time_member(root: Path) -> Member:
    """L0's time side at frozen weights, as a single member."""
    pairs = l0_time_pairs(root)
    return WeightedMember(
        "l0_time_frozen",
        [member for member, _ in pairs],
        [weight for _, weight in pairs],
    )
