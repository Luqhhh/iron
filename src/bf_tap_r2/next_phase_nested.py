"""Nested outer-evaluation harness for the round2 next-phase search.

V3.4's final-outer driver was never checked into the repository, so a fresh
search line has no way to evaluate a candidate under the same protocol the
recorded numbers used.  This module supplies that driver.

The nesting contract is the whole point.  For an outer fold, everything that
informs the fold's prediction is fit inside the fold's training part:

1. members are fitted on the inner training part to produce inner OOF
   predictions, so no row's weight is informed by its own label;
2. fusion weights are learned on that inner OOF alone;
3. members are then refitted on the full training part and applied to the
   validation rows.

Nothing reads the validation labels.  Members must honour the same rule
internally, which is why the frozen C2 recipe carries no outer-fold early
stopping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .metrics import wmape
from .splits import make_folds
from .v2_release import load_v2
from .v3_1_fusion import lp_simplex_weights


class Member:
    """A named predictor fitted per fold on the current training part only.

    Set ``targets`` to restrict a member to specific targets; the rest of the
    pool is then evaluated without it.  A target-specific expert should declare
    its target, otherwise it is fitted redundantly for the other one.
    """

    name: str
    targets: tuple[str, ...] | None = None

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        raise NotImplementedError


def applies_to(member: Member, target: str) -> bool:
    supported = getattr(member, "targets", None)
    return supported is None or target in supported


def fold_vector(frame: pd.DataFrame, seed: int, n_splits: int = 5) -> np.ndarray:
    """Fold assignment aligned to ``frame`` row order."""
    assignment = make_folds(frame, seed, n_splits=n_splits)
    if not assignment.sample_id.is_unique or set(assignment.sample_id) != set(frame.sample_id):
        raise ValueError("Fold identity mismatch")
    aligned = assignment.set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy()
    if set(aligned) != set(range(n_splits)):
        raise ValueError("Fold vector is incomplete")
    return aligned


def _member_names(members: Sequence[Member]) -> list[str]:
    if not members:
        raise ValueError("At least one member is required")
    names = [member.name for member in members]
    if len(set(names)) != len(names):
        raise ValueError("Member names must be unique")
    return names


def _aligned_predictions(values, expected_rows: int, member_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_rows,):
        raise ValueError(f"Member {member_name} returned shape {array.shape}, expected {(expected_rows,)}")
    if not np.isfinite(array).all():
        raise ValueError(f"Member {member_name} returned non-finite predictions")
    return array


def inner_oof(
    train_part: pd.DataFrame,
    target: str,
    members: Sequence[Member],
    inner_seed: int,
    inner_folds: int = 5,
) -> np.ndarray:
    """Inner OOF predictions for every member inside ``train_part``."""
    _member_names(members)
    folds = fold_vector(train_part, inner_seed, inner_folds)
    predictions = np.full((len(train_part), len(members)), np.nan, dtype=float)
    for fold in range(inner_folds):
        train_mask = folds != fold
        valid_mask = folds == fold
        inner_train = train_part.loc[train_mask].reset_index(drop=True)
        inner_valid = train_part.loc[valid_mask].reset_index(drop=True)
        for column, member in enumerate(members):
            values = _aligned_predictions(
                member.fit_predict(inner_train, inner_valid, target),
                int(valid_mask.sum()),
                member.name,
            )
            predictions[valid_mask, column] = values
    if np.isnan(predictions).any():
        raise ValueError("Inner OOF coverage is incomplete")
    return predictions


def fit_weights(predictions: np.ndarray, actual: np.ndarray, names: Sequence[str]) -> dict:
    """Nonnegative simplex weights over members, minimising pooled WMAPE."""
    matrix = np.asarray(predictions, dtype=float)
    y = np.asarray(actual, dtype=float)
    if matrix.ndim != 2 or y.ndim != 1 or matrix.shape[0] != y.shape[0]:
        raise ValueError("Prediction matrix and target must share a row axis")
    if matrix.shape[1] != len(names):
        raise ValueError("Candidate name count mismatch")
    fit = lp_simplex_weights({"inner": y}, {"inner": matrix})
    return {
        "names": list(names),
        "weights": fit["weights"],
        "objective": fit["objective"],
        "weights_by_name": {n: float(w) for n, w in zip(names, fit["weights"])},
    }


def evaluate_outer(
    train: pd.DataFrame,
    members: Sequence[Member],
    outer_seed: int,
    inner_seed: int,
    outer_folds: int = 5,
    inner_folds: int = 5,
    targets: Sequence[str] = TARGETS,
) -> dict:
    """Nested outer evaluation, pooled over the complete coverage set.

    ``targets`` narrows the work to the targets under test, which is what makes
    a target-isolated comparison cheap: the untouched target is simply not
    evaluated here.
    """
    names = _member_names(members)
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target}")
    folds = fold_vector(train, outer_seed, outer_folds)
    per_target: dict[str, dict] = {}
    for target in targets:
        active = [member for member in members if applies_to(member, target)]
        active_names = _member_names(active)
        y = train[target].to_numpy(dtype=float)
        predictions = np.full(len(train), np.nan, dtype=float)
        fold_wmape: dict[str, float] = {}
        fold_weights: dict[str, dict] = {}
        for fold in range(outer_folds):
            train_mask = folds != fold
            valid_mask = folds == fold
            outer_train = train.loc[train_mask].reset_index(drop=True)
            outer_valid = train.loc[valid_mask].reset_index(drop=True)
            inner = inner_oof(outer_train, target, active, inner_seed + fold, inner_folds)
            fit = fit_weights(inner, outer_train[target].to_numpy(dtype=float), active_names)
            columns = np.column_stack(
                [
                    _aligned_predictions(
                        member.fit_predict(outer_train, outer_valid, target),
                        int(valid_mask.sum()),
                        member.name,
                    )
                    for member in active
                ]
            )
            predictions[valid_mask] = columns @ fit["weights"]
            fold_wmape[str(fold)] = wmape(y[valid_mask], predictions[valid_mask])
            fold_weights[str(fold)] = fit["weights_by_name"]
        if np.isnan(predictions).any():
            raise ValueError("Outer coverage is incomplete")
        per_target[target] = {
            "pooled_wmape": wmape(y, predictions),
            "fold_wmape": fold_wmape,
            "fold_weights": fold_weights,
        }
    # Frozen formula: J = (W_I + W_T) / 2 and score = 100 - 100*J, which is
    # 100 - 50*(W_I + W_T).  With one target this reduces to 100 - 100*W_t.
    mean_wmape = float(np.mean([per_target[t]["pooled_wmape"] for t in targets]))
    return {
        "outer_seed": outer_seed,
        "inner_seed": inner_seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "members": names,
        "targets": per_target,
        "package_score": 100.0 - 100.0 * mean_wmape,
    }


C2_PARAMS = {
    "loss_function": "RMSE",
    "depth": 6,
    "iterations": 1500,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
}


def catboost_frame(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """C2 input frame; ``degree2`` appends every pairwise product.

    Built from one column mapping rather than repeated insertion: 210 inserts
    fragment the block and pandas warns on every fit.
    """
    columns = {name: frame[name].to_numpy(dtype=float) for name in FEATURES}
    if kind == "degree2":
        values = frame[list(FEATURES)].to_numpy(dtype=float)
        for i in range(len(FEATURES)):
            for j in range(i + 1, len(FEATURES)):
                columns[f"{FEATURES[i]}__x__{FEATURES[j]}"] = values[:, i] * values[:, j]
    columns["spout_no"] = frame["spout_no"].astype(int).to_numpy()
    return pd.DataFrame(columns, index=frame.index)


class CatBoostMember(Member):
    """Frozen C2 recipe from ``configs/round2_v0_1/models.yaml``.

    ``outer_fold_early_stopping: false`` there is what keeps this member legal
    inside the nesting contract.
    """

    def __init__(self, name: str = "c2_raw", kind: str = "raw", params: dict | None = None):
        self.name = name
        self.kind = kind
        self.params = dict(C2_PARAMS if params is None else params)

    def fit_predict(self, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**self.params)
        model.fit(
            catboost_frame(train, self.kind),
            train[target].to_numpy(dtype=float),
            cat_features=["spout_no"],
        )
        return np.asarray(model.predict(catboost_frame(valid, self.kind)), dtype=float)


def r0_members() -> list[Member]:
    """The reproducible reference anchor.

    Not V34_A: the historical member pool is unavailable on this machine, so
    this is the frozen C2 recipe, which was verified to reproduce the V2-era
    registered two-seed mean to eight decimals.
    """
    return [CatBoostMember("c2_raw")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--pool",
        choices=("r0", "r0+time-ebm", "r0+shrink", "r1", "r1-extended"),
        default="r0",
    )
    parser.add_argument("--outer-seed", type=int, default=16061)
    parser.add_argument("--inner-seed", type=int, default=7771)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.pool == "r0":
        members = r0_members()
    else:
        # Imported here so the module graph stays acyclic.
        from .next_phase_members import (
            r0_plus_shrink,
            r0_plus_time_ebm,
            r1_extended_members,
            r1_members,
        )

        builders = {
            "r0+time-ebm": r0_plus_time_ebm,
            "r0+shrink": r0_plus_shrink,
            "r1": r1_members,
            "r1-extended": r1_extended_members,
        }
        members = builders[args.pool]()
    label = args.pool.replace("+", "_plus_")
    output = args.output or Path(f"local/runs/round2-next-phase/{label}-outer-{args.outer_seed}")
    train = load_v2(args.root / "复赛_train", "train", 2754)
    report = evaluate_outer(
        train,
        members,
        outer_seed=args.outer_seed,
        inner_seed=args.inner_seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "outer.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{args.pool} outer seed {report['outer_seed']}  score = {report['package_score']:.6f}")
    for target, entry in report["targets"].items():
        weights = entry["fold_weights"]["0"]
        spread = ", ".join(f"{name}={value:.4f}" for name, value in sorted(weights.items()))
        print(f"  {target:14s} pooled WMAPE = {entry['pooled_wmape']:.8f}")
        print(f"  {'':14s} fold-0 weights: {spread}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
