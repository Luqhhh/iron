"""V4.3 frozen strong-base (V36 composition) loading and nested cross-fitting.

The parent is the frozen V36 package composition that produced the current
user-reported platform best.  Its time side is

    parent = w0 * A_dev + w1 * O-0057 + w2 * D-0048
    A_dev  = c0 * L1    + c1 * ebm_boundary-0089 + c2 * global_spout_shrink-0136

where ``L1`` is the V3.4 LP-simplex blend of the frozen five-member time pool.
This module rebuilds that composition from the private development caches and
refits every time-side member inside an outer training part so that residual
targets are fully cross-fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import TARGETS
from .metrics import wmape
from .v3_1_fusion import lp_simplex_weights
from .v3_1_models import fit_with_inner_early_stop
from .v3_4_bags import group_safe_inner_folds
from .v3_4_models import V34Regressor
from .v3_6_models import V36Regressor
from .v3_6_reference import A_EXPERTS, A_TARGET_WEIGHTS, build_reference_vectors
from .v3_local_search import TrialRegressor
from .v3_run import load_fold_vector, load_training_frame, read_complete_records, trial_details
from .v4_3_residual import TARGET, load_v43_config

PARENT_DEVELOPMENT_SCORE = 96.20376256899247


@dataclass(frozen=True)
class MemberSpec:
    """One time-side member of the parent composition and its effective weight."""

    name: str
    group: str
    weight: float


def parent_time_spec(root: Path | str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read and freeze the parent composition for the in-scope target."""
    root = Path(root)
    summary_path = root / str(config["parent"]["summary"])
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comp = summary[str(config["parent"]["summary_key"])]
    target = str(config["parent"]["target_in_scope"])
    recorded = comp["targets"][target]
    weights = [float(value) for value in recorded["weights"]]
    experts = [str(value) for value in recorded["selected_experts"]]
    frozen = config["parent"]["time"]
    if [float(v) for v in frozen["weights"]] != weights:
        raise ValueError("V4.3 parent time weights drifted from the frozen config")
    if [str(v) for v in frozen["experts"]] != experts:
        raise ValueError("V4.3 parent time experts drifted from the frozen config")
    if [float(v) for v in frozen["a_dev_weights"]] != [float(v) for v in A_TARGET_WEIGHTS[target]]:
        raise ValueError("V4.3 A_dev weights drifted from the recorded V3.6 reference")
    if [str(v) for v in frozen["a_dev_experts"]] != [str(v) for v in A_EXPERTS[target]]:
        raise ValueError("V4.3 A_dev experts drifted from the recorded V3.6 reference")
    a_weights = [float(value) for value in frozen["a_dev_weights"]]
    a_experts = [str(value) for value in frozen["a_dev_experts"]]
    return {
        "target": target,
        "package_weights": weights,
        "package_experts": experts,
        "a_weights": a_weights,
        "a_experts": a_experts,
        "l1_pool": [str(value) for value in frozen["l1_pool"]],
        "l1_early_stop_inner_seed": int(frozen["l1_early_stop_inner_seed"]),
        "complete_development_root": root / str(config["parent"]["development_predictions"]),
    }


def member_specs(config: Mapping[str, Any]) -> list[MemberSpec]:
    """Effective member weights of the parent composition (L1 pool shares w0*c0)."""
    frozen = config["parent"]["time"]
    weights = [float(value) for value in frozen["weights"]]
    a_weights = [float(value) for value in frozen["a_dev_weights"]]
    a_experts = [str(value) for value in frozen["a_dev_experts"]]
    specs = [
        MemberSpec(name=str(name), group="l1_pool", weight=weights[0] * a_weights[0])
        for name in frozen["l1_pool"]
    ]
    specs.append(MemberSpec(name=a_experts[0], group="a_expert", weight=weights[0] * a_weights[1]))
    specs.append(MemberSpec(name=a_experts[1], group="a_expert", weight=weights[0] * a_weights[2]))
    specs.append(MemberSpec(name=str(frozen["experts"][0]), group="parent_expert", weight=weights[1]))
    specs.append(MemberSpec(name=str(frozen["experts"][1]), group="parent_expert", weight=weights[2]))
    total = sum(spec.weight for spec in specs if spec.group != "l1_pool")
    if abs(total - (1.0 - weights[0] * a_weights[0])) > 1e-12:
        raise AssertionError("V4.3 member weights do not reconstruct the parent composition")
    return specs


def l1_group_weight(config: Mapping[str, Any]) -> float:
    frozen = config["parent"]["time"]
    return float(frozen["weights"][0]) * float(frozen["a_dev_weights"][0])


def _member_trial(root: Path, config: Mapping[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    """Resolve a member name to its frozen trial definition and source family."""
    frozen = config["parent"]["time"]
    if name in {str(value) for value in frozen["l1_pool"]}:
        family = "v31" if name.startswith("v31-") else "v3"
    elif name in {str(value) for value in frozen["a_dev_experts"]}:
        family = "v34"
    else:
        family = "v36"
    ledger = root / str(config["parent"]["member_trial_sources"][family])
    records = read_complete_records(ledger)
    for record in records:
        if str(record.get("trial_id")) == name:
            trial = trial_details(record) if family == "v3" else dict(record["trial"])
            return family, trial
    raise KeyError(f"Missing V4.3 member trial {name} in {ledger}")


def fit_member(root: Path | str, config: Mapping[str, Any], name: str,
               frame: pd.DataFrame, target_values: np.ndarray) -> Any:
    """Fit one frozen parent member on ``frame`` and return the fitted model."""
    root = Path(root)
    family, trial = _member_trial(root, config, name)
    y = np.asarray(target_values, dtype=float)
    if y.shape != (len(frame),) or not np.isfinite(y).all():
        raise ValueError(f"Invalid member fit target for {name}")
    if family == "v31":
        model, _meta = fit_with_inner_early_stop(
            trial, frame, TARGET, inner_seed=int(config["parent"]["time"]["l1_early_stop_inner_seed"])
        )
        return model
    if family == "v3":
        model = TrialRegressor(trial)
    elif family == "v34":
        model = V34Regressor(trial)
    elif family == "v36":
        model = V36Regressor(trial)
    else:  # pragma: no cover - guarded by _member_trial
        raise ValueError(f"Unsupported member family: {family}")
    model.fit(frame, y)
    return model


def predict_member(model: Any, frame: pd.DataFrame, name: str) -> np.ndarray:
    values = np.asarray(model.predict(frame), dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError(f"Invalid member prediction for {name}")
    return values


def compose_parent(config: Mapping[str, Any], *, l1_pool_predictions: Mapping[str, np.ndarray],
                   l1_weights: Sequence[float], a_expert_predictions: Sequence[np.ndarray],
                   parent_expert_predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Rebuild the frozen parent time prediction from member predictions."""
    frozen = config["parent"]["time"]
    pool = [str(value) for value in frozen["l1_pool"]]
    l1_weights_array = np.asarray(l1_weights, dtype=float)
    if l1_weights_array.shape != (len(pool),):
        raise ValueError("V4.3 L1 weight vector does not match the frozen pool")
    if abs(float(l1_weights_array.sum()) - 1.0) > 1e-7 or (l1_weights_array < 0).any():
        raise ValueError("V4.3 L1 weights must be a nonnegative simplex vector")
    columns = np.column_stack([np.asarray(l1_pool_predictions[name], dtype=float) for name in pool])
    l1 = columns @ l1_weights_array
    a_weights = [float(value) for value in frozen["a_dev_weights"]]
    if len(a_expert_predictions) != 2 or len(parent_expert_predictions) != 2:
        raise ValueError("V4.3 expects exactly two A-dev experts and two parent experts")
    a_dev = a_weights[0] * l1
    for weight, values in zip(a_weights[1:], a_expert_predictions):
        a_dev = a_dev + weight * np.asarray(values, dtype=float)
    package_weights = [float(value) for value in frozen["weights"]]
    parent = package_weights[0] * a_dev
    for weight, values in zip(package_weights[1:], parent_expert_predictions):
        parent = parent + weight * np.asarray(values, dtype=float)
    if not np.isfinite(parent).all():
        raise ValueError("V4.3 parent composition produced nonfinite values")
    return parent


def recorded_target_composition(root: Path | str, config: Mapping[str, Any],
                                target: str) -> tuple[list[float], list[str]]:
    """Recorded package weights and experts of the frozen parent for one target."""
    import json

    root = Path(root)
    summary = json.loads((root / str(config["parent"]["summary"])).read_text(encoding="utf-8"))
    comp = summary[str(config["parent"]["summary_key"])]
    if target not in comp["targets"]:
        raise ValueError(f"Parent composition has no target {target}")
    weights = [float(value) for value in comp["targets"][target]["weights"]]
    experts = [str(value) for value in comp["targets"][target]["selected_experts"]]
    return weights, experts


def load_recorded_parent_oof(root: Path | str, train: pd.DataFrame, config: Mapping[str, Any],
                             seed: int, target: str = TARGET) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild the recorded parent development OOF vector for one split seed."""
    root = Path(root)
    spec = parent_time_spec(root, config)
    weights, experts = recorded_target_composition(root, config, target)
    reference = build_reference_vectors(root, train)
    a_dev = np.asarray(reference["A"][target][str(seed)], dtype=float)
    member_vectors = {"A_dev": a_dev}
    values = weights[0] * a_dev
    for weight, name in zip(weights[1:], experts):
        path = spec["complete_development_root"] / f"seed-{int(seed)}" / f"pred-{name}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        vector = np.load(path).astype(float, copy=False)
        if vector.shape != (len(train),) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid recorded parent member vector: {path}")
        member_vectors[name] = vector
        values = values + float(weight) * vector
    return values, member_vectors


def parent_development_score(root: Path | str, train: pd.DataFrame,
                             config: Mapping[str, Any]) -> dict[str, Any]:
    """Score the recorded parent development composition on both split seeds."""
    target_wmape: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        target_wmape[target] = {}
        for seed in (42, 3407):
            parent, _members = load_recorded_parent_oof(root, train, config, seed, target)
            target_wmape[target][str(seed)] = float(
                wmape(train[target].to_numpy(dtype=float), parent)
            )
    seed_package = {
        seed: float(100.0 - 100.0 * np.mean([target_wmape[target][seed] for target in TARGETS]))
        for seed in ("42", "3407")
    }
    mean_package = float(
        100.0 - 100.0 * np.mean([np.mean(list(values.values())) for values in target_wmape.values()])
    )
    return {
        "per_seed_target_wmape": target_wmape,
        "per_seed_package_score": seed_package,
        "mean_package_score": mean_package,
        "recorded_mean_package_score": PARENT_DEVELOPMENT_SCORE,
    }


def nested_parent_oof(
    root: Path | str,
    config: Mapping[str, Any],
    train: pd.DataFrame,
    *,
    seed: int,
    fold_id: int,
) -> dict[str, Any]:
    """Cross-fit the parent inside one outer training part.

    Every member is refitted on two thirds of the outer training part and scored
    on the held-out third, so the residual target for each training row uses a
    prediction produced without that row.  Fold ``fold_id`` never enters any fit.
    """
    root = Path(root)
    outer_folds = load_fold_vector(root, train, int(seed))
    training_mask = outer_folds != int(fold_id)
    training = train.loc[training_mask].reset_index(drop=True)
    inner_folds = int(config["protocol"]["inner_folds"])
    inner_seed = int(config["protocol"]["inner_seed_base"]) + int(seed) * 10 + int(fold_id)
    folded = group_safe_inner_folds(training, n_splits=inner_folds, seed=inner_seed)
    fold_vector = np.asarray(folded["fold"], dtype=int)
    specs = member_specs(config)
    pool = [str(value) for value in config["parent"]["time"]["l1_pool"]]
    y_training = training[TARGET].to_numpy(dtype=float)

    member_oof = {spec.name: np.full(len(training), np.nan, dtype=float) for spec in specs}
    fit_count = 0
    for inner_fold in range(inner_folds):
        validation_mask = fold_vector == inner_fold
        if not validation_mask.any() or validation_mask.all():
            raise ValueError("V4.3 inner fold has an empty side")
        sub_train = training.loc[~validation_mask].reset_index(drop=True)
        sub_valid = training.loc[validation_mask].reset_index(drop=True)
        y_sub = sub_train[TARGET].to_numpy(dtype=float)
        for spec in specs:
            model = fit_member(root, config, spec.name, sub_train, y_sub)
            values = predict_member(model, sub_valid, spec.name)
            member_oof[spec.name][validation_mask] = values
            fit_count += 1

    incomplete = [name for name, values in member_oof.items() if not np.isfinite(values).all()]
    if incomplete:
        raise ValueError(f"V4.3 nested member OOF incomplete: {incomplete}")

    pool_matrix = np.column_stack([member_oof[name] for name in pool])
    l1_fit = lp_simplex_weights({"inner": y_training}, {"inner": pool_matrix})
    l1_weights = np.asarray(l1_fit["weights"], dtype=float)
    a_experts = [str(value) for value in config["parent"]["time"]["a_dev_experts"]]
    parent_experts = [str(value) for value in config["parent"]["time"]["experts"]]
    parent_oof = compose_parent(
        config,
        l1_pool_predictions=member_oof,
        l1_weights=l1_weights,
        a_expert_predictions=[member_oof[name] for name in a_experts],
        parent_expert_predictions=[member_oof[name] for name in parent_experts],
    )
    return {
        "parent_oof": parent_oof,
        "member_oof": member_oof,
        "l1_weights": l1_weights,
        "l1_objective": float(l1_fit["objective"]),
        "inner_fold": fold_vector,
        "inner_seed": int(inner_seed),
        "inner_fold_hash": str(folded["inner_fold_hash"]),
        "inner_group_hash": str(folded["group_hash"]),
        "fit_count": int(fit_count),
        "training_mask": training_mask,
    }


__all__ = [
    "MemberSpec",
    "PARENT_DEVELOPMENT_SCORE",
    "compose_parent",
    "fit_member",
    "l1_group_weight",
    "load_recorded_parent_oof",
    "member_specs",
    "nested_parent_oof",
    "parent_development_score",
    "parent_time_spec",
    "predict_member",
]
