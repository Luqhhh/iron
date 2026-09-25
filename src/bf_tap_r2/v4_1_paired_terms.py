"""V4.1 paired-term controls with explicit categorical spout handling.

The module implements the C line of ``V4_1_TASKBOOK.md``:

* C1 re-runs the V4 S5 automatic-interaction EBM with the registered
  ``spout_no`` categorical contract;
* C2 restores the selected V3.4 strong EBM parent recipe, including its target
  coordinate, category input, bags, bins and training budget;
* C3 extracts the pair set fitted by C2 on the current training part and
  refits with that pair set made explicit;
* C4 keeps the exact C3 pair set and budget and adds at most two triples
  selected only from training-part term contributions.

No function in this module reads validation labels for selection.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from .data import FEATURES, TARGETS
from .v3_4_models import V34Regressor
from .v3_4_sampler import sample_v34
from .v4_run import TRIPLES

__all__ = [
    "AUTO_PAIR_EBM_PARAMS",
    "PAIR_CONTROL_METHODS",
    "build_c_units",
    "fit_c2_c3_c4_on_training_part",
    "fit_c1_category_ebm",
    "fit_strong_ebm_parent",
    "select_triples_from_training_importances",
    "v34_trial_map",
]

CATEGORICAL = "spout_no"
V34_PARENT_TRIAL = {
    "tap_iron": "v34-s1-ebm_boundary-0016",
    "tap_time_len": "v34-s1-ebm_boundary-0089",
}
PAIR_CONTROL_METHODS = ("C2", "C3", "C4")
TIME_COARSE_INTERACTION_BINS = 16
C_METHODS = ("C0", "C1", "C2", "C3", "C4")

# Exactly the V4 S5 parameters, with the only C1 change being the explicit
# nominal ``spout_no`` column and InterpretML feature_types contract.
AUTO_PAIR_EBM_PARAMS = {
    "interactions": 60,
    "max_rounds": 1000,
    "early_stopping_rounds": 50,
    "max_bins": 64,
    "max_interaction_bins": 32,
    "learning_rate": 0.05,
    "outer_bags": 4,
    "inner_bags": 0,
    "random_state": 42,
    "n_jobs": 1,
}


def _categorical_feature_types() -> list[str]:
    return ["continuous"] * len(FEATURES) + ["nominal"]


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [*FEATURES, CATEGORICAL]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Training frame is missing columns: {missing}")
    x = frame.loc[:, columns].copy()
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("C-line feature frame must be finite")
    return x


def v34_trial_map(root: Path) -> dict[str, dict[str, Any]]:
    """Return the deterministic V3.4 trial catalogue keyed by ``trial_id``."""
    return {str(trial["trial_id"]): deepcopy(trial) for trial in sample_v34(root)}


def fit_c1_category_ebm(train: pd.DataFrame, valid: pd.DataFrame,
                        target: str) -> tuple[np.ndarray, dict[str, Any]]:
    """C1: V4 S5 EBM with correct nominal spout feature typing."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    x_train, x_valid = _feature_frame(train), _feature_frame(valid)
    model = ExplainableBoostingRegressor(
        feature_types=_categorical_feature_types(), **AUTO_PAIR_EBM_PARAMS
    )
    model.fit(x_train, np.asarray(train[target], dtype=float))
    prediction = np.asarray(model.predict(x_valid), dtype=float)
    terms = [tuple(int(v) for v in term) for term in getattr(model, "term_features_", ())]
    return prediction, {
        "estimator": "ebm_automatic_pairs_with_category",
        "category_feature": CATEGORICAL,
        "feature_types": _categorical_feature_types(),
        "actual_pair_terms": [list(term) for term in terms if len(term) == 2],
        "actual_term_count": len(terms),
        "parameters": dict(AUTO_PAIR_EBM_PARAMS),
    }


def fit_strong_ebm_parent(train: pd.DataFrame, valid: pd.DataFrame,
                          target: str, trials: Mapping[str, Mapping[str, Any]]) -> tuple[V34Regressor, np.ndarray, dict[str, Any]]:
    """C2: selected V3.4 strong EBM parent, fitted on the current training part."""
    trial_id = V34_PARENT_TRIAL[target]
    if trial_id not in trials:
        raise KeyError(f"Missing V3.4 parent trial spec: {trial_id}")
    model = V34Regressor(deepcopy(dict(trials[trial_id])))
    model.fit(train, np.asarray(train[target], dtype=float))
    prediction = np.asarray(model.predict(valid), dtype=float)
    term_features = [tuple(int(v) for v in term) for term in getattr(model.impl.estimator_, "term_features_", ())]
    return model, prediction, {
        "estimator": "v34_strong_ebm_parent",
        "source_trial_id": trial_id,
        "category_feature": CATEGORICAL,
        "target_transform": model.impl.target_transform,
        "bag_protocol": "group-safe-bags-v1",
        "parameters": deepcopy(model.impl.parameters),
        "actual_pair_terms": [list(term) for term in term_features if len(term) == 2],
        "actual_term_count": len(term_features),
    }


def _pair_terms(model: V34Regressor) -> list[tuple[int, ...]]:
    estimator = model.impl.estimator_
    pairs = [tuple(int(v) for v in term) for term in getattr(estimator, "term_features_", ()) if len(term) == 2]
    if not pairs:
        raise ValueError("C2 strong EBM produced no pair terms; C3/C4 controls are not defined")
    return pairs


def _pair_term_importances(model: V34Regressor) -> list[tuple[tuple[int, ...], float]]:
    estimator = model.impl.estimator_
    terms = [tuple(int(v) for v in term) for term in getattr(estimator, "term_features_", ())]
    importances = np.asarray(estimator.term_importances(), dtype=float)
    if len(importances) != len(terms):
        raise ValueError("Term features and importances have inconsistent lengths")
    pairs = [(term, float(value)) for term, value in zip(terms, importances) if len(term) == 2]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def _single_term_importances(model: V34Regressor) -> list[tuple[int, float]]:
    estimator = model.impl.estimator_
    terms = [tuple(int(v) for v in term) for term in getattr(estimator, "term_features_", ())]
    importances = np.asarray(estimator.term_importances(), dtype=float)
    singles = [(int(term[0]), float(value)) for term, value in zip(terms, importances) if len(term) == 1]
    singles.sort(key=lambda item: (-item[1], item[0]))
    return singles


def select_triples_from_training_importances(
    model: V34Regressor,
    *,
    max_triples: int = 2,
    n_pairs: int = 3,
    n_singles: int = 3,
    fallback_triples: Sequence[Sequence[int]] = TRIPLES,
) -> list[tuple[int, ...]]:
    """Select at most ``max_triples`` triples using training-part terms only.

    Candidates are the Cartesian product of the highest-contribution pair
    endpoints and the highest-contribution single features.  The score is the
    product of the two training-part importances.  Declared business triples
    are appended only if the contribution rule yields fewer than
    ``max_triples`` candidates.
    """
    paired = _pair_term_importances(model)
    singles = _single_term_importances(model)
    scored: list[tuple[float, tuple[int, ...]]] = []
    for pair, pair_importance in paired[: max(1, int(n_pairs))]:
        for feature, feature_importance in singles[: max(1, int(n_singles))]:
            if int(feature) in pair:
                continue
            triple = tuple(sorted((*pair, int(feature))))
            if len(set(triple)) != 3:
                continue
            scored.append((pair_importance * feature_importance, triple))
    # Deduplicate while keeping the highest score.
    best: dict[tuple[int, ...], float] = {}
    for score, triple in scored:
        if score > best.get(triple, -np.inf):
            best[triple] = score
    ordered = sorted(((score, triple) for triple, score in best.items()),
                     key=lambda item: (-item[0], item[1]))
    selected = [triple for _, triple in ordered[: max(0, int(max_triples))]]
    if len(selected) < int(max_triples):
        for raw in fallback_triples:
            triple = tuple(sorted(int(v) for v in raw))
            if len(set(triple)) != 3 or min(triple) < 0 or max(triple) >= len(FEATURES) + 1:
                continue
            if triple not in selected:
                selected.append(triple)
            if len(selected) >= int(max_triples):
                break
    return selected[: max(0, int(max_triples))]


def fit_c2_c3_c4_on_training_part(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    trials: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fit the shared C2/C3/C4 controls on one outer training/validation split."""
    c2_model, c2_pred, c2_meta = fit_strong_ebm_parent(train, valid, target, trials)
    pair_terms = _pair_terms(c2_model)
    # The time target inherited a 64-bin multidimensional partition.  With
    # explicit high-order terms that combination is computationally
    # intractable on the current CPU budget.  C3 and C4 therefore share a
    # coarser interaction bin count for that target and report the loss to C2.
    coarse_bins = TIME_COARSE_INTERACTION_BINS if target == "tap_time_len" else None
    max_triples = 1 if target == "tap_time_len" else 2
    selected_triples = select_triples_from_training_importances(c2_model, max_triples=max_triples)
    base_trial = deepcopy(dict(trials[V34_PARENT_TRIAL[target]]))

    c3_trial = deepcopy(base_trial)
    c3_params = {**dict(c3_trial["parameters"]), "interactions": [list(pair) for pair in pair_terms]}
    if coarse_bins is not None:
        c3_params["max_interaction_bins"] = int(coarse_bins)
    c3_trial["parameters"] = c3_params
    c3_model = V34Regressor(c3_trial)
    c3_model.fit(train, np.asarray(train[target], dtype=float))
    c3_pred = np.asarray(c3_model.predict(valid), dtype=float)
    c3_terms = [tuple(int(v) for v in term) for term in getattr(c3_model.impl.estimator_, "term_features_", ())]
    c3_pairs = [term for term in c3_terms if len(term) == 2]
    if set(c3_pairs) != set(pair_terms):
        raise AssertionError(
            f"C3 explicit-pair control did not preserve the C2 pair set: "
            f"expected={sorted(pair_terms)!r} actual={sorted(c3_pairs)!r}"
        )

    c4_trial = deepcopy(base_trial)
    c4_params = {
        **dict(c4_trial["parameters"]),
        "interactions": [list(pair) for pair in pair_terms] + [list(triple) for triple in selected_triples],
    }
    if coarse_bins is not None:
        c4_params["max_interaction_bins"] = int(coarse_bins)
    c4_trial["parameters"] = c4_params
    c4_model = V34Regressor(c4_trial)
    c4_model.fit(train, np.asarray(train[target], dtype=float))
    c4_pred = np.asarray(c4_model.predict(valid), dtype=float)
    c4_terms = [tuple(int(v) for v in term) for term in getattr(c4_model.impl.estimator_, "term_features_", ())]
    c4_pairs = [term for term in c4_terms if len(term) == 2]
    c4_triples = [term for term in c4_terms if len(term) == 3]
    if set(c4_pairs) != set(pair_terms):
        raise AssertionError(
            f"C4 did not preserve the C3 pair set: expected={sorted(pair_terms)!r} actual={sorted(c4_pairs)!r}"
        )
    if not set(selected_triples).issubset(set(c4_triples)):
        raise AssertionError(
            f"C4 did not preserve selected triples: expected={selected_triples!r} actual={c4_triples!r}"
        )
    return {
        "C2": {"prediction": c2_pred, "meta": c2_meta},
        "C3": {
            "prediction": c3_pred,
            "meta": {
                "estimator": "v34_strong_ebm_explicit_pairs",
                "source_trial_id": V34_PARENT_TRIAL[target],
                "pair_terms": [list(term) for term in pair_terms],
                "actual_pair_terms": [list(term) for term in c3_pairs],
                "pair_set_preserved": True,
                "category_feature": CATEGORICAL,
                "target_transform": c3_model.impl.target_transform,
                "bag_protocol": "group-safe-bags-v1",
                "parameters": deepcopy(c3_model.impl.parameters),
                "coarsened_interaction_bins_to": coarse_bins,
            },
        },
        "C4": {
            "prediction": c4_pred,
            "meta": {
                "estimator": "v34_strong_ebm_explicit_pairs_plus_triples",
                "source_trial_id": V34_PARENT_TRIAL[target],
                "pair_terms": [list(term) for term in pair_terms],
                "selected_triples": [list(term) for term in selected_triples],
                "actual_pair_terms": [list(term) for term in c4_pairs],
                "actual_triple_terms": [list(term) for term in c4_triples],
                "pair_set_preserved": True,
                "triple_terms_preserved": True,
                "category_feature": CATEGORICAL,
                "target_transform": c4_model.impl.target_transform,
                "bag_protocol": "group-safe-bags-v1",
                "parameters": deepcopy(c4_model.impl.parameters),
                "coarsened_interaction_bins_to": coarse_bins,
            },
        },
    }


def build_c_units() -> list[dict[str, Any]]:
    """Return the C0-C4 pre-registered unit table (10 method/target units)."""
    units: list[dict[str, Any]] = []
    for target in TARGETS:
        units.append({"method_id": "C0", "target": target, "kind": "reuse_v4_s5",
                      "description": "V4 S5 21-numeric control, reuse coarse-r2"})
        units.append({"method_id": "C1", "target": target, "kind": "ebm_category_spout",
                      "description": "same S5 + nominal spout_no"})
        units.append({"method_id": "C2", "target": target, "kind": "v34_parent",
                      "description": "restore selected strong EBM parent recipe"})
        units.append({"method_id": "C3", "target": target, "kind": "explicit_pairs",
                      "description": "explicit replay of C2 fitted pair set"})
        units.append({"method_id": "C4", "target": target, "kind": "explicit_pairs_triples",
                      "description": "C3 + at most two training-part triples"})
    if len(units) != 10:
        raise AssertionError(f"C line must contain 10 method/target units, got {len(units)}")
    return units
