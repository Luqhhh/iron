"""V4.1 strong-increment C2/C3/C4 controlled screen.

This module implements the first-round V4.1 budget from the 2026-09-25 task
book:

* C2 replays the exact selected V3.6 EBM parent on each outer training part;
* C3 makes the C2-fitted pair set explicit and fits the same parent budget;
* C4 keeps that C3 pair set and adds at most two triples chosen only from
  group-safe internal-training validation.

The outer validation frame is never passed to any selection rule.  The B36
baseline and the fixed-slot parent come from the frozen V3.6 development replay
cache; replacement models are fitted only on the outer training rows.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import TARGETS
from .metrics import wmape
from .v3_4_bags import group_safe_inner_folds
from .v3_6_models import V36Regressor
from .v3_run import load_fold_vector, load_training_frame
from .v4_1_models import (
    V41EBMRegressor,
    extract_pair_terms_by_name,
    fitted_feature_names,
    resolve_interactions,
    select_triples_from_parent_model,
    structure_audit,
)
from .v4_1_reference import V36DevelopmentReference

__all__ = [
    "apply_slot_replacement",
    "fit_c2_exact_parent",
    "fit_c3_explicit_pairs",
    "fit_c4_explicit_pairs_triples",
    "run_c234_screen",
    "select_c4_triple_count",
]

LOGGER = logging.getLogger(__name__)
DEFAULT_SEEDS = (42, 3407)
DEFAULT_FOLDS = (0, 1)
DEFAULT_INNER_FOLDS = 3
DEFAULT_INNER_SEED = 41017
MAX_TRIPLE_CANDIDATES = 8
MAX_SELECTED_TRIPLES = 2
TARGET_OTHER = {"tap_iron": "tap_time_len", "tap_time_len": "tap_iron"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_slot_replacement(
    baseline: Sequence[float] | np.ndarray,
    parent: Sequence[float] | np.ndarray,
    replacement: Sequence[float] | np.ndarray,
    weight: float,
    *,
    clip_nonnegative: bool = False,
) -> np.ndarray:
    """Apply ``baseline + weight * (replacement - parent)`` in original units."""
    b = np.asarray(baseline, dtype=float)
    p = np.asarray(parent, dtype=float)
    r = np.asarray(replacement, dtype=float)
    if b.shape != p.shape or b.shape != r.shape:
        raise ValueError("baseline, parent and replacement must have equal shape")
    if b.ndim != 1 or not len(b) or not np.isfinite(b).all() or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("slot replacement inputs must be finite one-dimensional vectors")
    w = float(weight)
    if not np.isfinite(w) or w < 0.0:
        raise ValueError("slot weight must be finite and nonnegative")
    out = b + w * (r - p)
    if clip_nonnegative:
        out = np.maximum(out, 0.0)
    return out


def _copy_trial_with_interactions(trial: Mapping[str, Any], terms: Sequence[Sequence[int]]) -> dict[str, Any]:
    copied = deepcopy(dict(trial))
    params = dict(copied.get("parameters", {}))
    params["interactions"] = [list(term) for term in terms]
    copied["parameters"] = params
    return copied


def _fit_parent(train_frame: pd.DataFrame, target: str, trial: Mapping[str, Any]) -> V36Regressor:
    model = V36Regressor(deepcopy(dict(trial)))
    y = np.asarray(train_frame[target], dtype=float)
    model.fit(train_frame.reset_index(drop=True), y)
    return model


def _fit_v41(
    train_frame: pd.DataFrame,
    target: str,
    trial: Mapping[str, Any],
    interactions: Sequence[Sequence[int]],
) -> V41EBMRegressor:
    model = V41EBMRegressor(_copy_trial_with_interactions(trial, interactions))
    y = np.asarray(train_frame[target], dtype=float)
    model.fit(train_frame.reset_index(drop=True), y)
    return model


def fit_c2_exact_parent(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target: str,
    trial: Mapping[str, Any],
) -> tuple[V36Regressor, np.ndarray, dict[str, Any]]:
    """Fit the exact selected V3.6 parent and predict the outer validation rows."""
    started = time.perf_counter()
    model = _fit_parent(train_frame, target, trial)
    prediction = np.asarray(model.predict(valid_frame.reset_index(drop=True)), dtype=float)
    if prediction.shape != (len(valid_frame),) or not np.isfinite(prediction).all():
        raise ValueError("C2 parent produced invalid validation predictions")
    return model, prediction, {
        "method": "C2",
        "trial_id": str(trial.get("trial_id")),
        "seconds": float(time.perf_counter() - started),
        "n_train": int(len(train_frame)),
        "n_valid": int(len(valid_frame)),
        "model_family": "v36_exact_parent_replay",
    }


def fit_c3_explicit_pairs(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target: str,
    trial: Mapping[str, Any],
    pair_indices: Sequence[Sequence[int]],
) -> tuple[V41EBMRegressor, np.ndarray, dict[str, Any]]:
    """Fit C3 as the explicit-pair version of the current C2 parent.

    ``pair_indices`` must be the terms extracted from the *current training
    part's* C2 model.  We deliberately do not accept a global pair list.
    """
    if not pair_indices:
        raise ValueError("C3 requires at least one explicit pair from the current C2 parent")
    started = time.perf_counter()
    model = _fit_v41(train_frame, target, trial, pair_indices)
    prediction = np.asarray(model.predict(valid_frame.reset_index(drop=True)), dtype=float)
    if prediction.shape != (len(valid_frame),) or not np.isfinite(prediction).all():
        raise ValueError("C3 model produced invalid validation predictions")
    audit = structure_audit(model, expected_pairs=pair_indices, expected_triples=[])
    return model, prediction, {
        "method": "C3",
        "trial_id": str(trial.get("trial_id")),
        "seconds": float(time.perf_counter() - started),
        "n_train": int(len(train_frame)),
        "n_valid": int(len(valid_frame)),
        "n_explicit_pairs": len(pair_indices),
        "pair_terms": [list(term) for term in pair_indices],
        **audit,
    }


def fit_c4_explicit_pairs_triples(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target: str,
    trial: Mapping[str, Any],
    pair_indices: Sequence[Sequence[int]],
    triple_indices: Sequence[Sequence[int]],
) -> tuple[V41EBMRegressor, np.ndarray, dict[str, Any]]:
    """Fit C4 with the exact C3 pair set plus selected training triples."""
    if len(triple_indices) > MAX_SELECTED_TRIPLES:
        raise ValueError("C4 may add at most two triples")
    interactions = [tuple(term) for term in pair_indices] + [tuple(term) for term in triple_indices]
    if len({tuple(sorted(term)) for term in interactions}) != len(interactions):
        raise ValueError("C4 interaction set contains duplicates")
    started = time.perf_counter()
    model = _fit_v41(train_frame, target, trial, interactions)
    prediction = np.asarray(model.predict(valid_frame.reset_index(drop=True)), dtype=float)
    if prediction.shape != (len(valid_frame),) or not np.isfinite(prediction).all():
        raise ValueError("C4 model produced invalid validation predictions")
    audit = structure_audit(model, expected_pairs=pair_indices, expected_triples=triple_indices)
    return model, prediction, {
        "method": "C4",
        "trial_id": str(trial.get("trial_id")),
        "seconds": float(time.perf_counter() - started),
        "n_train": int(len(train_frame)),
        "n_valid": int(len(valid_frame)),
        "n_explicit_pairs": len(pair_indices),
        "n_selected_triples": len(triple_indices),
        "pair_terms": [list(term) for term in pair_indices],
        "selected_triples": [list(term) for term in triple_indices],
        **audit,
    }


def select_c4_triple_count(
    train_frame: pd.DataFrame,
    target: str,
    trial: Mapping[str, Any],
    *,
    n_inner: int = DEFAULT_INNER_FOLDS,
    inner_seed: int = DEFAULT_INNER_SEED,
    max_candidates: int = MAX_TRIPLE_CANDIDATES,
) -> dict[str, Any]:
    """Select k in {0,1,2} using group-safe internal training validation.

    For every inner training subset we fit a fresh parent on that subset only,
    generate its own pair set and shared-edge triple candidates, fit explicit
    candidate models, and score the held-out inner rows in original units.  The
    outer validation frame is not accepted by this function.
    """
    folds = group_safe_inner_folds(
        train_frame.reset_index(drop=True),
        n_splits=int(n_inner),
        seed=int(inner_seed),
    )["fold"]
    unique_folds = sorted(int(v) for v in np.unique(folds))
    if len(unique_folds) < 2:
        raise ValueError("C4 internal selection needs at least two inner folds")
    per_k: dict[int, list[float]] = {0: []}
    per_fold_records: list[dict[str, Any]] = []
    n_parent_fits = 0
    n_candidate_fits = 0
    candidate_counts: list[int] = []
    for fold in unique_folds:
        inner_train = train_frame.loc[folds != fold].reset_index(drop=True)
        inner_valid = train_frame.loc[folds == fold].reset_index(drop=True)
        if inner_train.empty or inner_valid.empty:
            raise ValueError("C4 inner fold has an empty side")
        parent = _fit_parent(inner_train, target, trial)
        n_parent_fits += 1
        feature_names = fitted_feature_names(parent)
        pair_names = extract_pair_terms_by_name(parent, feature_names)
        if not pair_names:
            raise ValueError("C4 inner parent did not produce any pair terms")
        pair_indices = [
            tuple(int(feature_names.index(name)) for name in term)
            for term in pair_names
        ]
        candidate_indices = select_triples_from_parent_model(
            parent, max_candidates=int(max_candidates)
        )
        candidate_counts.append(len(candidate_indices))
        record: dict[str, Any] = {
            "inner_fold": int(fold),
            "n_train": int(len(inner_train)),
            "n_valid": int(len(inner_valid)),
            "pair_terms": [list(term) for term in pair_indices],
            "candidate_triples": [list(term) for term in candidate_indices],
            "inner_wmape": {},
        }
        actual = np.asarray(inner_valid[target], dtype=float)
        max_k = min(MAX_SELECTED_TRIPLES, len(candidate_indices))
        for k in range(0, max_k + 1):
            interactions = list(pair_indices) + [tuple(term) for term in candidate_indices[:k]]
            model = _fit_v41(inner_train, target, trial, interactions)
            n_candidate_fits += 1
            prediction = np.asarray(model.predict(inner_valid.reset_index(drop=True)), dtype=float)
            value = float(wmape(actual, prediction))
            per_k.setdefault(k, []).append(value)
            record["inner_wmape"][str(k)] = value
        per_fold_records.append(record)
    means = {
        k: float(np.mean(values))
        for k, values in per_k.items()
        if values and len(values) == len(unique_folds)
    }
    if 0 not in means:
        raise ValueError("C4 internal selection did not evaluate k=0")
    # Smaller k wins ties, as required by the task book.
    selected_k = min(sorted(means), key=lambda k: (means[k], k))
    return {
        "selected_k": int(selected_k),
        "inner_means_wmape": {str(k): float(v) for k, v in sorted(means.items())},
        "inner_folds": [int(v) for v in unique_folds],
        "inner_seed": int(inner_seed),
        "n_inner_parent_fits": int(n_parent_fits),
        "n_inner_candidate_fits": int(n_candidate_fits),
        "candidate_counts": [int(v) for v in candidate_counts],
        "records": per_fold_records,
        "selection_uses_outer_labels": False,
    }


def _package_metrics(
    actual_target: np.ndarray,
    baseline_target: np.ndarray,
    candidate_target: np.ndarray,
    actual_other: np.ndarray,
    baseline_other: np.ndarray,
) -> dict[str, float]:
    baseline_target_wmape = float(wmape(actual_target, baseline_target))
    baseline_other_wmape = float(wmape(actual_other, baseline_other))
    candidate_target_wmape = float(wmape(actual_target, candidate_target))
    baseline_score = 100.0 - 50.0 * (baseline_target_wmape + baseline_other_wmape)
    candidate_score = 100.0 - 50.0 * (candidate_target_wmape + baseline_other_wmape)
    return {
        "baseline_target_wmape": baseline_target_wmape,
        "baseline_other_wmape": baseline_other_wmape,
        "baseline_package_score": float(baseline_score),
        "candidate_target_wmape": float(candidate_target_wmape),
        "candidate_package_score": float(candidate_score),
        "package_delta_single": float(candidate_score - baseline_score),
    }


def _private_output(root: Path, output: Path) -> None:
    output = output.resolve()
    if not output.is_relative_to((root / "local/runs").resolve()):
        raise ValueError("V4.1 outputs must remain beneath local/runs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4.1 output: {output}")
    output.mkdir(parents=True, exist_ok=False)


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe_json(payload), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()



def run_c234_screen(
    root: Path | str = ".",
    output: Path | str | None = None,
    *,
    targets: Sequence[str] = TARGETS,
    methods: Sequence[str] = ("C2", "C3", "C4"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    folds: Sequence[int] = DEFAULT_FOLDS,
    n_inner: int = DEFAULT_INNER_FOLDS,
    inner_seed: int = DEFAULT_INNER_SEED,
    clip_nonnegative: bool = True,
    allow_intractable_high_dim_c4: bool = False,
) -> dict[str, Any]:
    """Run the first-round C2/C3/C4 six target-recipe screen.

    The six target recipes are two targets times three variants.  With two
    seeds and two outer folds this yields 24 outer evaluations, plus the real
    internal C4 parent/candidate fits recorded in the manifest.  The function
    never writes a submission, never uploads, and never reads platform labels.
    """
    root_path = Path(root).resolve()
    output_path = Path(output).resolve() if output is not None else (
        root_path / "local/runs/round2-v4.1-strong-increment/strong-increment-c234-r1"
    )
    _private_output(root_path, output_path)
    train = load_training_frame(root_path)
    reference = V36DevelopmentReference(root_path, train)
    target_values = [str(v) for v in targets]
    method_values = [str(v) for v in methods]
    if not set(target_values).issubset(set(TARGETS)) or not target_values:
        raise ValueError(f"Unknown or empty V4.1 target selection: {target_values}")
    if not set(method_values).issubset({"C2", "C3", "C4"}) or not method_values:
        raise ValueError(f"Unknown or empty V4.1 method selection: {method_values}")
    seed_values = [int(v) for v in seeds]
    fold_values = [int(v) for v in folds]
    fold_vectors = {seed: load_fold_vector(root_path, train, seed) for seed in seed_values}
    for seed, vector in fold_vectors.items():
        if not np.isin(fold_values, np.unique(vector)).all():
            raise ValueError(f"Requested folds are absent for seed {seed}: {fold_values}")

    ledger_path = output_path / "fit_ledger.jsonl"
    started_all = time.perf_counter()
    fold_records: list[dict[str, Any]] = []
    blocked_units: list[dict[str, Any]] = []
    fit_counts = {
        "outer_c2_fits": 0,
        "outer_c3_fits": 0,
        "outer_c4_fits": 0,
        "inner_c4_parent_fits": 0,
        "inner_c4_candidate_fits": 0,
    }
    array_template = {"candidate": np.full(len(train), np.nan, dtype=float),
                      "replacement": np.full(len(train), np.nan, dtype=float)}
    predictions = {
        method: {target: {seed: {key: value.copy() for key, value in array_template.items()}
                          for seed in seed_values}
                 for target in TARGETS}
        for method in ("C2", "C3", "C4")
    }
    selection_records: dict[tuple[int, str, int], dict[str, Any]] = {}

    for seed in seed_values:
        fold_vector = fold_vectors[seed]
        for target in target_values:
            other = TARGET_OTHER[target]
            trial = reference.trials[reference.slot_expert(target)]
            weight = reference.slot_weight(target)
            for fold in fold_values:
                mask = fold_vector == fold
                train_mask = ~mask
                if not mask.any() or not train_mask.any():
                    raise ValueError(f"Empty train/valid side for seed={seed} fold={fold}")
                outer_train = train.loc[train_mask].reset_index(drop=True)
                outer_valid = train.loc[mask].reset_index(drop=True)
                actual_target = np.asarray(outer_valid[target], dtype=float)
                actual_other = np.asarray(outer_valid[other], dtype=float)
                baseline_target = reference.baseline(seed, target)[mask]
                baseline_other = reference.baseline(seed, other)[mask]
                parent_cache = reference.parent(seed, target)[mask]

                c2_model, c2_pred, c2_meta = fit_c2_exact_parent(
                    outer_train, outer_valid, target, trial
                )
                fit_counts["outer_c2_fits"] += 1
                parent_max_abs = float(np.max(np.abs(c2_pred - parent_cache)))
                if parent_max_abs > 1e-6:
                    raise AssertionError(
                        f"C2 did not replay cached parent for seed={seed} fold={fold} "
                        f"target={target}: max_abs={parent_max_abs!r}"
                    )
                pair_names = extract_pair_terms_by_name(c2_model)
                feature_names = fitted_feature_names(c2_model)
                pair_indices = [
                    tuple(int(feature_names.index(name)) for name in term)
                    for term in pair_names
                ]
                if not pair_indices:
                    raise ValueError(f"C2 parent produced no pair terms for target {target}")

                need_c4 = "C4" in method_values
                selection: dict[str, Any] = {}
                selected_k: int | None = None
                c3_fitted = False
                full_candidates = (
                    select_triples_from_parent_model(c2_model, max_candidates=MAX_TRIPLE_CANDIDATES)
                    if need_c4 else []
                )
                max_interaction_bins = int(trial.get("parameters", {}).get("max_interaction_bins", 0))
                c4_blocked = bool(
                    not need_c4
                    or (
                        target == "tap_time_len"
                        and max_interaction_bins >= 64
                        and full_candidates
                        and not allow_intractable_high_dim_c4
                    )
                )
                if need_c4 and not c4_blocked:
                    selection = select_c4_triple_count(
                        outer_train, target, trial, n_inner=int(n_inner), inner_seed=int(inner_seed)
                    )
                    fit_counts["inner_c4_parent_fits"] += int(selection["n_inner_parent_fits"])
                    fit_counts["inner_c4_candidate_fits"] += int(selection["n_inner_candidate_fits"])
                    selected_k = int(selection["selected_k"])
                    selected_triples = [tuple(term) for term in full_candidates[:selected_k]]
                    if len(selected_triples) != selected_k:
                        selected_k = len(selected_triples)
                        selection = dict(selection)
                        selection["selected_k"] = int(selected_k)
                        selection["full_frame_candidate_fallback"] = True

                    c3_model, c3_pred, c3_meta = fit_c3_explicit_pairs(
                        outer_train, outer_valid, target, trial, pair_indices
                    )
                    c3_fitted = True
                    fit_counts["outer_c3_fits"] += 1
                    if selected_k == 0:
                        c4_pred = np.asarray(c3_pred, dtype=float).copy()
                        c4_meta: dict[str, Any] = {
                            "method": "C4",
                            "trial_id": str(trial.get("trial_id")),
                            "seconds": 0.0,
                            "n_train": int(len(outer_train)),
                            "n_valid": int(len(outer_valid)),
                            "n_explicit_pairs": len(pair_indices),
                            "n_selected_triples": 0,
                            "pair_terms": [list(term) for term in pair_indices],
                            "selected_triples": [],
                            "pair_set_preserved": True,
                            "triple_set_preserved": True,
                            "c4_equals_c3_control": True,
                        }
                    else:
                        _, c4_pred, c4_meta = fit_c4_explicit_pairs_triples(
                            outer_train, outer_valid, target, trial, pair_indices, selected_triples
                        )
                        fit_counts["outer_c4_fits"] += 1
                        c4_meta["c4_equals_c3_control"] = False
                elif need_c4:
                    # The exact time parent uses max_interaction_bins=64.  Native
                    # EBM multidimensional partitioning on an added triple is
                    # not operable at that budget on this environment.  Do not
                    # silently coarsen the bins: record the path as blocked.
                    selection = {
                        "blocked": True,
                        "reason": "exact_parent_max_interaction_bins_64_triple_partition_not_operable",
                        "selected_k": None,
                        "candidate_triples": [list(term) for term in full_candidates],
                        "selection_uses_outer_labels": False,
                    }
                    selected_k = None
                    c4_pred = None
                    c4_meta = {
                        "method": "C4",
                        "trial_id": str(trial.get("trial_id")),
                        "status": "blocked",
                        "reason": selection["reason"],
                        "n_explicit_pairs": len(pair_indices),
                        "selected_triples": [],
                        "candidate_triples": [list(term) for term in full_candidates],
                        "bins_changed": False,
                    }
                    c3_model, c3_pred, c3_meta = fit_c3_explicit_pairs(
                        outer_train, outer_valid, target, trial, pair_indices
                    )
                    c3_fitted = True
                    fit_counts["outer_c3_fits"] += 1
                    blocked_units.append({
                        "seed": int(seed),
                        "fold": int(fold),
                        "target": target,
                        "method_id": "C4",
                        "status": "blocked",
                        "reason": selection["reason"],
                        "candidate_triples": [list(term) for term in full_candidates],
                    })
                    _append_jsonl(ledger_path, {
                        "event": "complete",
                        "status": "blocked",
                        "evidence_level": "DESCRIPTIVE_FROZEN_V36_DEVELOPMENT_SCREEN",
                        "method_id": "C4",
                        "family": "C234",
                        "target": target,
                        "other_target": other,
                        "seed": int(seed),
                        "fold": int(fold),
                        "n_train": int(len(outer_train)),
                        "n_valid": int(len(outer_valid)),
                        "fixed_slot": reference.slot_expert(target),
                        "fixed_slot_weight": float(weight),
                        "method_meta": c4_meta,
                        "fit_accounting_this_outer_unit": {
                            "outer_c2": 1,
                            "outer_c3": 1,
                            "outer_c4": 0,
                            "inner_c4_parent": 0,
                            "inner_c4_candidate": 0,
                            "selected_k": None,
                        },
                    })
                if "C3" in method_values and not c3_fitted:
                    c3_model, c3_pred, c3_meta = fit_c3_explicit_pairs(
                        outer_train, outer_valid, target, trial, pair_indices
                    )
                    c3_fitted = True
                    fit_counts["outer_c3_fits"] += 1
                selection_records[(seed, target, fold)] = selection
                available_methods = [
                    method for method in ("C2", "C3", "C4")
                    if method in method_values and not (method == "C4" and c4_blocked)
                ]
                replacements = {"C2": c2_pred}
                if "C3" in method_values:
                    replacements["C3"] = c3_pred
                if "C4" in method_values and not c4_blocked:
                    replacements["C4"] = np.asarray(c4_pred, dtype=float)
                for method in available_methods:
                    replacement = np.asarray(replacements[method], dtype=float)
                    candidate_target = apply_slot_replacement(
                        baseline_target, parent_cache, replacement, weight,
                        clip_nonnegative=bool(clip_nonnegative),
                    )
                    package = _package_metrics(
                        actual_target, baseline_target, candidate_target, actual_other, baseline_other
                    )
                    predictions[method][target][seed]["candidate"][mask] = candidate_target
                    predictions[method][target][seed]["replacement"][mask] = replacement
                    if method == "C2":
                        method_meta = c2_meta
                    elif method == "C3":
                        method_meta = c3_meta
                    else:
                        method_meta = c4_meta
                    record = {
                        "event": "complete",
                        "status": "available",
                        "evidence_level": "DESCRIPTIVE_FROZEN_V36_DEVELOPMENT_SCREEN",
                        "method_id": method,
                        "family": "C234",
                        "target": target,
                        "other_target": other,
                        "seed": int(seed),
                        "fold": int(fold),
                        "n_train": int(len(outer_train)),
                        "n_valid": int(len(outer_valid)),
                        "fixed_slot": reference.slot_expert(target),
                        "fixed_slot_weight": float(weight),
                        "clip_nonnegative": bool(clip_nonnegative),
                        "replacement_wmape": float(wmape(actual_target, replacement)),
                        "parent_replay_max_abs": parent_max_abs,
                        "candidate_target_wmape": package["candidate_target_wmape"],
                        "baseline_target_wmape": package["baseline_target_wmape"],
                        "baseline_other_wmape": package["baseline_other_wmape"],
                        "baseline_package_score": package["baseline_package_score"],
                        "candidate_package_score": package["candidate_package_score"],
                        "package_delta_single": package["package_delta_single"],
                        "method_meta": method_meta,
                        "fit_accounting_this_outer_unit": {
                            "outer_c2": 1,
                            "outer_c3": 1 if c3_fitted else 0,
                            "outer_c4": 1 if ("C4" in method_values and not c4_blocked and selected_k is not None and selected_k > 0) else 0,
                            "inner_c4_parent": int(selection.get("n_inner_parent_fits", 0)) if ("C4" in method_values and not c4_blocked) else 0,
                            "inner_c4_candidate": int(selection.get("n_inner_candidate_fits", 0)) if ("C4" in method_values and not c4_blocked) else 0,
                            "selected_k": selected_k if "C4" in method_values else None,
                        },
                    }
                    if method == "C4":
                        record["c4_selection_NOT_OUTER_VALIDATION"] = selection
                    _append_jsonl(ledger_path, record)
                    fold_records.append(record)
                LOGGER.info(
                    "C234 complete seed=%s fold=%s target=%s selected_k=%s",
                    seed, fold, target, selected_k,
                )

    # ------------------------------------------------------------------
    # Pool each seed over the requested outer folds with sum(abs)/sum(abs).
    # ------------------------------------------------------------------
    pooled_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        mask_all = np.isin(fold_vectors[seed], fold_values)
        if not mask_all.any():
            raise ValueError(f"No pooled rows for seed {seed}")
        for target in TARGETS:
            other = TARGET_OTHER[target]
            actual_target = np.asarray(train.loc[mask_all, target], dtype=float)
            actual_other = np.asarray(train.loc[mask_all, other], dtype=float)
            baseline_target = reference.baseline(seed, target)[mask_all]
            baseline_other = reference.baseline(seed, other)[mask_all]
            for method in ("C2", "C3", "C4"):
                candidate = predictions[method][target][seed]["candidate"][mask_all]
                replacement = predictions[method][target][seed]["replacement"][mask_all]
                if not (np.isfinite(candidate).all() and np.isfinite(replacement).all()):
                    # A blocked path has no pooled prediction and is reported in
                    # ``blocked_units`` instead of being silently treated as C3.
                    continue
                package = _package_metrics(
                    actual_target, baseline_target, candidate, actual_other, baseline_other
                )
                pooled_rows.append({
                    "seed": int(seed),
                    "target": target,
                    "method_id": method,
                    "rows": int(mask_all.sum()),
                    "replacement_wmape": float(wmape(actual_target, replacement)),
                    **package,
                })
    pooled_frame = pd.DataFrame(pooled_rows)
    pooled_frame.to_csv(output_path / "pooled_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for method in ("C2", "C3", "C4"):
            rows = pooled_frame[(pooled_frame.target == target) & (pooled_frame.method_id == method)]
            if rows.empty:
                continue
            if len(rows) != len(seed_values):
                raise ValueError(f"Pooled metrics are incomplete for target={target} method={method}")
            deltas = rows.set_index("seed")["package_delta_single"].to_dict()
            summary_rows.append({
                "target": target,
                "method_id": method,
                "mean_package_delta": float(np.mean([deltas[int(s)] for s in seed_values])),
                "min_package_delta": float(np.min([deltas[int(s)] for s in seed_values])),
                "positive_seed_count": int(sum(1 for s in seed_values if deltas[int(s)] > 0.0)),
                "all_seeds_positive": bool(all(deltas[int(s)] > 0.0 for s in seed_values)),
                "mean_replacement_wmape": float(rows["replacement_wmape"].mean()),
                "mean_baseline_target_wmape": float(rows["baseline_target_wmape"].mean()),
                "mean_candidate_target_wmape": float(rows["candidate_target_wmape"].mean()),
                "seed": {str(int(s)): float(deltas[int(s)]) for s in seed_values},
            })
    summary_frame = pd.DataFrame(summary_rows)

    def _delta_for(target: str, method: str, seed: int) -> float:
        row = pooled_frame[(pooled_frame.target == target) & (pooled_frame.method_id == method)
                           & (pooled_frame.seed == seed)]
        return float(row["package_delta_single"].iloc[0])

    # Pairwise control differences are reported in package-delta points.
    bridge_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for left, right in (("C3", "C2"), ("C4", "C3")):
            available = set(pooled_frame[pooled_frame.target == target]["method_id"]) 
            if left not in available or right not in available:
                continue
            for seed in seed_values:
                bridge_rows.append({
                    "target": target,
                    "comparison": f"{left}-{right}",
                    "seed": int(seed),
                    "package_delta_difference": float(_delta_for(target, left, seed) - _delta_for(target, right, seed)),
                })
    bridge_frame = pd.DataFrame(bridge_rows)
    bridge_frame.to_csv(output_path / "pairwise_control_deltas.csv", index=False)
    summary_frame.to_csv(output_path / "summary.csv", index=False)
    (output_path / "summary.json").write_text(
        json.dumps(_safe_json(summary_rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    coarse_candidates = summary_frame[
        summary_frame["method_id"].isin(["C3", "C4"]) & summary_frame["all_seeds_positive"]
    ]
    manifest = {
        "status": "C234_DESCRIPTIVE_DEVELOPMENT_SCREEN_NOT_PROMOTION",
        "evidence_level": "DESCRIPTIVE_FROZEN_V36_DEVELOPMENT_SCREEN",
        "seeds": seed_values,
        "folds": fold_values,
        "n_inner": int(n_inner),
        "inner_seed": int(inner_seed),
        "clip_nonnegative": bool(clip_nonnegative),
        "methods": method_values,
        "targets": target_values,
        "blocked_units": blocked_units,
        "fixed_slots": {target: reference.slot_expert(target) for target in TARGETS},
        "fixed_slot_weights": {target: reference.slot_weight(target) for target in TARGETS},
        "reference": reference.meta(),
        "n_requested_target_method_outer_units": int(
            len(seed_values) * len(fold_values) * len(target_values) * len(method_values)
        ),
        "n_outer_model_evaluations": int(sum(1 for record in fold_records if record.get("status") == "available")),
        "actual_fit_counts": {key: int(value) for key, value in fit_counts.items()},
        "coarse_rule": "C3 or C4 fixed-slot package delta positive in BOTH seeds promotes to full five-fold development coverage",
        "all_seeds_positive_c3_or_c4": coarse_candidates[["target", "method_id", "mean_package_delta"]].to_dict("records"),
        "platform_uploads": 0,
        "submission_packages": 0,
        "source_hashes": {
            "train_samples": _sha256_file(root_path / "复赛_train/train_samples.csv"),
            "train_features": _sha256_file(root_path / "复赛_train/train_features.csv"),
            "v36_summary": _sha256_file(reference.summary_path),
            "v36_ledger": _sha256_file(reference.ledger_path),
        },
        "elapsed_seconds": float(time.perf_counter() - started_all),
        "warning": (
            "This is a frozen V36 development replay, not a new independent outer split. "
            "No result here is a platform-score forecast or a submission candidate by itself."
        ),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(_safe_json(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"output": str(output_path), **manifest}
