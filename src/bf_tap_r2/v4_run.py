"""V4 mechanism coarse-screen driver.

The driver reads the frozen V4 budget, materialises the pre-registered 40-unit
screen, and evaluates each unit on seeds 42/3407 and folds 0/1 with the frozen
group-isolated fold vectors.  It never reads model-platform labels, never writes
a submission, and never overwrites completed ledger entries.

The complete B_star process is not present in this checkout.  A reproducible
V34_A development replay is recovered from the private V3.4 OOF/cache trees and
is used as the allowed A anchor.  The P4 residual-on-B_star units are therefore
marked blocked rather than silently replaced with a different base model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import FEATURES, TARGETS
from .metrics import wmape
from .v3_run import load_fold_vector, load_training_frame
from .v4_cross_target import (
    CrossTargetChainRegressor,
    SharedOnlyCrossTargetRegressor,
    SharedPrivateCrossTargetRegressor,
)
from .v4_leaf_estimators import HonestLeafForest, LeafPartitionForest, fit_lightgbm_linear_tree
from .v4_projection_models import (
    CrossFittedResidualProjectionRegressor,
    ProjectionPursuitRegressor,
)
from .v4_smooth_models import (
    LinearControlRegressor,
    PairwiseTensorProductRegressor,
    SmoothAdditiveRegressor,
    TripleTensorProductRegressor,
    fit_ebm_with_explicit_triple,
)

SEEDS = (42, 3407)
COARSE_FOLDS = (0, 1)
TARGET_OTHER = {"tap_iron": "tap_time_len", "tap_time_len": "tap_iron"}

# Pre-declared feature groups.  Indices refer to ``FEATURES``.
PAIRS = ((3, 16), (3, 0), (11, 9), (13, 4))
TRIPLES = ((3, 16, 0), (11, 9, 10))
ALT_PAIRS = ((3, 11), (16, 0), (9, 13), (4, 5))


class _LeafUnit:
    def __init__(self, *, method: str, criterion: str = "squared_error") -> None:
        self.method = str(method)
        self.criterion = str(criterion)

    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_LeafUnit":
        self.model_ = LeafPartitionForest(
            n_estimators=64,
            max_depth=6,
            min_samples_leaf=10,
            criterion=self.criterion,
            random_state=42,
        ).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, x: Any) -> np.ndarray:
        return self.model_.predict(np.asarray(x, dtype=float), method=self.method)


class _LightGBMLinearUnit:
    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_LightGBMLinearUnit":
        self.model_ = fit_lightgbm_linear_tree(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            n_estimators=300,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _HistGBUnit:
    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_HistGBUnit":
        self.model_ = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _HonestUnit:
    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_HonestUnit":
        if groups is None:
            raise ValueError("Honest unit requires group ids")
        self.model_ = HonestLeafForest(
            n_estimators=32,
            max_depth=6,
            min_samples_leaf=10,
            criterion="squared_error",
            random_state=42,
            n_repeats=4,
            structure_fraction=0.5,
        ).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), groups)
        return self

    def predict(self, x: Any) -> np.ndarray:
        return self.model_.predict(np.asarray(x, dtype=float), method="median")


class _SmoothUnit:
    def __init__(self, *, kind: str, pairs: Sequence[Sequence[int]] = (), triples: Sequence[Sequence[int]] = ()) -> None:
        self.kind = str(kind)
        self.pairs = tuple(tuple(int(v) for v in pair) for pair in pairs)
        self.triples = tuple(tuple(int(v) for v in triple) for triple in triples)

    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_SmoothUnit":
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        if self.kind == "additive":
            self.model_ = SmoothAdditiveRegressor(n_knots=5, degree=3, alpha=10.0).fit(x_array, y_array)
        elif self.kind == "pair":
            self.model_ = PairwiseTensorProductRegressor(
                pairs=self.pairs, n_knots=4, degree=3, alpha=10.0
            ).fit(x_array, y_array)
        elif self.kind == "triple":
            self.model_ = TripleTensorProductRegressor(
                pairs=self.pairs, triples=self.triples, n_knots=4, degree=3, alpha=10.0
            ).fit(x_array, y_array)
        elif self.kind == "linear":
            self.model_ = LinearControlRegressor(alpha=10.0).fit(x_array, y_array)
        else:
            raise ValueError(f"Unknown smooth unit kind: {self.kind!r}")
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _EBMTupleUnit:
    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_EBMTupleUnit":
        self.model_ = fit_ebm_with_explicit_triple(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            [3, 16, 0],
            parameters={
                "max_rounds": 1000,
                "early_stopping_rounds": 50,
                "max_bins": 64,
                "max_interaction_bins": 32,
                "learning_rate": 0.05,
                "outer_bags": 4,
                "inner_bags": 0,
                "random_state": 42,
                "n_jobs": 1,
            },
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _EBMInteractionUnit:
    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_EBMInteractionUnit":
        from interpret.glassbox import ExplainableBoostingRegressor

        self.model_ = ExplainableBoostingRegressor(
            interactions=60,
            max_rounds=1000,
            early_stopping_rounds=50,
            max_bins=64,
            max_interaction_bins=32,
            learning_rate=0.05,
            outer_bags=4,
            inner_bags=0,
            random_state=42,
            n_jobs=1,
        ).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _PPRUnit:
    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_PPRUnit":
        self.model_ = ProjectionPursuitRegressor(
            n_components=3, n_knots=5, degree=3, alpha=10.0
        ).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _ResidualPPRUnit:
    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_ResidualPPRUnit":
        base = LinearControlRegressor(alpha=10.0)
        self.model_ = CrossFittedResidualProjectionRegressor(
            base,
            n_splits=5,
            random_state=42,
            projection_factory=lambda: ProjectionPursuitRegressor(
                n_components=3, n_knots=5, degree=3, alpha=10.0
            ),
        ).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), groups=groups)
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _ChainUnit:
    def __init__(self, target: str) -> None:
        self.target = str(target)

    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_ChainUnit":
        if aux is None:
            raise ValueError("Chain unit requires the other target")
        direction = "time_to_iron" if self.target == "tap_iron" else "iron_to_time"
        stage = SmoothAdditiveRegressor(n_knots=4, degree=2, alpha=1.0)
        self.model_ = CrossTargetChainRegressor(
            deepcopy(stage),
            deepcopy(stage),
            direction=direction,
            n_splits=5,
            random_state=42,
        )
        self.model_.fit(
            np.asarray(x, dtype=float),
            np.asarray(aux, dtype=float),
            np.asarray(y, dtype=float),
            groups=groups,
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class _SharedUnit:
    def __init__(self, target: str, *, private: bool) -> None:
        self.target = str(target)
        self.private = bool(private)

    def fit(self, x: Any, y: Any, *, aux: Any = None, groups: Any = None) -> "_SharedUnit":
        if aux is None:
            raise ValueError("Shared cross-target unit requires the other target")
        cls = SharedPrivateCrossTargetRegressor if self.private else SharedOnlyCrossTargetRegressor
        kwargs = {"n_components": 2, "n_knots": 3, "degree": 2}
        if self.private:
            kwargs.update({"shared_alpha": 1.0, "private_alpha": 10.0})
        else:
            kwargs.update({"shared_alpha": 10.0})
        self.model_ = cls(**kwargs)
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        aux_array = np.asarray(aux, dtype=float)
        if self.target == "tap_iron":
            self.model_.fit(x_array, y_array, aux_array)
        else:
            self.model_.fit(x_array, aux_array, y_array)
        return self

    def predict(self, x: Any) -> np.ndarray:
        output = self.model_.predict(np.asarray(x, dtype=float))
        return np.asarray(output[self.target], dtype=float)


def _build_unit(unit: Mapping[str, Any]) -> Any:
    kind = str(unit["estimator_kind"])
    if kind == "leaf":
        return _LeafUnit(method=str(unit["params"]["method"]), criterion=str(unit["params"]["criterion"]))
    if kind == "lightgbm_linear":
        return _LightGBMLinearUnit()
    if kind == "honest":
        return _HonestUnit()
    if kind == "hist_gradient_boosting":
        return _HistGBUnit()
    if kind == "smooth":
        return _SmoothUnit(
            kind=str(unit["params"]["kind"]),
            pairs=unit["params"].get("pairs", ()),
            triples=unit["params"].get("triples", ()),
        )
    if kind == "ebm_tuple":
        return _EBMTupleUnit()
    if kind == "ebm_interaction":
        return _EBMInteractionUnit()
    if kind == "ppr":
        return _PPRUnit()
    if kind == "residual_ppr":
        return _ResidualPPRUnit()
    if kind == "chain":
        return _ChainUnit(str(unit["target"]))
    if kind == "shared":
        return _SharedUnit(str(unit["target"]), private=bool(unit["params"]["private"]))
    raise ValueError(f"Unknown estimator kind: {kind!r}")


def unit_specs() -> list[dict[str, Any]]:
    """Return the pre-registered 40-unit first-round screen."""
    specs: list[dict[str, Any]] = []
    counter = 0

    def add(family: str, target: str, mechanism: str, kind: str,
            params: Mapping[str, Any] | None = None, *, blocked: bool = False,
            blocked_reason: str | None = None) -> None:
        nonlocal counter
        counter += 1
        specs.append({
            "method_id": f"v4-{family}-{counter:03d}",
            "family": family,
            "target": target,
            "mechanism": mechanism,
            "estimator_kind": kind,
            "params": dict(params or {}),
            "blocked": bool(blocked),
            "blocked_reason": blocked_reason,
        })

    for target in TARGETS:
        add("F", target, "F1_leaf_mean", "leaf", {"method": "mean", "criterion": "squared_error"})
        add("F", target, "F2_mixture_median", "leaf", {"method": "distribution_median", "criterion": "squared_error"})
        add("F", target, "F3_median_of_medians", "leaf", {"method": "median_of_medians", "criterion": "squared_error"})
        add("F", target, "F4_absolute_partition_F2", "leaf", {"method": "distribution_median", "criterion": "absolute_error"})
        add("F", target, "F5_honest_leaf", "honest", {})
        add("F", target, "F6_lightgbm_linear_leaf", "lightgbm_linear", {})

    for target in TARGETS:
        add("S", target, "S1_additive_spline", "smooth", {"kind": "additive"})
        add("S", target, "S2_pairwise_tensor", "smooth", {"kind": "pair", "pairs": PAIRS})
        add("S", target, "S3_triple_tensor", "smooth", {"kind": "triple", "pairs": PAIRS, "triples": TRIPLES})
        add("S", target, "S4_ebm_explicit_triple", "ebm_tuple", {})
        add("S", target, "S5_ebm_automatic_pairs", "ebm_interaction", {})
        add("S", target, "S6_alternative_pairwise_tensor", "smooth", {"kind": "pair", "pairs": ALT_PAIRS})

    for target in TARGETS:
        add("P", target, "P1_fixed_linear_coordinate", "smooth", {"kind": "linear"})
        add("P", target, "P2_learned_projection", "ppr", {})
        add("P", target, "P3_nested_linear_residual_projection", "residual_ppr", {})
        add("P", target, "P4_B_star_residual_projection", "blocked", {}, blocked=True,
            blocked_reason="full_B_star_process_not_recovered_in_this_checkout")

    for target in TARGETS:
        add("J", target, "J1_cross_target_chain", "chain", {})
        add("J", target, "J2_independent_target", "hist_gradient_boosting", {})
        add("J", target, "J3_shared_private", "shared", {"private": True})
        add("J", target, "J4_shared_only", "shared", {"private": False})

    if len(specs) != 40:
        raise AssertionError(f"V4 schedule must contain 40 units, got {len(specs)}")
    return specs


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


def _frame_hash(frame: pd.DataFrame, fold_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in frame["sample_id"].astype(str).tolist():
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    numeric = frame[[*FEATURES, *TARGETS]].to_numpy(dtype=float)
    digest.update(np.ascontiguousarray(numeric).tobytes())
    digest.update(",".join(str(int(v)) for v in fold_ids).encode("ascii"))
    return digest.hexdigest()


def _load_groups(root: Path, train: pd.DataFrame, seed: int) -> np.ndarray:
    path = root / "local/runs/round2-v2/comparison-r1" / f"folds-{seed}.csv"
    assignment = pd.read_csv(path, dtype={"group_id": str})
    if assignment.sample_id.duplicated().any() or set(assignment.sample_id) != set(train.sample_id):
        raise ValueError(f"Group assignment identity mismatch for seed {seed}")
    return assignment.set_index("sample_id").loc[train.sample_id, "group_id"].to_numpy(dtype=object)


def _load_a_dev_reference(root: Path, train: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    """Recover V34_A development predictions from the frozen private V3.4 cache."""
    run = root / "local/runs/round2-v3.4-ebm-and-constrained-composition"
    l1_weights = {"tap_iron": 0.5, "tap_time_len": 0.6192377408552066}
    expert_weights = {
        "tap_iron": {
            "v34-s1-ebm_boundary-0016": 0.3397038169347897,
            "v34-s1-ebm_boundary-0020": 0.16029618306521032,
        },
        "tap_time_len": {
            "v34-s1-ebm_boundary-0089": 0.28691289439355067,
            "v34-s1-global_spout_shrink-0136": 0.09384936475124295,
        },
    }
    out: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for target in TARGETS:
            parts = []
            for fold in range(5):
                path = run / "l1-oof-r1" / f"seed-{seed}" / f"l1-oof-fold-{fold}-{target}.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                parts.append(pd.read_csv(path, dtype={"sample_id": "string"}))
            frame = pd.concat(parts, ignore_index=True)
            if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(train.sample_id):
                raise ValueError(f"L1 OOF identity mismatch for seed {seed} target {target}")
            l1 = frame.set_index("sample_id").loc[train.sample_id, target].to_numpy(dtype=float)
            prediction = l1_weights[target] * l1
            for trial_id, weight in expert_weights[target].items():
                path = run / "refine-r1" / f"seed-{seed}" / f"pred-{trial_id}.npy"
                if not path.exists():
                    raise FileNotFoundError(path)
                arr = np.load(path).astype(float)
                if arr.shape != (len(train),):
                    raise ValueError(f"V34_A cache shape mismatch: {path}")
                prediction = prediction + weight * arr
            out[seed][target] = prediction
    return out


def _a_reference_metrics(train: pd.DataFrame, folds_by_seed: Mapping[int, np.ndarray],
                         a_ref: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for seed in SEEDS:
        folds = folds_by_seed[seed]
        payload[str(seed)] = {}
        for fold_ids in (COARSE_FOLDS, (0, 1, 2, 3, 4)):
            mask = np.isin(folds, list(fold_ids))
            target_wmape = {target: float(wmape(train.loc[mask, target], a_ref[seed][target][mask])) for target in TARGETS}
            payload[str(seed)][",".join(map(str, fold_ids))] = {
                "target_wmape": target_wmape,
                "package_score": float(100.0 - 50.0 * sum(target_wmape.values())),
            }
    target_wmape_full = {}
    for target in TARGETS:
        target_wmape_full[target] = float(np.mean([
            float(wmape(train[target], a_ref[seed][target])) for seed in SEEDS
        ]))
    payload["full_five_fold_seed_mean"] = {
        "target_wmape": target_wmape_full,
        "package_score": float(100.0 - 50.0 * sum(target_wmape_full.values())),
    }
    return payload


def _fit_unit(unit: Mapping[str, Any], train: pd.DataFrame, folds: np.ndarray,
              groups: np.ndarray, seed: int, fold: int) -> tuple[np.ndarray, dict[str, Any]]:
    target = str(unit["target"])
    other = TARGET_OTHER[target]
    train_mask = folds != int(fold)
    valid_mask = folds == int(fold)
    x_train = train.loc[train_mask, list(FEATURES)].to_numpy(dtype=float)
    x_valid = train.loc[valid_mask, list(FEATURES)].to_numpy(dtype=float)
    y_train = train.loc[train_mask, target].to_numpy(dtype=float)
    aux_train = train.loc[train_mask, other].to_numpy(dtype=float)
    model = _build_unit(unit)
    started = time.perf_counter()
    model.fit(x_train, y_train, aux=aux_train, groups=groups[train_mask])
    prediction = np.asarray(model.predict(x_valid), dtype=float)
    if prediction.shape != (int(valid_mask.sum()),) or not np.isfinite(prediction).all():
        raise ValueError(f"Unit {unit['method_id']} produced invalid predictions")
    return prediction, {
        "seed": int(seed),
        "fold": int(fold),
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "seconds": float(time.perf_counter() - started),
    }


def _metrics_for_unit(train: pd.DataFrame, folds: np.ndarray, seed: int, target: str,
                      prediction: np.ndarray, a_ref: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    mask = np.isin(folds, list(COARSE_FOLDS))
    actual = train.loc[mask, target].to_numpy(dtype=float)
    candidate = prediction[mask]
    baseline_target = a_ref[seed][target][mask]
    other = TARGET_OTHER[target]
    baseline_other = a_ref[seed][other][mask]
    candidate_wmape = float(wmape(actual, candidate))
    baseline_target_wmape = float(wmape(actual, baseline_target))
    baseline_other_wmape = float(wmape(train.loc[mask, other], baseline_other))
    baseline_package = float(100.0 - 50.0 * (baseline_target_wmape + baseline_other_wmape))
    candidate_package = float(100.0 - 50.0 * (candidate_wmape + baseline_other_wmape))
    blend = 0.5 * candidate + 0.5 * baseline_target
    blend_wmape = float(wmape(actual, blend))
    blend_package = float(100.0 - 50.0 * (blend_wmape + baseline_other_wmape))
    by_spout: dict[str, dict[str, float]] = {}
    spouts = np.asarray(train.loc[mask, "spout_no"], dtype=int)
    for spout in sorted(set(spouts.tolist())):
        selector = spouts == spout
        by_spout[str(spout)] = {
            "candidate_wmape": float(wmape(actual[selector], candidate[selector])),
            "baseline_wmape": float(wmape(actual[selector], baseline_target[selector])),
        }
    return {
        "target": target,
        "candidate_wmape": candidate_wmape,
        "baseline_target_wmape": baseline_target_wmape,
        "baseline_other_wmape": baseline_other_wmape,
        "baseline_package_score": baseline_package,
        "candidate_package_score": candidate_package,
        "package_delta_single": float(candidate_package - baseline_package),
        "equal_blend_wmape": blend_wmape,
        "equal_blend_package_score": blend_package,
        "package_delta_equal_blend": float(blend_package - baseline_package),
        "by_spout": by_spout,
    }


def _known_completed(ledger: Path) -> set[tuple[str, int]]:
    known: set[tuple[str, int]] = set()
    if not ledger.exists():
        return known
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            known.add((str(event["method_id"]), int(event["seed"])))
    return known


def _append_ledger(ledger: Path, event: Mapping[str, Any]) -> None:
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        handle.flush()


def _environment_fingerprint(root: Path, train: pd.DataFrame, folds_by_seed: Mapping[int, np.ndarray]) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "interpret", "lightgbm", "catboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        head = "no-git-head"
    data_hash = _frame_hash(train, (0, 1, 2, 3, 4))
    return {
        "created": "2026-09-24",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": head,
        "packages": packages,
        "data_hash": data_hash,
        "fold_hashes": {str(seed): _sha256_bytes(np.asarray(folds_by_seed[seed], dtype=np.int64).tobytes()) for seed in SEEDS},
        "n_train": int(len(train)),
        "feature_order": list(FEATURES),
    }


def _write_summary(output: Path, events: Sequence[Mapping[str, Any]]) -> None:
    if not events:
        return
    rows = []
    for event in events:
        metrics = event.get("metrics")
        if not metrics:
            continue
        rows.append({
            "method_id": event["method_id"],
            "family": event["family"],
            "target": event["target"],
            "mechanism": event["mechanism"],
            "seed": event["seed"],
            "status": event.get("status"),
            "candidate_wmape": metrics["candidate_wmape"],
            "baseline_target_wmape": metrics["baseline_target_wmape"],
            "package_delta_single": metrics["package_delta_single"],
            "equal_blend_wmape": metrics["equal_blend_wmape"],
            "package_delta_equal_blend": metrics["package_delta_equal_blend"],
            "n_fits_external": event.get("n_fits_external"),
        })
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(list(events), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run(root: Path, output: Path | None = None, *, limit: int | None = None,
        family: str | None = None, seed: int | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else (
        root / "local/runs/round2-v4-mechanism-search/coarse-r1"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "oof").mkdir(exist_ok=True)
    config = yaml.safe_load((root / "configs/round2_v4/mechanism_search.yaml").read_text(encoding="utf-8"))
    if config["status"] not in {"PLAN_ONLY_NOT_EXECUTED", "COARSE_SCREEN_COMPLETED_NO_EXTENSION"}:
        raise ValueError(f"Unexpected V4 mechanism config status: {config['status']!r}")

    train = load_training_frame(root)
    folds_by_seed = {seed_value: load_fold_vector(root, train, seed_value) for seed_value in SEEDS}
    groups_by_seed = {seed_value: _load_groups(root, train, seed_value) for seed_value in SEEDS}
    a_ref = _load_a_dev_reference(root, train)
    reference_dir = output / "reference"
    reference_dir.mkdir(exist_ok=True)
    reference_metrics = _a_reference_metrics(train, folds_by_seed, a_ref)
    (reference_dir / "a_dev_reference.json").write_text(
        json.dumps(reference_metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    np.savez_compressed(
        reference_dir / "a_dev_oof.npz",
        **{f"{seed_value}_{target}": a_ref[seed_value][target] for seed_value in SEEDS for target in TARGETS},
    )
    environment = _environment_fingerprint(root, train, folds_by_seed)
    (output / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    specs = unit_specs()
    if family:
        wanted = str(family).upper()
        specs = [unit for unit in specs if str(unit["family"]).upper() == wanted]
    if seed is not None:
        seed_values = (int(seed),)
    else:
        seed_values = SEEDS
    if limit is not None:
        specs = specs[: max(0, int(limit))]

    ledger = output / "fit_ledger.jsonl"
    known = _known_completed(ledger)
    all_events: list[dict[str, Any]] = []
    for seed_value in seed_values:
        folds = folds_by_seed[seed_value]
        groups = groups_by_seed[seed_value]
        for unit in specs:
            if unit["blocked"]:
                event = {
                    "event": "complete",
                    "status": "blocked",
                    "method_id": unit["method_id"],
                    "family": unit["family"],
                    "target": unit["target"],
                    "mechanism": unit["mechanism"],
                    "seed": int(seed_value),
                    "reason": unit["blocked_reason"],
                    "n_fits_external": 0,
                }
                if (str(unit["method_id"]), int(seed_value)) not in known:
                    _append_ledger(ledger, event)
                    known.add((str(unit["method_id"]), int(seed_value)))
                all_events.append(event)
                continue
            key = (str(unit["method_id"]), int(seed_value))
            if key in known:
                continue
            started = time.perf_counter()
            prediction = np.full(len(train), np.nan, dtype=float)
            fold_meta: list[dict[str, Any]] = []
            fold_wmape: dict[str, float] = {}
            for fold in COARSE_FOLDS:
                valid_mask = folds == int(fold)
                pred, meta = _fit_unit(unit, train, folds, groups, seed_value, fold)
                prediction[valid_mask] = pred
                fold_wmape[str(fold)] = float(wmape(train.loc[valid_mask, unit["target"]], pred))
                fold_meta.append(meta)
            if not np.isfinite(prediction[np.isin(folds, list(COARSE_FOLDS))]).all():
                raise RuntimeError(f"Unit {unit['method_id']} did not cover all coarse rows")
            np.save(output / "oof" / f"{unit['method_id']}-seed{seed_value}.npy", prediction)
            metrics = _metrics_for_unit(train, folds, seed_value, str(unit["target"]), prediction, a_ref)
            event = {
                "event": "complete",
                "status": "available",
                "method_id": unit["method_id"],
                "family": unit["family"],
                "target": unit["target"],
                "mechanism": unit["mechanism"],
                "seed": int(seed_value),
                "params": unit["params"],
                "unit_hash": _json_hash(unit),
                "data_hash": environment["data_hash"],
                "fold_hash": environment["fold_hashes"][str(int(seed_value))],
                "folds": list(COARSE_FOLDS),
                "fold_wmape": fold_wmape,
                "fold_meta": fold_meta,
                "n_fits_external": int(len(COARSE_FOLDS)),
                "metrics": metrics,
                "seconds_total": float(time.perf_counter() - started),
            }
            _append_ledger(ledger, event)
            known.add(key)
            all_events.append(event)
            print(json.dumps({
                "seed": int(seed_value),
                "method_id": unit["method_id"],
                "target": unit["target"],
                "mechanism": unit["mechanism"],
                "candidate_wmape": metrics["candidate_wmape"],
                "package_delta_single": metrics["package_delta_single"],
                "package_delta_equal_blend": metrics["package_delta_equal_blend"],
            }, ensure_ascii=False), flush=True)
    _write_summary(output, all_events)
    return {"output": str(output), "environment": environment, "n_events": len(all_events)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V4 40-unit coarse mechanism screen")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(args.root, args.output, limit=args.limit, family=args.family, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
