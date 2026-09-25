"""V4.1 nested residual protocol and E-line runner.

The protocol follows the taskbook pseudocode exactly: every base-of-fold used
to train a residual corrector is generated strictly inside the current outer
training part, and alpha is selected only on meta-fold predictions from inside
that outer training part.  The outer validation labels are touched only after
all fitting and alpha selection is complete.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .metrics import wmape
from .splits import make_folds
from .v3_run import load_fold_vector, load_training_frame
from .v4_1_reference import FrozenAReferenceFactory
from .v4_1_residuals import (
    ConstantMedianResidual,
    LADSplineResidual,
    LinearLeafResidual,
    LocalWeightedMedianResidual,
    exact_l1_alpha,
)

__all__ = [
    "E_METHODS",
    "build_or_load_nested_cache",
    "run_nested_residuals",
]


E_METHODS = ("E0", "E1", "E2", "E3")
STATE_COLUMNS = ("B_tap_iron", "B_tap_time_len", "disagreement_iron", "disagreement_time")


def _private_output(root: Path, output: Path) -> None:
    if not output.resolve().is_relative_to((root / "local/runs").resolve()):
        raise ValueError("V4.1 outputs must remain beneath local/runs")
    output.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        handle.flush()


def _state(bundle: Mapping[str, Any]) -> np.ndarray:
    return np.column_stack([
        np.asarray(bundle["predictions"]["tap_iron"], dtype=float),
        np.asarray(bundle["predictions"]["tap_time_len"], dtype=float),
        np.asarray(bundle["disagreement"]["tap_iron"], dtype=float),
        np.asarray(bundle["disagreement"]["tap_time_len"], dtype=float),
    ])


def _assemble_oof_indices(length: int, u_idx: np.ndarray, w_idx: np.ndarray,
                          u_values: np.ndarray, w_values: np.ndarray) -> np.ndarray:
    output = np.full(length, np.nan, dtype=float)
    output[u_idx] = u_values
    output[w_idx] = w_values
    if not np.isfinite(output).all():
        raise ValueError("Nested OOF assembly left uncovered rows")
    return output


def build_or_load_nested_cache(
    root: Path,
    train: pd.DataFrame,
    folds: np.ndarray,
    seed: int,
    outer_fold: int,
    factory: FrozenAReferenceFactory,
    *,
    cache_path: Path,
    n_meta: int = 2,
    n_inner: int = 2,
) -> dict[str, np.ndarray]:
    """Build (or load) one outer-fold nested base cache.

    ``n_meta`` and ``n_inner`` are deliberately fixed at the small values
    recorded by the runner; they are data-independent protocol constants, not
    a searched hyperparameter.
    """
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]) for key in loaded.files}
    if int(n_meta) < 2 or int(n_inner) < 2:
        raise ValueError("Nested protocol needs at least two meta and inner folds")

    outer_train_mask = folds != int(outer_fold)
    outer_valid_mask = folds == int(outer_fold)
    T = train.loc[outer_train_mask].reset_index(drop=True)
    V = train.loc[outer_valid_mask].reset_index(drop=True)
    if len(T) == 0 or len(V) == 0:
        raise ValueError("Empty outer training or validation part")

    meta_assignment = make_folds(T, seed=int(seed) + 1000, n_splits=int(n_meta))
    meta = meta_assignment.set_index("sample_id").loc[T.sample_id, "fold"].to_numpy(dtype=int)
    if set(meta) != set(range(int(n_meta))):
        raise ValueError("Invalid nested meta-fold assignment")

    cache: dict[str, np.ndarray] = {}
    b_u_by_meta: dict[int, dict[str, np.ndarray]] = {}
    b_w_by_meta: dict[int, dict[str, np.ndarray]] = {}
    oof_u_by_meta: dict[int, dict[str, np.ndarray]] = {}

    for k in range(int(n_meta)):
        u_idx = np.where(meta == k)[0]
        w_idx = np.where(meta != k)[0]
        U = T.iloc[u_idx].reset_index(drop=True)
        W = T.iloc[w_idx].reset_index(drop=True)
        cache[f"meta{k}_U_idx"] = u_idx.astype(np.int64)
        cache[f"meta{k}_W_idx"] = w_idx.astype(np.int64)

        inner_assignment = make_folds(U, seed=int(seed) + 2000 + k, n_splits=int(n_inner))
        inner = inner_assignment.set_index("sample_id").loc[U.sample_id, "fold"].to_numpy(dtype=int)
        if set(inner) != set(range(int(n_inner))):
            raise ValueError(f"Invalid inner-fold assignment for meta {k}")

        oof_predictions = {target: np.full(len(U), np.nan, dtype=float) for target in TARGETS}
        oof_state = np.full((len(U), 4), np.nan, dtype=float)
        for inner_fold in range(int(n_inner)):
            fit_mask = inner != inner_fold
            query_mask = inner == inner_fold
            fit_frame = U.loc[fit_mask].reset_index(drop=True)
            query_frame = U.loc[query_mask].reset_index(drop=True)
            bundle = factory.fit_predict(fit_frame, query_frame)
            for target in TARGETS:
                oof_predictions[target][query_mask] = np.asarray(bundle["predictions"][target], dtype=float)
            oof_state[query_mask] = _state(bundle)
        if any(not np.isfinite(values).all() for values in oof_predictions.values()) or not np.isfinite(oof_state).all():
            raise ValueError("Inner-fold base OOF assembly failed")
        oof_u_by_meta[k] = {"predictions": oof_predictions, "state": oof_state}

        # Complete B_U model fitted only on U; predicts meta validation W.
        b_u = factory.fit_predict(U, W)
        b_u_by_meta[k] = {"predictions": {t: np.asarray(b_u["predictions"][t], dtype=float) for t in TARGETS},
                          "state": _state(b_u)}

        # Complete B_W model fitted only on W; supplies the T-level OOF for U rows.
        b_w = factory.fit_predict(W, U)
        b_w_by_meta[k] = {"predictions": {t: np.asarray(b_w["predictions"][t], dtype=float) for t in TARGETS},
                          "state": _state(b_w)}

    # Assemble base OOF over the whole outer training part.  For each meta
    # fold, rows in U receive B_W(U) and rows in W receive B_U(W); these are
    # strictly out-of-fold base predictions inside T.
    base_oof_pred = {target: np.full(len(T), np.nan, dtype=float) for target in TARGETS}
    base_oof_state = np.full((len(T), 4), np.nan, dtype=float)
    for k in range(int(n_meta)):
        u_idx = cache[f"meta{k}_U_idx"]
        w_idx = cache[f"meta{k}_W_idx"]
        for target in TARGETS:
            base_oof_pred[target][u_idx] = b_w_by_meta[k]["predictions"][target]
            base_oof_pred[target][w_idx] = b_u_by_meta[k]["predictions"][target]
        base_oof_state[u_idx] = b_w_by_meta[k]["state"]
        base_oof_state[w_idx] = b_u_by_meta[k]["state"]
        for target in TARGETS:
            cache[f"meta{k}_oofU_pred_{target}"] = oof_u_by_meta[k]["predictions"][target]
            cache[f"meta{k}_oofU_state"] = oof_u_by_meta[k]["state"]
            cache[f"meta{k}_bW_pred_{target}"] = b_u_by_meta[k]["predictions"][target]
            cache[f"meta{k}_bW_state"] = b_u_by_meta[k]["state"]
    if not np.isfinite(base_oof_state).all():
        raise ValueError("T-level base OOF state assembly failed")
    for target in TARGETS:
        cache[f"base_oof_T_pred_{target}"] = base_oof_pred[target]
    cache["base_oof_T_state"] = base_oof_state

    # Final complete B_T model fitted on all of T; predicts V.
    final = factory.fit_predict(T, V)
    for target in TARGETS:
        cache[f"final_pred_{target}"] = np.asarray(final["predictions"][target], dtype=float)
    cache["final_state"] = _state(final)
    cache["outer_fold"] = np.asarray([int(outer_fold)], dtype=np.int64)
    cache["seed"] = np.asarray([int(seed)], dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    return cache


def _e_model(name: str) -> Any:
    if name == "E0":
        return ConstantMedianResidual()
    if name == "E1":
        return LADSplineResidual()
    if name == "E2":
        return LinearLeafResidual()
    if name == "E3":
        return LocalWeightedMedianResidual()
    raise ValueError(f"Unknown E method: {name}")


def _fit_residual_model(name: str, x_fit: pd.DataFrame, residual: np.ndarray,
                        x_query: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    model = _e_model(name)
    model.fit(x_fit, residual)
    prediction = np.asarray(model.predict(x_query), dtype=float)
    meta = {"method": name, "n_fit": int(len(x_fit)), "n_query": int(len(x_query))}
    if name == "E0":
        meta["median_residual"] = float(getattr(model, "median_", 0.0))
    if name == "E2":
        meta["linear_tree"] = getattr(getattr(model, "model_", None), "linear_tree_effective_", None)
    return prediction, meta


def _run_one_target(
    root: Path,
    train: pd.DataFrame,
    folds: np.ndarray,
    seed: int,
    outer_fold: int,
    cache: Mapping[str, np.ndarray],
    target: str,
) -> list[dict[str, Any]]:
    outer_train_mask = folds != int(outer_fold)
    outer_valid_mask = folds == int(outer_fold)
    T = train.loc[outer_train_mask].reset_index(drop=True)
    V = train.loc[outer_valid_mask].reset_index(drop=True)
    y_V = np.asarray(V[target], dtype=float)
    baseline_V = np.asarray(cache[f"final_pred_{target}"], dtype=float)
    baseline_wmape = float(wmape(y_V, baseline_V))
    x_T = T.loc[:, [*FEATURES, "spout_no"]].reset_index(drop=True)
    x_V = V.loc[:, [*FEATURES, "spout_no"]].reset_index(drop=True)

    meta_partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(2):
        meta_partitions.append((cache[f"meta{k}_U_idx"], cache[f"meta{k}_W_idx"]))

    results: list[dict[str, Any]] = []
    for method in E_METHODS:
        y_meta_parts: list[np.ndarray] = []
        b_meta_parts: list[np.ndarray] = []
        d_meta_parts: list[np.ndarray] = []
        fit_meta: list[dict[str, Any]] = []
        for k, (u_idx, w_idx) in enumerate(meta_partitions):
            U = T.iloc[u_idx].reset_index(drop=True)
            W = T.iloc[w_idx].reset_index(drop=True)
            residual_U = np.asarray(U[target], dtype=float) - np.asarray(cache[f"meta{k}_oofU_pred_{target}"], dtype=float)
            b_W = np.asarray(cache[f"meta{k}_bW_pred_{target}"], dtype=float)
            if method == "E3":
                bank_state = np.asarray(cache[f"meta{k}_oofU_state"], dtype=float)
                query_state = np.asarray(cache[f"meta{k}_bW_state"], dtype=float)
                model = LocalWeightedMedianResidual()
                model.fit(bank_state, residual_U)
                d_W = model.predict(query_state)
            else:
                d_W, meta = _fit_residual_model(method, U.loc[:, [*FEATURES, "spout_no"]], residual_U,
                                                W.loc[:, [*FEATURES, "spout_no"]])
                fit_meta.append({"meta_fold": k, **meta})
            y_meta_parts.append(np.asarray(W[target], dtype=float))
            b_meta_parts.append(b_W)
            d_meta_parts.append(np.asarray(d_W, dtype=float))
        alpha = exact_l1_alpha(np.concatenate(y_meta_parts), np.concatenate(b_meta_parts),
                               np.concatenate(d_meta_parts))

        residual_T = np.asarray(T[target], dtype=float) - np.asarray(cache[f"base_oof_T_pred_{target}"], dtype=float)
        if method == "E3":
            model = LocalWeightedMedianResidual()
            model.fit(np.asarray(cache["base_oof_T_state"], dtype=float), residual_T)
            h_V = model.predict(np.asarray(cache["final_state"], dtype=float))
            meta = {"method": method, "n_bank": int(len(T))}
        else:
            h_V, meta = _fit_residual_model(method, x_T, residual_T, x_V)
        fit_meta.append({"meta_fold": "full_T", **meta})
        candidate_V = baseline_V + float(alpha) * np.asarray(h_V, dtype=float)
        candidate_wmape = float(wmape(y_V, candidate_V))
        target_delta = float(50.0 * (baseline_wmape - candidate_wmape))
        results.append({
            "method_id": method,
            "target": target,
            "seed": int(seed),
            "outer_fold": int(outer_fold),
            "alpha": float(alpha),
            "baseline_wmape": baseline_wmape,
            "candidate_wmape": candidate_wmape,
            "nested_package_delta_single_target": target_delta,
            "n_corrector_fits": 0 if method == "E0" else (2 + 1),
            "fit_meta": fit_meta,
            "boundary": "outer validation scored only after alpha and h were frozen",
        })
    return results


def run_nested_residuals(root: Path | str = ".", output: Path | str | None = None,
                         *, seeds: Sequence[int] = (42, 3407),
                         folds: Sequence[int] = (0, 1), workers: int = 16) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else (
        root / "local/runs/round2-v4.1-strong-increment/nested-residual-r1"
    )
    _private_output(root, output)
    train = load_training_frame(root)
    factory = FrozenAReferenceFactory(root, workers=workers)
    ledger = output / "fit_ledger.jsonl"
    events: list[dict[str, Any]] = []
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                if event.get("event") == "complete":
                    events.append(event)
    completed = {(str(e["method_id"]), int(e["seed"]), int(e["outer_fold"]), str(e["target"])) for e in events}

    for seed in seeds:
        seed = int(seed)
        folds_vector = load_fold_vector(root, train, int(seed))
        for outer_fold in folds:
            outer_fold = int(outer_fold)
            cache_path = output / "nested-base-cache" / f"seed-{seed}" / f"fold-{outer_fold}.npz"
            started = time.perf_counter()
            cache = build_or_load_nested_cache(
                root, train, folds_vector, seed, outer_fold, factory,
                cache_path=cache_path, n_meta=2, n_inner=2,
            )
            cache_seconds = float(time.perf_counter() - started)
            for target in TARGETS:
                for event in _run_one_target(root, train, folds_vector, seed, outer_fold, cache, target):
                    key = (str(event["method_id"]), seed, outer_fold, target)
                    if key in completed:
                        continue
                    record = {
                        "event": "complete",
                        "status": "available",
                        "family": "E",
                        "mechanism": {
                            "E0": "E0_constant_median_residual",
                            "E1": "E1_lad_spline_residual",
                            "E2": "E2_linear_leaf_residual",
                            "E3": "E3_local_weighted_median_state",
                        }[event["method_id"]],
                        "n_base_fits_shared_per_cache": 9,
                        "cache_seconds": cache_seconds,
                        "agent_uploads": 0,
                        **event,
                    }
                    _append_jsonl(ledger, record)
                    events.append(record)
            print(json.dumps({"seed": seed, "outer_fold": outer_fold,
                              "cache_seconds": round(cache_seconds, 2)}, ensure_ascii=False), flush=True)

    table = pd.DataFrame([
        {
            "method_id": e["method_id"], "target": e["target"], "seed": e["seed"],
            "outer_fold": e["outer_fold"], "alpha": e["alpha"],
            "baseline_wmape": e["baseline_wmape"], "candidate_wmape": e["candidate_wmape"],
            "nested_package_delta_single_target": e["nested_package_delta_single_target"],
        }
        for e in events
    ])
    table.to_csv(output / "nested_residual_summary.csv", index=False)
    (output / "nested_residual_summary.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest = {
        "status": "E_LINE_NESTED_PRIVATE_NOT_SUBMISSION",
        "seeds": [int(v) for v in seeds],
        "outer_folds": [int(v) for v in folds],
        "methods": list(E_METHODS),
        "method_target_units": 8,
        "base_factory": "A_frozen_deployment_weights_v1",
        "base_fits_per_cache": 9,
        "alpha_selection": "training-part meta folds only, [0,1], zero fallback",
        "platform_uploads": 0,
        "submission_packages": 0,
        "warning": "E line reports A-relative nested increments; no platform promotion is implied.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), **manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    result = run_nested_residuals(args.root, args.output, seeds=args.seeds, folds=args.folds, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
