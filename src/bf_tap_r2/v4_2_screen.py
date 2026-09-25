"""V4.2 coarse screen: 24 target-recipe units x 2 split seeds x 2 outer folds.

Protocol (pre-registered, never relaxed at run time):

* every unit is evaluated against the same ``B_fit`` reference: ``B_fit`` and the
  new model are fitted on the same outer training part ``T`` and scored on the
  same outer evaluation part ``V``;
* two pre-declared paths are scored for each target: ``direct`` (the new model
  replaces that target entirely) and ``fixed_quarter``
  (``0.75 * B_fit_t + 0.25 * f_t``); the other target always stays ``B_fit``;
* ``S = max(0, 100 - 50 * (W_I + W_T))`` is recomputed on the combined
  prediction; per-seed WMAPEs pool numerators and denominators over the covered
  samples before the seed scores are averaged;
* the continuation gate requires the *same* fixed path to be positive in both
  seeds and to average at least ``+0.005`` on the full package.

Nothing here reads test labels, writes a submission or uploads anything.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import TARGETS
from .metrics import wmape
from .v3_run import load_fold_vector, load_training_frame
from .v4_2_reference import V42References, baseline_cache_identity
from .v4_2_spec import DEFAULT_SPEC_PATH, SearchSpec, load_search_spec

__all__ = ["SCREEN_PATHS", "aggregate_screen", "run_screen"]

SCREEN_PATHS = ("direct", "fixed_quarter")

_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def pooled_wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """WMAPE pooled over the covered rows: one numerator, one denominator."""
    return float(wmape(actual, predicted))


def package_score(iron_wmape: float, time_wmape: float) -> float:
    return float(max(0.0, 100.0 - 50.0 * (float(iron_wmape) + float(time_wmape))))


# ---------------------------------------------------------------------------
# worker plumbing
# ---------------------------------------------------------------------------

def _initialise_worker(train: pd.DataFrame, fold_vectors: Mapping[int, np.ndarray],
                       spec_raw: Mapping[str, Any], root: str, torch_threads: int | None) -> None:
    _WORKER.clear()
    _WORKER["train"] = train
    _WORKER["folds"] = {int(k): np.asarray(v, dtype=int) for k, v in fold_vectors.items()}
    _WORKER["spec_raw"] = dict(spec_raw)
    _WORKER["root"] = str(root)
    _WORKER["torch_threads"] = torch_threads


def _build_model(line: str, recipe: str, spec_raw: Mapping[str, Any], root: str,
                 seed: int, torch_threads: int | None):
    lines = spec_raw["lines"]
    if line == "R":
        from .v4_2_r_tabr import TabRRetrievalRegressor

        overrides = {
            key: value for key, value in lines["R"]["starting_point"].items()
            if key in {"repr_width", "n_blocks", "n_neighbors", "dropout", "ple_segments"}
        }
        return TabRRetrievalRegressor(
            recipe, overrides=overrides, n_inner_splits=5, inner_seed=20260925,
            torch_threads=torch_threads,
        )
    if line == "S":
        from .v4_2_s_symbolic import BoundedExpressionRegressor

        limits = {
            "max_nodes": int(lines["S"]["limits"]["max_nodes"]),
            "max_depth": int(lines["S"]["limits"]["max_depth"]),
            "eval_budget": int(lines["S"]["limits"]["eval_budget_per_training_subset"]),
            "max_candidates": int(lines["S"]["limits"]["max_candidates"]),
            "beam_width": int(lines["S"]["limits"]["beam_width"]),
            "no_improve_rounds": int(lines["S"]["limits"]["no_improve_rounds"]),
            "constant_refit_max_nfev": int(lines["S"]["limits"]["constant_refit_max_nfev"]),
            "constant_bound": float(lines["S"]["limits"]["constant_bound"]),
        }
        backend = str(lines["S"]["backend"]["primary"])
        return BoundedExpressionRegressor(
            recipe, limits=limits, n_inner_splits=5, inner_seed=20260925,
            backend=backend, root=root, allow_backend_fallback=True,
        )
    if line == "N":
        from .v4_2_n_node import NodeEnsembleRegressor

        return NodeEnsembleRegressor(
            recipe, n_inner_splits=5, inner_seed=20260925, torch_threads=torch_threads,
        )
    raise ValueError(f"Unknown V4.2 line: {line!r}")


def _neural_config(spec_raw: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered neural starting point, translated into runner parameters."""
    training = spec_raw["common_training"]["neural_starting_point"]
    return {
        "optimizer": str(training["optimizer"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "batch_size": int(training["batch_size"]),
        "max_epochs": int(training["max_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "seed": int(training["initialisation_seed"]),
    }


def _fit_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    line = str(payload["line"])
    recipe = str(payload["recipe"])
    target = str(payload["target"])
    seed = int(payload["seed"])
    fold = int(payload["fold"])
    train: pd.DataFrame = _WORKER["train"]
    folds = _WORKER["folds"][seed]
    spec_raw = _WORKER["spec_raw"]
    root = _WORKER["root"]
    torch_threads = _WORKER["torch_threads"]

    train_mask = folds != fold
    valid_mask = folds == fold
    train_frame = train.loc[train_mask].reset_index(drop=True)
    valid_frame = train.loc[valid_mask].reset_index(drop=True)
    model = _build_model(line, recipe, spec_raw, root, seed, torch_threads)
    started = time.perf_counter()
    try:
        model.fit(train_frame, target, train_config=_neural_config(spec_raw))
        prediction = np.asarray(model.predict(valid_frame), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "line": line, "recipe": recipe, "target": target, "seed": seed, "fold": fold,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": float(time.perf_counter() - started),
            "n_train": int(train_mask.sum()), "n_valid": int(valid_mask.sum()),
        }
    if prediction.shape != (int(valid_mask.sum()),) or not np.isfinite(prediction).all():
        return {
            "status": "failed",
            "line": line, "recipe": recipe, "target": target, "seed": seed, "fold": fold,
            "error": "invalid prediction vector",
            "seconds": float(time.perf_counter() - started),
            "n_train": int(train_mask.sum()), "n_valid": int(valid_mask.sum()),
        }
    full = np.full(len(train), np.nan, dtype=float)
    full[valid_mask] = prediction
    return {
        "status": "available",
        "line": line, "recipe": recipe, "target": target, "seed": seed, "fold": fold,
        "seconds": float(time.perf_counter() - started),
        "n_train": int(train_mask.sum()), "n_valid": int(valid_mask.sum()),
        "prediction": full,
        "fit_meta": model.fit_outcome_.as_dict(),
    }


# ---------------------------------------------------------------------------
# screen driver
# ---------------------------------------------------------------------------

def _private_output(root: Path, output: Path) -> None:
    if not output.resolve().is_relative_to((root / "local/runs").resolve()):
        raise ValueError("V4.2 outputs must remain beneath local/runs")


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _append_ledger(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _fit_key(event: Mapping[str, Any]) -> str:
    return "|".join(str(event[k]) for k in ("line", "recipe", "target", "seed", "fold"))


def run_screen(
    root: Path | str = ".",
    output: Path | str | None = None,
    *,
    spec_path: Path | str = DEFAULT_SPEC_PATH,
    lines: Sequence[str] | None = None,
    recipes: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    folds: Sequence[int] | None = None,
    limit: int | None = None,
    jobs: int = 1,
    torch_threads: int | None = None,
    b_fit_workers: int = 12,
    skip_baseline: bool = False,
    verify_replay: bool = True,
    start_method: str = "spawn",
    aggregate_only: bool = False,
    training_seed: int | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    spec: SearchSpec = load_search_spec(spec_path, root=root)
    output = Path(output).resolve() if output is not None else (
        root / "local/runs/round2-v4.2-structure-search/coarse-r1"
    )
    _private_output(root, output)
    output.mkdir(parents=True, exist_ok=True)

    train = load_training_frame(root)
    fold_vectors = {int(seed): load_fold_vector(root, train, int(seed)) for seed in spec.seeds}
    selected_lines = tuple(str(v) for v in (lines or ("R", "S", "N")))
    selected_recipes = None if recipes is None else {str(v).upper() for v in recipes}
    selected_targets = tuple(str(v) for v in (targets or spec.targets))
    selected_seeds = tuple(int(v) for v in (seeds or spec.seeds))
    selected_folds = tuple(int(v) for v in (folds or spec.folds))

    units = [
        (line, recipe, target)
        for line, recipe, target in spec.units
        if line in selected_lines and target in selected_targets
        and (selected_recipes is None or recipe in selected_recipes)
    ]
    if limit is not None:
        units = units[: int(limit)]

    # The training seed is a *randomness* control, kept separate from the split
    # seeds.  Overriding it never re-selects a structure or a blend weight.
    worker_spec = json.loads(json.dumps(spec.raw, default=str))
    if training_seed is not None:
        worker_spec["common_training"]["neural_starting_point"]["initialisation_seed"] = int(
            training_seed
        )

    ledger = output / "fit_ledger.jsonl"
    known = {
        _fit_key(event) for event in _read_ledger(ledger)
        if event.get("event") == "complete"
    }

    manifest_path = output / "manifest.json"
    started = time.perf_counter()

    # ---- stage A: the shared B_fit reference -----------------------------
    baselines: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    baseline_meta: dict[str, Any] = {"computed": 0, "reused": 0, "identity": {}}
    if aggregate_only:
        # Re-derive the summary from the already-persisted slots.  No model is
        # fitted and no prediction is recomputed.
        summary = aggregate_screen(
            root=root, output=output, spec=spec, train=train, fold_vectors=fold_vectors,
            units=units, seeds=selected_seeds, folds=selected_folds,
        )
        manifest = {
            "status": "V4_2_COARSE_SCREEN_AGGREGATE_ONLY",
            "output": str(output),
            "units": len(units),
            "seeds": list(selected_seeds),
            "folds": list(selected_folds),
            "refits": 0,
            "platform_uploads": 0,
            "submission_packages": 0,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(output), **manifest, "summary": summary}
    if not skip_baseline:
        references = V42References(
            root, train, workers=int(b_fit_workers), verify_replay=bool(verify_replay)
        )
        baseline_path = output / "baselines.npz"
        cached: dict[str, np.ndarray] = {}
        if baseline_path.is_file():
            cached = {k: v for k, v in np.load(baseline_path, allow_pickle=False).items()}
        for seed in selected_seeds:
            folds_vector = fold_vectors[seed]
            for fold in selected_folds:
                train_mask = folds_vector != fold
                valid_mask = folds_vector == fold
                key = f"s{seed}-f{fold}"
                identity = baseline_cache_identity(
                    train.loc[train_mask], seed=seed, fold=fold,
                    source_hash=references.source_hash,
                )
                baseline_meta["identity"][key] = identity
                if f"{key}-pred" in cached and f"{key}-id" in cached:
                    stored = str(cached[f"{key}-id"])
                    if stored == json.dumps(identity, sort_keys=True):
                        vector = np.asarray(cached[f"{key}-pred"], dtype=float)
                        baseline_meta["reused"] += 1
                    else:
                        vector = None
                else:
                    vector = None
                if vector is None:
                    bundle = references.fit_baseline(
                        train.loc[train_mask].reset_index(drop=True),
                        train.loc[valid_mask].reset_index(drop=True),
                    )
                    vector = np.full((len(train), len(TARGETS)), np.nan, dtype=float)
                    for column, target in enumerate(TARGETS):
                        vector[valid_mask, column] = np.asarray(bundle["b36"][target], dtype=float)
                    baseline_meta["computed"] += 1
                baselines[(int(seed), int(fold))] = {
                    target: vector[:, column] for column, target in enumerate(TARGETS)
                }
                cached[f"{key}-pred"] = vector
                cached[f"{key}-id"] = np.asarray(json.dumps(identity, sort_keys=True))
                np.savez_compressed(baseline_path, **cached)
        baseline_meta["file"] = str(baseline_path.relative_to(root))
        baseline_meta["source_hash"] = references.source_hash
        replay_path = output / "b_replay_diagnostics.json"
        if not replay_path.exists():
            replay_path.write_text(
                json.dumps(references.replay_diagnostics(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (output / "baseline_identity.json").write_text(
            json.dumps(baseline_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "spec_snapshot.json").write_text(
        json.dumps(spec.raw, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # ---- stage B: per-unit outer fits ------------------------------------
    tasks = [
        {"line": line, "recipe": recipe, "target": target, "seed": int(seed), "fold": int(fold)}
        for line, recipe, target in units
        for seed in selected_seeds
        for fold in selected_folds
    ]
    scheduled = [task for task in tasks if _fit_key(task) not in known]
    results: list[dict[str, Any]] = []

    def persist(outcome: Mapping[str, Any]) -> None:
        """Write one slot's evidence immediately so a crash cannot lose it."""
        record = {key: value for key, value in outcome.items() if key != "prediction"}
        # Only a successful fit is recorded as ``complete``; a failed slot stays
        # retryable on the next run instead of being silently treated as done.
        record["event"] = "complete" if outcome["status"] == "available" else "failed"
        _append_ledger(ledger, record)
        if outcome["status"] == "available":
            path = output / "predictions" / (
                f"{outcome['line']}-{outcome['recipe']}-{outcome['target']}"
                f"-s{outcome['seed']}-f{outcome['fold']}.npy"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(outcome["prediction"], dtype=float))
        results.append(dict(outcome))

    if jobs > 1 and len(scheduled) > 1:
        # The B_fit baseline stage trains a torch network *in this parent
        # process*, so a later ``fork`` would hand the children an already
        # initialised OpenMP runtime and can deadlock.  Fresh interpreters avoid
        # that entirely.
        context = multiprocessing.get_context(str(start_method))
        with ProcessPoolExecutor(
            max_workers=int(jobs),
            mp_context=context,
            initializer=_initialise_worker,
            initargs=(train, fold_vectors, worker_spec, str(root), torch_threads),
        ) as executor:
            futures = {executor.submit(_fit_one, task): task for task in scheduled}
            for future in as_completed(futures):
                outcome = future.result()
                persist(outcome)
                print(json.dumps({
                    "line": outcome["line"], "recipe": outcome["recipe"],
                    "target": outcome["target"], "seed": outcome["seed"],
                    "fold": outcome["fold"], "status": outcome["status"],
                    "seconds": round(float(outcome.get("seconds", 0.0)), 1),
                }, ensure_ascii=False), flush=True)
    else:
        _initialise_worker(train, fold_vectors, worker_spec, str(root), torch_threads)
        for task in scheduled:
            outcome = _fit_one(task)
            persist(outcome)
            print(json.dumps({
                "line": outcome["line"], "recipe": outcome["recipe"],
                "target": outcome["target"], "seed": outcome["seed"],
                "fold": outcome["fold"], "status": outcome["status"],
                "seconds": round(float(outcome.get("seconds", 0.0)), 1),
            }, ensure_ascii=False), flush=True)

    summary = aggregate_screen(
        root=root, output=output, spec=spec, train=train, fold_vectors=fold_vectors,
        units=units, seeds=selected_seeds, folds=selected_folds,
    )
    manifest = {
        "status": "V4_2_COARSE_SCREEN_PRIVATE_EVIDENCE_NOT_SUBMISSION",
        "spec_path": str(Path(spec_path)),
        "spec_status": spec.status,
        "lines": list(selected_lines),
        "targets": list(selected_targets),
        "seeds": list(selected_seeds),
        "folds": list(selected_folds),
        "units": len(units),
        "outer_recipe_slots": len(tasks),
        "scheduled_this_run": len(scheduled),
        "already_complete": len(tasks) - len(scheduled),
        "failed_slots": int(sum(1 for r in results if r["status"] != "available")),
        "worker_start_method": str(start_method),
        "training_seed": int(training_seed) if training_seed is not None
        else int(spec.raw["common_training"]["neural_starting_point"]["initialisation_seed"]),
        "split_seeds": [int(v) for v in selected_seeds],
        "baseline": baseline_meta,
        "seconds": round(float(time.perf_counter() - started), 2),
        "platform_uploads": 0,
        "submission_packages": 0,
        "promotion": "none",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), **manifest, "summary": summary}


# ---------------------------------------------------------------------------
# aggregation, gating and ranking
# ---------------------------------------------------------------------------

def _coverage_gate(
    spec: SearchSpec,
    coarse_gate: Mapping[str, Any],
    positive_folds: Mapping[str, Any],
    seeds: Sequence[int],
    folds: Sequence[int],
) -> dict[str, Any]:
    """Section-9 gate: may this recipe proceed to inner fusion selection?

    Requires, on the *complete* coverage, one and the same pre-declared path with
    a mean full-package gain of at least ``mean_gain_min``, positive gains in both
    split seeds, and at least ``positive_folds_min`` positive folds out of the
    covered ``seed x fold`` cells.
    """
    thresholds = dict(spec.raw.get("full_coverage", {}).get("fusion_eligibility_gate", {}))
    mean_min = float(thresholds.get("mean_gain_min", 0.02))
    folds_min = int(thresholds.get("positive_folds_min", 8))
    expected_total = int(len(seeds) * len(folds))
    per_path: dict[str, Any] = {}
    for path in SCREEN_PATHS:
        values = coarse_gate.get("per_path", {}).get(path)
        if values is None:
            per_path[path] = {"available": False, "passes": False}
            continue
        counted = positive_folds.get(path, {"positive": 0, "total": 0})
        per_path[path] = {
            "available": True,
            "mean_gain": float(values["mean_gain"]),
            "both_seeds_positive": bool(values["both_seeds_positive"]),
            "positive_folds": int(counted["positive"]),
            "covered_folds": int(counted["total"]),
            "expected_folds": expected_total,
            "passes": bool(
                float(values["mean_gain"]) >= mean_min
                and values["both_seeds_positive"]
                and int(counted["positive"]) >= folds_min
                and int(counted["total"]) == expected_total
            ),
        }
    passing = [path for path in SCREEN_PATHS if per_path[path].get("passes")]
    best = max(passing, key=lambda p: per_path[p]["mean_gain"]) if passing else None
    return {
        "mean_gain_min": mean_min,
        "positive_folds_min": folds_min,
        "expected_folds": expected_total,
        "per_path": per_path,
        "passes": bool(passing),
        "path": best,
        "note": (
            "Gate evaluated on the covered outer folds only; it authorises inner "
            "fusion selection and nothing else."
        ),
    }

def aggregate_screen(
    *,
    root: Path,
    output: Path,
    spec: SearchSpec,
    train: pd.DataFrame,
    fold_vectors: Mapping[int, np.ndarray],
    units: Sequence[tuple[str, str, str]],
    seeds: Sequence[int],
    folds: Sequence[int],
) -> dict[str, Any]:
    baseline_path = output / "baselines.npz"
    if not baseline_path.is_file():
        raise FileNotFoundError("B_fit baselines are required before aggregation")
    cached = {k: v for k, v in np.load(baseline_path, allow_pickle=False).items()}

    baseline: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        vector = np.full((len(train), len(TARGETS)), np.nan, dtype=float)
        for fold in folds:
            key = f"s{seed}-f{fold}-pred"
            if key not in cached:
                raise FileNotFoundError(f"Missing B_fit baseline for seed {seed} fold {fold}")
            stored = np.asarray(cached[key], dtype=float)
            rows = fold_vectors[int(seed)] == int(fold)
            vector[rows] = stored[rows]
        baseline[int(seed)] = {
            target: vector[:, column] for column, target in enumerate(TARGETS)
        }

    mask_by_seed = {
        int(seed): np.isin(fold_vectors[int(seed)], list(folds)) for seed in seeds
    }
    rows_by_seed = {
        int(seed): {"iron": train.loc[mask_by_seed[int(seed)], "tap_iron"].to_numpy(dtype=float),
                    "time": train.loc[mask_by_seed[int(seed)], "tap_time_len"].to_numpy(dtype=float)}
        for seed in seeds
    }
    baseline_scores: dict[str, dict[str, float]] = {}
    for seed in seeds:
        selector = mask_by_seed[int(seed)]
        w_iron = pooled_wmape(rows_by_seed[int(seed)]["iron"],
                              baseline[int(seed)]["tap_iron"][selector])
        w_time = pooled_wmape(rows_by_seed[int(seed)]["time"],
                              baseline[int(seed)]["tap_time_len"][selector])
        baseline_scores[str(int(seed))] = {
            "W_I": w_iron, "W_T": w_time, "package_score": package_score(w_iron, w_time),
        }

    weight = spec.fixed_quarter_weight
    unit_rows: list[dict[str, Any]] = []
    for line, recipe, target in units:
        other = "tap_time_len" if target == "tap_iron" else "tap_iron"
        candidate: dict[int, np.ndarray] = {}
        available = True
        failure_notes: list[str] = []
        for seed in seeds:
            vector = np.full(len(train), np.nan, dtype=float)
            for fold in folds:
                path = output / "predictions" / (
                    f"{line}-{recipe}-{target}-s{int(seed)}-f{int(fold)}.npy"
                )
                if not path.is_file():
                    available = False
                    failure_notes.append(f"missing prediction {path.name}")
                    continue
                stored = np.load(path)
                rows = fold_vectors[int(seed)] == int(fold)
                vector[rows] = stored[rows]
            candidate[int(seed)] = vector

        per_seed: dict[str, Any] = {}
        for seed in seeds:
            selector = mask_by_seed[int(seed)]
            actual_target = rows_by_seed[int(seed)]["iron" if target == "tap_iron" else "time"]
            other_key = "time" if target == "tap_iron" else "iron"
            actual_other = rows_by_seed[int(seed)][other_key]
            if not np.isfinite(candidate[int(seed)][selector]).all():
                available = False
            base_target = baseline[int(seed)][target][selector]
            base_other = baseline[int(seed)][other][selector]
            w_other = pooled_wmape(actual_other, base_other)
            base_score = package_score(pooled_wmape(actual_target, base_target), w_other)
            entry: dict[str, Any] = {
                "baseline_target_wmape": pooled_wmape(actual_target, base_target),
                "other_target_wmape": w_other,
                "baseline_package_score": base_score,
                "paths": {},
            }
            for path in SCREEN_PATHS:
                if not available:
                    continue
                values = candidate[int(seed)][selector]
                if path == "direct":
                    combined = values
                elif path == "fixed_quarter":
                    combined = (1.0 - weight) * base_target + weight * values
                else:  # pragma: no cover - guarded by SCREEN_PATHS
                    raise ValueError(path)
                w_target = pooled_wmape(actual_target, combined)
                score = package_score(w_target, w_other)
                entry["paths"][path] = {
                    "target_wmape": w_target,
                    "package_score": score,
                    "package_gain": float(score - base_score),
                    "single_target_increment": float(50.0 * (
                        entry["baseline_target_wmape"] - w_target
                    )),
                }
            per_seed[str(int(seed))] = entry
            # Fold-level accounting, needed by the section-9 coverage gate.
            fold_level: dict[str, Any] = {}
            for fold in folds:
                rows = (fold_vectors[int(seed)] == int(fold)) & selector
                if not rows.any() or not available:
                    continue
                actual_target_fold = train.loc[rows, target].to_numpy(dtype=float)
                other_name = "tap_time_len" if target == "tap_iron" else "tap_iron"
                actual_other_fold = train.loc[rows, other_name].to_numpy(dtype=float)
                base_target_fold = baseline[int(seed)][target][rows]
                base_other_fold = baseline[int(seed)][other][rows]
                w_other_fold = pooled_wmape(actual_other_fold, base_other_fold)
                base_score_fold = package_score(
                    pooled_wmape(actual_target_fold, base_target_fold), w_other_fold
                )
                values_fold = candidate[int(seed)][rows]
                fold_level[str(int(fold))] = {
                    "direct_gain": float(
                        package_score(pooled_wmape(actual_target_fold, values_fold), w_other_fold)
                        - base_score_fold
                    ),
                    "fixed_quarter_gain": float(
                        package_score(
                            pooled_wmape(
                                actual_target_fold,
                                (1.0 - weight) * base_target_fold + weight * values_fold,
                            ),
                            w_other_fold,
                        ) - base_score_fold
                    ),
                }
            entry["fold_level"] = fold_level
            per_seed[str(int(seed))] = entry

        positive_folds: dict[str, Any] = {}
        for path_key in SCREEN_PATHS:
            positives = sum(
                1
                for seed in seeds
                for fold_values in per_seed[str(int(seed))].get("fold_level", {}).values()
                if fold_values[f"{path_key}_gain"] > 0.0
            )
            total = sum(
                len(per_seed[str(int(seed))].get("fold_level", {})) for seed in seeds
            )
            positive_folds[path_key] = {"positive": int(positives), "total": int(total)}

        gate: dict[str, Any] = {"passes": False, "path": None, "per_path": {}}
        if available:
            for path in SCREEN_PATHS:
                gains = [per_seed[str(int(seed))]["paths"][path]["package_gain"] for seed in seeds]
                mean_gain = float(np.mean(gains))
                passes = bool(all(gain > 0.0 for gain in gains)
                              and mean_gain >= float(spec.gate["mean_full_package_gain_min"]))
                gate["per_path"][path] = {
                    "per_seed_gain": {str(int(seed)): per_seed[str(int(seed))]["paths"][path]["package_gain"]
                                      for seed in seeds},
                    "mean_gain": mean_gain,
                    "worse_seed_gain": float(min(gains)),
                    "both_seeds_positive": bool(all(gain > 0.0 for gain in gains)),
                    "passes": passes,
                }
            passing = [path for path in SCREEN_PATHS if gate["per_path"][path]["passes"]]
            if passing:
                best = max(passing, key=lambda p: gate["per_path"][p]["mean_gain"])
                gate["passes"] = True
                gate["path"] = best
        unit_rows.append({
            "trial_id": f"{line}-{recipe}-{target}",
            "line": line,
            "recipe": recipe,
            "target": target,
            "available": bool(available),
            "counts_as_retrieval_success": bool(
                line != "R" or spec.raw["lines"]["R"]["recipes"][recipe]["counts_as_retrieval_success"]
            ),
            "failure_notes": failure_notes,
            "per_seed": per_seed,
            "baseline_scores": baseline_scores,
            "gate": gate,
            "positive_folds": positive_folds,
            "coverage_gate": _coverage_gate(spec, gate, positive_folds, seeds, folds),
            "ranking_score": float(max(
                (gate["per_path"][p]["mean_gain"] for p in gate["per_path"]), default=float("-inf")
            )),
            # The ranking tie-break uses the *better pre-declared path*: that is
            # the path whose mean gain ranks the unit.  The worst value across
            # both paths is kept separately so the table is not misread.
            "worse_seed_gain": float(
                gate["per_path"][gate["path"]]["worse_seed_gain"]
                if gate["path"] is not None
                else min(
                    (gate["per_path"][p]["worse_seed_gain"] for p in gate["per_path"]),
                    default=float("-inf"),
                )
            ),
            "best_path": gate["path"],
            "worse_seed_gain_across_paths": float(min(
                (gate["per_path"][p]["worse_seed_gain"] for p in gate["per_path"]),
                default=float("-inf"),
            )),
        })

    ranked = sorted(
        unit_rows, key=lambda row: (-row["ranking_score"], -row["worse_seed_gain"], row["trial_id"])
    )
    selection = spec.selection
    per_family_target: dict[tuple[str, str], int] = {}
    finalists: list[dict[str, Any]] = []
    for row in ranked:
        if not row["gate"]["passes"]:
            continue
        key = (row["line"], row["target"])
        cap = int(selection["max_per_family_per_target"])
        if per_family_target.get(key, 0) >= cap:
            continue
        if len(finalists) >= int(selection["max_total_finalists"]):
            break
        if row["line"] == "R" and not row["counts_as_retrieval_success"]:
            row["note"] = "R0 control passes the numeric gate but is not retrieval success"
            continue
        per_family_target[key] = per_family_target.get(key, 0) + 1
        finalists.append(row)

    table = pd.DataFrame([
        {
            "trial_id": row["trial_id"], "line": row["line"], "recipe": row["recipe"],
            "target": row["target"], "available": row["available"],
            "gate_passes": row["gate"]["passes"], "gate_path": row["gate"]["path"],
            "best_path": row["best_path"],
            "ranking_score": row["ranking_score"],
            "best_path_worse_seed_gain": row["worse_seed_gain"],
            "worse_seed_gain_across_paths": row["worse_seed_gain_across_paths"],
            "direct_mean_gain": row["gate"]["per_path"].get("direct", {}).get("mean_gain"),
            "fixed_quarter_mean_gain": row["gate"]["per_path"].get("fixed_quarter", {}).get("mean_gain"),
        }
        for row in ranked
    ])
    table.to_csv(output / "coarse_summary.csv", index=False)
    (output / "coarse_summary.json").write_text(
        json.dumps(unit_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    selection_payload = {
        "gate": spec.gate,
        "fixed_quarter_weight": weight,
        "paths": list(SCREEN_PATHS),
        "ranked_trial_ids": [row["trial_id"] for row in ranked],
        "gate_passing_trial_ids": [row["trial_id"] for row in ranked if row["gate"]["passes"]],
        "finalists": [
            {
                "trial_id": row["trial_id"], "line": row["line"], "recipe": row["recipe"],
                "target": row["target"], "gate_path": row["gate"]["path"],
                "ranking_score": row["ranking_score"],
                "best_path_mean_gain": row["ranking_score"],
                "best_path_worse_seed_gain": row["worse_seed_gain"],
                "per_seed_gain": (
                    row["gate"]["per_path"][row["gate"]["path"]]["per_seed_gain"]
                    if row["gate"]["path"] is not None else {}
                ),
            }
            for row in finalists
        ],
        "note": (
            "Development screen only. Ranking is not independent validation and no "
            "finalist has been packaged or uploaded."
        ),
    }
    (output / "coarse_selection.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return selection_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=Path(DEFAULT_SPEC_PATH))
    parser.add_argument("--lines", nargs="+", choices=["R", "S", "N"], default=None)
    parser.add_argument("--recipes", nargs="+", default=None,
                        help="restrict to specific recipes, e.g. --recipes N2")
    parser.add_argument("--targets", nargs="+", choices=list(TARGETS), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--b-fit-workers", type=int, default=12)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--no-verify-replay", action="store_true")
    parser.add_argument("--training-seed", type=int, default=None,
                        help="randomness control only; never re-selects a structure")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="re-derive the summary from persisted slots without fitting")
    parser.add_argument("--start-method", choices=["spawn", "fork", "forkserver"],
                        default="spawn",
                        help="worker start method; spawn avoids fork-with-OpenMP deadlocks")
    args = parser.parse_args(argv)
    result = run_screen(
        args.root, args.output, spec_path=args.spec, lines=args.lines,
        recipes=args.recipes, targets=args.targets,
        seeds=args.seeds, folds=args.folds, limit=args.limit, jobs=args.jobs,
        torch_threads=args.torch_threads, b_fit_workers=args.b_fit_workers,
        skip_baseline=args.skip_baseline, verify_replay=not args.no_verify_replay,
        start_method=args.start_method, aggregate_only=args.aggregate_only,
        training_seed=args.training_seed,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
