"""Round2 V5 Stage 3b: the platform noise-floor experiment.

The V5 pre-registration treats a further submission as a declared experiment
rather than a routine slot.  This module builds the cheapest informative one:

* the **recipe is unchanged** — same V34_A endpoint, same four selected V3.6
  experts, same frozen blend weights, same split seeds;
* the **only** change is a pre-declared shift applied to every training seed
  inside the expert parameter blocks (``random_state`` for the EBM lines,
  ``random_seed`` / ``inner_validation_seed`` for the numeric-network line);
* the local out-of-fold counterpart of the same shift is measured on split
  seeds 42/3407, so the platform delta can be split into a local seed-sensitivity
  component and the residual platform component.

``shift = 0`` must reproduce the frozen expert vectors; the runner verifies that
before spending any full-data fit.  This module never uploads anything.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import digest
from .data import SUBMISSION_COLUMNS, TARGETS
from .submission import ZIP_NAME, package, validate_result
from .v5_library import fold_vector, load_column_reference, load_v5_training_frame
from .v5_package import DEFAULT_V34A_PACKAGE, read_parent_columns
from .v5_resolution import paired_cells, paired_summary, seed_gains, seed_level_summary
from .v5_spec import V5Spec, load_v5_spec

__all__ = [
    "SEED_KEYS",
    "shift_trial",
    "verify_zero_shift_reproduction",
    "load_v36_trials",
    "frozen_column_predictor",
    "shifted_expert_oof",
    "local_seed_sensitivity",
    "build_seed_swap_package",
    "main",
]

DEFAULT_OUTPUT = "local/runs/round2-v5-error-covariance/seed-swap-r1"
DEFAULT_SHIFT = 1000

#: Parameter keys that carry a training seed in the frozen V3.6 recipes.
SEED_KEYS = ("random_seed", "random_state", "inner_validation_seed")

_WORKER_TRAIN = None
_WORKER_FOLDS = None
_WORKER_TRIALS: dict[str, dict] | None = None


def _private_output(root: Path, output: Path) -> Path:
    allowed = (root / "local/runs/round2-v5-error-covariance").resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"V5 Stage 3b output must stay private under {allowed}")
    return resolved


def shift_trial(trial: Mapping[str, Any], shift: int) -> dict[str, Any]:
    """Return a copy of ``trial`` with every training seed shifted by ``shift``."""
    shifted = copy.deepcopy(dict(trial))
    parameters = dict(shifted.get("parameters") or {})
    touched: list[str] = []
    for key in SEED_KEYS:
        if key in parameters and isinstance(parameters[key], (int, np.integer)):
            parameters[key] = int(parameters[key]) + int(shift)
            touched.append(key)
    if not touched:
        raise ValueError(f"V5 seed swap: trial {trial.get('trial_id')!r} has no training seed key")
    shifted["parameters"] = parameters
    shifted["trial_id"] = f"{trial.get('trial_id')}::seed+{int(shift)}"
    shifted["seed_shift"] = {
        "source_trial_id": str(trial.get("trial_id")),
        "shift": int(shift),
        "keys": touched,
    }
    return shifted


def load_v36_trials(root: Path | str, spec: V5Spec | None = None) -> dict[str, dict[str, Any]]:
    """Load the frozen V3.6 trial specifications from the recorded ledger."""
    root = Path(root).resolve()
    spec = spec or load_v5_spec(root)
    path = root / str(spec.raw["reference"]["v36_coarse_ledger"])
    trials: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete" and isinstance(event.get("trial"), dict):
            trials[str(event["trial_id"])] = event["trial"]
    if not trials:
        raise ValueError(f"V5 seed swap: no complete trials in {path}")
    return trials


def _v34a_columns(root: Path, package: Path | str = DEFAULT_V34A_PACKAGE
                  ) -> tuple[list[str], dict[str, np.ndarray]]:
    path = Path(package)
    if not path.is_absolute():
        path = root / path
    ids, columns = read_parent_columns(path)
    return ids, {target: columns[target][0] for target in TARGETS}


def _fit_expert(trial: Mapping[str, Any], train, test, target: str) -> np.ndarray:
    from .v3_6_models import V36Regressor

    model = V36Regressor(dict(trial))
    model.fit(train.reset_index(drop=True), train[target].to_numpy(dtype=float))
    prediction = np.asarray(model.predict(test.reset_index(drop=True)), dtype=float)
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError(f"V5 seed swap: invalid expert prediction for {trial.get('trial_id')}")
    return prediction


def full_data_seed_swap_columns(root: Path | str, spec: V5Spec, shift: int,
                                workers: int = 12) -> dict[str, Any]:
    """Refit the four selected experts on all training rows under ``shift``."""
    from .v2_release import load_v2

    root = Path(root).resolve()
    train = load_v2(root / "复赛_train", "train", 2754)
    test = load_v2(root / "复赛_test", "test", 322)
    ids = test.sample_id.tolist()
    a_ids, a_columns = _v34a_columns(root)
    if set(a_ids) != set(ids):
        raise ValueError("V5 seed swap: V34_A package ID set mismatch")
    order = {sid: position for position, sid in enumerate(a_ids)}
    reference = load_column_reference(root, train, spec)
    trials = load_v36_trials(root, spec)

    tasks: list[dict[str, Any]] = []
    for target in TARGETS:
        for name in reference.members[target]:
            tasks.append({"target": target, "name": name,
                          "trial": shift_trial(trials[name], int(shift))})
    predictions: dict[str, np.ndarray] = {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(_fit_expert, task["trial"], train, test, task["target"]): task
                   for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            predictions[task["name"]] = future.result()

    columns: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for target in TARGETS:
        weights = [float(v) for v in reference.weights[target]]
        base = np.asarray([a_columns[target][order[sid]] for sid in ids], dtype=float)
        values = weights[0] * base
        for weight, name in zip(weights[1:], reference.members[target]):
            values = values + float(weight) * predictions[name]
        columns[target] = np.maximum(values, 0.0)
        diagnostics[target] = {
            "weights": weights,
            "experts": list(reference.members[target]),
            "clip": {
                "raw_min": float(values.min()),
                "negative_rows_clipped": int((values < 0.0).sum()),
            },
        }
    return {
        "ids": ids,
        "columns": columns,
        "diagnostics": diagnostics,
        "seconds": float(time.time() - started),
        "shift": int(shift),
    }


def _init_worker(train, folds) -> None:
    global _WORKER_TRAIN, _WORKER_FOLDS
    _WORKER_TRAIN, _WORKER_FOLDS = train, folds


def _worker_oof(payload: Mapping[str, Any]) -> dict[str, Any]:
    from .v3_6_models import evaluate_v36_outer_folds

    if _WORKER_TRAIN is None or _WORKER_FOLDS is None:
        raise RuntimeError("V5 seed-swap worker was not initialised")
    return evaluate_v36_outer_folds(_WORKER_TRAIN, _WORKER_FOLDS, payload["trial"],
                                    fold_ids=tuple(payload["fold_ids"]))


def shifted_expert_oof(root: Path | str, spec: V5Spec, target: str, name: str,
                       seed: int, shift: int, folds: Sequence[int], workers: int = 12) -> dict[str, Any]:
    """One experimental expert's complete OOF under the pre-declared seed shift."""
    root = Path(root).resolve()
    train = load_v5_training_frame(root)
    fold_vector_local = fold_vector(root, train, int(seed), spec)
    trial = shift_trial(load_v36_trials(root, spec)[name], int(shift))
    payload = {"trial": trial, "fold_ids": [int(v) for v in folds]}
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), initializer=_init_worker,
                             initargs=(train, fold_vector_local)) as executor:
        future = executor.submit(_worker_oof, payload)
        result = future.result()
    return {"trial_id": str(trial["trial_id"]), "name": name, "target": target, "seed": int(seed),
            "predictions": np.asarray(result["predictions"], dtype=float),
            "pooled_wmape": float(result["pooled_wmape"]),
            "fold_scores": dict(result["fold_scores"])}


def local_seed_sensitivity(root: Path | str, spec: V5Spec, shift: int,
                           seeds: Sequence[int] = (42, 3407),
                           folds: Sequence[int] = (0, 1, 2, 3, 4),
                           workers: int = 12) -> dict[str, Any]:
    """Compare the frozen and seed-shifted complete columns on recorded split seeds."""
    root = Path(root).resolve()
    train = load_v5_training_frame(root)
    reference = load_column_reference(root, train, spec)
    results: dict[str, Any] = {
        "shift": int(shift),
        "seeds": [int(s) for s in seeds],
        "folds": [int(f) for f in folds],
        "targets": {},
        "agent_uploads": 0,
    }
    for target in TARGETS:
        actual = train[target].to_numpy(dtype=float)
        frozen_cells: list[float] = []
        shifted_cells: list[float] = []
        per_seed: dict[str, Any] = {}
        shifted_vectors: dict[int, np.ndarray] = {}
        for seed in seeds:
            seed = int(seed)
            fold_vector_local = fold_vector(root, train, seed, spec)
            weights = [float(v) for v in reference.weights[target]]
            values = weights[0] * np.asarray(reference.a_dev[seed][target], dtype=float)
            for weight, name in zip(weights[1:], reference.members[target]):
                attempt = shifted_expert_oof(root, spec, target, name, seed, shift, folds, workers)
                values = values + float(weight) * attempt["predictions"]
            shifted = np.maximum(values, 0.0)
            shifted_vectors[seed] = shifted
            per_seed[str(seed)] = {
                "frozen_wmape": float(wmape(actual, reference.base_for(target, seed))),
                "shifted_wmape": float(wmape(actual, shifted)),
            }
            per_seed[str(seed)]["delta_score"] = float(50.0 * (
                per_seed[str(seed)]["frozen_wmape"] - per_seed[str(seed)]["shifted_wmape"]))
        cells = paired_cells(
            actual, {int(s): fold_vector(root, train, int(s), spec) for s in seeds},
            {int(s): reference.base_for(target, int(s)) for s in seeds},
            shifted_vectors,
        )
        for cell in cells:
            frozen_cells.append(cell.delta_score)
        gains = seed_gains(cells)
        results["targets"][target] = {
            "per_seed": per_seed,
            "fold_summary": paired_summary(frozen_cells),
            "seed_gains": gains,
            "seed_summary": seed_level_summary(gains),
            "cells": [cell.as_dict() for cell in cells],
        }
    return results


def wmape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    from .metrics import wmape as _wmape

    return float(_wmape(actual, predicted))


def _payload_both_columns(ids: Sequence[str], values: np.ndarray) -> bytes:
    ids = list(ids)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(ids), len(TARGETS)):
        raise ValueError("V5 seed swap: invalid prediction matrix")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("V5 seed swap: predictions must be finite and non-negative")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(SUBMISSION_COLUMNS))
    writer.writerows((sid, format(i, ".17g"), format(t, ".17g")) for sid, (i, t) in zip(ids, values))
    payload = stream.getvalue().encode("utf-8")
    validate_result(payload, ids)
    return payload


def verify_zero_shift_reproduction(root: Path | str, spec: V5Spec, workers: int = 12,
                                   rtol: float = 1e-6) -> dict[str, Any]:
    """Refit the recipe with ``shift = 0`` and compare it to the parent package.

    A seed-shift experiment is only interpretable if the unshifted rebuild
    reproduces the released columns; this check is recorded before the package is
    treated as evidence.
    """
    root = Path(root).resolve()
    built = full_data_seed_swap_columns(root, spec, 0, workers=workers)
    ids = list(built["ids"])
    parent_path = root / str(spec.raw["reference"]["v36_parent_package"]) / "result.csv"
    parent_ids, columns = read_parent_columns(parent_path)
    if set(parent_ids) != set(ids):
        raise ValueError("V5 zero-shift verification: parent ID set mismatch")
    order = {sid: position for position, sid in enumerate(parent_ids)}
    payload: dict[str, Any] = {
        "parent": str(parent_path.relative_to(root)),
        "rtol": float(rtol),
        "targets": {},
        "agent_uploads": 0,
    }
    passed = True
    for target in TARGETS:
        previous = np.asarray([columns[target][0][order[sid]] for sid in ids], dtype=float)
        rebuilt = np.asarray(built["columns"][target], dtype=float)
        difference = np.abs(rebuilt - previous)
        scale = max(1.0, float(np.max(np.abs(previous))))
        relative = float(difference.max() / scale)
        ok = bool(relative <= float(rtol))
        passed = passed and ok
        payload["targets"][target] = {
            "max_abs_diff": float(difference.max()),
            "max_rel_diff": relative,
            "mean_abs_diff": float(difference.mean()),
            "passed": ok,
        }
    payload["passed"] = bool(passed)
    if not passed:
        raise AssertionError("V5 zero-shift rebuild does not reproduce the parent package")
    return payload


def build_seed_swap_package(root: Path | str, spec: V5Spec, output: Path | str,
                            name: str, shift: int = DEFAULT_SHIFT, workers: int = 12,
                            status: str = "DECLARED_EXPERIMENT_PLATFORM_NOISE_FLOOR",
                            local_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write the seed-swap experiment package (both columns refit, recipe unchanged)."""
    root = Path(root).resolve()
    out = _private_output(root, Path(output))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing package: {out}")
    built = full_data_seed_swap_columns(root, spec, shift, workers=workers)
    ids = list(built["ids"])
    matrix = np.stack([built["columns"][target] for target in TARGETS], axis=1)
    out.mkdir(parents=True, exist_ok=False)
    package(out, _payload_both_columns(ids, matrix), ids)
    result_sha = digest(out / "result.csv")
    zip_sha = digest(out / ZIP_NAME)
    manifest = {
        "name": str(name),
        "candidate": str(name),
        "status": str(status),
        "purpose": (
            "Measure the platform noise floor: the frozen V36 recipe with only its "
            "training seeds shifted. The platform delta against V36 96.2734 equals the "
            "seed-induced score difference plus the platform's own resolution."
        ),
        "recipe": "V34_A endpoint (parent package column) + frozen V3.6 blend weights + shifted-seed experts",
        "changed": {
            "seed_shift": int(shift),
            "seed_keys": list(SEED_KEYS),
            "both_targets": True,
        },
        "unchanged": [
            "V34_A endpoint",
            "selected expert list",
            "blend weights",
            "split seeds 42/3407",
            "loss, iterations, features, post-processing",
        ],
        "diagnostics": built["diagnostics"],
        "seconds": float(built["seconds"]),
        "local_evidence": dict(local_evidence or {}),
        "rows": int(len(ids)),
        "columns": list(SUBMISSION_COLUMNS),
        "result_sha256": result_sha,
        "zip_sha256": zip_sha,
        "agent_uploads": 0,
        "platform_score_forecast": None,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "README.txt").write_text("\n".join([
        f"V5 seed-swap experiment package: {name}",
        "=" * (34 + len(str(name))),
        "",
        f"Status: {status}",
        f"Seed shift: +{int(shift)} applied to {', '.join(SEED_KEYS)}",
        "Recipe, weights, experts and split seeds are unchanged.",
        "",
        "Compare against the parent package V36_USER_REQUESTED_OUTER_FAILED (user-reported 96.2734).",
        "The difference bounds the platform resolution for this competition.",
        "",
        f"ZIP: {ZIP_NAME}",
        f"ZIP SHA-256: {zip_sha}",
        f"result.csv SHA-256: {result_sha}",
        "",
        "Upload is performed by the user.",
        "",
    ]), encoding="utf-8")
    with (out / "SHA256SUMS.txt").open("x", encoding="utf-8") as handle:
        handle.write(f"{result_sha}  result.csv\n{zip_sha}  {ZIP_NAME}\n")
    return {**manifest, "output": str(out)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--shift", type=int, default=DEFAULT_SHIFT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407])
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    spec = load_v5_spec(root)
    out = _private_output(root, args.output)
    payload: dict[str, Any] = {"shift": int(args.shift), "agent_uploads": 0}
    if not args.package_only:
        evidence = local_seed_sensitivity(root, spec, args.shift, seeds=tuple(args.seeds),
                                          workers=args.workers)
        payload["local_evidence"] = evidence
        out.mkdir(parents=True, exist_ok=True)
        (out / "local_seed_sensitivity.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    if not args.local_only:
        name = args.name or f"V5_SEED_SWAP_S{int(args.shift)}"
        manifest = build_seed_swap_package(root, spec, out / name, name, shift=int(args.shift),
                                           workers=int(args.workers),
                                           local_evidence=payload.get("local_evidence"))
        payload["package"] = manifest
    print(json.dumps({
        "shift": payload["shift"],
        "local_seed_sensitivity": {
            target: payload["local_evidence"]["targets"][target]["seed_summary"]
            for target in TARGETS
        } if "local_evidence" in payload else None,
        "package": payload.get("package", {}).get("output"),
        "zip_sha256": payload.get("package", {}).get("zip_sha256"),
    }, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
