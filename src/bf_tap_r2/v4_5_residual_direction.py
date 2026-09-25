"""Round2 V4.5: does the strong base's OOF residual carry usable direction?

Pre-registered in ``docs/round2_v4_5/PREREGISTRATION.md`` before this module was
run.  The question is deliberately split in two:

* **Q1 direction** -- can ``P(r > 0 | x)`` be predicted better than a calibrated
  constant?
* **Q2 utility** -- does a fitted conditional-median correction ``delta(x)``
  reduce absolute error on rows it has never seen?

Because WMAPE's denominator is fixed on a given evaluation set, each target is
an absolute-error problem, and ``dL/ddelta = 2 F_{r|x}(delta) - 1`` makes the
conditional median the L1-optimal correction.  Error *magnitude* being
predictable is explicitly not evidence for either question.

No base model is fitted here.  The module reads the frozen ``B_fit`` OOF
predictions that the V4.1-V4.4 screens already computed (both targets, two split
seeds, five folds) and verifies their structure before using them.

Nothing here writes outside ``local/`` and no submission artifact is produced.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import TARGETS
from .v3_run import load_fold_vector, load_training_frame
from .v4_2_prep import FEATURES, exclusion_group_keys

__all__ = [
    "DEFAULT_BASELINE_PATH",
    "DEFAULT_OUTPUT",
    "FOLDS",
    "SEEDS",
    "CellEvaluation",
    "evaluate_cells",
    "load_oof_dataset",
    "run_diagnostic",
]

DEFAULT_BASELINE_PATH = (
    "local/runs/round2-v4.4-mechanism-completion/N2-tap_iron/baselines.npz"
)
DEFAULT_OUTPUT = "local/runs/round2-v4.5-residual-direction/diagnostic-r1"
SEEDS: tuple[int, ...] = (42, 3407)
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def _require_private(root: Path, output: Path) -> Path:
    """The same private-path rule the other runners use, kept local on purpose.

    Importing the rule from the V4.2-r2 follow-up runner would make this module
    depend on a file that is still uncommitted in the shared checkout.
    """
    resolved = Path(output).resolve()
    private = (Path(root).resolve() / "local").resolve()
    if private not in resolved.parents and resolved != private:
        raise ValueError(f"V4.5 output must stay under {private}: {resolved}")
    return resolved


@dataclass(frozen=True)
class CellEvaluation:
    """One (seed, fold, target) held-out evaluation."""

    seed: int
    fold: int
    target: str
    feature_set: str
    n_rows: int
    sign_auc: float
    sign_accuracy: float
    sign_base_rate: float
    brier_model: float
    brier_constant: float
    mae_delta0: float
    mae_delta_const: float
    mae_delta_sign: float
    mae_delta_median: float
    residual_mean: float
    residual_median: float
    residual_std: float
    residual_positive_rate: float

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["delta_mae_const"] = self.mae_delta0 - self.mae_delta_const
        payload["delta_mae_sign"] = self.mae_delta0 - self.mae_delta_sign
        payload["delta_mae_median"] = self.mae_delta0 - self.mae_delta_median
        return payload


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC, defined for a single-class fold (returns 0.5)."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=float)
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[start:index] = ranks[start:index].mean()
            start = index
    # ``ranks`` is in sorted order; map back to the original positions.
    restored = np.empty(len(scores), dtype=float)
    restored[order] = ranks
    return float(
        (restored[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def load_oof_dataset(
    root: Path | str,
    baseline_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Load the frame, the fold vectors and the verified OOF base predictions.

    The structure is verified rather than assumed: every row is non-NaN in
    exactly one fold per seed and target, the non-NaN pattern equals the
    repository's own fold vector, and no duplicate-feature group straddles two
    folds.
    """
    root = Path(root)
    train = load_training_frame(root).reset_index(drop=True)
    path = Path(baseline_path or DEFAULT_BASELINE_PATH)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"OOF baseline cache not found: {path}")
    store = np.load(path, allow_pickle=True)

    folds = {
        seed: np.asarray(load_fold_vector(root, train, seed), dtype=int)
        for seed in SEEDS
    }
    frame = train.copy()
    for seed in SEEDS:
        frame[f"fold_{seed}"] = folds[seed]

    for seed in SEEDS:
        expected = folds[seed]
        for target_index, target in enumerate(TARGETS):
            covered = np.zeros(len(frame), dtype=int)
            values = np.full(len(frame), np.nan)
            for fold in FOLDS:
                key = f"s{seed}-f{fold}-pred"
                if key not in store:
                    raise KeyError(f"OOF baseline cache lacks {key}")
                column = np.asarray(store[key], dtype=float)
                if column.shape != (len(frame), len(TARGETS)):
                    raise ValueError(
                        f"{key} has shape {column.shape}, expected "
                        f"{(len(frame), len(TARGETS))}"
                    )
                present = np.isfinite(column[:, target_index])
                if not np.array_equal(present, expected == fold):
                    raise ValueError(
                        f"{key} column {target_index} does not match the fold vector"
                    )
                values[present] = column[present, target_index]
                covered[present] += 1
            if not (covered == 1).all():
                raise ValueError(
                    f"seed {seed} {target}: OOF coverage is not exactly one fold per row"
                )
            frame[f"y_hat_{target}"] = values

        keys = exclusion_group_keys(frame)
        for key in pd.unique(keys):
            member = keys == key
            if len(pd.unique(expected[member])) > 1:
                raise ValueError(
                    f"seed {seed}: duplicate-feature group {key!r} spans folds"
                )

    frame["spout_no_indicator"] = (
        pd.to_numeric(frame["spout_no"], errors="coerce").fillna(0.0) > 0
    ).astype(float)
    for target in TARGETS:
        frame[f"abs_y_hat_{target}"] = frame[f"y_hat_{target}"].abs()
    return frame, folds


def _feature_matrix(
    frame: pd.DataFrame, target: str, *, include_other_target: bool
) -> np.ndarray:
    other = TARGETS[1] if target == TARGETS[0] else TARGETS[0]
    columns = [
        *FEATURES, "spout_no_indicator", f"y_hat_{target}", f"abs_y_hat_{target}",
    ]
    if include_other_target:
        columns.append(f"y_hat_{other}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite corrector feature for {target}")
    return values


def evaluate_cells(
    frame: pd.DataFrame,
    folds: Mapping[int, np.ndarray],
    *,
    target: str,
    include_other_target: bool = False,
    seeds: Sequence[int] = SEEDS,
) -> list[CellEvaluation]:
    """Leave-one-fold-out evaluation for one target and one feature set."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_all = frame[target].to_numpy(dtype=float)
    y_hat_all = frame[f"y_hat_{target}"].to_numpy(dtype=float)
    residual_all = y_all - y_hat_all
    features = _feature_matrix(
        frame, target, include_other_target=include_other_target
    )
    feature_set = "plus_other" if include_other_target else "own"

    results: list[CellEvaluation] = []
    for seed in seeds:
        assignment = np.asarray(folds[int(seed)], dtype=int)
        for fold in FOLDS:
            test = assignment == int(fold)
            train = ~test
            if not test.any() or not train.any():
                continue
            scaler = StandardScaler().fit(features[train])
            x_train = scaler.transform(features[train])
            x_test = scaler.transform(features[test])
            r_train = residual_all[train]
            r_test = residual_all[test]

            positive = (r_train > 0).astype(int)
            base_rate = float(positive.mean())
            if positive.min() == positive.max():
                probability = np.full(int(test.sum()), base_rate)
            else:
                logistic = LogisticRegression(max_iter=2000, C=1.0)
                logistic.fit(x_train, positive)
                probability = logistic.predict_proba(x_test)[:, 1]

            quantile = GradientBoostingRegressor(
                loss="quantile", alpha=0.5, n_estimators=200, max_depth=3,
                learning_rate=0.05, random_state=0,
            )
            quantile.fit(x_train, r_train)
            correction_median = quantile.predict(x_test)

            constant = float(np.median(r_train))
            magnitude = float(np.median(np.abs(r_train)))
            correction_sign = np.where(probability > 0.5, magnitude, -magnitude)

            results.append(
                CellEvaluation(
                    seed=int(seed),
                    fold=int(fold),
                    target=str(target),
                    feature_set=feature_set,
                    n_rows=int(test.sum()),
                    sign_auc=_auc(r_test > 0, probability),
                    sign_accuracy=float(
                        ((probability > 0.5) == (r_test > 0)).mean()
                    ),
                    sign_base_rate=base_rate,
                    brier_model=float(np.mean((probability - (r_test > 0)) ** 2)),
                    brier_constant=float(np.mean((base_rate - (r_test > 0)) ** 2)),
                    mae_delta0=float(np.mean(np.abs(r_test))),
                    mae_delta_const=float(np.mean(np.abs(r_test - constant))),
                    mae_delta_sign=float(np.mean(np.abs(r_test - correction_sign))),
                    mae_delta_median=float(np.mean(np.abs(r_test - correction_median))),
                    residual_mean=float(r_test.mean()),
                    residual_median=float(np.median(r_test)),
                    residual_std=float(r_test.std(ddof=1)) if len(r_test) > 1 else 0.0,
                    residual_positive_rate=float((r_test > 0).mean()),
                )
            )
    return results


def decide(rows: Sequence[CellEvaluation]) -> dict[str, Any]:
    """Apply the pre-registered decision rule (section 6) verbatim."""
    decisions: dict[str, Any] = {}
    for target in TARGETS:
        cells = [row for row in rows if row.target == target]
        if not cells:
            continue
        auc = float(np.mean([row.sign_auc for row in cells]))
        improved = sum(1 for row in cells if row.mae_delta_median < row.mae_delta0)
        per_seed = {}
        for seed in SEEDS:
            seed_cells = [row for row in cells if row.seed == seed]
            if seed_cells:
                per_seed[str(seed)] = float(
                    np.mean(
                        [row.mae_delta0 - row.mae_delta_median for row in seed_cells]
                    )
                )
        values = list(per_seed.values())
        same_sign = len(values) == 2 and (all(v > 0 for v in values)
                                          or all(v < 0 for v in values))
        if auc <= 0.52 or improved < 8:
            verdict = "CLOSE_RESIDUAL_ROUTE"
        elif auc >= 0.55 and improved >= 8 and same_sign:
            verdict = "ESCALATE_TO_SCALE_DIAGNOSTIC"
        else:
            verdict = "WEAK_INCONCLUSIVE"
        decisions[target] = {
            "mean_sign_auc": round(auc, 6),
            "mean_sign_accuracy": round(
                float(np.mean([row.sign_accuracy for row in cells])), 6
            ),
            "mean_brier_model": round(
                float(np.mean([row.brier_model for row in cells])), 6
            ),
            "mean_brier_constant": round(
                float(np.mean([row.brier_constant for row in cells])), 6
            ),
            "cells_where_median_correction_improves_mae": int(improved),
            "cells_total": int(len(cells)),
            "mean_delta_mae_median_per_seed": {
                key: round(value, 8) for key, value in per_seed.items()
            },
            "per_seed_improvement_same_sign": bool(same_sign),
            "verdict": verdict,
        }
    return decisions


def run_diagnostic(
    root: Path | str = ".",
    output: Path | str | None = None,
    *,
    baseline_path: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    frame, folds = load_oof_dataset(root, baseline_path)
    target_output = _require_private(
        root, Path(output) if output is not None else root / DEFAULT_OUTPUT
    )
    target_output.mkdir(parents=True, exist_ok=True)

    primary: list[CellEvaluation] = []
    augmented: list[CellEvaluation] = []
    for target in TARGETS:
        primary.extend(evaluate_cells(frame, folds, target=target))
    for target in TARGETS:
        augmented.extend(
            evaluate_cells(frame, folds, target=target, include_other_target=True)
        )

    payload = {
        "kind": "V45_RESIDUAL_DIRECTION_DIAGNOSTIC",
        "preregistration": "docs/round2_v4_5/PREREGISTRATION.md",
        "baseline_path": str(baseline_path or DEFAULT_BASELINE_PATH),
        "n_rows": int(len(frame)),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "primary_feature_set": "own",
        "cells": [row.as_dict() for row in primary],
        "cells_plus_other": [row.as_dict() for row in augmented],
        "decisions": decide(primary),
        "decisions_plus_other": decide(augmented),
        "note": (
            "Magnitude predictability is not evidence for direction. The verdict "
            "follows docs/round2_v4_5/PREREGISTRATION.md section 6 verbatim."
        ),
        "platform_uploads": 0,
        "submission_packages": 0,
    }
    (target_output / "diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"output": str(target_output), **payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-path", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_diagnostic(
        args.root, args.output, baseline_path=args.baseline_path
    )
    print(json.dumps(result["decisions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
