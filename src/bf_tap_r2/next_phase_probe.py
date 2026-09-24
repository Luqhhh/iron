"""Batch-0 linear probe for the round2 next-phase search plan.

Answers one question before any reference rebuild: how much of each target's
variance is linearly explainable from the frozen raw21 features plus
``spout_no``?

That answer decides where the next-phase effort goes.  If a plain linear model
already lands close to the CatBoost families, the strong main effects have
absorbed the signal and the interaction-feature batch has little headroom.  If
it lands far away, pairwise structure is unexploited and the interaction batch
is worth running.

Three designs are compared, all fit on the training part of each fold only:

``raw``      intercept + 21 features + spout one-hot
``four``     ``raw`` plus the four ratio features already used by the V3 search
``degree2``  ``raw`` plus every pairwise product of the 21 features

Predictions are pooled across folds and scored with the frozen local package
score (pooled WMAPE, never a per-fold average).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .metrics import wmape
from .splits import make_folds
from .v2_release import load_v2
from .v3_local_search import package_local_score


def ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The four engineered ratios already searched over by the V3 rounds."""
    return pd.DataFrame(
        {
            "oxygen_per_air_volume": frame["oxygen"] / (60.0 * frame["air_volume"]),
            "pressure_per_air_volume": frame["total_press_diff"] / frame["air_volume"],
            "thermal_difference": frame["hot_air_temp"] - frame["furnace_throat_temp"],
            "upper_pressure_fraction": frame["upper_press_diff"] / frame["total_press_diff"],
        },
        index=frame.index,
    )


def build_design(train: pd.DataFrame, frame: pd.DataFrame, kind: str) -> np.ndarray:
    """Design matrix for ``frame`` using column statistics frozen from ``train``.

    Scaling comes from ``train`` only, so a validation fold never contributes to
    the centre or spread.  Without it the pairwise block would dominate the
    normal equations purely through unit magnitude.
    """
    columns = train[list(FEATURES)].to_numpy(dtype=float)
    centre = columns.mean(axis=0)
    scale = np.where((spread := columns.std(axis=0)) > 0, spread, 1.0)
    base = (frame[list(FEATURES)].to_numpy(dtype=float) - centre) / scale
    spout = (frame["spout_no"].to_numpy(dtype=float)[:, None] == np.array([[1.0, 2.0]])).astype(float)
    blocks = [base, spout]
    if kind in {"four", "degree2"}:
        blocks.append(ratio_features(frame).to_numpy(dtype=float))
    if kind == "degree2":
        blocks.append(
            np.hstack(
                [
                    (base[:, i] * base[:, j])[:, None]
                    for i in range(len(FEATURES))
                    for j in range(i + 1, len(FEATURES))
                ]
            )
        )
    return np.hstack([*blocks, np.ones((len(frame), 1))])


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, target: str, kind: str) -> np.ndarray:
    x_train = build_design(train, train, kind)
    x_valid = build_design(train, valid, kind)
    y_train = train[target].to_numpy(dtype=float)
    # Minimum-norm least squares: rank-deficient pairwise blocks are expected.
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return x_valid @ beta


def probe(root: Path, seeds: list[int], output: Path) -> dict:
    train = load_v2(root / "复赛_train", "train", 2754)
    results: dict[str, dict] = {}
    for kind in ("raw", "four", "degree2"):
        predictions = {target: np.full(len(train), np.nan) for target in TARGETS}
        per_seed: dict[str, dict] = {}
        for seed in seeds:
            folds = (
                make_folds(train, seed)
                .set_index("sample_id")
                .loc[train.sample_id, "fold"]
                .to_numpy()
            )
            seed_predictions = {target: np.full(len(train), np.nan) for target in TARGETS}
            for fold in range(5):
                training = train.loc[folds != fold].reset_index(drop=True)
                valid = train.loc[folds == fold].reset_index(drop=True)
                for target in TARGETS:
                    seed_predictions[target][folds == fold] = fit_predict(
                        training, valid, target, kind
                    )
            per_seed[str(seed)] = {
                target: wmape(train[target].to_numpy(dtype=float), seed_predictions[target])
                for target in TARGETS
            }
            for target in TARGETS:
                if np.isnan(predictions[target]).all():
                    predictions[target] = seed_predictions[target]
        w_i = wmape(train["tap_iron"].to_numpy(dtype=float), predictions["tap_iron"])
        w_t = wmape(train["tap_time_len"].to_numpy(dtype=float), predictions["tap_time_len"])
        results[kind] = {
            "pooled_W_I": w_i,
            "pooled_W_T": w_t,
            "package_score": package_local_score(w_i, w_t),
            "per_seed": per_seed,
        }
    payload = {
        "probe": "round2-next-phase batch 0 linear probe",
        "fold_seeds": seeds,
        "folds": 5,
        "note": "predictions pooled across folds; score uses the frozen pooled-WMAPE formula",
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "linear-probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407])
    parser.add_argument(
        "--output", type=Path, default=Path("local/runs/round2-next-phase/linear-probe-r1")
    )
    args = parser.parse_args()
    payload = probe(args.root, args.seeds, args.output)
    print(f"{'design':10s} {'W_I':>10s} {'W_T':>10s} {'score':>10s}")
    for kind, row in payload["results"].items():
        print(
            f"{kind:10s} {row['pooled_W_I']:10.6f} {row['pooled_W_T']:10.6f} "
            f"{row['package_score']:10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
