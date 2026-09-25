"""Round2 V4.5 residual-direction diagnostic: protocol and decision-rule tests.

These tests pin the parts that must not drift after the fact: the rank AUC
including ties and single-class folds, the pre-registered decision rule with all
three verdict branches, the private-output guard, and the corrector feature
matrix. The OOF loader is also checked end to end when the local baseline cache
is present, and skipped when it is not, so the suite stays environment-portable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.v4_2_prep import FEATURES
from bf_tap_r2.v4_5_residual_direction import (
    DEFAULT_BASELINE_PATH,
    FOLDS,
    SEEDS,
    CellEvaluation,
    _auc,
    _feature_matrix,
    _require_private,
    decide,
    load_oof_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# rank AUC
# ---------------------------------------------------------------------------

def test_auc_perfect_and_reversed_ranking() -> None:
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert _auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert _auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_ties_contribute_one_half_and_single_class_is_undefined() -> None:
    labels = np.array([0, 1, 0, 1], dtype=bool)
    # Every score identical -> no ranking information -> exactly chance.
    assert _auc(labels, np.zeros(4)) == pytest.approx(0.5)
    # One tied pair straddling the classes and one correctly ordered pair.
    scores = np.array([0.0, 0.0, 0.0, 1.0])
    labels = np.array([0, 1, 0, 1], dtype=bool)
    assert _auc(labels, scores) == pytest.approx(0.75)
    # A fold with only one class carries no information and must not crash.
    assert _auc(np.ones(5, dtype=bool), np.linspace(0, 1, 5)) == 0.5
    assert _auc(np.zeros(5, dtype=bool), np.linspace(0, 1, 5)) == 0.5


# ---------------------------------------------------------------------------
# pre-registered decision rule
# ---------------------------------------------------------------------------

def _cell(
    target: str,
    seed: int,
    fold: int,
    *,
    auc: float = 0.5,
    mae0: float = 1.0,
    mae_median: float = 1.0,
) -> CellEvaluation:
    return CellEvaluation(
        seed=seed, fold=fold, target=target, feature_set="own", n_rows=100,
        sign_auc=auc, sign_accuracy=0.5, sign_base_rate=0.5,
        brier_model=0.25, brier_constant=0.25,
        mae_delta0=mae0, mae_delta_const=mae0, mae_delta_sign=mae0,
        mae_delta_median=mae_median,
        residual_mean=0.0, residual_median=0.0, residual_std=1.0,
        residual_positive_rate=0.5,
    )


def _grid(target: str, *, auc: float, mae0: float, improved: int) -> list[CellEvaluation]:
    """Ten cells (2 seeds x 5 folds); ``improved`` of them beat delta0."""
    cells = []
    index = 0
    for seed in SEEDS:
        for fold in FOLDS:
            better = index < improved
            cells.append(
                _cell(
                    target, seed, fold, auc=auc, mae0=mae0,
                    mae_median=mae0 - 0.1 if better else mae0 + 0.1,
                )
            )
            index += 1
    return cells


def test_verdict_closes_when_auc_is_at_chance() -> None:
    rows = _grid("tap_iron", auc=0.50, mae0=1.0, improved=10)
    verdict = decide(rows)["tap_iron"]
    assert verdict["verdict"] == "CLOSE_RESIDUAL_ROUTE"
    assert verdict["mean_sign_auc"] == pytest.approx(0.50, abs=1e-9)


def test_verdict_closes_when_the_correction_rarely_helps() -> None:
    # High AUC but the correction fails on 3 of 10 cells -> still closed.
    rows = _grid("tap_iron", auc=0.60, mae0=1.0, improved=7)
    verdict = decide(rows)["tap_iron"]
    assert verdict["verdict"] == "CLOSE_RESIDUAL_ROUTE"
    assert verdict["cells_where_median_correction_improves_mae"] == 7


def test_verdict_escalates_only_with_a_stable_signal() -> None:
    rows = _grid("tap_iron", auc=0.58, mae0=1.0, improved=10)
    verdict = decide(rows)["tap_iron"]
    assert verdict["verdict"] == "ESCALATE_TO_SCALE_DIAGNOSTIC"
    assert verdict["per_seed_improvement_same_sign"] is True


def test_verdict_is_weak_when_seeds_disagree() -> None:
    """Eight cells improve, AUC is high, but the two seeds disagree in sign.

    Seed 42 improves on all five cells; seed 3407 improves on three by a little
    and loses on two by a lot, so the pooled count still clears the 8-cell bar
    while the seed directions have opposite signs. That is exactly the
    "not stable enough" case the rule must refuse to escalate.
    """
    rows = _grid("tap_iron", auc=0.58, mae0=1.0, improved=10)  # seed 42 half
    seed_3407 = [
        _cell("tap_iron", 3407, fold, auc=0.58, mae0=1.0,
              mae_median=1.0 - 0.01 if fold < 3 else 1.0 + 0.5)
        for fold in FOLDS
    ]
    rows = [row for row in rows if row.seed == 42] + seed_3407
    verdict = decide(rows)["tap_iron"]
    assert verdict["cells_where_median_correction_improves_mae"] == 8
    assert verdict["mean_sign_auc"] >= 0.55
    assert verdict["per_seed_improvement_same_sign"] is False
    assert verdict["verdict"] == "WEAK_INCONCLUSIVE"


def test_decision_reports_every_registered_target() -> None:
    rows = _grid(TARGETS[0], auc=0.5, mae0=1.0, improved=0)
    rows += _grid(TARGETS[1], auc=0.5, mae0=1.0, improved=0)
    decisions = decide(rows)
    assert set(decisions) == set(TARGETS)


# ---------------------------------------------------------------------------
# private output guard and feature matrix
# ---------------------------------------------------------------------------

def test_private_output_guard_rejects_a_public_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "local").mkdir(parents=True)
    assert _require_private(root, root / "local/runs/x").name == "x"
    with pytest.raises(ValueError):
        _require_private(root, root / "docs/x")


def test_feature_matrix_shapes_and_other_target_block() -> None:
    frame = pd.DataFrame({name: np.arange(6, dtype=float) for name in FEATURES})
    frame["spout_no_indicator"] = 1.0
    for target in TARGETS:
        frame[f"y_hat_{target}"] = 0.5
        frame[f"abs_y_hat_{target}"] = 0.5
    own = _feature_matrix(frame, TARGETS[0], include_other_target=False)
    other = _feature_matrix(frame, TARGETS[0], include_other_target=True)
    assert own.shape == (6, len(FEATURES) + 3)
    assert other.shape == (6, len(FEATURES) + 4)


# ---------------------------------------------------------------------------
# end-to-end OOF loader (needs the local, git-ignored baseline cache)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / DEFAULT_BASELINE_PATH).is_file(),
    reason="local OOF baseline cache is not present in this checkout",
)
def test_oof_loader_covers_every_row_exactly_once() -> None:
    frame, folds = load_oof_dataset(REPO_ROOT)
    assert len(frame) == 2754
    for seed in SEEDS:
        assert set(folds[seed]) == set(FOLDS)
        for target in TARGETS:
            covered = np.zeros(len(frame), dtype=int)
            for fold in FOLDS:
                covered[(folds[seed] == fold)] += 1
            assert (covered == 1).all()
            assert np.isfinite(frame[f"y_hat_{target}"]).all()
    # The verification inside the loader already asserts the fold/NaN pattern
    # and group safety; this test pins the coverage contract on top.
