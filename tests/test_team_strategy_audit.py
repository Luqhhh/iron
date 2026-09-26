import numpy as np
import pytest

from bf_tap_r2.team_strategy_audit import (
    alpha_curve_audit, audit_selection_boundary, holm_adjust, shared_label_probe,
)


def test_cross_seed_selection_changes_when_only_evaluation_fold_labels_change():
    probe = shared_label_probe()
    assert probe["prediction_arrays_unchanged"]
    assert probe["changed_rows_are_seed42_evaluation_fold0"]
    assert probe["seed42_alpha_before"] == 0
    assert probe["seed42_alpha_after"] == 1


def test_outer_boundary_refuses_same_rows_even_under_different_split_seeds():
    with pytest.raises(ValueError, match="Evaluation labels"):
        audit_selection_boundary(["a", "b"], ["b", "c"], [["a"]])


def test_selection_predictions_must_not_be_trained_on_outer_evaluation_rows():
    with pytest.raises(ValueError, match="outer evaluation rows"):
        audit_selection_boundary(["a", "b"], ["c"], [["a", "c"]])


def test_honest_selection_attestation_and_missing_provenance():
    result = audit_selection_boundary(["a", "b"], ["c"], [["a"], ["b"]])
    assert result["row_boundary"] == "PASS"
    with pytest.raises(ValueError, match="provenance"):
        audit_selection_boundary(["a", "b"], ["c"], [])


def test_held_out_labels_do_not_enter_honest_boundary_selection():
    from bf_tap_r2.v5_resolution import select_alpha
    y = np.array([10., 10., 11., 11.])
    selected = np.array([True, True, False, False])
    base, candidate = np.full(4, 10.), np.full(4, 12.)
    before = select_alpha(y[selected], base[selected], candidate[selected], [0., 1.])
    y[~selected] = 12.
    after = select_alpha(y[selected], base[selected], candidate[selected], [0., 1.])
    assert before == after


def test_alpha_curve_bound_does_not_claim_quadratic_peak():
    result = alpha_curve_audit([(0, 96.2734), (.05, 96.2844), (.2, 96.3143), (.35, 96.3366)])
    assert result["consistent_with_concavity_at_receipt_precision"]
    assert result["maximum_score_on_remaining_alpha_interval_upper_bound"] == pytest.approx(96.4332333333)
    assert result["peak_location"] == "not_identified_by_these_points"


def test_incompatible_curve_and_duplicate_alpha_are_detected():
    assert not alpha_curve_audit([(0, 96.), (.5, 96.1), (1., 97.)])["consistent_with_concavity_at_receipt_precision"]
    with pytest.raises(ValueError, match="Duplicate"):
        alpha_curve_audit([(0, 96.), (0, 97.)])


def test_holm_adjustment_preserves_original_feature_order():
    np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])
