"""Round2 V5 protocol tests: admissibility, resolution, backtest, packaging.

The suite pins the parts that must not drift after the fact:

* the frozen pre-registration constants and the refusal to load a weakened spec;
* the statistics (paired summaries, one-sided t quantile, seed-level unit);
* the resolution insight that a **fold-level** lower bound admits the historical
  N2 candidate, which is why promotion is defined on the split seed;
* the backtest decision function against the published per-cell gains;
* the single-target packaging column-preservation contract;
* the seed-shift construction of the platform noise-floor experiment.

Library-dependent checks are skipped when the private OOF caches are absent so
the suite stays environment-portable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.v5_package import (
    DEFAULT_PARENT_CSV,
    _verify_unchanged_column,
    payload_with_parent_other_column,
    read_parent_columns,
)
from bf_tap_r2.v5_resolution import (
    CellObservation,
    admit,
    fold_criteria_met,
    package_score,
    paired_cells,
    paired_summary,
    seed_gains,
    seed_level_summary,
    t_quantile_one_sided,
)
from bf_tap_r2.v5_seed_swap import SEED_KEYS, shift_trial

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The recorded N2 fold gains, in score points, from
#: ``local/runs/round2-v4.2-structure-search/n2-seed-init-v2/coarse_summary.json``.
N2_FOLD_GAINS = {
    42: [0.031218935768848155, -0.009845099967634496, 0.017396882141554215,
         0.005787154503721581, 0.027914851892930415],
    3407: [0.02142384106100792, -0.003404169392268841, 0.013302425444265964,
           0.020939687642268723, 0.01871764034764567],
}


def _has_private_caches() -> bool:
    return (REPO_ROOT / "local/runs/round2-v3.6-loss-training-and-numeric-encoding/"
            "complete-dev-r2-final/seed-42/batch.json").is_file()


def _template_ids(count: int) -> list[str]:
    """Real submission IDs: ``validate_result`` checks the template row order."""
    template = REPO_ROOT / "复赛_test/result_template.csv"
    if not template.is_file():
        pytest.skip("submission template is absent")
    import pandas as pd

    ids = pd.read_csv(template, dtype={"sample_id": "string"}).sample_id.tolist()
    return [str(value) for value in ids[:count]]


# ---------------------------------------------------------------------------
# pre-registration constants
# ---------------------------------------------------------------------------

def test_frozen_spec_loads_with_preserved_user_target() -> None:
    from bf_tap_r2.v5_spec import load_v5_spec

    spec = load_v5_spec(REPO_ROOT)
    assert spec.version == "round2-v5-error-covariance-resolution"
    raw = spec.raw["targets"]
    assert raw["user_requested_platform_score_verbatim"] == 93.5
    assert raw["operational_platform_target"] == "platform_gt_96.3"
    assert raw["current_platform_best"]["score"] == 96.2734
    assert raw["local_working_gate"] == 96.25
    assert spec.targets == tuple(TARGETS)
    assert spec.admissibility["residual_correlation_max"] == 0.95
    assert spec.admissibility["single_wmape_ratio_max"] == 1.25
    alpha = spec.alpha_grid
    assert alpha[0] == pytest.approx(0.05)
    assert alpha[-1] == pytest.approx(0.5)


def test_weakened_spec_is_refused() -> None:
    import copy

    from bf_tap_r2.v5_spec import load_v5_spec, validate_v5_spec

    spec = load_v5_spec(REPO_ROOT)
    weakened = copy.deepcopy(spec.raw)
    weakened["admissibility"]["residual_correlation_max"] = 0.99
    with pytest.raises(ValueError):
        validate_v5_spec(weakened)

    relaxed = copy.deepcopy(spec.raw)
    relaxed["resolution"]["seed_level"]["min_seeds_for_promotion"] = 2
    with pytest.raises(ValueError):
        validate_v5_spec(relaxed)

    renamed = copy.deepcopy(spec.raw)
    renamed["targets"]["user_requested_platform_score_verbatim"] = 96.5
    with pytest.raises(ValueError):
        validate_v5_spec(renamed)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_paired_summary_mean_se_and_lower_bound() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    summary = paired_summary(values)
    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["se"] == pytest.approx(0.6454972243679028, rel=1e-12)
    expected = 2.5 - t_quantile_one_sided(3, 0.95) * summary["se"]
    assert summary["lcb95"] == pytest.approx(expected)
    assert summary["positive"] == 4


def test_single_observation_has_no_lower_bound() -> None:
    summary = paired_summary([0.25])
    assert summary["n"] == 1
    assert summary["lcb95"] is None
    assert summary["positive"] == 1


def test_one_sided_t_quantile_matches_table_values() -> None:
    assert t_quantile_one_sided(9, 0.95) == pytest.approx(1.8331129, rel=1e-6)
    assert t_quantile_one_sided(1, 0.95) == pytest.approx(6.3138, rel=1e-4)
    with pytest.raises(ValueError):
        t_quantile_one_sided(0)


def test_package_score_is_equal_target_wmape() -> None:
    assert package_score(0.03754247, 0.03838228) == pytest.approx(96.2037625, abs=1e-6)


def test_seed_unit_summary_and_gains() -> None:
    cells = [
        CellObservation(seed=42, fold=0, rows=551, wmape_base=0.10, wmape_candidate=0.08),
        CellObservation(seed=42, fold=1, rows=551, wmape_base=0.10, wmape_candidate=0.09),
        CellObservation(seed=3407, fold=0, rows=551, wmape_base=0.10, wmape_candidate=0.11),
        CellObservation(seed=3407, fold=1, rows=551, wmape_base=0.10, wmape_candidate=0.10),
    ]
    gains = seed_gains(cells)
    assert gains[42] == pytest.approx(50.0 * 0.015)
    assert gains[3407] == pytest.approx(50.0 * -0.005)
    summary = seed_level_summary(gains)
    assert summary["n"] == 2
    assert summary["positive_seeds"] == 1
    assert summary["seeds"] == [42, 3407]


def test_paired_cells_use_fold_masks_and_shapes() -> None:
    y = np.arange(10, dtype=float)
    folds = {42: np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])}
    base = {42: np.full(10, 4.5)}      # constant mid-range prediction
    cand = {42: y.copy()}              # candidate matches the labels exactly
    cells = paired_cells(y, folds, base, cand)
    assert [cell.fold for cell in cells] == [0, 1]
    assert all(cell.rows == 5 for cell in cells)
    assert all(cell.delta_score > 0 for cell in cells)
    assert cells[0].wmape_candidate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the resolution insight
# ---------------------------------------------------------------------------

def test_fold_level_lower_bound_admits_the_recorded_n2_candidate() -> None:
    """Why promotion is defined on the split seed rather than the fold.

    The historical N2 candidate lost 0.0201 on the platform, yet its ten
    recorded cells give a positive fold-level 95% lower bound.  If the test ever
    starts failing, the resolution reasoning in the pre-registration must be
    revisited rather than silently kept.
    """
    values = N2_FOLD_GAINS[42] + N2_FOLD_GAINS[3407]
    summary = paired_summary(values)
    assert summary["n"] == 10
    assert summary["positive"] == 8
    assert summary["mean"] == pytest.approx(0.0143452, abs=1e-6)
    assert summary["lcb95"] > 0.0


def test_two_seed_rule_is_rejected_by_the_seed_count_regardless_of_evidence() -> None:
    gains = {42: float(np.mean(N2_FOLD_GAINS[42])), 3407: float(np.mean(N2_FOLD_GAINS[3407]))}
    record = {
        "fold_summary": paired_summary(N2_FOLD_GAINS[42] + N2_FOLD_GAINS[3407]),
        "seed_gains": gains,
    }
    decision = admit(record, min_seeds=4, positive_cells_min=8, positive_cells_total=10)
    assert decision["admitted"] is False
    assert "insufficient_split_seeds" in decision["reasons"]


def test_fold_criteria_flag_insufficient_positive_cells() -> None:
    ok, reasons = fold_criteria_met({"n": 10, "positive": 7, "mean": 0.01}, 8, 10, True)
    assert ok is False
    assert "positive_cells_below_minimum" in reasons
    ok, reasons = fold_criteria_met({"n": 10, "positive": 9, "mean": 0.01}, 8, 10, False)
    assert ok is False
    assert "initial_split_seed_not_positive" in reasons


# ---------------------------------------------------------------------------
# backtest against the published evidence
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_private_caches(), reason="private round2 caches are absent")
def test_backtest_replays_published_candidates_and_meets_the_requirement() -> None:
    from bf_tap_r2.v5_diagnostics import backtest_discrimination
    from bf_tap_r2.v5_spec import load_v5_spec

    spec = load_v5_spec(REPO_ROOT)
    payload = backtest_discrimination(REPO_ROOT, spec)
    assert payload["known_bad_total"] == 4
    assert payload["requirement_met"] is True
    assert payload["known_bad_rejected"] >= 3
    rows = {row["candidate"]: row for row in payload["rows"]}
    n2 = rows["V42_iron_N2_quarter_full_coverage"]
    # At complete ten-cell coverage the fold-level rule admits the candidate that
    # later lost 0.0201 on the platform; only the four-seed requirement refuses it.
    assert n2["v5_fold_only_admitted"] is True
    assert n2["v5_admitted"] is False
    assert "insufficient_split_seeds" in n2["v5_reasons"]
    assert n2["platform_note"]


@pytest.mark.skipif(not _has_private_caches(), reason="private round2 caches are absent")
def test_library_audit_is_fully_covered_and_duplicate_free() -> None:
    from bf_tap_r2.v5_library import build_candidate_library
    from bf_tap_r2.v5_spec import load_v5_spec

    spec = load_v5_spec(REPO_ROOT)
    library = build_candidate_library(REPO_ROOT, spec)
    audit = library.audit
    assert audit["train_rows"] == 2754
    assert audit["two_seed_full_per_target"]["tap_iron"] >= 100
    assert audit["two_seed_full_per_target"]["tap_time_len"] >= 100
    assert audit["source_counts"]["v36_coarse_diagnostic"] == 224
    # No time-side N member reached complete coverage in the frozen V3.6 round;
    # that gap is the Stage 2a motivation and must not be silently "fixed".
    time_library = library.complete_entries("tap_time_len")
    assert not any(entry.family == "N" for entry in time_library.values())
    for entry in library.complete_entries("tap_iron").values():
        for vector in entry.vectors.values():
            assert vector.shape == (2754,)
            assert np.isfinite(vector).all()


# ---------------------------------------------------------------------------
# packaging contract
# ---------------------------------------------------------------------------

def test_payload_keeps_the_unchanged_column_field_text() -> None:
    ids = _template_ids(322)
    other_strings = ["1.0000000000000001e+02", "0.1", "123456789.12345679"] + ["7"] * 319
    values = np.array([10.0, 20.0, 30.0] + [1.0] * 319)
    payload = payload_with_parent_other_column(ids, "tap_iron", values, other_strings)
    text = payload.decode("utf-8")
    for raw in other_strings:
        assert raw in text
    assert text.endswith("\n")
    assert text.splitlines()[0] == "sample_id,pred_tap_iron,pred_tap_time_len"


def test_payload_rejects_negative_and_nonfinite_values() -> None:
    ids = ["A", "B"]
    strings = ["1", "2"]
    with pytest.raises(ValueError):
        payload_with_parent_other_column(ids, "tap_iron", np.array([1.0, -1.0]), strings)
    with pytest.raises(ValueError):
        payload_with_parent_other_column(ids, "tap_iron", np.array([1.0, np.nan]), strings)
    with pytest.raises(ValueError):
        payload_with_parent_other_column(ids, "not_a_target", np.array([1.0, 2.0]), strings)


def test_unchanged_column_verification_detects_a_rewritten_field(tmp_path: Path) -> None:
    ids = _template_ids(322)
    original = ["1.0000000000000001e+02"] + ["7"] * 321
    payload = payload_with_parent_other_column(ids, "tap_iron",
                                               np.array([1.0] * 322), original)
    path = tmp_path / "result.csv"
    path.write_bytes(payload)
    check = _verify_unchanged_column(path, "tap_iron", original)
    assert check["byte_identical"] is True
    path.write_text(payload.decode("utf-8").replace("1.0000000000000001e+02", "100.0"),
                    encoding="utf-8")
    mutated = _verify_unchanged_column(path, "tap_iron", original)
    assert mutated["byte_identical"] is False
    assert mutated["mismatches"] == 1


@pytest.mark.skipif(not (REPO_ROOT / DEFAULT_PARENT_CSV).is_file(), reason="parent package is absent")
def test_real_parent_package_reads_without_reformatting() -> None:
    ids, columns = read_parent_columns(REPO_ROOT / DEFAULT_PARENT_CSV)
    assert len(ids) == 322
    assert len(set(ids)) == 322
    for target in TARGETS:
        values, strings = columns[target]
        assert values.shape == (322,)
        assert np.isfinite(values).all()
        assert len(strings) == 322
        assert all(float(text) == value for text, value in zip(strings, values))


# ---------------------------------------------------------------------------
# seed-swap experiment construction
# ---------------------------------------------------------------------------

def test_shift_trial_moves_every_training_seed_in_a_copy() -> None:
    trial = {
        "trial_id": "v36-s1-N-0005",
        "parameters": {"random_seed": 42, "inner_validation_seed": 42, "learning_rate": 0.001},
    }
    shifted = shift_trial(trial, 1000)
    assert shifted["parameters"]["random_seed"] == 1042
    assert shifted["parameters"]["inner_validation_seed"] == 1042
    assert shifted["parameters"]["learning_rate"] == 0.001
    assert shifted["seed_shift"]["keys"] == ["random_seed", "inner_validation_seed"]
    # the source specification is never mutated
    assert trial["parameters"]["random_seed"] == 42
    zero = shift_trial(trial, 0)
    assert zero["parameters"] == trial["parameters"]


def test_shift_trial_refuses_a_spec_without_a_training_seed() -> None:
    with pytest.raises(ValueError):
        shift_trial({"trial_id": "x", "parameters": {"learning_rate": 0.1}}, 10)


@pytest.mark.skipif(not _has_private_caches(), reason="private round2 caches are absent")
def test_seed_swap_covers_the_frozen_expert_seed_keys() -> None:
    from bf_tap_r2.v5_seed_swap import load_v36_trials
    from bf_tap_r2.v5_spec import load_v5_spec

    spec = load_v5_spec(REPO_ROOT)
    trials = load_v36_trials(REPO_ROOT, spec)
    assert {int(key) for key in spec.raw["reference"]["frozen_folds"]} == {42, 3407, 2026}
    for trial_id in ("v36-s1-D-0029", "v36-s1-N-0005", "v36-s1-O-0057", "v36-s1-D-0048"):
        parameters = trials[trial_id]["parameters"]
        assert any(key in parameters for key in SEED_KEYS)

# ---------------------------------------------------------------------------
# replication vector resolution
# ---------------------------------------------------------------------------

def test_recorded_vector_prefers_the_v5_refinement_tree(tmp_path: Path) -> None:
    from bf_tap_r2.v5_replicate import _recorded_vector
    from bf_tap_r2.v5_spec import load_v5_spec

    spec = load_v5_spec(REPO_ROOT)
    vector = np.arange(2754, dtype=float)
    path = (tmp_path / "local/runs/round2-v5-error-covariance/time-n-family-r1/seed-42"
            / "pred-v36-s1-N-0049.npy")
    path.parent.mkdir(parents=True)
    np.save(path, vector)
    loaded = _recorded_vector(tmp_path, spec, "v36", "v36-s1-N-0049", 42)
    assert loaded.shape == (2754,)
    assert np.array_equal(loaded, vector)
    with pytest.raises(FileNotFoundError):
        _recorded_vector(tmp_path, spec, "v36", "v36-s1-N-9999", 42)


def test_recorded_vector_rejects_a_nonfinite_vector(tmp_path: Path) -> None:
    from bf_tap_r2.v5_replicate import _recorded_vector
    from bf_tap_r2.v5_spec import load_v5_spec

    spec = load_v5_spec(REPO_ROOT)
    vector = np.ones(2754, dtype=float)
    vector[0] = np.nan
    path = (tmp_path / "local/runs/round2-v5-error-covariance/time-n-family-r1/seed-42"
            / "pred-v36-s1-N-0049.npy")
    path.parent.mkdir(parents=True)
    np.save(path, vector)
    with pytest.raises(ValueError):
        _recorded_vector(tmp_path, spec, "v36", "v36-s1-N-0049", 42)

def test_fold_criteria_scale_with_the_number_of_cells() -> None:
    """The 8-of-10 reference is a fraction: 20 cells require 16, not 8."""
    ok, reasons = fold_criteria_met({"n": 20, "positive": 14, "mean": 0.01}, 8, 10, True)
    assert ok is False
    assert "positive_cells_below_minimum" in reasons
    ok, _ = fold_criteria_met({"n": 20, "positive": 16, "mean": 0.01}, 8, 10, True)
    assert ok is True
    ok, _ = fold_criteria_met({"n": 10, "positive": 8, "mean": 0.01}, 8, 10, True)
    assert ok is True
    ok, reasons = fold_criteria_met({"n": 10, "positive": 8, "mean": 0.01}, 8, 10, True)
    assert ok is True and not reasons


def test_admission_is_decided_on_the_seed_level_and_reports_the_fold_profile() -> None:
    """The specification marks the fold level ``descriptive_only``.

    Promotion therefore follows the four-seed criterion, while a weaker fold
    profile is surfaced next to it instead of being hidden or silently binding.
    """
    gains = {1: 0.010, 2: 0.010, 3: 0.010, 4: 0.010}
    record = {
        "fold_summary": {"n": 20, "positive": 14, "mean": 0.010, "sd": 0.013, "se": 0.003,
                         "lcb95": 0.0045},
        "seed_gains": gains,
    }
    decision = admit(record, min_seeds=4, positive_cells_min=8, positive_cells_total=10)
    assert decision["admitted"] is True
    assert decision["reasons"] == []
    assert decision["fold_criteria"]["met"] is False
    assert decision["fold_criteria"]["required_positive"] == 16
    assert decision["fold_criteria"]["role"] == "descriptive_only"
    strict = admit(record, min_seeds=4, positive_cells_min=8, positive_cells_total=10,
                   enforce_fold_criteria=True)
    assert strict["admitted"] is False
    assert "positive_cells_below_minimum" in strict["reasons"]


def test_admission_rejects_a_negative_seed_lower_bound_even_with_four_seeds() -> None:
    gains = {1: 0.01, 2: 0.01, 3: -0.02, 4: 0.01}
    record = {
        "fold_summary": {"n": 20, "positive": 15, "mean": 0.003, "sd": 0.013, "se": 0.003,
                         "lcb95": -0.003},
        "seed_gains": gains,
    }
    decision = admit(record, min_seeds=4, positive_cells_min=8, positive_cells_total=10)
    assert decision["admitted"] is False
    assert "seed_level_lcb_not_positive" in decision["reasons"]
    assert "initial_split_seed_not_positive" in decision["reasons"]
