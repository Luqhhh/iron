"""Round2 V6 pre-registration tests: frozen spec, sampler identity, safety rails.

The V6 design rests on signed facts from V5 (the winning member's signature, the
sibling-duplication finding, the retracted constant transfer error, and the fact
that the iron capacity lever was never pulled).  These tests pin both the frozen
numbers and the fact that the design cannot be silently weakened.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from bf_tap_r2.v6_sampler import (
    V6_CAPACITIES,
    V6_SETTINGS,
    V6_STRUCTURES,
    V6_TRIAL_COUNT,
    sample_v6_iron_capacity_trials,
    v6_family_key,
    v6_trial_id,
    verify_no_v36_collision,
)
from bf_tap_r2.v6_spec import (
    CURRENT_PLATFORM_BEST,
    DEFAULT_SPEC_PATH,
    load_v6_spec,
    validate_v6_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------

def test_frozen_v6_spec_loads() -> None:
    spec = load_v6_spec(REPO_ROOT)
    assert spec.version == "round2-v6-iron-capacity-networks"
    assert spec.raw["targets"]["current_platform_best"]["score"] == CURRENT_PLATFORM_BEST
    assert spec.raw["targets"]["current_platform_best"]["candidate"] == "V5_TIME_N0048_Q20"
    assert spec.stage_a["accuracy_ratio_max"] == 1.25
    assert spec.stage_b["accuracy_ratio_max"] == 1.15
    assert spec.promotion["min_seeds"] == 4
    assert spec.promotion["fold_level_role"] == "descriptive_only"
    assert spec.diversity["max_members_per_structural_family"] == 1
    assert spec.budget["stage_a_new_fits"] == 0
    assert spec.budget["stage_b"]["max_trials_per_target"] == 6
    assert spec.budget["stage_c"]["reuse_cached_baselines"] is True
    assert spec.budget["stage_d"]["max_packages"] == 1
    assert (REPO_ROOT / DEFAULT_SPEC_PATH).is_file()


def test_weakened_v6_spec_is_refused() -> None:
    spec = load_v6_spec(REPO_ROOT)

    loosened = copy.deepcopy(spec.raw)
    loosened["signature"]["stage_a"]["accuracy_ratio_max"] = 2.0
    with pytest.raises(ValueError):
        validate_v6_spec(loosened)

    two_seeds = copy.deepcopy(spec.raw)
    two_seeds["gates"]["promotion"]["min_seeds"] = 2
    with pytest.raises(ValueError):
        validate_v6_spec(two_seeds)

    no_retraction = copy.deepcopy(spec.raw)
    no_retraction["context_from_v5"]["transfer_error"]["retracted"] = "nothing"
    with pytest.raises(ValueError):
        validate_v6_spec(no_retraction)

    magnitude_forecast = copy.deepcopy(spec.raw)
    magnitude_forecast["context_from_v5"]["local_magnitude"]["role"] = "platform_forecast"
    with pytest.raises(ValueError):
        validate_v6_spec(magnitude_forecast)

    time_iron_swap = copy.deepcopy(spec.raw)
    for source in time_iron_swap["candidate_sources"]:
        if source["name"] == "s2_iron_capacity_networks":
            source["target"] = "tap_time_len"
    with pytest.raises(ValueError):
        validate_v6_spec(time_iron_swap)

    screened_without_fits = copy.deepcopy(spec.raw)
    for source in screened_without_fits["candidate_sources"]:
        if source["name"] == "s2_iron_capacity_networks":
            source["zero_fit_screen"] = True
    with pytest.raises(ValueError):
        validate_v6_spec(screened_without_fits)

    relaxed_gate = copy.deepcopy(spec.raw)
    relaxed_gate["gates"]["platform_submission"]["gate_rederivation"] = "DONE_BY_AGENT"
    with pytest.raises(ValueError):
        validate_v6_spec(relaxed_gate)


def test_stage_a_spends_no_fits_and_iron_target_is_primary() -> None:
    """The zero-fit screen must be the first gate, and the new space is iron-only."""
    spec = load_v6_spec(REPO_ROOT)
    assert spec.budget["stage_a_new_fits"] == 0
    zero_fit = [source for source in spec.sources if source["zero_fit_screen"]]
    assert {source["target"] for source in zero_fit} == {"tap_iron", "tap_time_len"}
    new_space = [source for source in spec.sources if not source["zero_fit_screen"]]
    assert len(new_space) == 1
    assert new_space[0]["target"] == "tap_iron"
    assert new_space[0]["expected_trials"] == V6_TRIAL_COUNT


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

def test_trial_id_mapping_is_frozen_and_indexed_in_declared_order() -> None:
    assert v6_trial_id("medium", "raw_mlp", "mse_adam") == "v6-s1-N-0000"
    assert v6_trial_id("medium", "raw_tabm", "mse_adam") == "v6-s1-N-0008"
    assert v6_trial_id("large", "raw_tabm", "mse_adam") == "v6-s1-N-0024"
    assert v6_trial_id("large", "ple_tabm", "mse_adamw") == "v6-s1-N-0031"
    for capacity in V6_CAPACITIES:
        for structure in V6_STRUCTURES:
            for setting in V6_SETTINGS:
                trial_id = v6_trial_id(capacity, structure, setting)
                assert trial_id.startswith("v6-s1-N-")
                assert int(trial_id.rsplit("-", 1)[1]) < V6_TRIAL_COUNT


def test_trial_id_rejects_unknown_keys() -> None:
    for args in (("huge", "raw_mlp", "mse_adam"),
                 ("large", "transformer", "mse_adam"),
                 ("large", "raw_mlp", "lbfgs")):
        with pytest.raises(ValueError):
            v6_trial_id(*args)


def test_sampler_expands_the_frozen_iron_capacity_space() -> None:
    trials = sample_v6_iron_capacity_trials(REPO_ROOT)
    assert len(trials) == V6_TRIAL_COUNT == 32
    assert {trial["target"] for trial in trials} == {"tap_iron"}
    assert {trial["line"] for trial in trials} == {"N"}
    families = [v6_family_key(trial) for trial in trials]
    assert len(set(families)) == 8
    assert all(families.count(name) == 4 for name in set(families))
    ids = [trial["trial_id"] for trial in trials]
    assert len(set(ids)) == len(ids)

    # the numeric space is inherited from the frozen V3.6 configuration, not redefined
    by_id = {trial["trial_id"]: trial for trial in trials}
    large_tabm = by_id[v6_trial_id("large", "raw_tabm", "mse_adam")]["parameters"]
    assert large_tabm["k"] == 32 and large_tabm["n_blocks"] == 3 and large_tabm["d_block"] == 512
    medium_tabm = by_id[v6_trial_id("medium", "raw_tabm", "mse_adam")]["parameters"]
    assert medium_tabm["k"] == 16 and medium_tabm["d_block"] == 256
    large_mlp = by_id[v6_trial_id("large", "raw_mlp", "mse_adam")]["parameters"]
    assert large_mlp["hidden"] == [512, 256] and large_mlp["dropout"] == 0.15
    assert large_mlp["random_seed"] == 42 and large_mlp["batch_size"] == 256
    assert large_mlp["inner_validation_folds"] == 5 and large_mlp["max_epochs"] == 120


def test_sampler_is_deterministic_and_collision_free() -> None:
    first = sample_v6_iron_capacity_trials(REPO_ROOT)
    second = sample_v6_iron_capacity_trials(REPO_ROOT)
    assert first == second
    verify_no_v36_collision(REPO_ROOT)

# ---------------------------------------------------------------------------
# Stage A screen
# ---------------------------------------------------------------------------

def _synthetic_screen_inputs():
    import numpy as np

    rng = np.random.default_rng(7)
    rows = 40
    actual = rng.normal(100.0, 10.0, rows)
    base = actual + rng.normal(0.0, 1.0, rows)
    folds = np.array([0] * 20 + [1] * 20)
    mask = np.ones(rows, dtype=bool)
    close = actual + 0.1 * (actual - base)          # slightly decorrelated, same accuracy
    duplicate = close + 0.01 * rng.normal(size=rows)
    # a candidate whose residual points in an independent direction: not a scaled
    # copy of the base residual, but far less accurate
    far = actual + rng.normal(0.0, 5.0, rows)
    return actual, base, folds, mask, close, duplicate, far


def test_selection_enforces_family_and_mutual_correlation_constraints() -> None:
    import numpy as np

    from bf_tap_r2.v6_screen import StageARecord, select_with_diversity

    actual, base, folds, mask, close, duplicate, far = _synthetic_screen_inputs()
    del folds
    first = StageARecord("t1", "tap_iron", "raw_tabm", "large", "mse_adam", "raw_tabm|large",
                         "test", 0.8, 1.0, 1.0, 1.05, 0.01, admissible=True)
    same_family = StageARecord("t2", "tap_iron", "raw_tabm", "large", "mae_adam", "raw_tabm|large",
                               "test", 0.8, 1.0, 1.0, 1.05, 0.01, admissible=True)
    other_family_dup = StageARecord("t3", "tap_iron", "ple_tabm", "large", "mse_adam", "ple_tabm|large",
                                    "test", 0.8, 1.0, 1.0, 1.05, 0.01, admissible=True)
    other_family_far = StageARecord("t4", "tap_iron", "raw_mlp", "large", "mse_adam", "raw_mlp|large",
                                    "test", 0.8, 1.0, 1.0, 1.05, 0.01, admissible=True)
    wrong = StageARecord("t5", "tap_iron", "ple_mlp", "large", "mse_adam", "ple_mlp|large",
                         "test", 0.8, 1.0, 1.0, 1.05, 0.01, admissible=False,
                         reasons=["projected_gain_not_positive"])
    records = [first, same_family, other_family_dup, other_family_far, wrong]
    vectors = {"t1": close, "t2": duplicate, "t3": duplicate, "t4": far, "t5": far}
    chosen = select_with_diversity(records, vectors, actual, mask, mutual_max=0.95, limit=6)
    ids = [record.trial_id for record in chosen]
    assert ids == ["t1", "t4"]
    assert same_family.selection_note == "family_already_selected"
    assert other_family_dup.selection_note.startswith("mutual_residual_correlation_above_max")
    assert wrong.selection_note == ""


def test_screen_flags_each_admissibility_reason(tmp_path: Path) -> None:
    import numpy as np

    from bf_tap_r2.v6_screen import screen_records_for_target
    from bf_tap_r2.v6_spec import load_v6_spec

    spec = load_v6_spec(REPO_ROOT)
    actual, base, folds, mask, close, duplicate, far = _synthetic_screen_inputs()
    del duplicate
    candidates = {
        "good": {"structure": "raw_tabm", "capacity_name": "large", "training_setting": "mse_adam"},
        "bad_ratio": {"structure": "ple_tabm", "capacity_name": "small", "training_setting": "mse_adam"},
    }
    records = _screen_with_vectors(
        screen_records_for_target, spec, actual, base, folds, mask, candidates,
        {"good": close, "bad_ratio": far}, tmp_path)
    by_id = {record.trial_id: record for record in records}
    assert by_id["good"].admissible is True
    assert by_id["bad_ratio"].admissible is False
    assert "accuracy_loss_above_max" in by_id["bad_ratio"].reasons


def _screen_with_vectors(screen_fn, spec, actual, base, folds, mask, candidates, vectors, tmp: Path):
    """Run the screen against in-memory vectors written to a temporary directory."""
    import numpy as np

    for trial_id, vector in vectors.items():
        np.save(tmp / f"pred-{trial_id}.npy", vector)
    return screen_fn("tap_iron", actual, base, mask, folds, candidates, tmp, spec)


@pytest.mark.skipif(not (REPO_ROOT / "local/runs/round2-v3.6-loss-training-and-numeric-encoding/"
                         "fixed-r2-final/fit_ledger.jsonl").is_file(),
                    reason="private V3.6 caches are absent")
def test_screen_reproduces_the_known_winner_as_admissible() -> None:
    """The V5 winner must pass Stage A when judged from the recorded folds 0/1 files.

    If this ever fails, the screen is mis-calibrated and a Stage A negative cannot
    be read as evidence about the candidate space.
    """
    import numpy as np

    from bf_tap_r2.v3_6_sampler import sample_v36
    from bf_tap_r2.v5_library import fold_vector, load_column_reference, load_v5_training_frame
    from bf_tap_r2.v6_screen import RECORDED_FOLDS, RECORDED_SEED, V36_FIXED_DIR, screen_records_for_target
    from bf_tap_r2.v6_spec import load_v6_spec

    spec = load_v6_spec(REPO_ROOT)
    train = load_v5_training_frame(REPO_ROOT)
    reference = load_column_reference(REPO_ROOT, train, spec)
    trials = {str(t["trial_id"]): t for t in sample_v36(REPO_ROOT)}
    folds = fold_vector(REPO_ROOT, train, RECORDED_SEED, None)
    mask = np.isin(folds, list(RECORDED_FOLDS))
    witnesses = ("v36-s1-N-0048", "v36-s1-N-0049", "v36-s1-N-0050", "v36-s1-N-0051")
    records = screen_records_for_target(
        "tap_time_len", train["tap_time_len"].to_numpy(dtype=float),
        reference.base_for("tap_time_len", RECORDED_SEED), mask, folds,
        {name: trials[name] for name in witnesses}, REPO_ROOT / V36_FIXED_DIR, spec)
    assert len(records) == 4
    for record in records:
        assert record.admissible is True, record.reasons
        assert record.residual_correlation < 0.90
        assert record.accuracy_ratio < 1.10
        assert record.projected_gain_score > 0.01
