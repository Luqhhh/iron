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
