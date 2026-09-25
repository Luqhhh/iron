"""Tests for replaying the V3.1 schedule without its run ledger.

The replay is a lookup, so what matters is that the lookup resolves to real
trials, that the schedule is well formed, and that the default path is
untouched: omitting ``centers`` must still go to the ledger and still fail
when it is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bf_tap_r2.next_phase_v31_ref import (
    L1_IRON_MEMBERS,
    L1_TIME_MEMBERS,
    reconstructed_centers,
    v31_s1_trials,
    v31_trial_by_id,
)
from bf_tap_r2.v3_1_sampler import IRON_CENTERS, TIME_CENTERS, sample_s1

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CENTRES = set(TIME_CENTERS) | set(IRON_CENTERS)


def test_centres_resolve_to_first_batch_v3_trials():
    centres = reconstructed_centers(ROOT)
    assert set(centres) == EXPECTED_CENTRES
    for trial_id, trial in centres.items():
        assert trial["trial_id"] == trial_id
        assert trial["family"] == "catboost"
        assert "parameters" in trial


def test_replayed_schedule_is_well_formed():
    trials = v31_s1_trials(ROOT)
    assert len(trials) == 320
    ids = [trial["trial_id"] for trial in trials]
    assert len(set(ids)) == len(ids)
    families = set()
    for trial in trials:
        assert trial["target"] in {"tap_iron", "tap_time_len"}
        families.add(trial.get("family"))
        # Only the tree family takes spout_no as a declared categorical; the
        # residual/spline members carry different parameter keys.
        if trial.get("family") == "catboost":
            assert trial["parameters"]["cat_features"] == ["spout_no"]
    assert "catboost" in families


def test_every_l1_member_is_in_the_replayed_schedule():
    ids = {trial["trial_id"] for trial in v31_s1_trials(ROOT)}
    for wanted in (*L1_IRON_MEMBERS, *L1_TIME_MEMBERS):
        assert wanted in ids, wanted


def test_mutants_inherit_their_centre_target():
    iron_ids = {tid for tid in IRON_CENTERS}
    for trial in v31_s1_trials(ROOT):
        if trial.get("line") in {"iron_core", "iron_wide"}:
            assert trial["target"] == "tap_iron"
        if trial.get("line") == "time_core":
            assert trial["target"] == "tap_time_len"
    assert iron_ids  # the centres exist as a set, sanity on the import


def test_trial_lookup_reports_a_missing_id():
    with pytest.raises(ValueError):
        v31_trial_by_id(ROOT, "v31-s1-not-a-trial-9999")


def test_default_path_still_requires_the_ledger(tmp_path: Path):
    """The refactor must not have made the ledger optional by accident.

    The assertion is about the *missing-ledger* path, so it must not depend on
    whether this checkout still holds the V3.1 run ledger.  An empty root is
    used instead: ``load_centers`` then resolves nothing and must fail loudly
    rather than silently substituting a reconstruction.
    """
    with pytest.raises(ValueError, match="Missing V3.1 centers"):
        sample_s1(tmp_path)
