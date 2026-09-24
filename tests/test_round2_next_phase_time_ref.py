"""Contract tests for the time-side L0 reconstruction.

The fitted members are deliberately left to the real run: their recipe
fidelity is judged by whether the reconstructed L0 reproduces its recorded
score, not by a unit test.  What is checked here is the part that can be
checked cheaply and would silently corrupt the reconstruction if wrong -- the
frozen weights, the trial lookup, and the target guards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.next_phase_members import EBMMember
from bf_tap_r2.next_phase_nested import Member
from bf_tap_r2.next_phase_time_ref import (
    B3_SEEDS,
    L0_TIME_RECIPE,
    PREFIX_TREES,
    AORDMember,
    B3Member,
    C2SeedMember,
    ORDMember,
    T1PrefixMember,
    V3TrialMember,
    WeightedMember,
    l0_time_components,
    l0_time_member,
    v3_time_0021,
)

ROOT = Path(__file__).resolve().parents[1]


class Constant(Member):
    def __init__(self, name: str, value: float, target: str = "tap_time_len"):
        self.name = name
        self.value = float(value)
        self.targets = (target,)

    def fit_predict(self, train, valid, target):
        return np.full(len(valid), self.value)


def test_l0_time_recipe_is_a_full_precision_simplex():
    assert len(L0_TIME_RECIPE) == 3
    assert sum(L0_TIME_RECIPE.values()) == pytest.approx(1.0, abs=1e-12)
    assert L0_TIME_RECIPE["v3-catboost-tap_time_len-0021"] == 0.6688582528038586
    assert L0_TIME_RECIPE["AORD"] == 0.17410974011267225
    assert L0_TIME_RECIPE["T1_B3_PREFIX1000"] == 0.15703200708346923


def test_v3_time_0021_is_reproduced_from_the_frozen_schedule():
    trial = v3_time_0021(ROOT)
    assert trial["trial_id"] == "v3-catboost-tap_time_len-0021"
    assert trial["family"] == "catboost"
    assert trial["target"] == "tap_time_len"
    assert trial["feature_set"] == "raw"
    assert trial["target_transform"] == "log1p"
    params = trial["parameters"]
    assert params["iterations"] == 3000
    assert params["depth"] == 4
    assert params["boosting_type"] == "Ordered"
    assert params["bootstrap_type"] == "Bernoulli"
    assert params["cat_features"] == ["spout_no"]


def test_b3_seeds_match_the_v2_1_contract():
    assert B3_SEEDS == (42, 2026, 2027)


def test_l0_time_members_and_weights_line_up():
    components = l0_time_components(ROOT)
    assert [member.name for member in components] == [
        "v3-catboost-tap_time_len-0021",
        "aord",
        "t1_b3_prefix1000",
    ]
    frozen = l0_time_member(ROOT)
    assert isinstance(frozen, WeightedMember)
    assert frozen.members and len(frozen.members) == len(frozen.weights) == 3
    assert frozen.targets == ("tap_time_len",)


def test_weighted_member_returns_the_frozen_weighted_mean():
    members = [Constant("a", 10.0), Constant("b", 20.0), Constant("c", 40.0)]
    combined = WeightedMember("mix", members, [0.5, 0.25, 0.25])
    values = combined.fit_predict(None, pd.DataFrame(index=range(4)), "tap_time_len")
    assert np.allclose(values, 0.5 * 10.0 + 0.25 * 20.0 + 0.25 * 40.0)


def test_weighted_member_rejects_weights_that_are_not_a_simplex():
    members = [Constant("a", 1.0), Constant("b", 2.0)]
    with pytest.raises(ValueError):
        WeightedMember("mix", members, [0.5, 0.6])
    with pytest.raises(ValueError):
        WeightedMember("mix", members, [1.5, -0.5])
    with pytest.raises(ValueError):
        WeightedMember("mix", members, [1.0])


def test_weighted_member_refuses_mixed_targets():
    with pytest.raises(ValueError):
        WeightedMember("mix", [Constant("a", 1.0), Constant("b", 1.0, "tap_iron")], [0.5, 0.5])


@pytest.mark.parametrize(
    "member",
    [
        C2SeedMember(42),
        ORDMember(),
        B3Member(),
        AORDMember(),
        T1PrefixMember(),
        V3TrialMember(v3_time_0021(ROOT)),
    ],
)
def test_time_members_reject_the_iron_target_before_fitting(member):
    """A wrong-target call must fail on the guard, not after a wasted fit."""
    dummy = pd.DataFrame({name: [0.0] for name in FEATURES} | {"spout_no": [1]})
    with pytest.raises(ValueError):
        member.fit_predict(dummy, dummy, "tap_iron")


def test_l1_time_members_are_the_four_replayed_s1_trials():
    from bf_tap_r2.next_phase_v31_ref import L1_TIME_MEMBERS
    from bf_tap_r2.next_phase_time_ref import V31TrialMember, l1_time_members

    members = l1_time_members(ROOT)
    assert [member.name for member in members] == list(L1_TIME_MEMBERS)
    for member in members:
        assert isinstance(member, V31TrialMember)
        assert member.targets == ("tap_time_len",)
        assert member.trial["trial_id"] == member.name
        assert member.trial["target"] == "tap_time_len"


def test_l1_time_members_reject_the_iron_target_before_fitting():
    from bf_tap_r2.next_phase_time_ref import l1_time_members

    dummy = pd.DataFrame({name: [0.0] for name in FEATURES} | {"spout_no": [1]})
    for member in l1_time_members(ROOT):
        with pytest.raises(ValueError):
            member.fit_predict(dummy, dummy, "tap_iron")


def test_prefix_member_rejects_an_impossible_prefix_at_construction():
    member = T1PrefixMember(ntree_end=PREFIX_TREES)
    assert member.ntree_end == 1000
    with pytest.raises(ValueError):
        T1PrefixMember(ntree_end=99999)
    with pytest.raises(ValueError):
        T1PrefixMember(ntree_end=0)


def test_time_members_are_target_scoped_so_pools_skip_them_for_iron():
    from bf_tap_r2.next_phase_nested import applies_to

    assert applies_to(C2SeedMember(42), "tap_time_len")
    assert not applies_to(C2SeedMember(42), "tap_iron")
    assert not applies_to(EBMMember("x", target="tap_time_len"), "tap_iron")
