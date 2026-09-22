from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bf_tap_r2.candidate_tiers import classify_candidates


def setup():
    policy = yaml.safe_load(Path("configs/candidate_tiers.yaml").read_text())
    spec = {"split_seeds": [42, 3407], "folds": 5,
            "reference_by_target": {"iron": "anchor"},
            "candidates": {"iron": ["stable", "small", "flat"]},
            "tie_preference_by_target": {"iron": ["stable", "small", "flat"]}}
    def score(v):
        return {"wmape": v, "by_fold": {str(i): v for i in range(5)}, "by_spout": {"1": v}}
    metrics = {"iron": {r: {"42": score(v), "3407": score(v)}
                        for r, v in [("anchor", .04), ("stable", .039), ("small", .0399), ("flat", .04)]}}
    metrics["iron"]["small"]["3407"] = score(.04005)
    return metrics, spec, policy


def test_stable_first_exploration_retained_without_mutation():
    m, s, p = setup()
    before = deepcopy(m)
    result = classify_candidates(m, s, p)
    assert [r["candidate"] for r in result["submission_priority"]] == ["stable", "small"]
    small = result["decisions"]["iron"]["small"]
    assert small["tier"] == "exploration" and not small["eligible"]
    assert small["failed_conditions"] == ["both_splits_improve", "minimum_improved_folds"]
    assert result["decisions"]["iron"]["flat"]["tier"] == "not_shortlisted"
    assert not result["release_authorized"] and m == before


def test_spout_failure_is_visible_and_exploration_limit_is_global():
    m, s, p = setup()
    m["iron"]["stable"]["42"]["by_spout"]["1"] = .042
    result = classify_candidates(m, s, p)
    assert not result["formal_selected"]
    assert len(result["exploration_selected"]) == 1
    assert result["exploration_selected"][0]["candidate"] == "stable"
    assert result["exploration_selected"][0]["failed_conditions"] == ["spout_risk"]
    assert result["decisions"]["iron"]["small"]["tier"] == "exploration"


@pytest.mark.parametrize("change", ["nan", "fold", "split"])
def test_invalid_evidence_cannot_become_exploration(change):
    m, s, p = setup()
    if change == "nan":
        m["iron"]["small"]["42"]["wmape"] = float("nan")
    elif change == "fold":
        del m["iron"]["small"]["42"]["by_fold"]["0"]
    else:
        del m["iron"]["small"]["3407"]
    with pytest.raises(ValueError):
        classify_candidates(m, s, p)


def test_global_exploration_cap_and_target_specific_anchor():
    m, s, p = setup()
    m["time"] = deepcopy(m["iron"])
    m["time"]["new_anchor"] = m["time"].pop("anchor")
    s["reference_by_target"]["time"] = "new_anchor"
    s["candidates"]["time"] = list(s["candidates"]["iron"])
    s["tie_preference_by_target"]["time"] = list(s["tie_preference_by_target"]["iron"])
    result = classify_candidates(m, s, p)
    assert len(result["formal_selected"]) == 2
    assert len(result["exploration_selected"]) == 1
    assert result["decisions"]["time"]["small"]["reference"] == "new_anchor"


def test_no_gain_within_numerical_tolerance_is_not_shortlisted():
    m, s, p = setup()
    for score in m["iron"]["flat"].values():
        score["wmape"] -= 1e-13
    assert classify_candidates(m, s, p)["decisions"]["iron"]["flat"]["tier"] == "not_shortlisted"
