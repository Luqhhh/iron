import importlib
import numpy as np
import pytest


def protocol():
    return importlib.import_module("bf_tap_r2.mean2_cycle_swa_protocol")


def test_mean2_cancels_opposite_member_errors_in_aligned_rows():
    p=protocol()
    result=p.mean2_predictions(["a","b"],[80.,140.],["a","b"],[120.,100.])
    np.testing.assert_array_equal(result,[100.,120.])


def test_mean2_refuses_cross_split_row_reordering():
    p=protocol()
    with pytest.raises(ValueError,match="IDs"):
        p.mean2_predictions(["a","b"],[80.,140.],["b","a"],[120.,100.])


def test_mean2_refuses_nonfinite_or_incomplete_members():
    p=protocol()
    for ids,values in [(["a","b"],[np.nan,1.]),(["a"],[1.])]:
        with pytest.raises(ValueError):
            p.mean2_predictions(["a","b"],[2.,3.],ids,values)
    with pytest.raises(ValueError):
        p.mean2_predictions(["a","a"],[2.,3.],["a","a"],[4.,5.])


def test_member_seed_changes_rng_without_changing_inner_partition_or_other_settings():
    p=protocol();old={"random_seed":42,"inner_seed":42,"width":256,"max_epochs":240}
    new=p.member_settings(old)
    assert old=={"random_seed":42,"inner_seed":42,"width":256,"max_epochs":240}
    assert new=={"random_seed":3407,"inner_seed":42,"width":256,"max_epochs":240}
    with pytest.raises(ValueError):p.member_settings(dict(old,inner_seed=3407))


def test_positive_parent_gain_cannot_hide_worse_ensemble_than_control():
    p=protocol()
    d=p.decide([.004,.006],[.005,.006],False)
    assert d["selected_for_confirmation"]==[]
    assert "nonpositive_mechanism_advantage" in d["failure_reasons"]
    assert not d["formal_promoted"]


def test_confirm_requires_each_seed_and_positive_seed_lower_bound():
    p=protocol()
    d=p.decide([.004,.005,-.001,.004],[.002,.002,-.002,.001],True)
    assert not d["formal_promoted"]
    assert "nonpositive_seed_gain" in d["failure_reasons"]
    d=p.decide([.004,.005,.004,.005],[.002,.002,.002,.002],True)
    assert d["formal_promoted"] and d["seed_lcb95"]>0
    assert not d["release_authorized"]
