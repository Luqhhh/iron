from copy import deepcopy
import numpy as np
import pytest

from bf_tap_r2.realmlp_time_development import combine,select,recipe_for_init,CANDIDATES


def test_fixed_blend_keeps_native_member_order_and_no_hidden_weight_scan():
    base=np.array([100.,110.]);members=np.array([[95.,100.],[105.,115.],[100.,115.]])
    p=combine(base,members)
    np.testing.assert_allclose(p[CANDIDATES[0]],[99.,108.])
    np.testing.assert_allclose(p[CANDIDATES[1]],[100.,110.])
    with pytest.raises(ValueError):combine(base,members[:2])
    with pytest.raises(ValueError):combine(base,np.full((3,2),np.nan))
    with pytest.raises(ValueError):combine(base,np.full((3,2),-10000.))


def test_choice_requires_two_positive_splits_and_uses_preregistered_worst_split():
    assert select({CANDIDATES[0]:{'42':.004,'3407':.01},CANDIDATES[1]:{'42':.006,'3407':.007}})==CANDIDATES[1]
    assert select({CANDIDATES[0]:{'42':.004,'3407':.01},CANDIDATES[1]:{'42':-.001,'3407':.04}})==CANDIDATES[0]
    assert select({c:{'42':0.,'3407':1.} for c in CANDIDATES}) is None
    assert select({c:{'42':.001,'3407':.001} for c in CANDIDATES})==CANDIDATES[0]
    with pytest.raises(ValueError):select({c:{'42':.01} for c in CANDIDATES})


def test_initialization_change_preserves_every_other_recipe_field():
    recipe={'constructor':{'random_state':42,'n_epochs':256},'resolved':{'random_state':42,'lr':.2}}
    original=deepcopy(recipe);changed=recipe_for_init(recipe,1042)
    assert recipe==original
    changed['constructor']['random_state']=42;changed['resolved']['random_state']=42
    assert changed==recipe
    with pytest.raises(ValueError):recipe_for_init(recipe,999)
