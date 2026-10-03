import numpy as np
import pytest
from bf_tap_r2.tabm_correction_fusion import affine, assemble, choose

def test_delta_preserves_parent_when_candidate_equals_base():
    p=np.array([10.,20.]);b=np.array([3.,7.])
    np.testing.assert_array_equal(affine([p,b,b,b],[1,.2,0,-.2]),p)

def test_negative_affine_rejected_without_clipping():
    with pytest.raises(ValueError):affine([np.array([1.]),np.array([0.]),np.array([0.]),np.array([20.])],[1,.1,0,-.1])

def test_fold_identity_and_unique_coverage():
    ids=np.array(['a','b']);folds=np.array([0,1]);records=[(np.array([0]),np.array(['a']),np.array([3.]),0),(np.array([1]),np.array(['b']),np.array([7.]),1)]
    np.testing.assert_array_equal(assemble(ids,folds,records),[3.,7.])
    with pytest.raises(ValueError):assemble(ids,folds,[records[0],records[0]])
    with pytest.raises(ValueError):assemble(ids,folds,[(np.array([0]),np.array(['b']),np.array([3.]),0),records[1]])

def test_no_cross_fold_or_nonfinite_prediction():
    ids=np.array(['a']);folds=np.array([1])
    with pytest.raises(ValueError):assemble(ids,folds,[(np.array([0]),ids,np.array([3.]),0)])
    with pytest.raises(ValueError):assemble(ids,folds,[(np.array([0]),ids,np.array([np.nan]),1)])

def test_selection_is_exploration_positive_mean_fixed_tie_order():
    rows={'b':{'mean_gain':.2,'minimum_gain':-.1},'a':{'mean_gain':.2,'minimum_gain':-.1},'c':{'mean_gain':-.1,'minimum_gain':-.2}}
    assert choose(rows,['a','b','c'])==['a']
    assert choose({'c':rows['c']},['c'])==[]
