import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.q75_pressure_calibration import apply_calibration, fit_offsets, learn_calibration
from bf_tap_r2.data import FEATURES


def test_sparse_condition_falls_back_to_training_global_median():
    residual = np.array([-4.,-3.,2.,3.,4.])
    bins = np.array([0,0,1,1,1])
    np.testing.assert_array_equal(fit_offsets(residual,bins,'PRESSURE',3), [2,3,2,2])


def test_application_is_label_free_and_zero_gamma_exactly_preserves_parent():
    query = pd.DataFrame({'total_press_diff':[1.,2.,3.]})
    parent = np.array([10.,20.,30.])
    fitted = dict(cuts=[1.5,2.5,3.5],offsets=[-2.,-1.,1.,2.],gamma=0.)
    np.testing.assert_array_equal(apply_calibration(query,parent,fitted),parent)
    query['tap_time_len'] = 0.
    with pytest.raises(ValueError,match='labels'): apply_calibration(query,parent,fitted)


def test_calibration_selects_on_inner_held_residuals_and_has_no_outer_argument():
    n=120
    frame = pd.DataFrame({f:np.arange(n,dtype=float)+i for i,f in enumerate(FEATURES)})
    frame['sample_id'] = [f's{i}' for i in range(n)]; frame['spout_no']=np.arange(n)%2+1
    p=np.full(n,100.);y=p+2
    settings=dict(cv_seed=961048,minimum_bin_rows=30,gamma_grid=[0,.25,.5,1])
    fitted=learn_calibration(frame,y,p,[40.,70.,100.],'GLOBAL',settings)
    assert fitted['gamma']==1
    np.testing.assert_array_equal(apply_calibration(frame,p,fitted),y)
    fitted_zero=learn_calibration(frame,p,p,[40.,70.,100.],'GLOBAL',settings)
    assert fitted_zero['gamma']==0
    frame['tap_time_len']=y
    with pytest.raises(ValueError,match='Label-free'): learn_calibration(frame,y,p,[40.,70.,100.],'GLOBAL',settings)
