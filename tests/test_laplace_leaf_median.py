"""Pure-array and fake-tree tests: zero actual sklearn fitting calls."""
import numpy as np
import pytest
import json
from pathlib import Path

from bf_tap_r2.laplace_leaf_median import LeafMedianRegressor, leaf_medians, best_epoch


def test_even_and_odd_leaf_medians():
    values=leaf_medians([1.,3.,7.,0.,4.],[1,1,2,2,2],3)
    assert np.isnan(values[0])
    np.testing.assert_array_equal(values[1:],[2.,4.])


def test_descent_and_exact_leaf_optimum():
    residual=np.array([-9.,-1.,2.,40.,-2.,3.,8.])
    leaves=np.array([1,1,1,1,2,2,2])
    values=leaf_medians(residual,leaves,3)
    initial=np.abs(residual).sum()
    for eta in (.0,.05,.5,1.):
        assert np.abs(residual-eta*values[leaves]).sum()<=initial
    for leaf in (1,2):
        r=residual[leaves==leaf]
        optimum=np.abs(r-values[leaf]).sum()
        assert all(np.abs(r-v).sum()>=optimum for v in (-100.,-5.,0.,10.,100.))


@pytest.mark.parametrize("r,l,n",[([],[],1),([np.nan],[0],1),([1.],[1.5],2),([1.],[-1],2),([1.],[2],2),([1.],[0],0)])
def test_invalid_leaf_arrays(r,l,n):
    with pytest.raises(ValueError):
        leaf_medians(r,l,n)


def test_earliest_checkpoint_including_zero():
    assert best_epoch([dict(calibration_mae=1.),dict(calibration_mae=1.)],1.)==0
    assert best_epoch([dict(calibration_mae=.5),dict(calibration_mae=.5)],1.)==1


class FakeTree:
    def apply(self,x):
        return np.where(x[:,0]<0,1,2)


def fake_model():
    model=LeafMedianRegressor(3)
    model.x_mean_=np.array([0.,0.])
    model.x_scale_=np.array([1.,1.])
    model.y_median_=100.
    model.y_scale_=10.
    model.trees_=[FakeTree()]
    model.leaf_values_=[np.array([np.nan,-2.,4.])]
    model.selected_epoch_=1
    return model


def test_query_order_chunk_empty_and_epoch_zero():
    model=fake_model()
    x=np.array([[-1.,0.],[1.,0.],[-2.,4.]])
    p=model.predict(x)
    np.testing.assert_array_equal(p,[99.,102.,99.])
    np.testing.assert_array_equal(model.predict(x[::-1])[::-1],p)
    np.testing.assert_array_equal(np.r_[model.predict(x[:1]),model.predict(x[1:])],p)
    np.testing.assert_array_equal(model.predict(x,epoch=0),[100.,100.,100.])
    assert model.predict(x[:0]).shape==(0,)


@pytest.mark.parametrize("epoch",[-1,2,True,1.5])
def test_invalid_checkpoint(epoch):
    with pytest.raises(ValueError):
        fake_model().predict(np.zeros((2,2)),epoch=epoch)


@pytest.mark.parametrize("depth",[True,2,4,7])
def test_invalid_depth(depth):
    with pytest.raises(ValueError):
        LeafMedianRegressor(depth)


def test_fresh_instance_and_input_gate_no_tree_fits():
    with pytest.raises(ValueError):
        fake_model().fit(np.zeros((2,2)),np.ones(2))
    with pytest.raises(ValueError):
        LeafMedianRegressor(3).fit(np.zeros((2,2)),np.ones(2),epochs=True)


def test_whole_frozen_specification():
    from bf_tap_r2.laplace_leaf_experiment import validate_spec
    spec=json.loads((Path(__file__).parents[1]/"configs/laplace_leaf_median/SPEC.json").read_text())
    validate_spec(spec)
    spec["fixed_weight"]=.1
    with pytest.raises(ValueError):
        validate_spec(spec)
