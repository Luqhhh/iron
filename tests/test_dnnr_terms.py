import numpy as np
import pytest

from bf_tap_r2.dnnr_terms import (first_order_coefficients,first_order_prediction,
    scaling_cost_gradient,local_linear_errors)


def test_local_taylor_extrapolation_recovers_affine_function_outside_neighbors():
    rng=np.random.default_rng(57311);anchor=rng.normal(size=4);x=rng.normal(size=(40,4));slope=np.array([3.,-2.,.5,7.])
    value=lambda z:4.+z@slope
    coefficient=first_order_coefficients(anchor,value(anchor),x,value(x))
    np.testing.assert_allclose(coefficient,slope,rtol=0,atol=1e-13)
    query=np.array([20.,-10.,5.,30.])
    assert abs(first_order_prediction(query,anchor,value(anchor),coefficient)-value(query))<1e-12
    assert abs(value(x).mean()-value(query))>100


def test_rank_deficient_local_derivative_is_finite_minimum_norm_solution():
    x=np.arange(1.,20.)[:,None]*np.ones((1,2));y=2*x[:,0]
    coefficient=first_order_coefficients(np.zeros(2),0.,x,y)
    np.testing.assert_allclose(coefficient,[1.,1.],atol=1e-13)
    assert abs(first_order_prediction(np.array([10.,10.]),np.zeros(2),0.,coefficient)-20)<1e-12


@pytest.mark.parametrize('scale',[np.array([.7,1.2,2.]),np.array([-1.,.3,1.5])])
def test_scale_derivative_matches_independent_central_difference(scale):
    rng=np.random.default_rng(57312);delta=rng.normal(size=(50,3));errors=rng.uniform(.1,3.,size=50)
    cost,gradient=scaling_cost_gradient(scale,delta,errors)
    def objective(s):
        h=np.sqrt(np.sum((delta*s)**2,axis=1));h=np.maximum(h,1e-6)
        a=errors-errors.mean();b=h-h.mean()
        return -sum(a*b)/(np.sqrt(sum(a*a))*np.sqrt(sum(b*b)))
    difference=[]
    for j in range(3):
        direction=np.zeros(3);direction[j]=1e-5
        difference.append((objective(scale+direction)-objective(scale-direction))/2e-5)
    np.testing.assert_allclose(gradient,difference,rtol=1e-7,atol=1e-9)
    assert abs(cost-objective(scale))<1e-14


def test_scale_invariance_is_visible_in_gradient_and_degenerate_cases():
    rng=np.random.default_rng(57313);delta=rng.normal(size=(30,3));errors=rng.uniform(.1,2,size=30);scale=np.array([.7,1.2,2.])
    cost,gradient=scaling_cost_gradient(scale,delta,errors)
    assert abs(gradient@scale)<1e-12
    assert abs(scaling_cost_gradient(scale*2,delta,errors)[0]-cost)<1e-14
    for d,e in [(delta,np.ones(30)),(np.ones((30,3)),errors),(np.zeros((30,3)),errors)]:
        c,g=scaling_cost_gradient(scale,d,e);assert c==0;np.testing.assert_array_equal(g,np.zeros(3))


def test_distance_floor_derivative_handles_duplicate_points_and_extremely_small_steps():
    rng=np.random.default_rng(57314);delta=rng.normal(size=(30,3));delta[0]=0;delta[1]=1e-12;errors=rng.uniform(.1,2,size=30)
    _,gradient=scaling_cost_gradient(np.ones(3),delta,errors)
    assert np.isfinite(gradient).all()


def test_scaling_linear_fit_uses_training_anchor_and_raw_linearization():
    rng=np.random.default_rng(57315);x=rng.normal(size=(40,3));center=rng.normal(size=3);y=2+x@np.array([1.,2.,3.])
    delta,error=local_linear_errors(center,2+center@np.array([1.,2.,3.]),x,y)
    np.testing.assert_array_equal(delta,x-center);assert error.max()<1e-13


@pytest.mark.parametrize('defect',['nonfinite','shape','negative_error','epsilon'])
def test_invalid_scaling_inputs_fail_before_any_update(defect):
    scale=np.ones(3);delta=np.ones((5,3));errors=np.arange(5,dtype=float);epsilon=1e-6
    if defect=='nonfinite':scale[0]=np.nan
    elif defect=='shape':delta=np.ones((4,3))
    elif defect=='negative_error':errors[0]=-1
    else:epsilon=0
    with pytest.raises(ValueError):scaling_cost_gradient(scale,delta,errors,epsilon)
