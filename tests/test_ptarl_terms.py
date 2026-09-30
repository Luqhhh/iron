import math

import numpy as np
import pytest
from scipy.optimize import linprog
import torch

from bf_tap_r2.ptarl_terms import (auxiliary_terms, coordinate_diversity, cosine_matrix,
                                 projection_cost, prototype_orthogonalization,
                                 regression_bins, sampled_rows)


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def np_cosine(a, b):
    return np.array([[sum(x*y for x,y in zip(u,v))/max(np.linalg.norm(u),1e-12)
                      /max(np.linalg.norm(v),1e-12) for v in b] for u in a])


def independent_terms(h, r, b, y, indices):
    projection = sum(sum(r[i,k]*(1-np_cosine(h[i:i+1],b)[0,k]) for k in range(len(b)))
                     for i in range(len(h)))/len(h)
    count = 1+int(math.log2(len(y)))
    bins = np.zeros(len(y),int) if max(y)==min(y) else np.minimum(
        np.floor((y-min(y))/((max(y)-min(y))/count)).astype(int), count-1)
    similarities = np_cosine(r[indices],r[indices])
    denominator = math.log(sum(math.exp(float(v)) for v in similarities.flatten()))
    positives = [denominator-similarities[i,j] for i in range(len(indices))
                 for j in range(len(indices)) if bins[indices[i]]==bins[indices[j]]]
    diversity = sum(positives)/len(positives)
    matrix = np.abs(np_cosine(b,b))
    l1 = sum(matrix.flatten()); l2 = sum(v*v for v in matrix.flatten())
    orthogonalization = l1/l2+.5*abs(l1-len(b))
    return np.array([projection,diversity,orthogonalization])


def arrays():
    rng = np.random.default_rng(54)
    h = rng.normal(size=(7,4)); b = rng.normal(size=(3,4))
    r = rng.uniform(.1,1.,size=(7,3)); r /= r.sum(1,keepdims=True)
    y = np.array([1.,2.,2.5,3.,4.,4.1,5.])
    return h,r,b,y,[0,2,4]


def test_three_terms_against_independent_scalar_equations():
    h,r,b,y,indices = arrays()
    actual = auxiliary_terms(t(h),t(r),t(b),t(y),indices)
    np.testing.assert_allclose([v.item() for v in actual.values()],
                               independent_terms(h,r,b,y,indices),rtol=0,atol=2e-14)


def test_single_atom_transport_matches_independent_linear_program():
    h,r,b,_,_ = arrays()
    costs = 1-np_cosine(h,b)
    solutions = []
    for i in range(len(h)):
        # Independent full OT constraints: every destination column mass=r_ik.
        result = linprog(costs[i],A_eq=np.eye(3),b_eq=r[i],bounds=(0,None),method="highs")
        assert result.success
        np.testing.assert_allclose(result.x,r[i],rtol=0,atol=1e-12)
        solutions.append(result.fun)
    assert projection_cost(t(h),t(r),t(b)).item()==pytest.approx(np.mean(solutions),abs=1e-14)


@pytest.mark.parametrize("argument", ["hidden","logits","prototypes"])
def test_full_loss_autograd_matches_finite_difference(argument):
    h,r,b,y,indices = arrays()
    logits = np.log(r)
    tensors = {"hidden":t(h,True),"logits":t(logits,True),"prototypes":t(b,True)}
    terms = auxiliary_terms(tensors["hidden"],tensors["logits"].softmax(1),tensors["prototypes"],t(y),indices)
    sum(terms.values()).backward()
    actual = tensors[argument].grad.numpy()
    arrays_by_name = {"hidden":h,"logits":logits,"prototypes":b}
    eps = 1e-6
    for index in [(0,0),(1,1)]:
        shifted = []
        for direction in [1,-1]:
            a = {k:v.copy() for k,v in arrays_by_name.items()}
            a[argument][index] += direction*eps
            probabilities = np.exp(a["logits"]-a["logits"].max(1,keepdims=True))
            probabilities /= probabilities.sum(1,keepdims=True)
            shifted.append(independent_terms(a["hidden"],probabilities,a["prototypes"],y,indices).sum())
        assert actual[index]==pytest.approx((shifted[0]-shifted[1])/(2*eps),abs=1e-7,rel=1e-6)
    assert np.isfinite(actual).all() and np.linalg.norm(actual)>0


def test_global_diversity_denominator_and_diagonal_are_intentional():
    # All equal coordinates -> similarities1; positive pairs include diagonals.
    r = t([[.5,.5]]*4); y = t([0,1,2,3])
    assert coordinate_diversity(r,y,[0,3]).item()==pytest.approx(math.log(4),abs=1e-14)
    # Row-wise InfoNCE would be log2, so the test distinguishes the implementation.
    assert coordinate_diversity(r,y,[2]).item()==0


def test_bins_use_full_batch_before_subset_and_handle_constant_singleton():
    y = t([0.,1.,2.,3.,4.,5.,6.,7.])
    assert regression_bins(y).tolist()==[0,0,1,1,2,2,3,3]
    r = t([[.6,.4],[.5,.5],[.4,.6],[.3,.7],[.2,.8],[.1,.9],[.8,.2],[.9,.1]])
    indices = [2,3]
    # Same full-batch bin; subset min/max would wrongly split these two labels.
    value = coordinate_diversity(r,y,indices)
    expected = independent_terms(np.ones((8,2)),r.numpy(),np.eye(2),y.numpy(),indices)[1]
    assert value.item()==pytest.approx(expected,abs=1e-14)
    subset = coordinate_diversity(r[indices],y[indices],[0,1])
    assert abs(value.item()-subset.item())>1e-4
    assert regression_bins(t([7.,7.,7.])).tolist()==[0,0,0]
    assert regression_bins(t([7.])).tolist()==[0]
    assert coordinate_diversity(t([[.5,.5]]),t([7.]),[0]).item()==0


def test_orthogonal_and_collapsed_banks_have_correct_nonzero_baseline():
    assert prototype_orthogonalization(t(np.eye(3))).item()==pytest.approx(1.,abs=1e-14)
    assert prototype_orthogonalization(t([[1,0]]*3)).item()==pytest.approx(4.,abs=1e-14)
    with pytest.raises(ValueError,match="Nonzero"):
        prototype_orthogonalization(t([[0,0],[1,0]]))


def test_auxiliary_scales_and_permutations_preserve_geometry():
    h,r,b,y,indices = arrays()
    original = auxiliary_terms(t(h),t(r),t(b),t(y),indices)
    scaled = auxiliary_terms(t(h*3),t(r),t(b*.2),t(y*4+9),indices)
    for name in original:
        torch.testing.assert_close(original[name],scaled[name],rtol=0,atol=1e-14)
    order = [2,0,1]
    permuted = auxiliary_terms(t(h),t(r[:,order]),t(b[order]),t(y),indices)
    for name in original:
        torch.testing.assert_close(original[name],permuted[name],rtol=0,atol=1e-14)


def test_sampled_half_rows_do_not_consume_global_randomness():
    state = torch.random.get_rng_state().clone()
    before = np.random.get_state()
    result = sampled_rows(17,42,7)
    assert len(result)==8 and len(set(result))==8
    assert result==sampled_rows(17,42,7) and result!=sampled_rows(17,42,8)
    assert sampled_rows(1,42,7)==[0]
    assert torch.equal(state,torch.random.get_rng_state())
    after = np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_zero_hidden_has_finite_gradient_and_simplex_checks_reject_bad_inputs():
    h = t([[0,0]],True); b = t([[1,0],[0,1]],True)
    loss = projection_cost(h,t([[.5,.5]]),b)
    loss.backward()
    assert torch.isfinite(h.grad).all() and loss.item()==1
    for r in [t([[.5,.4]]),t([[-.1,1.1]])]:
        with pytest.raises(ValueError,match="simplex"):
            projection_cost(t([[1,1]]),r,b)
    with pytest.raises(ValueError,match="Unique"):
        coordinate_diversity(t([[.5,.5]]*2),t([1,2]),[0,0])
    with pytest.raises(ValueError,match="float64"):
        cosine_matrix(torch.ones(1,2,dtype=torch.float32),t([[1,0]]))
