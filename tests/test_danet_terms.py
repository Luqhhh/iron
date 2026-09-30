import numpy as np
import pytest
import torch

from bf_tap_r2.danet_terms import entmax15,AbstractLayer,GhostNormalization


def bisection_simplex(logits):
    values=np.asarray(logits,float)/2
    lo,hi=values.max()-1,values.max()
    for _ in range(100):
        threshold=(lo+hi)/2
        if np.maximum(values-threshold,0).dot(np.maximum(values-threshold,0))>1:lo=threshold
        else:hi=threshold
    return np.maximum(values-(lo+hi)/2,0)**2


def test_entmax_matches_independent_constraint_root_and_has_exact_sparse_support():
    rng=np.random.default_rng(57359)
    values=rng.normal(size=(8,21))*3
    x=torch.tensor(values,dtype=torch.float64,requires_grad=True)
    before=x.detach().clone();actual=entmax15(x)
    expected=np.stack([bisection_simplex(row) for row in values])
    np.testing.assert_allclose(actual.detach(),expected,atol=1e-13,rtol=0)
    torch.testing.assert_close(actual.sum(-1),torch.ones(8,dtype=torch.float64),atol=1e-13,rtol=0)
    assert (actual==0).any() and torch.equal(x.detach(),before)
    torch.testing.assert_close(entmax15(x+100),actual,atol=1e-13,rtol=0)


def test_entmax_support_aware_gradient_passes_finite_differences():
    torch.manual_seed(57359)
    logits=torch.randn(3,9,dtype=torch.float64,requires_grad=True)*2
    assert torch.autograd.gradcheck(entmax15,(logits,),eps=1e-6,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('value',[torch.tensor([1,2]),torch.empty(0),torch.tensor([float('nan')])])
def test_entmax_rejects_nonfloating_empty_and_nonfinite_input(value):
    with pytest.raises(ValueError):entmax15(value)


def test_ghost_statistics_are_training_only_and_match_sequential_batches():
    torch.manual_seed(57359);x=torch.randn(10,7,dtype=torch.float64)
    ghost=GhostNormalization(7,4).double();direct=torch.nn.BatchNorm1d(7).double()
    direct.load_state_dict(ghost.bn.state_dict())
    expected=torch.cat([direct(part) for part in x.chunk(3)])
    torch.testing.assert_close(ghost(x),expected,atol=0,rtol=0)
    for name,value in direct.state_dict().items():torch.testing.assert_close(ghost.bn.state_dict()[name],value,atol=0,rtol=0)
    ghost.eval();before={k:v.clone() for k,v in ghost.bn.state_dict().items()}
    ghost(x*1e6)
    assert all(torch.equal(v,ghost.bn.state_dict()[k]) for k,v in before.items())


def test_singleton_ghost_batch_is_refused_before_any_statistics_change():
    layer=GhostNormalization(3,2);before={k:v.clone() for k,v in layer.bn.state_dict().items()}
    with pytest.raises(ValueError,match='Singleton'):layer(torch.ones(3,3))
    assert all(torch.equal(v,layer.bn.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('bias',[False,True])
def test_abstract_folding_matches_independent_numpy_gating_and_is_row_local(bias):
    torch.manual_seed(57359)
    layer=AbstractLayer(21,8,5,4,bias=bias).double()
    layer(torch.randn(16,21,dtype=torch.float64));layer.eval()
    query=torch.randn(8,21,dtype=torch.float64)
    folded=layer.folded();actual=folded(query)
    with torch.no_grad():
        torch.testing.assert_close(actual,layer(query),atol=1e-12,rtol=0)
        torch.testing.assert_close(actual,torch.cat([folded(row[None]) for row in query]),atol=1e-12,rtol=0)
        torch.testing.assert_close(actual,folded(query.flip(0)).flip(0),atol=1e-12,rtol=0)
        weight=folded.weight.numpy();intercept=folded.bias.numpy();answers=[]
        for row in query.numpy():
            branches=np.stack([w@row+b for w,b in zip(weight,intercept)])
            gate=1/(1+np.exp(-branches[:,:8]));value=branches[:,8:]
            answers.append(np.maximum(gate*value,0).sum(0))
        np.testing.assert_allclose(actual.numpy(),np.asarray(answers),atol=1e-12,rtol=0)
    assert not list(folded.parameters())
    with pytest.raises(ValueError,match='inference'):folded.train()


def test_fixed_mask_control_shares_initial_state_and_only_omits_mask_updates():
    torch.manual_seed(57359);learned=AbstractLayer(21,8,5,4).double()
    torch.manual_seed(57359);control=AbstractLayer(21,8,5,4,learn_masks=False).double()
    for name,value in learned.state_dict().items():assert torch.equal(value,control.state_dict()[name])
    assert learned.logits.requires_grad and not control.logits.requires_grad
    x=torch.randn(8,21,dtype=torch.float64)
    expected=learned(x);actual=control(x)
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    expected.square().mean().backward();actual.square().mean().backward()
    assert learned.logits.grad is not None and learned.logits.grad.abs().max()>0
    assert control.logits.grad is None
    for name,param in learned.named_parameters():
        if name!='logits':torch.testing.assert_close(param.grad,dict(control.named_parameters())[name].grad,atol=0,rtol=0)


def test_training_layer_cannot_fold_using_live_query_batch_statistics():
    layer=AbstractLayer(21,8)
    with pytest.raises(ValueError,match='evaluation'):layer.folded()
