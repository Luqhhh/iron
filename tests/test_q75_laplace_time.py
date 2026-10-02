import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import laplace
import torch

from bf_tap_r2.laplace_time_model import ARMS, LaplaceRegressor, laplace_nll
from bf_tap_r2.q75_laplace_time import admission, validate_spec


def test_likelihood_matches_independent_scipy_and_torch_distribution():
    y=torch.tensor([-3.,.2,4.],dtype=torch.float64)
    mu=torch.tensor([[-1.],[.4],[2.]],dtype=torch.float64)
    b=torch.tensor([[.3],[1.2],[2.]],dtype=torch.float64)
    result=float(laplace_nll(y,mu,b))
    expected=-laplace.logpdf(y.numpy(),loc=mu.numpy().ravel(),scale=b.numpy().ravel()).mean()
    assert result==pytest.approx(expected,abs=1e-12)
    assert result==pytest.approx(float(-torch.distributions.Laplace(mu[:,0],b[:,0]).log_prob(y).mean()),abs=1e-12)


def test_location_and_scale_gradients_match_finite_differences_away_from_kink():
    y=torch.tensor([-.8,1.9],dtype=torch.float64)
    mu=torch.tensor([[.1],[1.2]],dtype=torch.float64,requires_grad=True)
    log_b=torch.tensor([[-.2],[.3]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda m,l:laplace_nll(y,m,l.exp()),(mu,log_b),eps=1e-6,atol=1e-6)


def test_fixed_scale_loss_is_scaled_mae_and_invariant_to_scale_head():
    model=LaplaceRegressor('LAPLACE_FIXED',{'initial_scale':.5})
    y=torch.tensor([-1.,2.],dtype=torch.float64)
    mu=torch.tensor([[.2],[1.8]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.1],[3.]],dtype=torch.float64,requires_grad=True)
    output=(torch.zeros_like(mu),mu,b)
    loss=model.loss(y,output)
    assert float(loss.detach())==pytest.approx(float((y[:,None]-mu).abs().mean().detach())/.5)
    grad_mu,grad_b=torch.autograd.grad(loss,(mu,b),allow_unused=True)
    assert grad_b is None and torch.isfinite(grad_mu).all()
    assert float(model.loss(y,(output[0],mu,b*100)).detach())==float(loss.detach())


def test_adaptive_scale_is_not_silently_fixed_and_scale_gradient_is_nonzero():
    model=LaplaceRegressor('LAPLACE_SCALE',{'initial_scale':.5})
    y=torch.tensor([-1.,2.],dtype=torch.float64)
    mu=torch.tensor([[.2],[1.8]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.1],[3.]],dtype=torch.float64,requires_grad=True)
    loss=model.loss(y,(torch.zeros_like(mu),mu,b))
    assert torch.autograd.grad(loss,b)[0].abs().sum()>0


@pytest.mark.parametrize('bad',[0.,-1.,np.nan,np.inf])
def test_invalid_adaptive_scale_is_rejected(bad):
    with pytest.raises(ValueError,match='positive scale'):
        laplace_nll(torch.zeros(1),torch.ones((1,1)),torch.full((1,1),bad))


def test_misaligned_distribution_shapes_are_rejected():
    with pytest.raises(ValueError,match='Aligned'):
        laplace_nll(torch.zeros(2),torch.ones((2,3)),torch.ones((2,3)))


def test_fixed_control_can_advance_when_adaptive_scale_does_not():
    gains={ARMS[0]:{'42':.012,'3407':.011},ARMS[1]:{'42':.003,'3407':.004}}
    over={ARMS[0]:{'42':.003,'3407':.001},ARMS[1]:{'42':-.006,'3407':-.006}}
    result=admission(gains,over)
    assert result['confirmation_finalist']==ARMS[0] and not result['formal_promoted']
    assert 'nonpositive_mean_advantage_over_GAUSS1_A20' in result['failed_conditions'][ARMS[1]]


def test_positive_mean_cannot_replace_two_positive_complete_development_seeds():
    gains={a:{'42':.02,'3407':-.001} for a in ARMS}
    over={a:{'42':.01,'3407':.001} for a in ARMS}
    assert admission(gains,over)['confirmation_finalist'] is None
    gains[ARMS[0]].pop('3407')
    with pytest.raises(ValueError,match='Complete'):admission(gains,over)


def test_registered_scope_rejects_extra_budget_seed_or_weight():
    spec=json.loads(Path('configs/q75_laplace_time/SPEC.json').read_text());validate_spec(spec)
    for key,value in [('optimizer_runs',44),('weight',.5),('engineering_optimizer_runs',6),('split_seeds',[42,271828])]:
        with pytest.raises(ValueError,match='scope or budget'):validate_spec(dict(spec,**{key:value}))
