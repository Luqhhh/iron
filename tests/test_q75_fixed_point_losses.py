import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm
import torch

from bf_tap_r2.fixed_point_model import FixedPointRegressor, fixed_gaussian_nll
from bf_tap_r2.laplace_time_model import LaplaceRegressor
from bf_tap_r2.q75_fixed_point_losses import ARMS, RECIPES, admission, reference_vectors, validate_spec


def test_gaussian_value_and_gradient_match_independent_distribution():
    y=torch.tensor([-3.,.2,4.],dtype=torch.float64)
    mu=torch.tensor([[-1.],[.4],[2.]],dtype=torch.float64,requires_grad=True)
    result=fixed_gaussian_nll(y,mu,.5)
    expected=-norm.logpdf(y.numpy(),loc=mu.detach().numpy().ravel(),scale=.5).mean()
    assert result.item()==pytest.approx(expected,abs=1e-12)
    assert result.item()==pytest.approx(-torch.distributions.Normal(mu[:,0],.5).log_prob(y).mean().item(),abs=1e-12)
    torch.testing.assert_close(torch.autograd.grad(result,mu)[0],(mu-y[:,None])/(.5**2*len(y)))
    assert torch.autograd.gradcheck(lambda m:fixed_gaussian_nll(y,m,.5),(mu,),eps=1e-6,atol=1e-6)


@pytest.mark.parametrize('arm',['GAUSS_FIXED','LAPLACE_FIXED'])
def test_unused_heads_cannot_change_loss_or_receive_gradient(arm):
    y=torch.tensor([-1.,2.],dtype=torch.float64)
    mu=torch.tensor([[.2],[1.8]],dtype=torch.float64,requires_grad=True)
    scale=torch.tensor([[.1],[3.]],dtype=torch.float64,requires_grad=True)
    logits=torch.zeros_like(mu,requires_grad=True)
    model=FixedPointRegressor(arm,{'initial_scale':.5})
    loss=model.loss(y,(logits,mu,scale))
    other=model.loss(y,(logits+100,mu,scale*100))
    assert loss.item()==other.item()
    grad_mu,grad_scale,grad_logits=torch.autograd.grad(loss,(mu,scale,logits),allow_unused=True)
    assert grad_scale is None and grad_logits is None and torch.isfinite(grad_mu).all()


def test_laplace_arm_exactly_reuses_prior_loss_and_training_loop():
    y=torch.tensor([-1.,2.],dtype=torch.float64)
    mu=torch.tensor([[.2],[1.8]],dtype=torch.float64,requires_grad=True)
    output=(torch.zeros_like(mu),mu,torch.ones_like(mu))
    old=LaplaceRegressor('LAPLACE_FIXED',{'initial_scale':.5})
    new=FixedPointRegressor('LAPLACE_FIXED',{'initial_scale':.5})
    assert new.loss(y,output).item()==old.loss(y,output).item()
    assert FixedPointRegressor.train is LaplaceRegressor.train
    assert FixedPointRegressor.initialize is LaplaceRegressor.initialize


@pytest.mark.parametrize('scale',[0.,-1.,np.nan,np.inf])
def test_invalid_fixed_scale_rejected(scale):
    with pytest.raises(ValueError,match='positive fixed scale'):
        fixed_gaussian_nll(torch.zeros(1),torch.zeros((1,1)),scale)


def test_wrong_shape_and_nonfinite_target_rejected():
    with pytest.raises(ValueError,match='Aligned'):
        fixed_gaussian_nll(torch.zeros(2),torch.zeros((2,2)),.5)
    with pytest.raises(ValueError,match='Finite'):
        fixed_gaussian_nll(torch.tensor([float('nan')]),torch.zeros((1,1)),.5)


def test_target_mapping_keeps_iron_truth_and_parent_separate_from_time():
    data={k:np.array([i+1.,i+2.]) for i,k in enumerate(['y','y_iron','q75_42','iron_42','laplace_42'])}
    yi,bi,ri=reference_vectors(data,42,'tap_iron')
    yt,bt,rt=reference_vectors(data,42,'tap_time_len')
    assert yi is data['y_iron'] and bi is ri is data['iron_42']
    assert yt is data['y'] and bt is data['q75_42'] and rt is data['laplace_42']
    assert RECIPES['IRON_LAPLACE_FIXED']==('tap_iron','LAPLACE_FIXED')


def test_positive_time_control_cannot_advance_if_existing_ready_arm_better():
    gains={a:{'42':.01,'3407':.02} for a in ARMS}
    over={a:dict(gains[a]) for a in ARMS};over['TIME_GAUSS_FIXED']={'42':-.01,'3407':.001}
    result=admission(gains,over)
    assert result['confirmation_finalist']=='IRON_LAPLACE_FIXED'
    assert result['failed_conditions']['TIME_GAUSS_FIXED']==['nonpositive_mean_advantage_over_ready_LAPLACE_FIXED_A20']
    assert not result['formal_promoted']


def test_negative_seed_or_incomplete_coverage_cannot_advance():
    gains={a:{'42':.03,'3407':-.001} for a in ARMS};over={a:dict(gains[a]) for a in ARMS}
    assert admission(gains,over)['confirmation_finalist'] is None
    gains[ARMS[0]].pop('3407')
    with pytest.raises(ValueError,match='Complete'):admission(gains,over)


def test_scope_is_exact_before_training():
    spec=json.loads(Path('configs/q75_fixed_point_losses/SPEC.json').read_text());validate_spec(spec)
    for key,value in [('optimizer_runs',62),('weight',.5),('engineering_optimizer_runs',4),('split_seeds',[42,271828])]:
        with pytest.raises(ValueError,match='scope or budget'):validate_spec(dict(spec,**{key:value}))
