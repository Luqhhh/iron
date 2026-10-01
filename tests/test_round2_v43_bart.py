import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import multivariate_normal
import yaml

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v43_bart import (BartRegressor,NodeSpace,Tree,fit_partition,leaf_conditional,
                               leaf_log_integral,log_acceptance,mixture_median,proposal_probability)


def settings():
    s=yaml.safe_load((Path(__file__).parents[1]/'configs/round2_v43/SPEC.yaml').read_text())['training']
    s.update(trees=8,burn_sweeps=8,retained_draws=4,thin=2,min_leaf_rows=2)
    return s


def test_integrated_likelihood_and_leaf_conditional_against_dense_gaussian():
    y=np.array([-.3,.7,1.1]);sigma2=.7;tau2=.2;n=len(y)
    covariance=sigma2*np.eye(n)+tau2*np.ones((n,n))
    expected=multivariate_normal.logpdf(y,cov=covariance)-multivariate_normal.logpdf(y,cov=sigma2*np.eye(n))
    assert leaf_log_integral(n,y.sum(),sigma2,tau2)==pytest.approx(expected,abs=1e-13)
    mean,var=leaf_conditional(n,y.sum(),sigma2,tau2)
    assert mean==pytest.approx(tau2*np.ones(n)@np.linalg.solve(covariance,y),abs=1e-13)
    assert var==pytest.approx(tau2-tau2**2*np.ones(n)@np.linalg.solve(covariance,np.ones(n)),abs=1e-13)


@pytest.mark.parametrize('depth',[1,3])
def test_grow_prune_detailed_balance_including_forced_actions(depth):
    x=np.arange(12).reshape(-1,1).astype(float);s=settings()
    space=NodeSpace(x,[[0,3.5],[0,5.5],[0,7.5]],depth,s)
    old=Tree(space.root);new=old.changed('grow',0,space,1)
    y=np.sin(x[:,0]);sigma2=.3;tau2=.1
    forward=log_acceptance(old,new,'grow',0,space,y,sigma2,tau2)
    reverse=log_acceptance(new,old,'prune',0,space,y,sigma2,tau2)
    assert forward == pytest.approx(-reverse, abs=1e-12)
    pi_old=np.exp(old.log_prior(space)+old.log_integral(y,sigma2,tau2))
    pi_new=np.exp(new.log_prior(space)+new.log_integral(y,sigma2,tau2))
    flow=pi_old*proposal_probability(old,'grow',0)*np.exp(min(0,forward))
    reverse_flow=pi_new*proposal_probability(new,'prune',0)*np.exp(min(0,reverse))
    assert flow==pytest.approx(reverse_flow,rel=1e-12)
    assert proposal_probability(old,'grow',0)==pytest.approx(1/3)
    assert proposal_probability(new,'prune',0)==pytest.approx(1 if depth==1 else .5)
    left=(x[:,0]<=5.5).astype(float);right=1-left
    old_cov=sigma2*np.eye(len(y))+tau2*np.ones((len(y),len(y)))
    new_cov=sigma2*np.eye(len(y))+tau2*(np.outer(left,left)+np.outer(right,right))
    leaf_stop=1 if depth==1 else 1-.95/4
    prior_ratio=(.95/3*leaf_stop**2)/.05
    independent=(multivariate_normal.logpdf(y,cov=new_cov)-multivariate_normal.logpdf(y,cov=old_cov)
                 +np.log(prior_ratio)+np.log((1 if depth==1 else .5)/(1/3)))
    assert forward==pytest.approx(independent,abs=1e-11)


def test_mixture_median_against_independent_scalar_root_and_row_order():
    means=np.array([[-3.,1.,2.],[4.,4.,4.],[-1.,5.,9.]])
    scales=np.array([.2,1.,3.])
    pred=mixture_median(means,scales)
    for i,row in enumerate(means):
        expected=brentq(lambda z:ndtr((z-row)/scales).mean()-.5,row.min()-40,row.max()+40,xtol=1e-12)
        assert pred[i]==pytest.approx(expected,abs=1e-10)
    assert np.array_equal(pred,mixture_median(means[::-1],scales)[::-1])
    assert np.array_equal(pred,np.concatenate([mixture_median(row[None,:],scales) for row in means]))


def frame(n=60):
    rng=np.random.default_rng(4)
    data=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    data['sample_id']=[f's{i}' for i in range(n)];data['spout_no']=np.arange(n)%2+1
    return data


def test_tiny_chain_reproducibility_persistence_and_train_only_transform(tmp_path):
    data=frame();y=10+3*data[FEATURES[0]].to_numpy()
    first=BartRegressor('BART',settings()).fit(data,y)
    second=BartRegressor('BART',settings()).fit(data,y)
    assert first.draws_==second.draws_
    assert len(first.draws_)==4 and len(first.trace_)==16
    first.save(tmp_path/'model.json');cold=BartRegressor.load(tmp_path/'model.json')
    pred=first.predict(data)
    assert np.array_equal(pred,cold.predict(data))
    assert np.array_equal(pred,cold.predict(data.iloc[::-1])[::-1])
    assert np.array_equal(pred,np.concatenate([cold.predict(data.iloc[i:i+7]) for i in range(0,len(data),7)]))
    assert np.mean(np.abs(y-pred))<np.mean(np.abs(y-np.median(y)))
    original=first.preprocessor_.metadata();query=data.iloc[:3].copy();query['spout_no']=99
    query.loc[:,list(FEATURES)]=1e4
    assert not first.preprocessor_.transform(query)[:,len(FEATURES):].any()
    assert original==first.preprocessor_.metadata()
    with pytest.raises(FileExistsError): first.save(tmp_path/'model.json')


def test_queries_with_labels_cannot_start_sampler():
    data=frame(20);query=data.iloc[18:].copy();query['tap_time_len']=1
    with pytest.raises(ValueError,match='Query labels'):
        fit_partition(data.iloc[:12],data.iloc[12:18],data.iloc[:18],query,'tap_iron','BART',{},None,[0,1])
