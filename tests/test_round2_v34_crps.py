from pathlib import Path
import math

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
from scipy.integrate import quad
from scipy.stats import norm
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v34_crps import CRPSRegressor, normal_crps, crps_gradient, crps_metric_diagonal, fit_partition
from bf_tap_r2.v34_run import select_finalists, partitions


def options():
    s=yaml.safe_load(Path("configs/round2_v34/SPEC.yaml").read_text())["training"]
    return dict(s,max_epochs=8,min_samples_leaf=3,patience=3)


def synthetic(n=100):
    x=np.random.default_rng(534).normal(size=(n,len(FEATURES)))
    d=pd.DataFrame(x,columns=FEATURES)
    d["sample_id"]=[f"synthetic-{i}" for i in range(n)]
    d["spout_no"]=1+np.arange(n)%2
    d["tap_iron"]=100+5*x[:,0]+np.sin(x[:,1])
    d["tap_time_len"]=30+2*x[:,2]
    return d


def test_crps_gradient_matches_independent_finite_differences():
    y=np.array([-.3,1.,4.])
    p=np.array([[0.,-.6],[2.,.3],[3.,1.]])
    analytic=crps_gradient(y,p)
    for j in range(2):
        plus,minus=p.copy(),p.copy()
        plus[:,j]+=1e-6;minus[:,j]-=1e-6
        numerical=(normal_crps(y,plus)-normal_crps(y,minus))/2e-6
        assert np.allclose(analytic[:,j],numerical,atol=1e-9,rtol=1e-8)
    # Degenerate normal limit approaches absolute error.
    small=p.copy();small[:,1]=-20
    assert np.allclose(normal_crps(y,small),np.abs(y-p[:,0]),atol=2e-9,rtol=0)


@pytest.mark.parametrize("mu,sigma",[(0.,.4),(2.,1.7)])
def test_metric_matches_numerical_integrated_cdf_derivatives(mu,sigma):
    def dmu(t): return -norm.pdf((t-mu)/sigma)/sigma
    def dlogsigma(t): return -(t-mu)/sigma*norm.pdf((t-mu)/sigma)
    expected=np.array([[2*quad(lambda t:f(t)*g(t),-np.inf,np.inf,epsabs=1e-11)[0]
                        for g in (dmu,dlogsigma)] for f in (dmu,dlogsigma)])
    observed=np.diag(crps_metric_diagonal(np.array([[mu,math.log(sigma)]]))[0])
    assert np.allclose(expected,observed,atol=1e-9,rtol=1e-9)


@pytest.mark.parametrize("recipe",["CRPS_FIXED","CRPS_SCALE"])
def test_training_score_fixed_scale_and_cold_predictions(recipe,tmp_path):
    d=synthetic();fit=d.iloc[:75];query=d.iloc[75:].drop(columns=list(TARGETS))
    m=CRPSRegressor(recipe,options()).initialize(fit,fit.tap_iron.to_numpy())
    initial=float(normal_crps(m.y_train_,np.zeros((len(fit),2))).mean())
    m.train(8)
    scores=[initial]+[r["training_crps"] for r in m.history_]
    assert all(b<=a+1e-12 for a,b in zip(scores,scores[1:]))
    assert scores[-1]<scores[0]
    params=m.predict_params(query)
    if recipe=="CRPS_FIXED":assert np.array_equal(params[:,1],np.zeros(len(query)))
    else:assert np.any(params[:,1]!=0)
    assert np.allclose(m.preprocessor_.means_,fit[list(FEATURES)].mean())
    path=tmp_path/"model.pkl";m.save(path)
    cold=CRPSRegressor.load(path);expected=m.predict(query)
    assert np.array_equal(cold.predict(query),expected)
    assert np.array_equal(cold.predict(query.iloc[::-1])[::-1],expected)
    assert np.array_equal(np.concatenate([cold.predict(query.iloc[i:i+1]) for i in range(len(query))]),expected)
    assert not hasattr(cold,"y_train_")
    with pytest.raises(FileExistsError):m.save(path)


def test_calibration_selection_fresh_refit_and_outer_labels_absent():
    d=synthetic();fv=np.arange(len(d))%5
    spec=yaml.safe_load(Path("configs/round2_v34/SPEC.yaml").read_text())
    training,query,fitting,cal=partitions(d,fv,0,spec)
    changed=d.copy();changed.loc[fv==0,list(TARGETS)]=1e9
    for a,b in zip((training,query,fitting,cal),partitions(changed,fv,0,spec)):
        pd.testing.assert_frame_equal(a,b)
    m,pred,meta,_=fit_partition(fitting,cal,training,query,"tap_iron","CRPS_SCALE",options(),np.full(len(cal),100),[0.,.5,1.])
    assert meta["refit"]["selected_epoch"]==meta["calibration"]["selected_epoch"]
    assert len(m.trees_)==meta["calibration"]["selected_epoch"]
    assert np.allclose(meta["refit"]["preprocessing"]["means"],training[list(FEATURES)].mean())
    assert np.isfinite(pred).all()
    with pytest.raises(ValueError,match="Query labels"):
        fit_partition(fitting,cal,training,query.assign(tap_iron=100),"tap_iron","CRPS_SCALE",options(),np.zeros(len(cal)),[0.,1.])


def test_both_arms_prospectively_eligible_but_both_seeds_required():
    spec=yaml.safe_load(Path("configs/round2_v34/SPEC.yaml").read_text())
    rows=[dict(target="tap_iron",recipe="CRPS_FIXED",both_seeds_positive=True,paired_seed_summary={"mean":.02}),
          dict(target="tap_iron",recipe="CRPS_SCALE",both_seeds_positive=True,paired_seed_summary={"mean":.01})]
    assert select_finalists(rows,spec)["tap_iron"]=="CRPS_FIXED"
    rows[1]["paired_seed_summary"]["mean"]=.03
    assert select_finalists(rows,spec)["tap_iron"]=="CRPS_SCALE"
    rows[1]["both_seeds_positive"]=False
    assert select_finalists(rows,spec)["tap_iron"]=="CRPS_FIXED"
