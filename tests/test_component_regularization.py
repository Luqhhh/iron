import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from test_round2_v12 import sample
from bf_tap_r2.data import TARGETS
from bf_tap_r2.v12_joint import JointRegressor
from bf_tap_r2.v7_periodic import PeriodicRegressor
from bf_tap_r2.component_regularization import (ComponentRegressor, clone_state,
    update_ema, sam_step, replacement, paired_row_bootstrap)
from bf_tap_r2.component_regularization_audit import verify_saved
from bf_tap_r2.component_regularization_run import choose

RECIPE={"backbone":"tabm","frequency":.01}
MECH={"ema_beta":.99,"sam_rho":.05,"sam_epsilon":1e-12}


@pytest.mark.parametrize("outputs",[1,2])
def test_base_exact_native_trajectory_and_epoch(tmp_path,outputs):
    frame,settings=sample();y=frame[["tap_time_len"] if outputs==1 else list(TARGETS)].to_numpy()
    old=(PeriodicRegressor(RECIPE,settings).fit(frame,y[:,0]) if outputs==1 else JointRegressor(RECIPE,settings).fit(frame,y))
    new=ComponentRegressor(RECIPE,settings,"BASE",MECH,tmp_path).fit(frame,y)
    query=frame.drop(columns=list(TARGETS))
    expected=old.predict(query)
    np.testing.assert_array_equal(new.predict(query),expected[:,None] if outputs==1 else expected)
    assert old.metadata_["selected_epoch"]==new.metadata_["selected_epoch"]
    for k,v in old.model_.state_dict().items():assert torch.equal(v,new.model_.state_dict()[k])


def test_sam_analytic_update_zero_norm_and_exception_restores():
    model=torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad():model.weight.fill_(2.)
    opt=torch.optim.SGD(model.parameters(),lr=.1)
    sam_step(model,opt,lambda:.5*model.weight.square().sum(),.05)
    assert model.weight.item()==pytest.approx(2-.1*2.05,abs=1e-12)
    before=clone_state(model)
    sam_step(model,opt,lambda:(model.weight*0).sum(),.05)
    assert torch.equal(before["weight"],model.weight)
    calls=0
    def fail():
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError("second forward")
        return model.weight.square().sum()
    with pytest.raises(RuntimeError):sam_step(model,opt,fail,.05)
    assert torch.equal(before["weight"],model.weight)


def test_zero_rho_matches_adamw_and_dropout_rng():
    torch.manual_seed(17)
    a=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1))
    b=deepcopy(a);x=torch.randn(8,3);y=torch.randn(8,1)
    oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
    before=torch.get_rng_state();oa.zero_grad();((a(x)-y)**2).mean().backward();oa.step();after=torch.get_rng_state()
    torch.set_rng_state(before);sam_step(b,ob,lambda:((b(x)-y)**2).mean(),0.)
    assert torch.equal(after,torch.get_rng_state())
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])


def test_ema_recursion_buffers_and_no_rng():
    m=torch.nn.Linear(1,1,bias=False)
    m.register_buffer("counter",torch.tensor(0));m.register_buffer("running",torch.tensor(1.))
    with torch.no_grad():m.weight.fill_(2.)
    state=clone_state(m);rng=torch.get_rng_state()
    with torch.no_grad():m.weight.fill_(4.);m.counter.fill_(2);m.running.fill_(5.)
    update_ema(m,state,.99)
    assert state["weight"].item()==pytest.approx(2.02)
    assert state["counter"].item()==2 and state["running"].item()==5
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize("arm",["BASE","EMA","SAM"])
def test_saved_state_query_guard_epoch_and_tamper(tmp_path,arm):
    frame,settings=sample();y=frame[list(TARGETS)].to_numpy()
    m=ComponentRegressor(RECIPE,settings,arm,MECH,tmp_path).fit(frame,y)
    cold=verify_saved(tmp_path/"refit.pt",frame,y,arm,settings,MECH,expected_epoch=m.metadata_["selected_epoch"])
    query=frame.drop(columns=list(TARGETS))
    np.testing.assert_array_equal(cold.predict(query),m.predict(query))
    with pytest.raises(ValueError,match="targets"):cold.predict(frame)
    payload=torch.load(tmp_path/"refit.pt",weights_only=True);payload["trace"]["selected_epoch"]+=1
    torch.save(payload,tmp_path/"bad.pt")
    with pytest.raises(ValueError):verify_saved(tmp_path/"bad.pt",frame,y,arm,settings,MECH)
    with pytest.raises(FileExistsError):m.save(tmp_path/"refit.pt",m.traces["refit"])


def test_beta_zero_and_rho_zero_reduce_to_base():
    frame,settings=sample();y=frame[list(TARGETS)].to_numpy();query=frame.drop(columns=list(TARGETS))
    base=ComponentRegressor(RECIPE,settings,"BASE",MECH).fit(frame,y)
    for arm,mech in [("EMA",dict(MECH,ema_beta=0.)),("SAM",dict(MECH,sam_rho=0.))]:
        other=ComponentRegressor(RECIPE,settings,arm,mech).fit(frame,y)
        np.testing.assert_array_equal(base.predict(query),other.predict(query))
        assert base.metadata_["selected_epoch"]==other.metadata_["selected_epoch"]


def test_outer_labels_never_needed_and_fit_ids_train_only(tmp_path):
    frame,settings=sample();train=frame.iloc[:80];query=frame.iloc[80:].drop(columns=list(TARGETS))
    m=ComponentRegressor(RECIPE,settings,"SAM",MECH,tmp_path).fit(train,train[list(TARGETS)].to_numpy())
    assert m.metadata_["fit_rows"]==80
    np.testing.assert_allclose(m.preprocessor_.means_,train[list(m.preprocessor_.feature_names_)].mean().to_numpy(),atol=1e-14)
    assert m.predict(query).shape==(20,2)
    with pytest.raises(ValueError):m.predict(query.assign(tap_time_len=999))


def test_replacement_keeps_other_components_and_bootstrap_binds_rows():
    b=np.array([10.,20.]);old=np.array([6.,8.]);new=np.array([8.,4.])
    np.testing.assert_array_equal(replacement(b,old,new),[11.,18.])
    np.testing.assert_array_equal(replacement(b,old,old),b)
    y=np.array([1.,2.,7.]);base=np.array([[2.,3.,9.],[0.,1.,5.]])
    candidate=np.array([[1.,2.,7.],[1.,2.,7.]])
    report=paired_row_bootstrap(y,base,candidate,100,8)
    rng=np.random.default_rng(8);values=[]
    for _ in range(100):
        ix=rng.integers(0,3,3)
        values.append(np.mean([50*(np.abs(y[ix]-p[ix]).sum()-np.abs(y[ix]-q[ix]).sum())/np.abs(y[ix]).sum() for p,q in zip(base,candidate)]))
    assert report["mean"]==pytest.approx(np.mean(values))
    np.testing.assert_allclose(report["percentile95"],np.quantile(values,[.025,.975]))


def test_finalist_requires_both_seeds_and_control_advantage():
    spec=yaml.safe_load(Path("configs/strong_component_regularization/SPEC.yaml").read_text())
    rows=[]
    for target in TARGETS:
        for arm,gains in [("BASE",[0.,0.]),("EMA",[.03,-.001]),("SAM",[.011,.012])]:
            rows.append(dict(target=target,arm=arm,seeds={str(i):{"gain":v} for i,v in enumerate(gains)},paired={"mean":np.mean(gains)}))
    assert set(choose(rows,spec).values())=={"SAM"}


@pytest.mark.parametrize("seeds",[[42,3407],[271828,314159]])
def test_complete_summary_and_confirmation_use_their_own_seed_identity(monkeypatch,tmp_path,seeds):
    import bf_tap_r2.component_regularization_run as runner
    frame,_=sample();spec=yaml.safe_load(Path("configs/strong_component_regularization/SPEC.yaml").read_text())
    spec["diagnostics"]["bootstrap_replicates"]=20
    folds={s:np.arange(len(frame))%5 for s in seeds}
    candidates=spec["candidates"] if seeds[0]==42 else {t:["BASE","SAM"] for t in TARGETS}
    base={s:{t:frame[t].to_numpy()+1 for t in TARGETS} for s in seeds};members={}
    for s in seeds:
        base[s].update(v12_iron=base[s]["tap_iron"],v7_time=base[s]["tap_time_len"])
        for t,arms in candidates.items():
            for a in arms:members[s,t,a]=base[s][t]-(.5 if a!="BASE" else 0)
    monkeypatch.setattr(runner,"collect",lambda *args:(base,members))
    result=runner.summarize(tmp_path,frame,folds,spec,candidates)
    assert result["tiers"] is not None if seeds[0]==42 else result["tiers"] is None
    for row in result["records"]:
        assert set(row["seeds"])==set(map(str,seeds))
        assert row["paired"]["mean"]==0 if row["arm"]=="BASE" else row["paired"]["mean"]>0
