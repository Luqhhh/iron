"""Pure arrays/source specs only; no native model fits or label reads."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import joint_density_gmr_experiment as e
from bf_tap_r2.data import FEATURES


def spec():
    return json.loads((Path(__file__).resolve().parents[1]/e.SPEC).read_text())


def test_entire_spec_frozen():
    original=spec()
    e.validate_spec(original)
    for key in original:
        changed=copy.deepcopy(original)
        changed[key]=None if changed[key] is not None else 0
        with pytest.raises(ValueError,match="specification"):
            e.validate_spec(changed)


def test_arrays_synthetic_only_reproducible():
    a,b=e.synthetic_data(spec()),e.synthetic_data(spec())
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x,y)
    assert a[0].shape==(240,23)
    assert a[1].shape==(240,)


@pytest.mark.parametrize("values",[[1,3],[2,4],[1,1],[2,2]])
def test_spout_unsupported_or_incomplete_rejected(values):
    frame=pd.DataFrame(np.ones((2,21)),columns=FEATURES)
    frame["spout_no"]=values
    with pytest.raises(ValueError,match="spouts"):
        e.frame_arrays(frame)


def test_all_numeric_plus_two_onehot():
    frame=pd.DataFrame(np.arange(42).reshape(2,21),columns=FEATURES)
    frame["spout_no"]=[2,1]
    x=e.frame_arrays(frame)
    np.testing.assert_array_equal(x[:,:21],frame[list(FEATURES)].to_numpy())
    np.testing.assert_array_equal(x[:,21:],[[0,1],[1,0]])


def test_fixed_gain_unit_and_scalar():
    y=np.array([100.,100.])
    p=y+1
    assert e.gain(y,p,y)==pytest.approx(.1)
    with pytest.raises(ValueError,match="Invalid fixed blend"):
        e.gain(y,-100*y,-100*y)


def test_independent_likelihood_known_single_gaussian():
    z=np.zeros((2,3))
    got=e.independent_score(z,np.ones(1),np.zeros((1,3)),np.eye(3)[None])
    assert got==pytest.approx(-1.5*np.log(2*np.pi))


def test_development_full_seed_coverage_and_control_identity():
    n=2754
    q={"targets":np.ones((n,2))*100,"spout":1+np.arange(n)%2}
    vectors={}
    for s in (42,3407):
        q[f"fold-{s}"]=np.arange(n)%5
        q[f"current-{s}"]=np.ones((n,2))*101
        vectors[s,1]=np.ones(n)*101
        vectors[s,8]=np.ones(n)*100
    result=e.describe(q,vectors)
    for metrics in result.values():
        assert metrics["arms"]["K1"]["gain_vs_Q75"]==pytest.approx(0,abs=1e-12)
        assert metrics["arms"]["K8"]["gain_vs_Q75"]==pytest.approx(.1)
        assert metrics["K8_minus_K1_gain"]==pytest.approx(.1)
    vectors[42,8][0]=np.nan
    with pytest.raises(ValueError,match="complete"):
        e.describe(q,vectors)


def test_ledger_native_fits_and_kmeans_are_separate(tmp_path):
    units=[dict(key="K8",metadata={"fit_rows":10})]
    events=[dict(event="procedure_started",key="K8",fit_rows=10)]
    for i in range(5):
        events.extend(dict(event=ev,key="K8",start_index=i,random_state=42+i,fit_rows=10) for ev in
            ("native_fit_started","native_kmeans_started","native_kmeans_completed","native_fit_completed"))
    events.append(dict(event="procedure_completed",key="K8",metadata={"fit_rows":10}))
    for event in events:
        e.log(tmp_path,event)
    e.audit_ledger(tmp_path,units)
    e.log(tmp_path,dict(event="native_fit_completed",key="K8",start_index=0))
    with pytest.raises(ValueError,match="ledger"):
        e.audit_ledger(tmp_path,units)


@pytest.mark.parametrize("kind",["unknown_key","unknown_event"])
def test_ledger_rejects_hidden_native_entries(tmp_path,kind):
    e.log(tmp_path,dict(event="unregistered" if kind=="unknown_event" else "native_fit_started",
        key="missing" if kind=="unknown_key" else "K8",start_index=0))
    with pytest.raises(ValueError,match="Unknown ledger"):
        e.audit_ledger(tmp_path,[dict(key="K8",metadata={})])


def test_runtime_refuses_thread_drift(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key,"1")
    monkeypatch.setenv("OMP_NUM_THREADS","2")
    with pytest.raises(ValueError,match="threads"):
        e.runtime()


@pytest.mark.parametrize("constant",[False,True])
def test_synthetic_learning_gate_without_any_fit(tmp_path,monkeypatch,constant):
    _,y=e.synthetic_data(spec())
    def fake_fit(out,key,k,x,training_y,query_x,outer,query):
        pred=np.full(len(query),np.median(training_y)) if constant else y[query].copy()
        return dict(key=key,k=k,metadata={}),pred
    monkeypatch.setattr(e,"fit_one",fake_fit)
    if constant:
        with pytest.raises(ValueError,match="learning gate"):
            e.engineering(tmp_path,spec())
    else:
        result=e.engineering(tmp_path,spec())
        assert result["official_label_parses"]==0
        assert all(m["query_MAE"]==0 and m["fit_median_constant_query_MAE"]>0
            for m in result["learning"].values())
