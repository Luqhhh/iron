from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.kernel_geometry import GeometryRegressor, numeric_weights
from test_round2_v2_robust_joint import synthetic


def test_positive_geometry_keeps_all_features_and_total_weight():
    a=np.zeros(len(FEATURES));a[0]=100
    w=numeric_weights(a)
    assert (w>=0.1).all()
    np.testing.assert_allclose(w.sum(),len(FEATURES))
    np.testing.assert_allclose(w,numeric_weights(a*7))
    for bad in (a[:-1],-a,a*np.nan,np.zeros_like(a)):
        with pytest.raises(ValueError):
            numeric_weights(bad)


@pytest.mark.parametrize('route',['KSM','KWT'])
def test_geometry_matches_explicit_kernel_and_survives_feature_only_inference(route,tmp_path):
    f=synthetic();tr=f.iloc[:40];va=f.iloc[40:].drop(columns=list(TARGETS))
    spec=yaml.safe_load(Path('configs/round2_v2_8/experiment.yaml').read_text())['models'][route]
    w=np.ones(len(FEATURES)) if route=='KSM' else numeric_weights(np.arange(1,len(FEATURES)+1))
    with threadpool_limits(limits=1):
        m=GeometryRegressor(route,spec,'tap_iron',w).fit(tr,tr.tap_iron)
        np.testing.assert_allclose(m.scales_,tr.tap_iron.mean())
        np.testing.assert_allclose(m.preprocessor_.named_transformers_['numeric'].mean_,tr[list(FEATURES)].mean())
        # Independent RBF distance/system calculation verifies sqrt weights,
        # unweighted one-hot geometry, centering and restoration in original units.
        mu=tr[list(FEATURES)].mean().to_numpy();sd=tr[list(FEATURES)].std(ddof=0).to_numpy()
        categories=sorted(tr.spout_no.unique())
        def explicit(frame):
            numeric=(frame[list(FEATURES)].to_numpy()-mu)/sd*np.sqrt(w)
            onehot=np.column_stack([frame.spout_no.to_numpy()==c for c in categories])
            return np.column_stack([numeric,onehot])
        x=explicit(tr);v=explicit(va)
        z=(tr.tap_iron.to_numpy()-tr.tap_iron.mean())/tr.tap_iron.mean()
        dual=np.linalg.solve(rbf_kernel(x,gamma=spec['gamma'])+spec['alpha']*np.eye(len(tr)),z)
        expected=tr.tap_iron.mean()*(1+rbf_kernel(v,x,gamma=spec['gamma'])@dual)
        np.testing.assert_allclose(m.predict(va),expected,rtol=1e-11,atol=1e-10)
        path=tmp_path/'model.joblib';joblib.dump(m,path);loaded=joblib.load(path)
        pred=loaded.predict(va)
        assert pred.shape==(20,)
        np.testing.assert_allclose(pred,loaded.predict(va.iloc[::-1])[::-1],rtol=1e-12)
        np.testing.assert_allclose(pred[:1],loaded.predict(va.iloc[:1]),rtol=1e-12)
        va['tap_iron']=-999;va['tap_time_len']=np.nan
        np.testing.assert_array_equal(pred,loaded.predict(va))


def test_geometry_rejects_wrong_shapes_and_uniform_control_mismatch():
    w=np.ones(len(FEATURES))
    with pytest.raises(ValueError,match='uniform'):
        GeometryRegressor('KSM',{},TARGETS[0],w*2)
    with pytest.raises(ValueError,match='Positive finite'):
        GeometryRegressor('KWT',{},TARGETS[0],w*0)
    f=synthetic();m=GeometryRegressor('KWT',{},TARGETS[0],w)
    with pytest.raises(ValueError,match='single target'):
        m.fit(f,f[list(TARGETS)])
    with pytest.raises(ValueError,match='Positive target mean'):
        m.fit(f,-np.ones(len(f)))
