from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml
from threadpoolctl import threadpool_limits

from bf_tap_r2.nonlinear_models import NonlinearRegressor
from bf_tap_r2.data import TARGETS,FEATURES
from test_round2_v2_robust_joint import synthetic


@pytest.mark.parametrize('route',['JM1','KR1','KS1'])
def test_nonlinear_fold_scales_serialization_and_feature_only_inference(route,tmp_path):
    spec=yaml.safe_load(Path('configs/round2_v2_7/experiment.yaml').read_text())['models'][route]
    if route=='JM1':
        spec.update(hidden_layer_sizes=[4,3],tol=1e-5)
    frame=synthetic();training=frame.iloc[:40];valid=frame.iloc[40:]
    target='tap_iron' if route=='KS1' else None
    labels=training[target] if target else training[list(TARGETS)]
    with threadpool_limits(limits=1):
        model=NonlinearRegressor(route,spec,target).fit(training,labels)
        np.testing.assert_allclose(model.scales_,training[[target] if target else list(TARGETS)].mean())
        np.testing.assert_allclose(model.preprocessor_.named_transformers_['numeric'].mean_,training[list(FEATURES)].mean())
        path=tmp_path/'model.joblib';joblib.dump(model,path)
        restored=joblib.load(path)
        x=valid.drop(columns=list(TARGETS))
        pred=restored.predict(x)
        assert pred.shape==((20,) if target else (20,2))
        np.testing.assert_allclose(pred,model.predict(valid),rtol=1e-12)
        np.testing.assert_allclose(pred,restored.predict(x.iloc[::-1])[::-1],rtol=1e-12)
        np.testing.assert_allclose(pred[:1],restored.predict(x.iloc[:1]),rtol=1e-12)
        assert not model.fit_report_['convergence_warning']


def test_joint_target_order_rejected():
    frame=synthetic()
    with pytest.raises(ValueError,match='order'):
        NonlinearRegressor('KR1',{}).fit(frame,frame[list(TARGETS)[::-1]])


def test_nonpositive_training_mean_rejected():
    frame=synthetic()
    with pytest.raises(ValueError,match='Positive'):
        NonlinearRegressor('KS1',{},'tap_iron').fit(frame,np.zeros(len(frame)))


def test_completed_solver_cap_can_be_reused_but_not_other_parameters():
    from bf_tap_r2.v2_nonlinear import compatible_parameters
    old={'C':10.,'tol':1e-5,'max_iter':100000}
    new={**old,'max_iter':1000000}
    report={'iterations':90000,'fit_status':0,'convergence_warning':False}
    assert compatible_parameters(old,new,'KS1',report,True)
    assert not compatible_parameters(old,new,'KS1',report,False)
    assert not compatible_parameters(old,{**new,'C':20.},'KS1',report,True)
    assert not compatible_parameters(old,new,'KS1',{**report,'convergence_warning':True},True)
    assert not compatible_parameters(old,new,'KS1',{**report,'iterations':100000},True)
