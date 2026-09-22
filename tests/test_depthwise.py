from pathlib import Path
import joblib
import numpy as np
import yaml
from bf_tap_r2.normalized_models import NormalizedSnapshotRegressor
from bf_tap_r2.data import TARGETS
from test_round2_v2_robust_joint import synthetic


def test_depthwise_serialization_and_feature_only_prediction(tmp_path):
    parameters=yaml.safe_load(Path('configs/round2_v2_10/experiment.yaml').read_text())['models']['DW']
    parameters={**parameters,'iterations':8,'thread_count':1}
    frame=synthetic();training=frame.iloc[:40];valid=frame.iloc[40:].drop(columns=list(TARGETS))
    model=NormalizedSnapshotRegressor(parameters,'tap_iron').fit(training,training.tap_iron)
    assert model.actual_parameters_['grow_policy']=='Depthwise'
    assert model.actual_parameters_['min_data_in_leaf']==20
    np.testing.assert_allclose(model.target_scales_,training.tap_iron.mean())
    p=tmp_path/'depthwise.joblib';joblib.dump(model,p);restored=joblib.load(p)
    pred=restored.predict(valid)
    assert pred.shape==(20,)
    np.testing.assert_array_equal(pred,model.predict(valid))
    np.testing.assert_array_equal(pred,restored.predict(valid.iloc[::-1])[::-1])
    np.testing.assert_array_equal(pred[:1],restored.predict(valid.iloc[:1]))
    np.testing.assert_array_equal(pred,np.concatenate([restored.predict(valid.iloc[:7]),restored.predict(valid.iloc[7:])]))
