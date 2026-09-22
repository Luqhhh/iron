from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

from test_round2_v2_robust_joint import synthetic
from bf_tap_r2.smooth_residual import make_model


@pytest.mark.parametrize('route',['QRC','BAY','R30'])
def test_fold_only_transform_and_serialized_prediction(route,tmp_path):
    spec=yaml.safe_load(Path('configs/round2_v2_4/experiment.yaml').read_text())
    spec['common'].update(iterations=10,thread_count=1)
    frame=synthetic()
    model=make_model(route,'tap_iron',spec).fit(frame.iloc[:40],frame.tap_iron.iloc[:40])
    scale=model.target_scale_ if route=='QRC' else model.target_scales_[0]
    assert scale==pytest.approx(frame.tap_iron.iloc[:40].mean())
    path=tmp_path/'model.joblib'
    joblib.dump(model,path)
    recovered=joblib.load(path)
    valid=frame.iloc[40:].drop(columns=['tap_iron','tap_time_len'])
    pred=recovered.predict(valid)
    assert pred.shape==(20,)
    np.testing.assert_allclose(pred,model.predict(frame.iloc[40:]),rtol=1e-12)
    np.testing.assert_allclose(pred,recovered.predict(valid.iloc[::-1])[::-1],rtol=1e-12)
    np.testing.assert_allclose(pred,np.concatenate([recovered.predict(valid.iloc[:1]),recovered.predict(valid.iloc[1:])]),rtol=1e-12)
    if route=='QRC':
        transform=recovered.smooth_.named_steps['transform'].named_transformers_['numeric']
        np.testing.assert_allclose(transform.named_steps['input_scale'].mean_,frame.iloc[:40][transform.feature_names_in_].mean())


def test_invalid_residual_target_rejected():
    spec=yaml.safe_load(Path('configs/round2_v2_4/experiment.yaml').read_text())
    frame=synthetic()
    with pytest.raises(ValueError,match='labels'):
        make_model('QRC','tap_iron',spec).fit(frame,np.zeros(len(frame)))
