from pathlib import Path
import joblib
import numpy as np
import pytest
import yaml
from bf_tap_r2.engineered_models import EngineeredRegressor,engineered_inputs,ENGINEERED_FIELDS
from bf_tap_r2.data import TARGETS
from test_round2_v2_robust_joint import synthetic


def test_four_feature_model_fold_scale_and_serialized_feature_only_inference(tmp_path):
    f=synthetic();f['air_volume']=f.air_volume.abs()+1;f['total_press_diff']=f.total_press_diff.abs()+1
    tr,va=f.iloc[:40],f.iloc[40:].drop(columns=list(TARGETS))
    spec=yaml.safe_load(Path('configs/round2_v2_12/experiment.yaml').read_text())['models']['FE4']
    m=EngineeredRegressor({**spec,'iterations':8,'thread_count':1},'tap_iron').fit(tr,tr.tap_iron)
    np.testing.assert_allclose(m.target_scales_,tr.tap_iron.mean())
    assert m.estimator_.feature_names_==list(ENGINEERED_FIELDS)
    x=engineered_inputs(va)
    np.testing.assert_allclose(x.thermal_difference,va.hot_air_temp-va.furnace_throat_temp)
    np.testing.assert_allclose(x.upper_pressure_fraction,va.upper_press_diff/va.total_press_diff)
    np.testing.assert_allclose(x.pressure_per_air_volume,va.total_press_diff/va.air_volume)
    path=tmp_path/'m.joblib';joblib.dump(m,path);restored=joblib.load(path)
    pred=restored.predict(va)
    np.testing.assert_array_equal(pred,m.predict(va))
    np.testing.assert_array_equal(pred,restored.predict(va.iloc[::-1])[::-1])
    np.testing.assert_array_equal(pred[:1],restored.predict(va.iloc[:1]))
    np.testing.assert_array_equal(pred,np.concatenate([restored.predict(va.iloc[:7]),restored.predict(va.iloc[7:])]))
    va['tap_iron']=np.nan;va['tap_time_len']=-999
    np.testing.assert_array_equal(pred,restored.predict(va))
    restored.input_fields_=tuple(reversed(ENGINEERED_FIELDS))
    with pytest.raises(ValueError,match='Stored engineered'):
        restored.predict(va)
