from pathlib import Path

import numpy as np
import pytest
import yaml

from bf_tap_r2.smooth_residual import make_model
from test_round2_v2_robust_joint import synthetic


@pytest.mark.parametrize('route',['R3','D8','ORD'])
def test_capacity_routes_use_fold_means_and_fixed_parameters(route):
    spec=yaml.safe_load(Path('configs/round2_v2_5/experiment.yaml').read_text())
    spec['common'].update(iterations=8,thread_count=1)
    frame=synthetic()
    model=make_model(route,'tap_time_len',spec).fit(frame.iloc[:40],frame.tap_time_len.iloc[:40])
    np.testing.assert_allclose(model.target_scales_,[frame.tap_time_len.iloc[:40].mean()])
    assert model.predict(frame.iloc[40:41]).shape==(1,)
    for key,value in spec['routes'][route].items():
        assert model.actual_parameters_[key]==value
