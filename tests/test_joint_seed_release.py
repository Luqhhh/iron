from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.v2_selected_release import predict_node


def test_joint_seed_nodes_average_only_iron_and_reject_target_order(tmp_path,monkeypatch):
    model=SimpleNamespace(target_fields_=('tap_iron','tap_time_len'),estimator_=SimpleNamespace(tree_count_=1500),
                         predict=lambda frame: np.tile([501.,999.],(len(frame),1)))
    monkeypatch.setattr('bf_tap_r2.v2_selected_release.digest',lambda path:'sha')
    monkeypatch.setattr('bf_tap_r2.v2_selected_release.joblib.load',lambda path:model)
    node={'kind':'joint_iron','target':'tap_iron','path':'joint.joblib','sha256':'sha'}
    blend={'kind':'mean','weights':[1/3]*3,'members':[node,node,node],'aggregation':'arithmetic_mean'}
    frame=pd.DataFrame({'sample_id':['one','two']})
    np.testing.assert_allclose(predict_node(tmp_path,blend,frame),[501,501])
    model.target_fields_=('tap_time_len','tap_iron')
    with pytest.raises(ValueError,match='identity'):
        predict_node(tmp_path,node,frame)
