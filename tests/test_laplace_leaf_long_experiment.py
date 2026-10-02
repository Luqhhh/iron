"""Prefix identity and frozen scope tests without model fitting."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bf_tap_r2.laplace_leaf_long_experiment import SPEC, check_prefix, validate_spec


class FakeTree:
    def __init__(self, threshold=0.):
        self.tree_=self
        self.threshold=threshold

    def get_params(self):
        return {"max_depth":3,"random_state":42}

    def __getstate__(self):
        return {"nodes":np.array([(self.threshold,)],dtype=[("threshold","f8")]),"values":np.array([1.])}


def model():
    return SimpleNamespace(trees_=[FakeTree()],leaf_values_=[np.array([np.nan,1.])],history_=[{"epoch":1}],
        x_mean_=np.array([0.]),x_scale_=np.array([1.]),y_median_=1.,y_scale_=1.,fit_rows_=2,
        initial_train_mae_=1.,initial_calibration_mae_=1.)


def test_exact_prefix_accepts_semantic_tree_state():
    check_prefix(model(),model(),1)


@pytest.mark.parametrize("change",["threshold","leaf","history","normalizer"])
def test_changed_prefix_rejected(change):
    left,right=model(),model()
    if change=="threshold": right.trees_[0].threshold=1.
    elif change=="leaf": right.leaf_values_[0][1]=2.
    elif change=="history": right.history_[0]["epoch"]=2
    else: right.x_mean_[0]=1.
    with pytest.raises((ValueError,AssertionError)):
        check_prefix(left,right,1)


def test_every_spec_field_is_frozen():
    root=Path(__file__).resolve().parents[1]
    original=json.loads((root/SPEC).read_text())
    validate_spec(original)
    for key in original:
        changed=dict(original)
        changed[key]="changed"
        with pytest.raises(ValueError): validate_spec(changed)
