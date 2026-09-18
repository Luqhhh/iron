from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "qrf_v029"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qrf_v015"))

from append_forest import (  # noqa: E402
    PARENT_PARAMETERS,
    PARENT_TREES,
    TOTAL_TREES,
    AppendedTimeForest,
    append_forest,
    full_leaf_mapping,
)
from oob_append import (  # noqa: E402
    ATTACHMENT_PROTOCOL,
    REGRESSION_TREES,
    RestrictedView,
    assert_prefix_equal,
    full_leaf_arrays,
    require_1024_identity,
    slice_prefix_arrays,
)
from oob_response import derive_attachment, predict  # noqa: E402


def toy_pair():
    rng = np.random.default_rng(7)
    rows, features = 80, 4
    x = rng.normal(size=(rows, features)).astype(np.float32)
    y = np.abs(rng.normal(loc=100.0, scale=20.0, size=rows))
    ids = [f"id-{index}" for index in range(rows)]
    parent = RandomForestRegressor(**PARENT_PARAMETERS).fit(x, y)
    forest, certificate = append_forest(parent, ids, x, y)
    leaves = full_leaf_mapping(parent, x) + full_leaf_mapping(forest, x)[PARENT_TREES:]
    model = AppendedTimeForest(forest, ids, y, np.arange(rows) % 3, leaves, certificate)
    parent_model = SimpleNamespace(
        forest=parent, y=y, ids=ids, training_months=np.arange(rows) % 3,
        leaves=full_leaf_mapping(parent, x),
    )
    return parent_model, model, x


@pytest.fixture(scope="module")
def toy():
    return toy_pair()


def test_slice_and_prefix_match_full_attachment(toy):
    parent_model, model, x = toy
    full, _ = derive_attachment(model, x)
    prefix, _ = derive_attachment(parent_model, x)
    sliced = slice_prefix_arrays(full, REGRESSION_TREES)
    assert sliced["draws"].shape == (REGRESSION_TREES, len(x))
    assert np.array_equal(sliced["draws"], prefix["draws"])
    assert np.array_equal(sliced["members"], prefix["members"])
    receipt = assert_prefix_equal(full, prefix, REGRESSION_TREES, label="toy")
    assert receipt["prefix_leaf_members_exact"] is True


def test_prefix_difference_is_rejected(toy):
    parent_model, model, x = toy
    full, _ = derive_attachment(model, x)
    prefix, _ = derive_attachment(parent_model, x)
    broken = {name: np.array(value) for name, value in prefix.items()}
    broken["members"][0] = (broken["members"][0] + 1) % len(x)
    with pytest.raises(ValueError, match="members"):
        assert_prefix_equal(full, broken, REGRESSION_TREES, label="toy")


def test_restricted_view_matches_parent_oob(toy):
    parent_model, model, x = toy
    parent_arrays, _ = derive_attachment(parent_model, x)
    view = RestrictedView(model, REGRESSION_TREES)
    parent_median, parent_mean, _ = predict(parent_model, x, parent_arrays, parent_model.training_months)
    view_median, view_mean, _ = predict(view, x, parent_arrays, view.training_months)
    assert np.array_equal(parent_median, view_median)
    assert np.array_equal(parent_mean, view_mean)


def test_full_leaf_arrays_reproduce_full_leaf_qrf(toy):
    parent_model, model, x = toy
    view = RestrictedView(model, REGRESSION_TREES)
    arrays = full_leaf_arrays(view)
    median, mean, _ = predict(view, x, arrays, view.training_months)
    expected_median, expected_mean, _ = predict(parent_model, x, full_leaf_arrays(parent_model), parent_model.training_months)
    assert np.array_equal(median, expected_median)
    assert np.array_equal(mean, expected_mean)


def test_require_1024_identity_rejects_old_loader(toy):
    parent_model, model, _ = toy
    assert require_1024_identity(model)["new_trees"] == 768
    assert ATTACHMENT_PROTOCOL == "QRF_OOB_LEAF_APPENDED_1024_v030"
    old = SimpleNamespace(forest=parent_model.forest, leaves=parent_model.leaves, protocol=model.protocol)
    with pytest.raises(ValueError, match="1024"):
        require_1024_identity(old)
    model.prefix_verified = False
    with pytest.raises(ValueError, match="prefix"):
        require_1024_identity(model)
    model.prefix_verified = True
