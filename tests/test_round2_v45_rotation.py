from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.tree import DecisionTreeRegressor

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v45_rotation import (RotationRegressor, export_tree, fit_partition,
                                    pca_block, predict_tree, project, tree_design)
from bf_tap_r2.v45_verify import verify_forest


@pytest.fixture
def cfg():
    s = yaml.safe_load(Path('configs/round2_v45/SPEC.yaml').read_text())['training']
    return dict(s, trees=6)


@pytest.fixture
def sample():
    rng = np.random.default_rng(901)
    x = rng.normal(size=(120, len(FEATURES)))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame['sample_id'] = [f'unit-{i}' for i in range(len(x))]
    frame['spout_no'] = rng.integers(1, 3, len(x))
    y = 20+3*x[:, 0]+2*x[:, 1]**2
    return frame, y


def test_full_rank_pca_preserves_distances_and_training_covariance():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(80, 6))
    x[:, 1] = 3*x[:, 0]
    x[:, 2] = 7  # singular and constant columns; never divide by singular values
    block = pca_block(x, np.arange(3), np.arange(60))
    axes = np.asarray(block['axes'])
    np.testing.assert_allclose(axes.T@axes, np.eye(3), atol=1e-14)
    centered = x[:, :3]-block['center']
    np.testing.assert_allclose(np.linalg.norm(centered@axes, axis=1),
                               np.linalg.norm(centered, axis=1), atol=1e-14)
    expected = np.column_stack([centered@axes, x[:, 3:]]).astype(np.float32)
    np.testing.assert_allclose(project(x, [block]), expected, atol=1e-6)
    np.testing.assert_array_equal(project(x[::-1], [block])[::-1], project(x, [block]))


def test_export_matches_independent_sklearn_tree_on_boundaries():
    x = np.array([[i, i%3] for i in range(40)], np.float32)
    y = np.sin(x[:, 0])
    estimator = DecisionTreeRegressor(min_samples_leaf=3, random_state=31).fit(x, y)
    tree = export_tree(estimator)
    query = np.concatenate([x, x+.00001, x-.00001])
    np.testing.assert_array_equal(predict_tree(query, tree), estimator.predict(query))


@pytest.mark.parametrize('arm', ['AXIS', 'ROTATE'])
def test_saved_forest_sampling_provenance_and_cold_predictions(cfg, sample, tmp_path, arm):
    frame, y = sample
    model = RotationRegressor(arm, cfg).fit(frame.iloc[:100], y[:100])
    assert verify_forest(model, frame.iloc[:100], y[:100])['trees_verified'] == cfg['trees']
    for i, member in enumerate(model.members_):
        bag, _, _, _ = tree_design(100, cfg, i)
        expected = np.random.default_rng(np.random.SeedSequence([42, i, 0])).integers(100, size=100)
        np.testing.assert_array_equal(bag, expected)
    p = tmp_path/'forest.json'
    model.save(p)
    with pytest.raises(FileExistsError):
        model.save(p)
    loaded = RotationRegressor.load(p)
    query = frame.iloc[100:]
    expected = model.predict(query)
    np.testing.assert_array_equal(loaded.predict(query), expected)
    np.testing.assert_array_equal(loaded.predict(query.iloc[::-1])[::-1], expected)
    np.testing.assert_array_equal(np.concatenate([loaded.predict(query.iloc[i:i+3])
                                                 for i in range(0, len(query), 3)]), expected)
    metadata = deepcopy(loaded.preprocessor_.metadata())
    extreme = query.copy();extreme[list(FEATURES)] = 1000000
    assert np.isfinite(loaded.predict(extreme)).all()
    assert loaded.preprocessor_.metadata() == metadata
    with pytest.raises(ValueError, match='Query targets'):
        loaded.predict(query.assign(tap_iron=0))
    mutated = RotationRegressor.load(p)
    mutated.members_[0]['tree']['value'][0] += 1
    with pytest.raises(ValueError, match='means'):
        verify_forest(mutated, frame.iloc[:100], y[:100])


def test_axis_and_rotation_share_sampling_and_identity_basis_control(cfg, sample):
    frame, y = sample
    axis = RotationRegressor('AXIS', cfg).fit(frame, y)
    rotated = RotationRegressor('ROTATE', cfg).fit(frame, y)
    assert [(m['bag_digest'],m['tree_seed']) for m in axis.members_] == [
        (m['bag_digest'],m['tree_seed']) for m in rotated.members_]
    identity = deepcopy(axis)
    for member in identity.members_:
        member['blocks'] = [{'indices': list(range(i,i+3)), 'center': [0.,0.,0.],
                             'axes': np.eye(3).tolist()} for i in range(0,len(FEATURES),3)]
    np.testing.assert_array_equal(identity.predict(frame), axis.predict(frame))


def test_audit_rejects_pca_center_and_nonorthogonal_basis(cfg, sample):
    frame, y = sample
    model = RotationRegressor('ROTATE', dict(cfg,trees=1)).fit(frame,y)
    changed = deepcopy(model);changed.members_[0]['blocks'][0]['center'][0] += .1
    with pytest.raises(ValueError, match='center'):
        verify_forest(changed, frame, y)
    changed = deepcopy(model);changed.members_[0]['blocks'][0]['axes'][0][0] += .1
    with pytest.raises(ValueError, match='orthogonal'):
        verify_forest(changed, frame, y)


def test_partition_rejects_query_labels_before_fitting(cfg, sample):
    frame, y = sample
    frame = frame.assign(tap_iron=y, tap_time_len=y)
    with pytest.raises(ValueError, match='Query labels'):
        fit_partition(frame.iloc[:70],frame.iloc[70:100],frame.iloc[:100],frame.iloc[100:],
                      'tap_iron','ROTATE',cfg,y[70:100],[0.,.5,1.])


def test_constant_features_keep_full_basis_and_finite_leaf_mean(cfg, sample):
    frame, y = sample
    frame = frame.copy();frame[list(FEATURES)] = 1.;frame['spout_no'] = 1
    model = RotationRegressor('ROTATE', dict(cfg,trees=2)).fit(frame,y)
    assert verify_forest(model,frame,y)['trees_verified']==2
    expected = []
    for i in range(2):
        bag,_,_,_ = tree_design(len(frame),cfg,i)
        expected.append(y[bag].mean())
    np.testing.assert_allclose(model.predict(frame),np.mean(expected),rtol=1e-14)
