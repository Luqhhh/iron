from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip('torch')
pytest.importorskip('tabm')
pytest.importorskip('rtdl_num_embeddings')
pytest.importorskip('rtdl_revisiting_models')

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_periodic import PeriodicRegressor
from bf_tap_r2.v7_release import save_and_cold
from bf_tap_r2.v11_quantile import QuantilePreprocessor, QuantileRegressor, choose_confirmation, evaluate_fold


def sample():
    rng = np.random.default_rng(711)
    frame = pd.DataFrame(rng.lognormal(size=(100, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 3 + 1
    frame['tap_time_len'] = 100 + 5*np.sin(frame.air_volume) + rng.normal(size=len(frame))
    frame['tap_iron'] = 4*frame.tap_time_len
    spec = yaml.safe_load(Path('configs/round2_v11/SPEC.yaml').read_text())
    settings = spec['training']
    settings.update(width=16, tabm_k=2, max_epochs=3, patience=2, embedding_dim=4, n_frequencies=4)
    return frame, settings, spec['quantile_policy']


@pytest.mark.parametrize('distribution', ['normal', 'uniform'])
def test_training_landmarks_monotonicity_and_bounded_extrapolation(distribution):
    frame, _, policy = sample(); training = frame.iloc[:80]
    fitted = QuantilePreprocessor(distribution, policy).fit(training)
    raw = training[list(FEATURES)].to_numpy(dtype=np.float64)
    noisy = raw + np.random.RandomState(42).normal(0., 1e-5, raw.shape)
    np.testing.assert_allclose(fitted.transformer_.quantiles_, np.percentile(noisy, np.linspace(0., 100., 10), axis=0), atol=1e-12, rtol=0)
    query = frame.iloc[80:].copy()
    query.loc[:, 'spout_no'] = 999
    query.loc[:, 'tap_time_len'] = 1e9; query.loc[:, 'tap_iron'] = -1e9
    unchanged = fitted.quantile_digest_
    x, cat = fitted.transform_tabm(query)
    np.testing.assert_array_equal(x, fitted.transform_tabm(query.drop(columns=['sample_id', 'tap_time_len', 'tap_iron']))[0])
    assert np.count_nonzero(cat) == 0
    for j, feature in enumerate(FEATURES):
        ordered = query.sort_values(feature)
        assert (np.diff(fitted.transform_tabm(ordered)[0][:, j]) >= 0).all()
    extremes = query.iloc[:2].copy()
    extremes.loc[extremes.index[0], list(FEATURES)] = -1e9
    extremes.loc[extremes.index[1], list(FEATURES)] = 1e9
    values, _ = fitted.transform_tabm(extremes)
    assert np.isfinite(values).all() and (values[0] < 0).all() and (values[1] > 0).all()
    assert np.max(np.abs(values)) < (5.3 if distribution == 'normal' else 1.733)
    assert fitted.quantile_digest_ == unchanged


def test_standard_adapter_preserves_original_v7_training():
    frame, settings, policy = sample()
    recipe = {'backbone': 'tabm', 'frequency': .01, 'distribution': 'standard'}
    old = PeriodicRegressor(recipe, settings).fit(frame, frame.tap_time_len.values)
    new = QuantileRegressor(recipe, settings, policy).fit(frame, frame.tap_time_len.values)
    np.testing.assert_array_equal(old.predict(frame), new.predict(frame))
    for key, value in old.metadata_.items():
        assert value == new.metadata_[key]


@pytest.mark.parametrize('distribution', ['normal', 'uniform'])
@pytest.mark.parametrize('frequency', [None, .01])
def test_inner_only_landmarks_repeat_fit_and_cold_inference(tmp_path, distribution, frequency):
    frame, settings, policy = sample()
    recipe = {'backbone': 'tabm', 'frequency': frequency, 'distribution': distribution}
    model = QuantileRegressor(recipe, settings, policy).fit(frame, frame.tap_time_len.values)
    again = QuantileRegressor(recipe, settings, policy).fit(frame, frame.tap_time_len.values)
    np.testing.assert_array_equal(model.predict(frame), again.predict(frame))
    inner = group_safe_inner_folds(frame, seed=42)['fold'] != 0
    for trace, part in zip(model.metadata_['preprocessing_trace'], [frame.loc[inner], frame]):
        expected = QuantilePreprocessor(distribution, policy).fit(part)
        assert trace['rows'] == len(part)
        assert trace['noisy_fit_digest'] == expected.noisy_fit_digest_
        assert trace['quantile_digest'] == expected.quantile_digest_
    json.dumps(model.metadata_, allow_nan=False)
    result = save_and_cold(model, frame, model.predict(frame), tmp_path/'cold', 1e-6)
    assert result['bit_identical'] and result['training_reads_prohibited']


def test_outer_features_and_labels_cannot_change_fitted_quantile_landmarks():
    frame, settings, policy = sample(); folds = np.arange(len(frame)) % 5
    recipe = {'backbone': 'tabm', 'frequency': .01, 'distribution': 'normal'}
    pred, metadata = evaluate_fold(frame, folds, 'tap_time_len', recipe, settings, policy, 0)
    changed = frame.copy()
    changed.loc[folds == 0, ['tap_iron', 'tap_time_len']] = 1e9
    other, other_meta = evaluate_fold(changed, folds, 'tap_time_len', recipe, settings, policy, 0)
    np.testing.assert_array_equal(pred, other); assert metadata == other_meta
    changed.loc[folds == 0, list(FEATURES)] *= 1000
    _, transformed_meta = evaluate_fold(changed, folds, 'tap_time_len', recipe, settings, policy, 0)
    assert metadata == transformed_meta


def test_selection_requires_increment_on_both_complete_splits():
    spec = {'tie_preference_by_target': {t: ['raw_qnormal', 'plr_qnormal'] for t in ['tap_iron', 'tap_time_len']}}
    def row(target, name, gains, base):
        return {'target': target, 'recipe': name, 'comparisons': {
            'A35': {'seed_gains': dict(zip([42, 3407], base))},
            'CURRENT': {'seed_gains': dict(zip([42, 3407], gains)), 'seed_summary': {'mean': np.mean(gains)}}}}
    bad = row('tap_time_len', 'plr_qnormal', [.01, -.001], [.04, .04])
    good = row('tap_iron', 'raw_qnormal', [.002, .003], [.01, .01])
    assert choose_confirmation([bad], spec) is None
    assert choose_confirmation([bad, good], spec) == {'target': 'tap_iron', 'recipe': 'raw_qnormal'}
