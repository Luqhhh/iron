from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.dnnr_model import (ARMS, Encoder, Regressor, Settings,
                                exact_neighbors, metric_epoch)
from bf_tap_r2.dnnr_calibration import fit_pair
from bf_tap_r2.dnnr_audit import verify_saved, verify_pair
from bf_tap_r2.v7_periodic import digest, file_hash


def sample(rows=75, seed=57321, start=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-dnnr-{i:05d}' for i in range(start, start+rows)]
    frame['spout_no'] = np.arange(rows) % 3+1
    return frame


def test_train_only_encoder_handles_unseen_spout_constants_and_query_outliers():
    fit = sample(); fit[FEATURES[-1]] = 7.
    query = sample(13, 57322, 200); query.spout_no = 999; query[FEATURES[0]] = 1e6
    encoder = Encoder().fit(fit); before = encoder.metadata(); x = encoder.transform(query)
    assert encoder.metadata() == before
    assert encoder.means_[0] != 1e6 and encoder.stds_[-1] == 1
    np.testing.assert_array_equal(x[:, len(FEATURES)], 1)
    np.testing.assert_array_equal(x[:, len(FEATURES)+1:], 0)
    np.testing.assert_array_equal(Encoder.from_metadata(before).transform(query), x)


def test_tied_duplicate_neighbors_use_ids_and_actual_anchor_identity():
    x = np.zeros((5, 3)); ids = ['z', 'b', 'a', 'c', 'd']
    np.testing.assert_array_equal(exact_neighbors(x, x[0], ids, 3), [2, 1, 3])
    np.testing.assert_array_equal(exact_neighbors(x, x[0], ids, 3, exclude=0), [2, 1, 3])
    np.testing.assert_array_equal(exact_neighbors(x, x[1], ids, 3, include_anchor=1), [1, 2, 3])
    with pytest.raises(ValueError): exact_neighbors(x, x[0], ids, 5, exclude=0)


def test_affine_taylor_extrapolation_beats_knn_without_target_clipping():
    fit = sample(100); query = sample(13, 57322, 200)
    query.air_volume = 15.
    y = 50+fit.air_volume.to_numpy()*3-fit.cold_air_press.to_numpy()*2
    expected = 50+query.air_volume.to_numpy()*3-query.cold_air_press.to_numpy()*2
    events = []
    fixed = Regressor('DNNR_FIXED').fit(fit, y, observer=lambda k, v: events.append(k))
    knn = Regressor('KNN_FIXED').fit(fit, y)
    np.testing.assert_allclose(fixed.predict(query), expected, rtol=0, atol=1e-10)
    assert abs(knn.predict(query)-expected).mean() > 30
    assert fixed.predict(query).min() > y.max()
    assert events == ['estimator', 'derivative_bank']


def test_supervised_metric_uses_fixed_initial_graph_and_all_training_updates():
    fit = sample(100); encoder = Encoder().fit(fit); x = encoder.transform(fit)
    y = 20+2*fit.air_volume.to_numpy()**2+.2*fit.cold_air_press.to_numpy()
    settings = Settings(); events = []
    scale, witnesses = metric_epoch(x, y, fit.sample_id.tolist(), settings,
                                    lambda k, m: events.append((k, m)))
    assert events[0][0] == 'metric_epoch' and events[0][1]['updates'] == len(fit)
    assert np.max(abs(scale-1)) > 1e-5
    for i, g in enumerate(witnesses['metric_graph']):
        assert i not in g
        np.testing.assert_array_equal(g, exact_neighbors(x, x[i], fit.sample_id.tolist(), len(g), exclude=i))
    np.testing.assert_array_equal(witnesses['metric_scales_before'][0], np.ones(x.shape[1]))
    for step in range(1, len(fit)):
        np.testing.assert_array_equal(witnesses['metric_scales_before'][step],
            witnesses['metric_scales_before'][step-1]-settings.learning_rate*witnesses['metric_gradients'][step-1])
    np.testing.assert_array_equal(scale, witnesses['metric_scales_before'][-1]-settings.learning_rate*witnesses['metric_gradients'][-1])
    # Deterministic local random generator must leave NumPy global RNG untouched.
    before = np.random.get_state()
    other, proof = metric_epoch(x, y, fit.sample_id.tolist(), settings, lambda _k, _m: None)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1]); assert before[2:] == after[2:]
    np.testing.assert_array_equal(other, scale)
    np.testing.assert_array_equal(proof['metric_order'], witnesses['metric_order'])


@pytest.mark.parametrize('arm,epochs', [('KNN_FIXED', 0), ('DNNR_FIXED', 0), ('DNNR_LEARNED', 1), ('DNNR_LEARNED', 0)])
def test_saved_array_cold_independent_and_witness_audit(tmp_path, arm, epochs):
    fitting = sample(60); query = sample(13, 57322, 200)
    y = 30+fitting.air_volume.to_numpy()**2+fitting.hot_air_press.to_numpy()
    model = Regressor(arm).fit(fitting, y, metric_epochs=epochs)
    path = tmp_path / 'model.npz'; sha = model.save(path)
    cold = Regressor.load(path, sha)
    np.testing.assert_array_equal(model.predict(query), cold.predict(query))
    result = verify_saved(path, sha, fitting, y, query, model.predict(query))
    assert result['status'] == 'passed' and result['new_estimator_fits'] == 0
    assert result['numerical_audit_metric_solutions'] == len(fitting)*epochs
    assert result['numerical_audit_derivative_solutions'] == len(fitting)*(arm != 'KNN_FIXED')
    with pytest.raises(FileExistsError): model.save(path)
    with pytest.raises(ValueError): Regressor.load(path, '0'*64)
    with pytest.raises(ValueError): Regressor.load(path, None)
    with pytest.raises(ValueError): cold.fit(fitting, y)


@pytest.mark.parametrize('defect', ['label', 'statistics', 'derivative', 'gradient'])
def test_external_hash_does_not_replace_independent_training_witness_audit(tmp_path, defect):
    fitting = sample(60); query = sample(13, 57322, 200)
    y = 30+fitting.air_volume.to_numpy()**2+fitting.hot_air_press.to_numpy()
    model = Regressor('DNNR_LEARNED').fit(fitting, y, metric_epochs=1)
    path = tmp_path / 'original.npz'; sha = model.save(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        arrays = {k: data[k].copy() for k in data.files if k != 'metadata'}
    if defect == 'statistics': metadata['encoder']['means'][0] += .1
    else:
        key = {'label': 'y', 'derivative': 'derivatives', 'gradient': 'metric_gradients'}[defect]
        arrays[key].flat[0] += .1
        metadata['array_digests'][key] = digest(arrays[key].tolist())
    corrupt = tmp_path / 'corrupt.npz'
    np.savez(corrupt, metadata=np.array(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError):
        verify_saved(corrupt, file_hash(corrupt), fitting, y, query, model.predict(query))


def test_pair_epoch_zero_tie_uses_group_safe_holdout_and_exact_matched_refits(tmp_path):
    frame = sample(100)
    # One exact duplicate group with distinct IDs must stay on one inner side.
    frame.loc[1, list(FEATURES)+['spout_no']] = frame.loc[0, list(FEATURES)+['spout_no']].to_numpy()
    y = 30+2*frame.air_volume.to_numpy()-frame.hot_air_press.to_numpy()
    events = []
    pair = fit_pair(frame, y, 'tap_iron', observer=lambda k, m: events.append((k, m)))
    receipt = pair.receipt
    assert receipt['selected_metric_epochs'] == 0
    assert receipt['inner_folds'][0] == receipt['inner_folds'][1]
    assert not set(receipt['fitting_ids']) & set(receipt['calibration_ids'])
    assert len([e for e in events if e[0] == 'estimator']) == 6
    assert len([e for e in events if e[0] == 'derivative_bank']) == 4
    assert len([e for e in events if e[0] == 'metric_epoch']) == 1
    assert receipt['metric_updates'] == len(receipt['fitting_ids'])
    for key, value in pair.outer['DNNR_FIXED'].arrays_.items():
        np.testing.assert_array_equal(value, pair.outer['DNNR_LEARNED'].arrays_[key])
    hashes = pair.save(tmp_path / 'pair')
    assert len(hashes) == 7
    with pytest.raises(FileExistsError): pair.save(tmp_path / 'pair')


def test_complete_pair_cold_audit_refuses_changed_selection_and_budget_receipt(tmp_path, monkeypatch):
    frame = sample(100); query = sample(13, 57322, 200)
    y = 30+frame.air_volume.to_numpy()**2+frame.hot_air_press.to_numpy()**2
    pair = fit_pair(frame, y, 'tap_time_len')
    hashes = pair.save(tmp_path/'pair')
    predictions = {arm: m.predict(query) for arm, m in pair.outer.items()}
    def no_fitting(*_args, **_kwargs): raise AssertionError('Auditor attempted fitting')
    monkeypatch.setattr(Regressor, 'fit', no_fitting)
    monkeypatch.setattr(Encoder, 'fit', no_fitting)
    audit = verify_pair(tmp_path/'pair', hashes['receipt.json'], frame, y, query, predictions)
    assert audit['status'] == 'passed' and audit['new_estimator_fits'] == 0
    receipt_path = tmp_path/'pair'/'receipt.json'
    payload = json.loads(receipt_path.read_text())
    payload['receipt']['metric_updates'] += 1
    changed = tmp_path/'changed'; changed.mkdir()
    for name in payload['model_hashes']:
        (changed/name).write_bytes((tmp_path/'pair'/name).read_bytes())
    (changed/'receipt.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='accounting'):
        verify_pair(changed, file_hash(changed/'receipt.json'), frame, y, query, predictions)
    payload['receipt']['metric_updates'] -= 1
    payload['receipt']['selected_metric_epochs'] = 1-payload['receipt']['selected_metric_epochs']
    (changed/'receipt.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='selected'):
        verify_pair(changed, file_hash(changed/'receipt.json'), frame, y, query, predictions)


def test_pair_calibration_reselection_is_separate_from_fresh_outer_metric():
    frame = sample(100); y = 30+frame.air_volume.to_numpy()**2+frame.hot_air_press.to_numpy()**2
    pair = fit_pair(frame, y, 'tap_time_len')
    receipt = pair.receipt
    mae = receipt['calibration_maes']
    selected = int(mae['DNNR_LEARNED'] < mae['DNNR_FIXED'])
    assert receipt['selected_metric_epochs'] == selected
    assert pair.outer['DNNR_LEARNED'].metric_epochs_ == selected
    assert receipt['metric_epoch_runs'] == 1+selected
    assert receipt['metric_updates'] == len(receipt['fitting_ids'])+selected*len(frame)
    assert len(pair.inner['DNNR_LEARNED'].arrays_['x']) == len(receipt['fitting_ids'])
    assert len(pair.outer['DNNR_LEARNED'].arrays_['x']) == len(frame)
    query = sample(13, 57322, 200)
    for arm in ARMS: assert np.isfinite(pair.outer[arm].predict(query)).all()


@pytest.mark.parametrize('defect', ['targets', 'extra', 'duplicate_id', 'nonfinite', 'fractional_spout'])
def test_labeled_queries_or_invalid_training_schema_cannot_enter_encoder(defect):
    frame = sample(30)
    if defect == 'targets': frame['tap_iron'] = 1
    elif defect == 'extra': frame['some_label'] = 1
    elif defect == 'duplicate_id': frame.loc[1, 'sample_id'] = frame.loc[0, 'sample_id']
    elif defect == 'nonfinite': frame.loc[0, FEATURES[0]] = np.nan
    else: frame['spout_no'] = 1.5
    with pytest.raises(ValueError): Encoder().fit(frame)


def test_failed_fit_is_consumed_and_query_training_identity_overlap_refused():
    frame = sample(30); y = np.arange(30, dtype=float); model = Regressor('DNNR_LEARNED')
    events = []
    def deny(kind, metadata):
        events.append(kind)
        if kind == 'metric_epoch': raise RuntimeError('budget exhausted')
    with pytest.raises(RuntimeError): model.fit(frame, y, metric_epochs=1, observer=deny)
    assert events == ['estimator', 'metric_epoch']
    with pytest.raises(ValueError): model.fit(frame, y, metric_epochs=1)
    with pytest.raises(ValueError): model.predict(frame)
    good = Regressor('DNNR_FIXED').fit(frame, y)
    with pytest.raises(ValueError, match='overlap'): good.predict(frame)


def test_invalid_settings_epochs_or_arm_never_silently_change_protocol():
    for change in ({'neighbors': 0}, {'random_seed': -1}, {'learning_rate': np.nan}, {'epsilon': 0}):
        with pytest.raises(ValueError): replace(Settings(), **change)
    frame = sample(30); y = np.arange(30, dtype=float)
    with pytest.raises(ValueError): Regressor('DNNR_FIXED').fit(frame, y, metric_epochs=1)
    with pytest.raises(ValueError): Regressor('DNNR_LEARNED').fit(frame, y, metric_epochs=2)
    with pytest.raises(ValueError): Regressor('residual_corrector')
