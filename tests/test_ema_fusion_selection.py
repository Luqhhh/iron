import numpy as np
import pytest
import torch
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

from test_round2_v12 import sample
from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.component_regularization_run import RECIPE
from bf_tap_r2.data import TARGETS
from bf_tap_r2.ema_fusion_selection import paired_selector, select_step, validate_partitions
from bf_tap_r2.q75_combination_review import combine, PROBES, REFERENCE_WEIGHTS, select_development

MECHANISMS = {'ema_beta': .99, 'sam_rho': .05, 'sam_epsilon': 1e-12}


def test_shared_component_policy_exactly_matches_original_selector():
    frame, settings = sample()
    fitting, calibration = frame.iloc[:80].copy(), frame.iloc[80:].copy()
    y = fitting[['tap_time_len']].to_numpy()
    query = calibration.drop(columns=list(TARGETS))
    actual = calibration.tap_time_len.to_numpy()
    original = ComponentRegressor(RECIPE, settings, 'EMA', MECHANISMS)
    original._initialize(fitting, y)
    epoch = original._train(fitting, y, settings['max_epochs'], (calibration, actual[:, None]))
    paired = ComponentRegressor(RECIPE, settings, 'EMA', MECHANISMS)
    paired._initialize(fitting, y)
    states, checkpoints, history = paired_selector(paired, fitting, y, query, actual, .25*actual)
    assert states['COMPONENT_MAE']['selected_epoch'] == epoch
    assert states['COMPONENT_MAE']['stopped_epoch'] == original.selection_stopped_epoch_
    for key, value in original.model_.state_dict().items():
        assert torch.equal(value, checkpoints['COMPONENT_MAE'][key])
    assert states['FUSION_MAE']['selected_epoch'] == epoch


def test_stopped_policy_never_reopens_and_ties_keep_first():
    state = dict(best=float('inf'), selected_epoch=0, stale=0, stopped_epoch=None)
    assert select_step(state, 2., 1, 2, 0.)
    assert not select_step(state, 2., 2, 2, 0.)
    assert not select_step(state, 2., 3, 2, 0.)
    assert state['stopped_epoch'] == 3
    assert not select_step(state, 0., 4, 2, 0.)
    assert state['selected_epoch'] == 1
    with pytest.raises(ValueError):
        select_step(state, float('nan'), 5, 2, 0.)


def test_component_and_fusion_can_prefer_opposite_checkpoints():
    actual = np.array([10., 20.])
    remainder = np.array([4., 8.])
    exact_component = actual.copy()
    complementary = (actual-remainder)/.75
    assert np.abs(actual-exact_component).mean() < np.abs(actual-complementary).mean()
    assert np.abs(actual-(remainder+.75*complementary)).mean() < np.abs(actual-(remainder+.75*exact_component)).mean()


def test_partition_guards_block_rows_groups_and_targets():
    frame, _ = sample()
    training = frame.iloc[:60]
    calibration = frame.iloc[60:80].drop(columns=list(TARGETS))
    query = frame.iloc[80:].drop(columns=list(TARGETS))
    validate_partitions(training, calibration, query)
    with pytest.raises(ValueError):
        validate_partitions(training, training.iloc[:1].drop(columns=list(TARGETS)), query)
    with pytest.raises(ValueError):
        validate_partitions(training, frame.iloc[60:80], query)
    duplicate = training.iloc[:1].drop(columns=list(TARGETS)).copy()
    duplicate['sample_id'] = 'other-id'
    with pytest.raises(ValueError, match='duplicate-group'):
        validate_partitions(training, duplicate, query)


def test_sparse_affine_directions_and_invalid_prediction_rejected():
    parts = dict(v36_time=np.array([10., 20.]), n_time=np.array([12., 18.]),
        v7_time=np.array([8., 24.]), ema=np.array([11., 21.]))
    baseline = combine(parts, REFERENCE_WEIGHTS)
    np.testing.assert_allclose(combine(parts, PROBES['V36_TO_N005'])-baseline,
        .05*(parts['n_time']-parts['v36_time']), atol=1e-14, rtol=0)
    with pytest.raises(ValueError):
        combine(parts, (.2, .3, 0., .5))
    with pytest.raises(ValueError, match='clipping'):
        combine(dict(parts, v7_time=np.array([1000., 1000.])), REFERENCE_WEIGHTS)


def test_development_selection_needs_both_seeds_and_keeps_frozen_tie_order():
    records = {n: {'42': {'gain': .01}, '3407': {'gain': .01}} for n in PROBES}
    assert select_development(records) == next(iter(PROBES))
    records[next(iter(PROBES))]['3407']['gain'] = -1.
    assert select_development(records) == list(PROBES)[1]
    for values in records.values():
        values['42']['gain'] = -1.
    assert select_development(records) is None


def test_synthetic_paired_worker_native_budget_and_independent_cold(tmp_path, monkeypatch):
    import bf_tap_r2.ema_fusion_selection as module
    from bf_tap_r2.ema_reference_ledger import binding_sources, reference_bindings
    from bf_tap_r2.ema_reference_artifacts import sha
    from bf_tap_r2.ema_average_span import task_frames
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    from bf_tap_r2.v7_periodic import digest
    frame, settings = sample()
    settings = dict(settings, max_epochs=3)
    fv = np.arange(len(frame)) % 5
    training, query = task_frames(frame, fv, 0)
    inner = np.asarray(group_safe_inner_folds(training, seed=27001)['fold'])
    fitting = training.loc[inner != 0].reset_index(drop=True)
    calibration = training.loc[inner == 0].reset_index(drop=True)
    partitions = {'s42-f0': {n: digest(d.sample_id.tolist()) for n, d in
        [('fitting', fitting), ('calibration', calibration), ('training', training), ('query', query)]}}
    import bf_tap_r2.component_regularization as original_module
    sources = binding_sources(reference_bindings())
    for name in ('component_regularization', 'v12_joint', 'v3_6_networks', 'ema_fusion_selection'):
        path = Path(module.__file__).with_name(name+'.py'); sources[str(path)] = sha(path)
    spec = dict(training=settings, mechanisms=MECHANISMS, split_seeds=[42], calibration_seed=27001,
        calibration_cache='fake', old_development='fake', max_worker_rss_mib=1536,
        cold_predict_atol=.0005, runtime_versions={})
    manifest = dict(spec=spec, main_root=str(tmp_path), workspace=str(Path(module.__file__).parents[2]),
        output=str(tmp_path), files=sources, model_sources=sources, partitions=partitions)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    config = tmp_path/'configs/strong_component_regularization'; config.mkdir(parents=True)
    (config/'SPEC.yaml').write_text(yaml.safe_dump({}))
    monkeypatch.setattr(module, 'context', lambda *a, **kw: (manifest, frame))
    monkeypatch.setattr(module, 'fold_vector', lambda *a, **kw: fv)
    monkeypatch.setattr(module, 'load_v5_spec', lambda *a: {})
    parent = np.full(len(calibration), 100.)
    monkeypatch.setattr(module, 'read_reference', lambda *a: (
        dict(v36_time=parent, n_time=parent, v7_time=parent), {'query_labels_received': False}))
    old = dict(q75=np.full(len(query), 100.), ema=np.full(len(query), 100.), iron=np.full(len(query), 500.))
    monkeypatch.setattr(module, 'old_cache', lambda *a: old)
    module.worker(tmp_path, 42, 0)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    child = subprocess.run([sys.executable, '-m', 'bf_tap_r2.ema_fusion_selection', '--cold',
        '--output', str(tmp_path), '--seed', '42', '--fold', '0'], env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout+child.stderr
    checked = json.loads((tmp_path/'s42-f0/cold-complete.json').read_text())
    assert checked['retained_states'] == 4 and checked['native_counts']['torch_optimizer'] == 3
