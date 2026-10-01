"""Scientific contrasts, exact update prefixes and executable cold artifacts."""
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from test_round2_v12 import sample
from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.component_regularization_run import RECIPE
from bf_tap_r2.data import TARGETS
from bf_tap_r2.ema_average_span import task_frames
from bf_tap_r2.ema_reference_artifacts import sha
from bf_tap_r2.ema_reference_ledger import binding_sources, reference_bindings
import bf_tap_r2.ema_retraining_validation as module
from bf_tap_r2.v7_periodic import digest

WORK = Path(module.__file__).parents[2]
MECHANISMS = dict(ema_beta=.99, sam_rho=.05, sam_epsilon=1e-12)


@pytest.mark.parametrize('phase', module.PHASES)
def test_phase_freezes_independent_scope_and_original_scientific_recipe(phase):
    spec = json.loads((WORK/module.SPEC.format(phase=phase)).read_text())
    original = yaml.safe_load((WORK/'configs/strong_component_regularization/SPEC.yaml').read_text())
    module.validate_scope(spec, original)
    for key, value in [('replacement_weight', .5), ('optimizer_runs', spec['optimizer_runs']+1),
            ('maximum_runtime_seconds', 600), ('training_seeds', [42, 1042])]:
        changed = copy.deepcopy(spec); changed[key] = value
        with pytest.raises(ValueError, match='scope'): module.validate_scope(changed, original)
    changed = copy.deepcopy(spec); changed['training']['inner_seed'] = 1042
    with pytest.raises(ValueError, match='scope'): module.validate_scope(changed, original)


def test_selected_update_budget_excludes_patience_tail():
    trace = dict(selected_epoch=2, fit_rows=513,
        history=[dict(epoch=i, updates=3) for i in range(1, 7)])
    assert module.selected_updates(trace, 256) == 6
    trace['history'][1]['updates'] = 2
    with pytest.raises(ValueError): module.selected_updates(trace, 256)


def test_exact_whole_epoch_refit_is_bitwise_original_ema(tmp_path):
    frame, settings = sample(); y = frame[['tap_time_len']].to_numpy()
    settings = dict(settings, batch_size=31)
    original = ComponentRegressor(RECIPE, settings, 'EMA', MECHANISMS)
    original._initialize(frame, y); original._train(frame, y, 2)
    model = module.StepMatchedRegressor(RECIPE, settings, 'EMA', MECHANISMS, tmp_path)
    model._initialize(frame, y); trace = model.train_updates(frame, y, 8)
    assert trace['updates'] == 8
    for key, value in original.model_.state_dict().items():
        assert torch.equal(value, model.model_.state_dict()[key])
    np.testing.assert_array_equal(model.predict(frame.drop(columns=list(TARGETS))),
        original.predict(frame.drop(columns=list(TARGETS))))


def test_partial_epoch_stops_exactly_and_metadata_rejects_extra_update(tmp_path, monkeypatch):
    frame, settings = sample(); y = frame[['tap_time_len']].to_numpy()
    settings = dict(settings, batch_size=31)
    model = module.StepMatchedRegressor(RECIPE, settings, 'EMA', MECHANISMS, tmp_path)
    model._initialize(frame, y)
    counts = []; original = model.optimizer_.step
    def counted(*a, **kw):
        counts.append(1); return original(*a, **kw)
    monkeypatch.setattr(model.optimizer_, 'step', counted)
    trace = model.train_updates(frame, y, 6)
    assert len(counts) == 6 and [r['updates'] for r in trace['history']] == [4, 2]
    assert trace['history'][-1]['examples'] == 62
    broken = copy.deepcopy(trace); broken['history'][-1]['updates'] = 3
    with pytest.raises(ValueError): module.validate_update_trace(broken, 6, len(frame), 31)
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError): model.train_updates(frame, y, invalid)


def test_paired_means_enter_final_affine_combination_without_selecting_seed():
    actual = np.array([[500., 100.], [600., 120.], [550., 130.], [580., 140.], [590., 150.]])
    ema = actual[:, 1]+1
    parts = {f'{a}_{i}': actual[:, 1]+(2 if a == 'BASE' else 1)+j
        for j, i in enumerate(module.INITIALIZATIONS) for a in ('BASE', 'EMA')}
    q75 = actual[:, 1]+3
    cols, report = module.summarize_initialization(actual, parts, q75, ema, actual[:, 0], np.arange(5), np.array([1, 2, 1, 2, 1]))
    np.testing.assert_array_equal(cols['EMA_42'], q75)
    np.testing.assert_array_equal(cols['EMA_MEAN3'], np.mean([cols[f'EMA_{i}'] for i in module.INITIALIZATIONS], axis=0))
    assert all(r['fusion_gain'] > 0 for r in report['ema_minus_paired_base'].values())
    assert report['equal_mean_ema_minus_equal_mean_base'] > 0
    del parts['BASE_2042']
    with pytest.raises(ValueError, match='Complete'): module.summarize_initialization(actual, parts, q75, ema, actual[:, 0], np.arange(5), np.ones(5))


def synthetic_manifest(tmp_path, phase):
    frame, settings = sample(); settings = dict(settings, max_epochs=3)
    fv = np.arange(len(frame)) % 5
    training, query = task_frames(frame, fv, 0)
    fitting, calibration, hashes = module.partitions(training, query, settings)
    sources = binding_sources(reference_bindings())
    for name in ('ema_retraining_validation', 'component_regularization', 'v12_joint', 'v3_6_networks'):
        p = Path(module.__file__).with_name(name+'.py'); sources[str(p)] = sha(p)
    spec = dict(phase=phase, training=settings, mechanisms=MECHANISMS,
        training_seeds=[42, 1042, 2042] if phase == 'initialization' else [42], split_seeds=[42],
        max_worker_rss_mib=1536, cold_predict_atol=.0005, runtime_versions={}, old_development='fake')
    manifest = dict(spec=spec, main_root=str(tmp_path), output=str(tmp_path), files=sources,
        model_sources=sources, partitions={'s42-f0': hashes}, fold_hashes={'42': digest(fv.tolist())})
    manifest['identity'] = digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return manifest, frame, fv, training, query, fitting, calibration


@pytest.mark.parametrize('phase', module.PHASES)
def test_native_worker_and_new_process_cold_artifact_path(tmp_path, monkeypatch, phase):
    manifest, frame, fv, training, query, fitting, calibration = synthetic_manifest(tmp_path, phase)
    monkeypatch.setattr(module, 'context', lambda *a, **kw: (manifest, frame))
    monkeypatch.setattr(module, 'load_v5_spec', lambda *a: {})
    monkeypatch.setattr(module, 'fold_vector', lambda *a, **kw: fv)
    old = dict(q75=np.full(len(query), 100.), ema=np.full(len(query), 100.), iron=np.full(len(query), 500.))
    monkeypatch.setattr(module, 'old_cache', lambda *a: old)
    if phase == 'steps':
        cache = tmp_path/'cached'; cache.mkdir()
        model = ComponentRegressor(RECIPE, manifest['spec']['training'], 'EMA', MECHANISMS, cache)
        model._initialize(fitting, fitting[['tap_time_len']].to_numpy())
        epoch = model._train(fitting, fitting[['tap_time_len']].to_numpy(), 3, (calibration, calibration[['tap_time_len']].to_numpy()))
        selector = ComponentRegressor.load(cache/'selection.pt')
        model._initialize(training, training[['tap_time_len']].to_numpy()); model._train(training, training[['tap_time_len']].to_numpy(), epoch)
        refit = ComponentRegressor.load(cache/'refit.pt')
        old['ema'] = refit.predict(query)[:, 0]
        def cached(manifest, directory, *a):
            witnesses = {}
            for name, m, q, ids in [('selection', selector, calibration.drop(columns=list(TARGETS)), fitting.sample_id.tolist()),
                                  ('refit', refit, query, training.sample_id.tolist())]:
                witnesses[name] = module.save_prediction_witness(m, q, m.predict(q), directory, manifest, 42, 0, 'EMA_42', name, ids, m.saved['trace'])
            return selector, refit, refit.predict(query)[:, 0], witnesses, cache
        monkeypatch.setattr(module, 'original_states', cached)
    init, arm = (1042, 'EMA') if phase == 'initialization' else (42, 'EMA_UPDATES')
    module.worker(tmp_path, 42, 0, init, arm)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    child = subprocess.run([sys.executable, '-m', 'bf_tap_r2.ema_retraining_validation', '--cold', '--output', str(tmp_path),
        '--seed', '42', '--fold', '0', '--init', str(init), '--arm', arm], env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout+child.stderr
    receipt = json.loads((tmp_path/module.unit_name(42, 0, init, arm)/'cold-complete.json').read_text())
    assert receipt['cold_states'] == (2 if phase == 'initialization' else 3)
    assert receipt['optimizer_runs'] == (2 if phase == 'initialization' else 1)


def test_real_module_main_dispatches_importable_subclass(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module, 'worker', lambda *args: calls.append((args, module.StepMatchedRegressor.__module__)))
    monkeypatch.setattr(sys, 'argv', ['ema_retraining_validation', '--worker', '--output', str(tmp_path),
        '--seed', '42', '--fold', '0', '--init', '42', '--arm', 'EMA_UPDATES'])
    runpy.run_module(module.__name__, run_name='__main__')
    assert calls[0][1] == 'bf_tap_r2.ema_retraining_validation'
