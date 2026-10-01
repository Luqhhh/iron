"""Matched application, strict phase gates and real cold CLI for cached states."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_round2_v12 import sample
from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.component_regularization_run import RECIPE
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.ema_reference_artifacts import save_witness
import bf_tap_r2.ema_component_calibration as module


def inputs(n=120):
    f = pd.DataFrame({k: np.arange(n, dtype=float)+i for i, k in enumerate(FEATURES)})
    f['sample_id'] = [f'id{i}' for i in range(n)]; f['spout_no'] = np.arange(n)%2+1
    return f


def test_correction_is_learned_on_component_and_enters_fusion_at_original_weight():
    f = inputs(); prediction = np.full(len(f), 100.); truth = prediction+2
    q75 = np.full(len(f), 140.); old = np.full(len(f), 110.)
    settings = dict(cv_seed=961048, minimum_bin_rows=30, gamma_grid=[0, .25, .5, 1])
    columns, fitted = module.component_heads(f, truth, prediction, f, prediction, f, q75, old, settings)
    np.testing.assert_array_equal(columns['UNCORRECTED'], q75+.75*(prediction-old))
    for family in module.ORDER:
        assert fitted[family]['gamma'] == 1
        np.testing.assert_array_equal(columns[family], columns['UNCORRECTED']+1.5)
    columns, fitted = module.component_heads(f, prediction, prediction, f, prediction, f, q75, old, settings)
    for family in module.ORDER:
        assert fitted[family]['gamma'] == 0
        np.testing.assert_array_equal(columns[family], columns['UNCORRECTED'])


def test_outer_labels_are_rejected_before_any_calibration_learning(monkeypatch):
    f = inputs(); query = f.assign(tap_time_len=0.); p = np.full(len(f), 100.)
    def forbidden(*a, **kw): raise AssertionError('Learning started before label guard')
    monkeypatch.setattr(module, 'learn_calibration', forbidden)
    with pytest.raises(ValueError, match='label-free'):
        module.component_heads(f, p, p, query, p, f, p, p, {})


def test_confirmation_requires_both_complete_splits_and_gain_over_matched_control():
    gains = {'GLOBAL': {'42': .03, '3407': .02}, 'PRESSURE': {'42': .02, '3407': .01}}
    paired = {'GLOBAL': {'42': .01, '3407': -.001}, 'PRESSURE': {'42': .01, '3407': .01}}
    assert module.choose_candidate(gains, paired) == 'PRESSURE'
    paired['PRESSURE']['3407'] = 0
    assert module.choose_candidate(gains, paired) is None
    tied = {k: {'42': .01, '3407': .01} for k in module.ORDER}
    assert module.choose_candidate(tied, tied) == 'GLOBAL'
    tied['PRESSURE'].pop('3407')
    with pytest.raises(ValueError, match='within-split'): module.choose_candidate(tied, paired)


@pytest.mark.parametrize('key,value', [('optimizer_runs', 1), ('correction_fit_calls', 40),
    ('split_seeds', [42]), ('maximum_runtime_seconds', 600), ('replacement_weight', .5), ('previous_terminals', [])])
def test_scope_preserves_zero_base_fit_complete_pool_and_no_time_limit(key, value):
    spec = json.loads((module.WORK/module.SPEC).read_text()); module.validate_scope(spec)
    spec[key] = value
    with pytest.raises(ValueError, match='scope'): module.validate_scope(spec)


def test_missing_or_failed_previous_phase_blocks_freeze(tmp_path):
    with pytest.raises(FileNotFoundError): module.previous_success({'previous_terminals': [str(tmp_path/'missing.json')]})
    p = tmp_path/'terminal.json'; p.write_text(json.dumps(dict(status='failed', actual_exit_codes=[0, 1])))
    with pytest.raises(ValueError, match='actual terminals'): module.previous_success({'previous_terminals': [str(p)]})


def test_copied_manifest_cannot_write_predictions_outside_frozen_private_run(tmp_path, monkeypatch):
    spec = json.loads((module.WORK/module.SPEC).read_text())
    manifest = dict(spec=spec, output=str(Path(spec['main_root'])/'local/runs/frozen-example'), files={})
    manifest['identity'] = module.digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs): raise AssertionError('Official data read before output guard')
    monkeypatch.setattr(module, 'load_v5_training_frame', forbidden)
    with pytest.raises(ValueError, match='run location'): module.run(tmp_path)


def test_no_model_training_guard_blocks_optimizer_and_training_initializer():
    import torch
    frame, settings = sample(); y = frame[['tap_time_len']].to_numpy()
    model = ComponentRegressor(RECIPE, settings, 'EMA', dict(ema_beta=.99))
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with module.no_model_training():
        with pytest.raises(ValueError, match='training'): model._initialize(frame, y)
        with pytest.raises(ValueError, match='training'): optimizer.step()


def test_saved_synthetic_component_has_real_new_process_cold_path(tmp_path):
    frame, settings = sample(); training = frame.iloc[:90].copy(); query = frame.iloc[90:].drop(columns=list(TARGETS))
    model = ComponentRegressor(RECIPE, settings, 'EMA', dict(ema_beta=.99), tmp_path)
    model._initialize(training, training[['tap_time_len']].to_numpy()); model._train(training, training[['tap_time_len']].to_numpy(), 1)
    model = ComponentRegressor.load(tmp_path/'refit.pt'); observed = model.predict(query)
    witness = tmp_path/'witness'
    receipt = save_witness(model, query, observed, witness,
        identity=dict(source_directory=str(tmp_path), split_seed=42, fold=0, trial_id='SYNTHETIC_EMA', fit_call_id='refit'),
        training_ids=training.sample_id.tolist(), source_hashes=module.source_files(),
        full_batch_atol=0., row_atol=.0005, fit_metadata=model.saved['trace'])
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-m', module.__name__, '--cold', '--directory', str(witness),
        '--receipt-sha256', receipt], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    cold = json.loads((witness/'cold-audit.json').read_text()); rss = json.loads((witness/'cold-rss.json').read_text())
    assert cold['status'] == 'passed' and cold['differences']['full'] == 0
    assert rss['status'] == 'passed' and rss['new_model_fits'] == rss['optimizer_runs'] == 0
