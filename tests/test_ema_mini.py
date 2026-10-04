import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import ema_mini as run
from bf_tap_r2.ema_mini_model import (
    MiniEMARegressor, make_mini_network,
)
from bf_tap_r2.data import FEATURES, TARGETS


def test_legacy_manifest_paths_resolve_against_original_root(tmp_path):
    files = {}
    run.merge_legacy_files(files, {'local/checks.json': 'a'}, tmp_path)
    name = str(tmp_path/'local/checks.json')
    assert files == {name: 'a'}
    run.merge_legacy_files(files, {name: 'a'}, tmp_path/'other-worktree')
    assert files == {name: 'a'}
    with pytest.raises(ValueError, match='Conflicting'):
        run.merge_legacy_files(files, {'local/checks.json': 'b'}, tmp_path)


def test_native_mini_shares_hidden_layers_and_matches_direct_author_factory():
    import torch
    from rtdl_num_embeddings import PeriodicEmbeddings
    from tabm import TabM
    from bf_tap_r2.v12_joint import make_network
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.component_regularization_run import RECIPE
    torch.set_num_threads(1)
    spec = run.read(run.ROOT/'configs/ema_mini/SPEC.json')
    settings = dict(spec['training'], arch_type='tabm-mini')
    torch.manual_seed(17)
    mini = make_mini_network(RECIPE, settings, 2, 1).eval()
    torch.manual_seed(17)
    direct = TabM.make(n_num_features=len(FEATURES), cat_cardinalities=[2], d_out=1,
        k=16, n_blocks=2, d_block=256, dropout=.1, arch_type='tabm-mini',
        num_embeddings=PeriodicEmbeddings(len(FEATURES), d_embedding=16,
            n_frequencies=16, frequency_init_scale=.01, lite=True)).eval()
    full = make_network(RECIPE, spec['training'], 2, 1).eval()
    assert sum(p.numel() for p in mini.parameters()) < sum(p.numel() for p in full.parameters())
    assert set(mini.state_dict()) != set(full.state_dict())
    assert MiniEMARegressor._train is ComponentRegressor._train
    x = torch.linspace(-1, 1, 31*len(FEATURES)).reshape(31, len(FEATURES))
    cat = torch.arange(31).remainder(2).reshape(-1, 1)
    with torch.no_grad():
        a, b = mini(x, cat), direct(x, cat)
    assert a.shape == (31,16,1)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        full.load_state_dict(mini.state_dict(), strict=True)


def test_mini_refuses_undeclared_architecture_and_batch_change():
    from bf_tap_r2.component_regularization_run import RECIPE
    spec = run.read(run.ROOT/'configs/ema_mini/SPEC.json')
    with pytest.raises(ValueError):
        MiniEMARegressor(RECIPE, spec['training'], 'EMA', spec['mechanisms'])
    with pytest.raises(ValueError):
        MiniEMARegressor(RECIPE, dict(spec['training'], arch_type='tabm-mini',
            batch_order='independent_without_replacement'), 'EMA', spec['mechanisms'])
    with pytest.raises(ValueError):
        make_mini_network(RECIPE, dict(spec['training'], arch_type='tabm'), 2, 1)


def test_fixed_candidates_and_no_partial_seed_admission():
    old = np.array([[8., 10.], [10., 12.], [12., 14.]])
    new = old + [1., -2.]
    base, candidates = run.columns([10., 15.], [9., 11.], old, new)
    np.testing.assert_array_equal(base, [11., 16.])
    np.testing.assert_array_equal(candidates['MINI_A100'], [12., 14.])
    np.testing.assert_allclose(candidates['MINI_A20'], [11.2, 15.6], rtol=0, atol=1e-14)
    assert run.choose({'MINI_A100': {'42': 1., '3407': -1.},
                       'MINI_A20': {'42': .1, '3407': .2}}) == 'MINI_A20'
    assert run.choose({'MINI_A100': {'42': .1, '3407': .2},
                       'MINI_A20': {'42': .1, '3407': .2}}) == 'MINI_A100'
    with pytest.raises(ValueError):
        run.choose({'MINI_A100': {'42': 1.}})
    with pytest.raises(ValueError):
        run.columns([10., 15.], [9., 11.], old, new[:2])
    with pytest.raises(ValueError, match='clipping'):
        run.columns([10., 15.], [9., 11.], old, new-100.)


def test_task_roster_failure_stops_and_no_overwrite(tmp_path, monkeypatch):
    tasks = run.tasks()
    assert len(tasks) == 72
    assert sum(x[0] == 'worker' for x in tasks) == 30
    assert sum(x[0] == 'cold' for x in tasks) == 30
    assert sum(x[0] == 'reuse' for x in tasks) == 10
    assert [x[0] for x in tasks[-2:]] == ['report', 'audit']
    monkeypatch.setattr(run, 'RUN', tmp_path)
    with pytest.raises(FileExistsError, match='consumed'):
        run.prepare('missing-checks.json')
    with pytest.raises(FileNotFoundError):
        run.verify_model_events()
    run.write(tmp_path/'manifest.json', {'files': {}})
    launched = []
    def launch(*args, **kwargs):
        launched.append(args[0]); return SimpleNamespace(pid=1001, returncode=None)
    monkeypatch.setattr(run.subprocess, 'Popen', launch)
    monkeypatch.setattr(run.os, 'wait4', lambda *_: (1001, 3 << 8, SimpleNamespace(ru_maxrss=800*1024)))
    with pytest.raises(RuntimeError, match='child failed'):
        run.controller()
    assert len(launched) == 1
    terminal = run.read(tmp_path/'execution/terminal.json')
    assert terminal['status'] == 'failed' and terminal['automatic_retry'] is False
    assert not (tmp_path/'report.json').exists()


def test_outer_queries_never_include_labels_or_training_ids():
    from bf_tap_r2.ema_nested_residual import split_training
    rows = 20
    frame = pd.DataFrame({x: np.arange(rows, dtype=float) for x in FEATURES})
    frame['sample_id'] = [str(i) for i in range(rows)]
    frame['spout_no'] = 1
    for t in TARGETS:
        frame[t] = np.arange(rows)
    folds = np.arange(rows) % 5
    a, b = split_training(frame, folds, 0)
    changed = frame.copy(); changed.loc[folds == 0, list(TARGETS)] = 10**12
    c, d = split_training(changed, folds, 0)
    pd.testing.assert_frame_equal(a, c); pd.testing.assert_frame_equal(b, d)
    plan = {'training_ids': a.sample_id.tolist(), 'query_ids': b.sample_id.tolist()}
    run.validate_frames(a, b, plan)
    with pytest.raises(ValueError):
        run.validate_frames(a, frame.loc[folds == 0], plan)


@pytest.mark.parametrize('field,value', [('weights', {'MINI_A100':.5,'MINI_A20':.2}),
    ('split_seeds',[42]), ('architecture','tabm'), ('time_budget_seconds',3600),
    ('automatic_retries',True)])
def test_freeze_rejects_changed_scientific_scope(field, value):
    spec=run.read(run.ROOT/'configs/ema_mini/SPEC.json')
    run.validate_spec(spec,spec['training'],spec['mechanisms'])
    changed=dict(spec);changed[field]=value
    with pytest.raises(ValueError):
        run.validate_spec(changed,spec['training'],spec['mechanisms'])


def test_synthetic_full_shape_selection_refit_and_independent_cold(tmp_path):
    """One 2204-row synthetic estimator; cold audit runs in another process."""
    import os
    import pickle
    import subprocess
    import sys
    from bf_tap_r2.component_regularization_run import RECIPE
    spec = run.read(run.ROOT/'configs/ema_mini/SPEC.json')
    rng = np.random.default_rng(964506)
    frame = pd.DataFrame(rng.normal(size=(2204, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 2 + 1
    frame['tap_time_len'] = 100 + 10*frame[FEATURES[0]] + rng.normal(size=len(frame))
    frame['tap_iron'] = 1000 + rng.normal(size=len(frame))
    query = frame.iloc[:74].drop(columns=list(TARGETS)).copy()
    query['sample_id'] = [f'query-{i}' for i in range(len(query))]
    settings = dict(spec['training'], max_epochs=2, patience=25, arch_type='tabm-mini')
    identity = {'source_directory': str(tmp_path), 'split_seed': -1, 'trial_id': 'synthetic_full_shape'}
    with run.optimizer_ledger(tmp_path, identity) as counts:
        model = MiniEMARegressor(RECIPE, settings, 'EMA', spec['mechanisms'], tmp_path)
        model.fit(frame.drop(columns=list(TARGETS)), frame[['tap_time_len']].to_numpy())
    run.verify_counts(counts, model.traces)
    assert counts['constructors'] == ['selection', 'refit']
    prediction = model.predict(query)[:,0]
    for name, data in [('training.pkl',frame), ('query.pkl',query)]:
        with (tmp_path/name).open('xb') as stream:
            pickle.dump(data, stream, protocol=5)
    run.save_arrays(tmp_path/'predictions.npz', prediction=prediction)
    run.write(tmp_path/'settings.json', dict(settings=settings, mechanisms=spec['mechanisms'],
        warm_pid=os.getpid(), counts=counts))
    code = """
import json,os,pickle,sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from bf_tap_r2.ema_mini_audit import audit_models
p=Path(sys.argv[1]); settings=json.loads((p/'settings.json').read_text())
assert settings['warm_pid']!=os.getpid()
with (p/'training.pkl').open('rb') as f: training=pickle.load(f)
with (p/'query.pkl').open('rb') as f: query=pickle.load(f)
with np.load(p/'predictions.npz',allow_pickle=False) as a:
    receipt=audit_models(p,training,query,{'refit':a['prediction']},settings['settings'],settings['mechanisms'])
receipt['cold_pid']=os.getpid();receipt['warm_pid']=settings['warm_pid']
with (p/'cold.json').open('x') as f:json.dump(receipt,f,indent=2)
"""
    completed = subprocess.run([sys.executable,'-c',code,str(tmp_path)],
        env=os.environ.copy(), text=True, capture_output=True)
    run.write(tmp_path/'cold-terminal.json',dict(exit_code=completed.returncode,
        stdout=completed.stdout,stderr=completed.stderr))
    assert completed.returncode==0,completed.stderr
    receipt=run.read(tmp_path/'cold.json')
    assert receipt['status']=='passed' and receipt['states']==2
    assert receipt['peak_rss_mib']<=1536 and receipt['maximum_difference']<=.0005
