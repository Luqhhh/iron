import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import ema_independent_batches as run
from bf_tap_r2.ema_independent_batches_model import (
    IndependentBatchRegressor, member_loss, member_orders,
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


def test_member_permutations_have_complete_coverage_and_reproducible_rng():
    rng = np.random.default_rng(42)
    orders = [member_orders(rng, 551, 16) for _ in range(3)]
    for a in orders:
        assert a.shape == (551, 16)
        np.testing.assert_array_equal(np.sort(a, axis=0), np.tile(np.arange(551)[:, None], (1, 16)))
        assert len({tuple(v) for v in a.T}) == 16
    assert not np.array_equal(orders[0], orders[1])
    rng2 = np.random.default_rng(42)
    for a in orders:
        np.testing.assert_array_equal(a, member_orders(rng2, 551, 16))
    with pytest.raises(ValueError):
        member_orders(rng, 0, 16)


def test_row_head_target_alignment_gradient_and_tabm_three_dimensional_input():
    import torch
    from bf_tap_r2.v12_joint import make_network
    from bf_tap_r2.component_regularization_run import RECIPE
    torch.set_num_threads(1)
    pred = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2).requires_grad_()
    target = torch.flip(pred.detach(), [1]) + 1
    loss = member_loss(pred, target)
    expected = math.fsum((float(pred.detach()[i, j, k])-float(target[i, j, k]))**2
                         for i in range(2) for j in range(3) for k in range(2))/12
    assert float(loss.detach()) == expected
    loss.backward()
    torch.testing.assert_close(pred.grad, 2*(pred.detach()-target)/12, rtol=0, atol=5e-16)
    with pytest.raises(ValueError, match='Aligned'):
        member_loss(pred, target[:, 0])
    spec = run.read(run.ROOT/'configs/ema_independent_batches/SPEC.json')
    torch.manual_seed(9)
    model = make_network(RECIPE, spec['training'], 2, 1).eval()
    x = torch.randn(31, len(FEATURES)); cat = torch.zeros((31, 1), dtype=torch.long)
    idx = member_orders(np.random.default_rng(9), len(x), 16)[:7]
    with torch.no_grad():
        standard = model(x, cat)
        separate = model(x[idx], cat[idx])
    expected = torch.stack([standard[idx[:, j], j] for j in range(16)], dim=1)
    torch.testing.assert_close(separate, expected, rtol=1e-5, atol=1e-6)


def test_fixed_candidates_and_no_partial_seed_admission():
    old = np.array([[8., 10.], [10., 12.], [12., 14.]])
    new = old + [1., -2.]
    base, candidates = run.columns([10., 15.], [9., 11.], old, new)
    np.testing.assert_array_equal(base, [11., 16.])
    np.testing.assert_array_equal(candidates['IBATCH_A100'], [12., 14.])
    np.testing.assert_allclose(candidates['IBATCH_A20'], [11.2, 15.6], rtol=0, atol=1e-14)
    assert run.choose({'IBATCH_A100': {'42': 1., '3407': -1.},
                       'IBATCH_A20': {'42': .1, '3407': .2}}) == 'IBATCH_A20'
    assert run.choose({'IBATCH_A100': {'42': .1, '3407': .2},
                       'IBATCH_A20': {'42': .1, '3407': .2}}) == 'IBATCH_A100'
    with pytest.raises(ValueError):
        run.choose({'IBATCH_A100': {'42': 1.}})
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


def test_synthetic_full_shape_selection_refit_and_cold_native_audit(tmp_path):
    """The only engineering fit: 2204 synthetic rows, two optimizer constructors."""
    import torch
    from bf_tap_r2.component_regularization_run import RECIPE
    from bf_tap_r2.component_regularization import ComponentRegressor
    spec = run.read(run.ROOT/'configs/ema_independent_batches/SPEC.json')
    rng = np.random.default_rng(964504)
    frame = pd.DataFrame(rng.normal(size=(2204, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 2 + 1
    frame['tap_time_len'] = 100 + 10*frame[FEATURES[0]] + rng.normal(size=len(frame))
    frame['tap_iron'] = 1000 + rng.normal(size=len(frame))
    query = frame.iloc[:37].drop(columns=list(TARGETS)).copy()
    query['sample_id'] = [f'query-{i}' for i in range(len(query))]
    settings = dict(spec['training'], max_epochs=2, patience=25, batch_order='independent_without_replacement')
    identity = {'source_directory': str(tmp_path), 'split_seed': -1, 'trial_id': 'synthetic_full_shape'}
    with run.optimizer_ledger(tmp_path, identity) as counts:
        model = IndependentBatchRegressor(RECIPE, settings, 'EMA', spec['mechanisms'], tmp_path)
        model.fit(frame.drop(columns=list(TARGETS)), frame[['tap_time_len']].to_numpy())
    run.verify_counts(counts, model.traces)
    assert counts['constructors'] == ['selection', 'refit']
    for role in counts['constructors']:
        t = model.traces[role]
        assert len(t['member_order_digests']) == t['stopped_epoch']
        assert t['samples_per_head_per_epoch'] == t['fit_rows']
    prediction = model.predict(query)[:, 0]
    receipt = run.audit_models(tmp_path, frame, query, {'refit': prediction}, settings, spec['mechanisms'])
    assert receipt['status'] == 'passed' and receipt['states'] == 2
    assert receipt['peak_rss_mib'] <= 1536
    cold = ComponentRegressor.load(tmp_path/'refit.pt')
    np.testing.assert_array_equal(cold.predict(query)[:, 0], prediction)
    with pytest.raises(ValueError):
        IndependentBatchRegressor(RECIPE, spec['training'], 'EMA', spec['mechanisms'])
