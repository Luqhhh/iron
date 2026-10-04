import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import ema_silu as run
from bf_tap_r2.ema_silu_model import (
    SiLUEMARegressor, make_silu_network,
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


def test_native_silu_matches_author_factory_and_paired_relu_initialization():
    import torch
    from rtdl_num_embeddings import PeriodicEmbeddings
    from tabm import TabM
    from bf_tap_r2.v12_joint import make_network
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.component_regularization_run import RECIPE
    torch.set_num_threads(1)
    spec = run.read(run.ROOT/'configs/ema_silu/SPEC.json')
    settings = dict(spec['training'], arch_type='tabm', hidden_activation='SiLU')
    torch.manual_seed(17)
    candidate = make_silu_network(RECIPE, settings, 2, 1).eval()
    candidate_rng = torch.get_rng_state()
    torch.manual_seed(17)
    direct = TabM.make(n_num_features=len(FEATURES), cat_cardinalities=[2], d_out=1,
        k=16, n_blocks=2, d_block=256, dropout=.1, arch_type='tabm', activation='SiLU',
        num_embeddings=PeriodicEmbeddings(len(FEATURES), d_embedding=16,
            n_frequencies=16, frequency_init_scale=.01, lite=True)).eval()
    torch.testing.assert_close(candidate_rng, torch.get_rng_state(), rtol=0, atol=0)
    torch.manual_seed(17)
    original = make_network(RECIPE, spec['training'], 2, 1).eval()
    torch.testing.assert_close(candidate_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert SiLUEMARegressor._train is ComponentRegressor._train
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in original.parameters())
    assert set(candidate.state_dict()) == set(original.state_dict())
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
    assert sum(isinstance(m, torch.nn.SiLU) for m in candidate.backbone.modules()) == 2
    assert sum(isinstance(m, torch.nn.ReLU) for m in original.backbone.modules()) == 2
    assert not any(isinstance(m, torch.nn.ReLU) for m in candidate.backbone.modules())
    assert {k:type(v) for k,v in candidate.num_module.named_modules()} == {
        k:type(v) for k,v in original.num_module.named_modules()}
    assert sum(isinstance(m, torch.nn.ReLU) for m in candidate.num_module.modules()) == 1
    x = torch.linspace(-1, 1, 31*len(FEATURES)).reshape(31, len(FEATURES))
    cat = torch.arange(31).remainder(2).reshape(-1, 1)
    with torch.no_grad():
        a, b, c = candidate(x, cat), direct(x, cat), original(x, cat)
    assert a.shape == (31,16,1)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)


def test_silu_refuses_undeclared_architecture_activation_and_batch_change(tmp_path):
    import torch
    from bf_tap_r2.component_regularization_run import RECIPE
    spec = run.read(run.ROOT/'configs/ema_silu/SPEC.json')
    settings = dict(spec['training'], arch_type='tabm', hidden_activation='SiLU')
    bad_settings = [spec['training'], dict(settings, arch_type='tabm-packed'),
        dict(settings, batch_order='independent_without_replacement'),
        dict(settings, hidden_activation='ReLU'), dict(settings, hidden_activation='GELU')]
    for i,bad in enumerate(bad_settings):
        with pytest.raises(ValueError):
            SiLUEMARegressor(RECIPE, bad, 'EMA', spec['mechanisms'])
        # A parameter-free activation is not identifiable from tensor shapes.
        # Reject the native header before accepting any same-shaped state.
        path=tmp_path/f'bad-header-{i}.pt'
        torch.save(dict(recipe=RECIPE, settings=bad, arm='EMA', mechanisms=spec['mechanisms']), path)
        with pytest.raises(ValueError):
            SiLUEMARegressor.load(path)
    with pytest.raises(ValueError):
        make_silu_network(RECIPE, dict(settings, hidden_activation='ReLU'), 2, 1)
    with pytest.raises(ValueError):
        make_silu_network(RECIPE, dict(settings, arch_type='tabm-packed'), 2, 1)


def test_preceding_packed_stage_requires_actual_exits_and_immutable_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'PREVIOUS_PACKED', tmp_path)
    with pytest.raises(FileNotFoundError):
        run.require_packed_closed()
    execution = tmp_path/'execution'; execution.mkdir()
    terminal = dict(status='passed', events=[dict(task=f'{i:03d}-{stage}-s{seed}-f{fold}-i{init}', exit_code=0)
        for i,(stage,seed,fold,init) in enumerate(run.tasks())])
    run.write(execution/'terminal.json', terminal)
    for name in ('report.json', 'independent-audit.json', 'manifest.json'):
        run.write(tmp_path/name, {})
    bindings = {'original_terminal_sha256': execution/'terminal.json',
        'report_sha256': tmp_path/'report.json', 'arithmetic_audit_sha256': tmp_path/'independent-audit.json',
        'manifest_sha256': tmp_path/'manifest.json'}
    audit = dict(status='passed', actual_child_exit_codes=[0]*72,
        controller_and_owned_children_absent=True, **{k:run.sha(p) for k,p in bindings.items()})
    run.write(execution/'independent-terminal-audit.json', audit)
    receipt = dict(status='passed', owned_processes_absent=True, owned_pids=[],
        actual_supervisor_exec_exit_code=0, actual_followup_exec_exit_code=0,
        actual_independent_audit_exit_code=0,
        independent_terminal_audit_sha256=run.sha(execution/'independent-terminal-audit.json'),
        **{k:run.sha(p) for k,p in bindings.items()})
    final = execution/'final-reconciliation.json'; run.write(final, receipt)
    assert len(run.require_packed_closed()) == 6
    for update in ({'status':'failed'}, {'actual_supervisor_exec_exit_code':None},
            {'actual_followup_exec_exit_code':1}, {'owned_processes_absent':False},
            {'manifest_sha256':'stale'}):
        final.write_text(json.dumps(dict(receipt, **update)))
        with pytest.raises(ValueError, match='packed stage'):
            run.require_packed_closed()


def test_fixed_candidates_and_no_partial_seed_admission():
    old = np.array([[8., 10.], [10., 12.], [12., 14.]])
    new = old + [1., -2.]
    base, candidates = run.columns([10., 15.], [9., 11.], old, new)
    np.testing.assert_array_equal(base, [11., 16.])
    np.testing.assert_array_equal(candidates['SILU_A100'], [12., 14.])
    np.testing.assert_allclose(candidates['SILU_A20'], [11.2, 15.6], rtol=0, atol=1e-14)
    assert run.choose({'SILU_A100': {'42': 1., '3407': -1.},
                       'SILU_A20': {'42': .1, '3407': .2}}) == 'SILU_A20'
    assert run.choose({'SILU_A100': {'42': .1, '3407': .2},
                       'SILU_A20': {'42': .1, '3407': .2}}) == 'SILU_A100'
    with pytest.raises(ValueError):
        run.choose({'SILU_A100': {'42': 1.}})
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


@pytest.mark.parametrize('field,value', [('weights', {'SILU_A100':.5,'SILU_A20':.2}),
    ('split_seeds',[42]), ('architecture','tabm-packed'), ('hidden_activation','ReLU'), ('time_budget_seconds',3600),
    ('automatic_retries',True)])
def test_freeze_rejects_changed_scientific_scope(field, value):
    spec=run.read(run.ROOT/'configs/ema_silu/SPEC.json')
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
    spec = run.read(run.ROOT/'configs/ema_silu/SPEC.json')
    rng = np.random.default_rng(964508)
    frame = pd.DataFrame(rng.normal(size=(2204, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 2 + 1
    frame['tap_time_len'] = 100 + 10*frame[FEATURES[0]] + rng.normal(size=len(frame))
    frame['tap_iron'] = 1000 + rng.normal(size=len(frame))
    query = frame.iloc[:74].drop(columns=list(TARGETS)).copy()
    query['sample_id'] = [f'query-{i}' for i in range(len(query))]
    settings = dict(spec['training'], max_epochs=2, patience=25, arch_type='tabm', hidden_activation='SiLU')
    identity = {'source_directory': str(tmp_path), 'split_seed': -1, 'trial_id': 'synthetic_full_shape'}
    with run.optimizer_ledger(tmp_path, identity) as counts:
        model = SiLUEMARegressor(RECIPE, settings, 'EMA', spec['mechanisms'], tmp_path)
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
from bf_tap_r2.ema_silu_audit import audit_models
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
