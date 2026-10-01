"""Scientific scope, outer isolation and complete paired OOF regressions."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap_r2.data import FEATURES
from bf_tap_r2.ema_average_span import (
    ORDER, bind_original_sources, candidate_column, choose_confirmation, summarize, task_frames, validate_scope, worker,
)

REPO = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((REPO/'configs/ema_average_span/SPEC.json').read_text())


def frame():
    n = 20
    data = {name: np.arange(n, dtype=float)+i for i, name in enumerate(FEATURES)}
    data.update(sample_id=[f's{i}' for i in range(n)], spout_no=np.arange(n)%2+1,
                tap_time_len=np.arange(n, dtype=float)+100, tap_iron=np.arange(n, dtype=float)+500)
    return pd.DataFrame(data)


def test_frozen_scope_refuses_trainer_weight_budget_or_time_gate_changes():
    spec = configuration(); original = yaml.safe_load((REPO/'configs/strong_component_regularization/SPEC.yaml').read_text())
    validate_scope(spec, original)
    for key, value in [('replacement_weight', .5), ('new_optimizer_runs', 42), ('maximum_runtime_seconds', 3600)]:
        changed = copy.deepcopy(spec); changed[key] = value
        with pytest.raises(ValueError, match='scope'):
            validate_scope(changed, original)
    changed = copy.deepcopy(spec); changed['training']['max_epochs'] += 1
    with pytest.raises(ValueError, match='trainer'):
        validate_scope(changed, original)
    changed = copy.deepcopy(spec); changed['candidate_beta']['LONG_SPAN'] = .995
    with pytest.raises(ValueError, match='spans'):
        validate_scope(changed, original)


def test_original_private_configs_bind_to_main_and_do_not_allow_missing_code(tmp_path):
    workspace, main = tmp_path/'workspace', tmp_path/'main'
    (workspace/'src').mkdir(parents=True); (main/'configs').mkdir(parents=True)
    public = workspace/'src/model.py'; public.write_text('frozen model')
    private = main/'configs/data.local.yaml'; private.write_text('private configuration')
    hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {'src/model.py': hash_file(public), 'configs/data.local.yaml': hash_file(private)}
    assert bind_original_sources(workspace, main, hashes) == {'configs/data.local.yaml': hash_file(private)}
    assert not (workspace/'configs/data.local.yaml').exists()
    public.unlink()
    with pytest.raises(FileNotFoundError):
        bind_original_sources(workspace, main, hashes)
    public.write_text('frozen model'); private.write_text('changed configuration')
    with pytest.raises(ValueError, match='identity'):
        bind_original_sources(workspace, main, hashes)


def test_affine_replacement_preserves_control_and_rejects_invalid_extrapolation():
    parent = np.array([100., 120.]); old = np.array([101., 121.])
    np.testing.assert_array_equal(candidate_column(parent, old, old), parent)
    np.testing.assert_array_equal(candidate_column(parent, old, old+4), parent+3)
    with pytest.raises(ValueError, match='clipping'):
        candidate_column(parent, old, [-1000, -1000])
    with pytest.raises(ValueError, match='Nonfinite'):
        candidate_column(parent, old, [np.nan, 1])
    with pytest.raises(ValueError, match='Aligned'):
        candidate_column(parent, old[:, None], old)


def test_confirmation_requires_full_positive_seeds_and_uses_frozen_tie_order():
    gains = {ORDER[0]: {'42': .01, '3407': .01}, ORDER[1]: {'42': .02, '3407': .02}}
    assert choose_confirmation(gains) == ORDER[1]
    gains[ORDER[1]] = {'42': .01+5e-13, '3407': .01+5e-13}
    assert choose_confirmation(gains) == ORDER[0]
    gains[ORDER[0]]['3407'] = 0; gains[ORDER[1]]['42'] = -.01
    assert choose_confirmation(gains) is None
    del gains[ORDER[1]]['3407']
    with pytest.raises(ValueError, match='coverage'):
        choose_confirmation(gains)


def test_held_labels_never_enter_training_or_inference_query():
    f = frame(); folds = np.arange(len(f))%5
    training, query = task_frames(f, folds, 0)
    changed = f.copy(); changed.loc[folds == 0, ['tap_time_len', 'tap_iron']] = -1e12
    changed_training, changed_query = task_frames(changed, folds, 0)
    pd.testing.assert_frame_equal(training, changed_training)
    pd.testing.assert_frame_equal(query, changed_query)
    assert 'tap_time_len' not in query and 'tap_iron' not in query


def test_worker_passes_only_outer_training_to_backend_and_keeps_original_settings(tmp_path, monkeypatch):
    pytest.importorskip("torch", reason="This backend path imports the optional neural trainer")
    import bf_tap_r2.ema_average_span as module
    spec = configuration(); f = frame(); fv = np.arange(len(f))%5
    plan = {'s42-f0': dict(training='synthetic', query='synthetic')}
    (tmp_path/'manifest.json').write_text(json.dumps(dict(spec=spec, plan=plan)))
    monkeypatch.setattr(module, 'context', lambda *a: (spec, f, {42: fv}))
    monkeypatch.setattr(module, 'require_memory', lambda *a: None)
    monkeypatch.setattr(module, 'old_cache', lambda *a: dict(q75=np.full(4, 100.), ema=np.full(4, 100.), iron=np.full(4, 500.)))
    seen = []

    class Backend:
        def __init__(self, recipe, settings, arm, mechanisms, directory):
            assert settings == spec['training'] and arm == 'EMA'
            assert mechanisms == dict(spec['mechanisms'], ema_beta=spec['candidate_beta'][ORDER[0]])
            self.directory = directory; self.traces = {}

        def _train(self, inputs, targets, epochs, validation=None):
            phase = 'selection' if validation is not None else 'refit'
            trace = dict(fit_rows=len(inputs), selected_epoch=1, stopped_epoch=1, updates=1)
            self.traces[phase] = trace
            (self.directory/f'{phase}.pt').write_bytes(b'synthetic backend, no real fitting')
            return 1

        def fit(self, training, target):
            seen.append(training.copy())
            np.testing.assert_array_equal(target[:, 0], f.loc[fv != 0, 'tap_time_len'])
            self._train(training, target, 1, (training.iloc[:1], target[:1]))
            self._train(training, target, 1)
            self.metadata_ = dict(traces=self.traces)

        def predict(self, query):
            assert 'tap_time_len' not in query and 'tap_iron' not in query
            return np.full((len(query), 1), 100.)

    monkeypatch.setattr('bf_tap_r2.component_regularization.ComponentRegressor', Backend)
    worker(REPO, tmp_path, 42, 0, ORDER[0])
    assert set(seen[0].sample_id) == set(f.loc[fv != 0, 'sample_id'])
    with np.load(tmp_path/f'{ORDER[0]}-s42-f0/predictions.npz') as saved:
        np.testing.assert_array_equal(saved['candidate'], np.full(4, 100.))
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events].count('optimizer_completed') == 2


def test_summary_closes_each_seed_independently_and_requires_every_unit(tmp_path, monkeypatch):
    pytest.importorskip("torch", reason="This backend path imports the optional neural trainer")
    import bf_tap_r2.ema_average_span as module
    spec = configuration(); spec['main_root'] = str(REPO)
    f = frame(); folds = {42: np.arange(len(f))%5, 3407: (np.arange(len(f))+1)%5}
    (tmp_path/'manifest.json').write_text(json.dumps({'spec': spec}))
    monkeypatch.setattr(module, 'context', lambda *a: (spec, f, folds))
    for seed, fv in folds.items():
        for fold in range(5):
            held = fv == fold; y = f.loc[held, 'tap_time_len'].to_numpy()
            for index, name in enumerate(ORDER):
                directory = tmp_path/f'{name}-s{seed}-f{fold}'; directory.mkdir()
                parent = y+(2 if seed == 42 else 4)
                prediction = y+(1 if index == 0 else .5)
                np.savez(directory/'predictions.npz', query_ids=f.loc[held, 'sample_id'].to_numpy(str),
                    q75=parent, candidate=prediction, ema=prediction, iron_reference=f.loc[held, 'tap_iron'].to_numpy()+1)
                metadata = dict(averaging={}, refit_training_mae=.25, model=dict(selected_epoch=1,
                    traces={'selection': {'history': [dict(validation_mae=1., training_eval_mae=.5)]}}))
                (directory/'metadata.json').write_text(json.dumps(metadata))
                (directory/'complete.json').write_text(json.dumps({'hashes': {}}))
    summary = summarize(REPO, tmp_path)
    assert summary['selected_for_confirmation'] == 'LONG_SPAN'
    assert summary['gains']['SHORT_SPAN']['42'] == pytest.approx(50/109.5)
    assert summary['gains']['SHORT_SPAN']['3407'] == pytest.approx(150/109.5)
    assert summary['diagnostics']['42']['SHORT_SPAN-f0']['outer_component_mae'] == 1
    # A missing outer unit is refused even when all other predictions look valid.
    (tmp_path/'LONG_SPAN-s3407-f4/complete.json').unlink()
    with pytest.raises(FileNotFoundError):
        summarize(REPO, tmp_path)
