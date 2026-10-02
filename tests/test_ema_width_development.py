import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap_r2.data import FEATURES
from bf_tap_r2.ema_width_development import ARMS, report_arrays, summarize, validate_scope, worker

WORK = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((WORK/'configs/ema_width512/SPEC.json').read_text())


def fixture_frame(n=100):
    rng = np.random.default_rng(960531)
    frame = pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    frame['sample_id'] = [f'test-{i}' for i in range(n)]; frame['spout_no'] = np.arange(n)%2+1
    frame['tap_iron'] = 500.; frame['tap_time_len'] = 100.
    return frame


def test_scope_preserves_native_trainer_gate_budget_and_no_time_limit():
    original = yaml.safe_load((WORK/'configs/strong_component_regularization/SPEC.yaml').read_text()); spec = configuration()
    validate_scope(spec,original)
    for key,value in [('new_optimizer_runs',22),('maximum_runtime_seconds',3600),('replacement_weight',.5)]:
        changed = copy.deepcopy(spec); changed[key] = value
        with pytest.raises(ValueError,match='scope'): validate_scope(changed,original)
    changed = copy.deepcopy(spec); changed['training']['width'] = 256
    with pytest.raises(ValueError,match='protocol'): validate_scope(changed,original)
    changed = copy.deepcopy(spec); changed['training']['max_epochs'] += 1
    with pytest.raises(ValueError,match='protocol'): validate_scope(changed,original)
    changed = copy.deepcopy(spec); changed['confirmation_gate']['both_complete_development_seeds_positive_vs_Q75'] = False
    with pytest.raises(ValueError,match='scope'): validate_scope(changed,original)


def test_complete_both_seed_gate_and_real_classifier_api():
    arrays = {}
    for seed,delta in [('42',.5),('3407',3.)]:
        arrays[seed] = dict(actual=np.tile([500.,100.],(100,1)),iron=np.full(100,500.),Q75=np.full(100,102.),
            EMA_WIDTH512=np.full(100,100.+delta),folds=np.arange(100)%5,spouts=np.arange(100)%2+1)
    policy = yaml.safe_load((WORK/'configs/candidate_tiers.yaml').read_text())
    result = report_arrays(arrays,policy)
    assert result['records']['42']['gains']['EMA_WIDTH512']>0
    assert result['records']['3407']['gains']['EMA_WIDTH512']<0
    assert not result['confirmation_eligible'] and result['selected_for_confirmation'] is None
    assert result['formal_promotion'] is False
    assert set(result['tiers']['decisions']['tap_time_len'])=={'EMA_WIDTH512'}
    arrays['3407']['EMA_WIDTH512'][:] = 100.5
    assert report_arrays(arrays,policy)['selected_for_confirmation']=='EMA_WIDTH512'
    with pytest.raises(ValueError): report_arrays({'42':arrays['42']},policy)
    arrays['3407']['EMA_WIDTH512'][0] = np.nan
    with pytest.raises(ValueError): report_arrays(arrays,policy)


def test_worker_training_boundary_and_complete_summary_never_use_query_targets(tmp_path,monkeypatch):
    import bf_tap_r2.ema_width_development as module
    frame = fixture_frame(); folds = {42:np.arange(len(frame))%5,3407:(np.arange(len(frame))+1)%5}
    spec = configuration(); spec['main_root'] = str(WORK)
    manifest = dict(spec=spec,files={},plan={f's{s}-f{f}':{'training':'spy','query':'spy'} for s in folds for f in range(5)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(module,'context',lambda out:(manifest,spec,frame,folds)); monkeypatch.setattr(module,'require_memory',lambda spec:None)
    monkeypatch.setattr(module,'old_cache',lambda spec,s,f,training,query:dict(q75=np.full(len(query),102.),ema=np.full(len(query),100.),iron=np.full(len(query),500.)))
    seen = []

    class Backend:
        def __init__(self,recipe,settings,arm,mechanisms,directory):
            self.directory = directory; self.traces = {}; self.mechanisms = mechanisms
            assert arm=='EMA' and settings==spec['training']
        def _train(self,frame,y,epochs,validation=None):
            phase = 'selection' if validation is not None else 'refit'; self.traces[phase] = {'fake':True}
            (self.directory/f'{phase}.pt').write_bytes(b'fake backend: no optimizer or labels loaded')
            return 1
        def fit(self,training,y):
            seen.append(training.copy()); np.testing.assert_array_equal(y,training[['tap_time_len']].to_numpy())
            self._train(training,y,1,(training.iloc[:1],y[:1])); self._train(training,y,1)
            self.metadata_ = {'traces':self.traces}
        def predict(self,query):
            assert 'tap_iron' not in query and 'tap_time_len' not in query
            offset = 1.
            return np.full((len(query),1),100.-offset/.75)

    monkeypatch.setattr(module,'ComponentRegressor',Backend)
    original = frame.copy(); frame.loc[folds[42]==0,['tap_iron','tap_time_len']] = -1e12
    worker(tmp_path,42,0,ARMS[0])
    pd.testing.assert_frame_equal(seen[0],original.loc[folds[42]!=0].reset_index(drop=True))
    frame.loc[:,:] = original
    for seed in folds:
        for fold in range(5):
            for arm in ARMS:
                if (seed,fold,arm)!=(42,0,ARMS[0]): worker(tmp_path,seed,fold,arm)
    summarize(tmp_path)
    result = json.loads((tmp_path/'report.json').read_text()); assert result['selected_for_confirmation']=='EMA_WIDTH512'
    for seed,fv in folds.items():
        with np.load(tmp_path/f'oof-s{seed}.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['folds'],fv); np.testing.assert_array_equal(saved['query_ids'],frame.sample_id.to_numpy(str))
            np.testing.assert_array_equal(saved['EMA_WIDTH512'],np.full(len(frame),101.))
    rows = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert sum(row['event']=='optimizer_completed' for row in rows)==20
    assert sum(row['event']=='unit_completed' for row in rows)==10
    assert all(row['source_directory']==str(tmp_path.resolve()) for row in rows)


def test_summary_refuses_missing_unit_before_creating_any_complete_oof(tmp_path,monkeypatch):
    import bf_tap_r2.ema_width_development as module
    frame = fixture_frame(); spec = configuration()
    monkeypatch.setattr(module,'context',lambda out:({},spec,frame,{42:np.arange(len(frame))%5,3407:(np.arange(len(frame))+1)%5}))
    with pytest.raises(FileNotFoundError): summarize(tmp_path)
    assert not list(tmp_path.glob('oof-*.npz')) and not (tmp_path/'report.json').exists()


def test_independent_auditor_import_and_scalar_math_use_unrounded_arrays():
    spec = importlib.util.spec_from_file_location('dropout_auditor',WORK/'scripts/audit_ema_width_development.py')
    auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
    actual = np.tile([500.,100.],(10,1))
    assert auditor.scalar_score(actual,np.full(10,500.),np.full(10,102.))==99.
    assert auditor.scalar_score(actual,np.full(10,505.),np.full(10,100.))==99.5


def test_observer_reaps_real_success_and_failure_without_between_check_progress_reads(tmp_path):
    spec = importlib.util.spec_from_file_location('dropout_observer',WORK/'scripts/observe_ema_width_development.py')
    observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
    for exit_code in (0,7):
        child = subprocess.Popen([sys.executable,'-c',f'raise SystemExit({exit_code})'])
        phase = f'exit{exit_code}'
        receipt = observer.wait_actual(child,tmp_path,phase)
        assert receipt['pid']==child.pid and receipt['exit_code']==exit_code and child.returncode==exit_code
        assert json.loads((tmp_path/f'{phase}-terminal.json').read_text())==receipt
    assert not list(tmp_path.glob('*-observation-*.json'))
