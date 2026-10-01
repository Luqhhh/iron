"""Capture plumbing uses synthetic data and a solver fixture without updates.

The original Trial/V31 fit, prediction and selection helper remain in use.
The fixture replaces the numerical estimator; it cannot establish model quality.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.ema_reference_artifacts import audit_witness,sha
from bf_tap_r2.ema_reference_ledger import Binding,KINDS
from bf_tap_r2.ema_reference_capture import ReferenceCapture,capture_original_factory
from bf_tap_r2.ema_reference_capture_audit import audit_capture
import bf_tap_r2.ema_reference_capture as capture_module
import bf_tap_r2.v3_local_search as v3
import bf_tap_r2.v3_1_models as v31
import bf_tap_r2.v4_1_reference as reference_module


class NativeSolverFixture:
    def __init__(self,**params):
        self.params=params;self.predict_calls=0

    def fit(self,x,y,eval_set=None):
        # Recording fixture, with no numerical optimization or parameter update.
        self.rows=len(x);self.tree_count_=3
        return self

    def get_best_iteration(self):return 2

    def predict(self,x):
        self.predict_calls+=1
        return .02*x.air_volume.to_numpy(dtype=float)+2.


def trial():
    return dict(trial_id='fixture',family='catboost',target='tap_time_len',
        feature_set='raw',target_transform='mean_std',
        parameters=dict(iterations=7,thread_count=1))


def frames():
    n=30
    training=pd.DataFrame({name:np.arange(1,n+1,dtype=float)*(j+1)/10 for j,name in enumerate(FEATURES)})
    training.insert(0,'sample_id',[f'F{i:02d}' for i in range(n)])
    training['spout_no']=np.arange(n)%2+1
    training['tap_iron']=50+np.arange(n,dtype=float)
    training['tap_time_len']=20+np.arange(n,dtype=float)
    query=training.iloc[:9].drop(columns=['tap_iron','tap_time_len']).copy()
    query['sample_id']=[f'Q{i:02d}' for i in range(len(query))]
    return training,query


def plan(tree=False):
    roles=[dict(name=f'role-{i:02d}',family='legacy_tree',recipe=trial(),
        expected_native_calls={k:int(k=='catboost_fit') for k in KINDS}) for i in range(32)]
    if tree:
        roles[0]['family']='legacy_inner_select_refit_tree'
        roles[0]['inner_seed']=7771
        roles[0]['expected_native_calls']['catboost_fit']=2
    return dict(roles=roles,pipelines=32,expected_native_calls_per_factory={
        k:sum(r['expected_native_calls'][k] for r in roles) for k in KINDS})


def manager(tmp_path,*,tree=False):
    training,query=frames()
    paths=[__file__,capture_module.__file__,v3.__file__,v31.__file__]
    return ReferenceCapture(tmp_path/'capture',source_directory=str(tmp_path.resolve()),
        split_seed=271828,fold=0,training=training,query=query,plan=plan(tree),
        source_hashes={str(Path(p).resolve()):sha(p) for p in paths},row_atol=0),training,query


def fixture_fit_model(name,training,query):
    model=v3.TrialRegressor(trial())
    model.fit(training,training.tap_time_len.to_numpy())
    return 'tap_time_len',model.predict(query)


def fixture_initializer(training,query,dummy):
    global _TRAIN,_QUERY
    _TRAIN=training;_QUERY=query


def fixture_worker(task):
    target,prediction=_RECOVERY.fit_model(task['name'],_TRAIN,_QUERY)
    return dict(target=target,pred=prediction)


@pytest.fixture
def solver_fixture(monkeypatch):
    monkeypatch.setattr(v3,'CatBoostRegressor',NativeSolverFixture)
    monkeypatch.setattr(capture_module,'reference_bindings',lambda:[Binding(NativeSolverFixture,'fit','catboost_fit')])


def test_original_factory_single_worker_closes_all_roles_with_child_owned_ledgers(tmp_path,monkeypatch,solver_fixture):
    global _RECOVERY
    capture,training,query=manager(tmp_path)
    original_fit=v3.TrialRegressor.fit;original_predict=v3.TrialRegressor.predict
    original_v31_fit=v31.V31Regressor.fit
    recovery=SimpleNamespace(BASE_NAMES=list(capture.roles),fit_model=fixture_fit_model,
        init_worker=fixture_initializer,worker=fixture_worker)
    monkeypatch.setattr(reference_module,'_load_l1_recovery',lambda root:recovery)
    loader=reference_module._load_l1_recovery
    with capture_original_factory(capture):
        _RECOVERY=reference_module._load_l1_recovery(tmp_path)
        factory=SimpleNamespace(_recovery=_RECOVERY,workers=1)
        store=reference_module.FrozenAReferenceFactory._fit_base_store(factory,training,query)
    assert reference_module._load_l1_recovery is loader
    assert v3.TrialRegressor.fit is original_fit and v3.TrialRegressor.predict is original_predict
    assert v31.V31Regressor.fit is original_v31_fit
    assert set(store)==set(capture.roles)
    result=capture.close()
    assert result['pipelines']==32 and result['native_counts']['catboost_fit']==32
    assert result['status'].endswith('cold_audits_pending')
    worker_pids=set()
    for name in capture.roles:
        directory=capture.directory/name
        native_start=json.loads((directory/'native/scope-start.json').read_text())
        worker_pids.add(native_start['pid'])
        witness=json.loads((directory/'final/complete.json').read_text())
        assert witness['model_class']=='bf_tap_r2.v3_local_search.TrialRegressor'
        # It saved the one original observed prediction, without warm replay.
        import pickle
        model=pickle.loads((directory/'final/model.pkl').read_bytes())
        assert model.estimator_.predict_calls==1
        np.testing.assert_array_equal(np.load(directory/'final/observed.npy'),store[name])
    assert len(worker_pids)==1 and os.getpid() not in worker_pids
    script='''import json,os,sys,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from bf_tap_r2.ema_reference_capture_audit import audit_capture
print(json.dumps(audit_capture(sys.argv[1],sys.argv[2])))
'''
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH','')
    process=subprocess.run([sys.executable,'-c',script,str(capture.directory),sha(capture.directory/'complete.json')],env=env,
        capture_output=True,text=True,check=True)
    cold=json.loads(process.stdout)
    assert cold['cold_pid'] not in worker_pids|{os.getpid()} and cold['retained_states']==32
    assert cold['pipelines']==32 and cold['native_counts']['catboost_fit']==32
    assert all(v==0 for v in cold['differences'].values())
    with pytest.raises(FileExistsError):audit_capture(capture.directory,sha(capture.directory/'complete.json'))
    with pytest.raises(FileExistsError):capture.close()


def test_original_v31_selection_return_and_probe_state_are_preserved(tmp_path,solver_fixture):
    capture,training,query=manager(tmp_path,tree=True)
    with capture_original_factory(capture),capture.role('role-00'):
        model,metadata=v31.fit_with_inner_early_stop(trial(),training,'tap_time_len',inner_seed=7771)
        prediction=model.predict(query)
        assert model.trial['parameters']['iterations']==3 and metadata['best_iteration']==3
        assert model._delegate.estimator_.predict_calls==1
    directory=capture.directory/'role-00'
    complete=json.loads((directory/'complete.json').read_text())
    assert complete['top_fits']==2 and complete['native_counts']['catboost_fit']==2
    assert set(complete['witnesses'])=={'final','tree-selector'}
    probe=json.loads((directory/'tree-selector/complete.json').read_text())
    assert not probe['fit_metadata']['original_factory_prediction']
    assert set(probe['training_ids'])|set(probe['query_ids'])==set(training.sample_id)
    assert probe['fit_metadata']['selection_fields']['selected_num_boost_round']==metadata['best_iteration']
    for key,h in complete['witnesses'].items():
        assert audit_witness(directory/key,h)['differences']['full']==0
    np.testing.assert_array_equal(np.load(directory/'final/observed.npy'),prediction)


def test_original_v31_selector_cannot_change_inner_seed(tmp_path,solver_fixture):
    capture,training,query=manager(tmp_path,tree=True)
    with capture_original_factory(capture),pytest.raises(ValueError,match='selector partition changed'):
        with capture.role('role-00'):
            v31.fit_with_inner_early_stop(trial(),training,'tap_time_len',inner_seed=7772)
    assert not list((capture.directory/'role-00/native').glob('call-*'))


def test_caller_plan_mutation_cannot_change_already_bound_recipe_or_budget(tmp_path,solver_fixture):
    training,query=frames();p=plan()
    capture=ReferenceCapture(tmp_path/'capture',source_directory=str(tmp_path.resolve()),
        split_seed=271828,fold=0,training=training,query=query,plan=p,
        source_hashes={str(Path(s).resolve()):sha(s) for s in (__file__,capture_module.__file__,v3.__file__)})
    p['roles'][0]['recipe']['parameters']['iterations']=100
    p['roles'][0]['expected_native_calls']['catboost_fit']=9
    with capture_original_factory(capture),capture.role('role-00'):
        v3.TrialRegressor(trial()).fit(training,training.tap_time_len.to_numpy()).predict(query)
    complete=json.loads((capture.directory/'role-00/complete.json').read_text())
    assert complete['native_counts']['catboost_fit']==1
    assert capture.roles['role-00']['recipe']['parameters']['iterations']==7


def test_fit_failure_is_preserved_without_masking_or_retry(tmp_path,solver_fixture):
    capture,training,query=manager(tmp_path)
    altered=trial();altered['parameters']['iterations']=8
    with capture_original_factory(capture),pytest.raises(ValueError,match='constructor recipe changed'):
        with capture.role('role-00'):
            v3.TrialRegressor(altered).fit(training,training.tap_time_len.to_numpy())
    directory=capture.directory/'role-00'
    assert 'constructor recipe changed' in (directory/'failure.json').read_text()
    assert not list((directory/'native').glob('call-*'))
    with pytest.raises(FileExistsError),capture.role('role-00'):pass


@pytest.mark.parametrize('change',['fit-values','query-values','query-order'])
def test_features_and_query_order_are_bound_before_success(tmp_path,solver_fixture,change):
    capture,training,query=manager(tmp_path)
    fit=training.copy();request=query.copy()
    if change=='fit-values':fit.loc[0,'air_volume']+=1
    if change=='query-values':request.loc[0,'air_volume']+=1
    if change=='query-order':request=request.iloc[::-1]
    with capture_original_factory(capture),pytest.raises(ValueError,match='features|feature values|query changed'):
        with capture.role('role-00'):
            model=v3.TrialRegressor(trial()).fit(fit,fit.tap_time_len.to_numpy())
            model.predict(request)
    assert (capture.directory/'role-00/failure.json').is_file()
    assert not (capture.directory/'role-00/complete.json').exists()


def test_original_prediction_error_is_preserved(tmp_path,solver_fixture,monkeypatch):
    capture,training,query=manager(tmp_path)
    def fail(self,frame):raise RuntimeError('original fixture inference error')
    monkeypatch.setattr(v3.TrialRegressor,'predict',fail)
    with capture_original_factory(capture),pytest.raises(RuntimeError,match='original fixture inference error'):
        with capture.role('role-00'):
            v3.TrialRegressor(trial()).fit(training,training.tap_time_len.to_numpy()).predict(query)
    assert 'original fixture inference error' in (capture.directory/'role-00/failure.json').read_text()
    assert v3.TrialRegressor.predict is fail


@pytest.mark.parametrize('change',['duplicate-role','role-path','totals','negative-tolerance','bool-seed'])
def test_invalid_scope_is_refused_before_consuming_directory(tmp_path,change):
    training,query=frames();p=deepcopy(plan());options={}
    if change=='duplicate-role':p['roles'].append(deepcopy(p['roles'][0]))
    if change=='role-path':p['roles'][0]['name']='../outside'
    if change=='totals':p['expected_native_calls_per_factory']['catboost_fit']+=1
    if change=='negative-tolerance':options['row_atol']=-1
    if change=='bool-seed':options['split_seed']=True
    with pytest.raises(ValueError):
        ReferenceCapture(tmp_path/'bad',source_directory=str(tmp_path.resolve()),
            split_seed=options.pop('split_seed',271828),fold=0,training=training,query=query,plan=p,
            source_hashes={str(Path(__file__).resolve()):sha(__file__)},**options)
    assert not (tmp_path/'bad').exists()


def test_missing_roles_and_tampered_manager_start_cannot_close(tmp_path):
    capture,_,_=manager(tmp_path)
    with pytest.raises(ValueError,match='role coverage'):capture.close()
    (capture.directory/'start.json').write_text('{}')
    with pytest.raises(ValueError,match='start identity'):capture.close()


@pytest.mark.parametrize('change',['outer-start','native-call','extra-native-call','model-receipt','same-warm-process'])
def test_cold_factory_rejects_tampering_and_same_process_before_inference(tmp_path,solver_fixture,change):
    capture,training,query=manager(tmp_path)
    with capture_original_factory(capture):
        for name in capture.roles:
            with capture.role(name):
                v3.TrialRegressor(trial()).fit(training,training.tap_time_len.to_numpy()).predict(query)
    capture.close();receipt=sha(capture.directory/'complete.json')
    first=capture.directory/'role-00'
    if change=='outer-start':(capture.directory/'start.json').write_text('{}')
    if change=='native-call':(first/'native/call-0001/start.json').write_text('{}')
    if change=='extra-native-call':
        d=first/'native/call-0002';d.mkdir();(d/'failure.json').write_text('{}')
    if change=='model-receipt':(first/'final/complete.json').write_text('{}')
    with pytest.raises(ValueError,match='identity changed|coverage changed|artifact changed|different process'):
        audit_capture(capture.directory,receipt)
    assert not list(capture.directory.glob('*/final/cold-audit.json'))
    assert not (capture.directory/'cold-complete.json').exists()
