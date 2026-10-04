"""No-neural-fit guards; full-shape native parity/cold admission is separate."""
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import realmlp_state_adapter as adapter
from bf_tap_r2.data import FEATURES


@pytest.fixture(autouse=True)
def isolated_private_root(tmp_path,monkeypatch):
    monkeypatch.setattr(adapter,'PRIVATE_ROOT',tmp_path)


def frame():
    x=pd.DataFrame(np.random.default_rng(964510).normal(size=(100,len(FEATURES))),columns=FEATURES)
    x['sample_id']=[f'synthetic-{i:03d}' for i in range(100)]
    x['spout_no']=np.arange(100)%2+1
    return x


RECIPE={'class':'RealMLP_TD_Regressor','constructor':{'n_epochs':3},
        'resolved':{'n_epochs':3,'val_fraction':.2}}


class RecordingEstimator:
    def __init__(self,fail=False):
        self.fail=fail
        self.fit_params_={'stop_epoch':{'mae':2}}
        self.calls=[]
    def fit(self,*args):
        self.calls.append(args)
        if self.fail:raise RuntimeError('deliberate native-fit boundary failure')
        return self
    def predict(self,x):return x[:,0].astype(float)


def test_lightning_empty_trainer_slot_is_safe_but_live_state_is_rejected():
    from types import SimpleNamespace
    adapter.validate_cleanup(SimpleNamespace())
    adapter.validate_cleanup(SimpleNamespace(_trainer=None))
    for module in [SimpleNamespace(_trainer=object()),SimpleNamespace(train_dl=None),
                   SimpleNamespace(val_dl=None),SimpleNamespace(callbacks=[])]:
        with pytest.raises(ValueError,match='cleanup'):
            adapter.validate_cleanup(module)


def test_calls_original_fit_once_and_captures_actual_two_objects(monkeypatch):
    factory_calls=[]
    def factory(recipe,**kwargs):
        obj=RecordingEstimator();factory_calls.append((obj,kwargs));return obj
    monkeypatch.setattr(adapter.original,'make_estimator',factory)
    original_encoder=adapter.original.InputEncoder
    native_fit=adapter.original.RealMLPRegressor.fit
    calls=[]
    def observe(self,*args):
        calls.append(self);return native_fit(self,*args)
    monkeypatch.setattr(adapter.original.RealMLPRegressor,'fit',observe)
    x=frame();y=np.arange(100,dtype=float)
    captured=adapter.capture_fit(RECIPE,x,y)
    assert calls==[captured.regressor]
    assert [kwargs for _,kwargs in factory_calls]==[{}, {'stop_epoch':2,'val_fraction':0.}]
    assert tuple(obj for obj,_ in factory_calls)==captured.estimators
    assert len(captured.estimators[0].calls)==len(captured.estimators[1].calls)==1
    assert [len(captured.estimators[i].calls[0][0]) for i in range(2)]==[80,100]
    assert len(captured.estimators[0].calls[0])==4 and len(captured.estimators[1].calls[0])==2
    assert [len(ids) for ids in captured.fitting_ids]==[80,100]
    assert adapter.original.make_estimator is factory
    assert adapter.original.InputEncoder is original_encoder
    assert captured.regressor.model_ is captured.estimators[1]
    assert captured.regressor.encoder_ is captured.encoders[1]
    assert captured.regressor.metadata_['selected_epoch']==2


def test_factories_restore_after_native_failure(monkeypatch):
    def factory(*args,**kwargs):return RecordingEstimator(fail=True)
    monkeypatch.setattr(adapter.original,'make_estimator',factory)
    encoder=adapter.original.InputEncoder
    with pytest.raises(RuntimeError,match='deliberate'):
        adapter.capture_fit(RECIPE,frame(),np.arange(100,dtype=float))
    assert adapter.original.make_estimator is factory and adapter.original.InputEncoder is encoder


@pytest.mark.parametrize('bad',['duplicate','target_shape','nonfinite'])
def test_bad_fitting_identity_rejected_before_estimator(monkeypatch,bad):
    x=frame();y=np.arange(100,dtype=float)
    if bad=='duplicate':x.loc[1,'sample_id']=x.loc[0,'sample_id']
    elif bad=='target_shape':y=y[:-1]
    else:y[0]=np.nan
    with patch.object(adapter.original,'make_estimator',side_effect=AssertionError('must not construct')):
        with pytest.raises(ValueError):adapter.capture_fit(RECIPE,x,y)


def test_role_config_keeps_horizon_and_selects_fresh_refit():
    before=deepcopy(RECIPE)
    assert adapter.expected_native_config(RECIPE,'selection',2)==RECIPE['resolved']
    assert adapter.expected_native_config(RECIPE,'refit',2)=={'n_epochs':3,'val_fraction':0.,'stop_epoch':2}
    assert RECIPE==before
    with pytest.raises(ValueError):adapter.expected_native_config(RECIPE,'wrong',2)


def identity(parent):return dict(source_directory=str(parent.resolve()),split_seed=42,fold=0,trial_id='synthetic-interface')


@pytest.mark.parametrize('bad',['source','seed_bool','fold_string','empty_trial'])
def test_cache_identity_requires_exact_source_and_types(tmp_path,bad):
    value=identity(tmp_path)
    if bad=='source':value['source_directory']=str(tmp_path/'elsewhere')
    elif bad=='seed_bool':value['split_seed']=True
    elif bad=='fold_string':value['fold']='0'
    else:value['trial_id']=''
    with pytest.raises(ValueError):adapter.validate_identity(value,tmp_path)


@pytest.mark.parametrize('bad',['role','trial','recipe','source','runtime','bytes'])
def test_sidecar_faults_refused_before_pickle_load(tmp_path,bad):
    path=tmp_path/'selection.pkl';path.write_bytes(b'not a pickle')
    ident=identity(tmp_path)
    h=dict(version=adapter.VERSION,identity=deepcopy(ident),role='selection',recipe=deepcopy(RECIPE),
        original_source_sha256=adapter.sha(adapter.original.__file__),adapter_source_sha256=adapter.sha(adapter.__file__),
        runtime_versions={p:importlib.metadata.version(p) for p in adapter.DEPENDENCIES})
    record={'header':h,'state_sha256':adapter.sha(path)}
    if bad=='role':h['role']='refit'
    elif bad=='trial':h['identity']['trial_id']='other-fit'
    elif bad=='recipe':h['recipe']['resolved']['n_epochs']=99
    elif bad=='source':h['adapter_source_sha256']='0'*64
    elif bad=='runtime':h['runtime_versions']['torch']='not-the-frozen-version'
    else:record['state_sha256']='0'*64
    path.with_suffix('.pkl.json').write_text(json.dumps(record))
    with patch.object(adapter.pickle,'load',side_effect=AssertionError('unpickle must not run')):
        with pytest.raises(ValueError):adapter.load_snapshot(path,expected_identity=ident,expected_role='selection',expected_recipe=RECIPE)


def test_existing_state_cannot_be_overwritten(tmp_path):
    p=tmp_path/'selection.pkl';p.write_bytes(b'preserved failure')
    with pytest.raises(FileExistsError):adapter.save_snapshot(None,'selection',p,identity(tmp_path))
    assert p.read_bytes()==b'preserved failure'


def test_snapshot_query_guards_and_order():
    x=frame();enc=adapter.original.InputEncoder().fit(x)
    payload={'encoder':enc,'estimator':RecordingEstimator()}
    expected=adapter.predict_snapshot(payload,x)
    np.testing.assert_array_equal(adapter.predict_snapshot(payload,x.iloc[::-1])[::-1],expected)
    x['tap_time_len']=1.
    with pytest.raises(ValueError,match='targets'):adapter.predict_snapshot(payload,x)


@pytest.mark.parametrize('bad',[np.array([np.nan]),np.zeros((1,1))])
def test_bad_saved_predictions_rejected(bad):
    x=frame().iloc[:1]
    enc=adapter.original.InputEncoder().fit(frame())
    class Bad:
        def predict(self,x):return bad
    with pytest.raises(ValueError,match='Invalid saved'):
        adapter.predict_snapshot({'encoder':enc,'estimator':Bad()},x)
