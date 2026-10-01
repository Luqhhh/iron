"""Original component plumbing with synthetic frames and no parameter updates."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.ema_reference_artifacts import sha
from bf_tap_r2.ema_reference_ledger import Binding
from bf_tap_r2.ema_span_confirmation_models import fit_component,audit_component
import bf_tap_r2.ema_span_confirmation_models as module
import bf_tap_r2.component_regularization as original_component
import bf_tap_r2.v12_joint as original_joint
from test_ema_reference_capture import frames


class FixedNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(.1))

    def forward(self,x,cat):return (x[:,:1]*self.value)[:,None,:]


class NoUpdateOptimizer:
    def __init__(self,params,lr,weight_decay):
        self.params=list(params);self.steps=0

    def zero_grad(self,**kwargs):
        for p in self.params:p.grad=None

    def step(self):self.steps+=1


@pytest.fixture
def fixture_backend(monkeypatch):
    monkeypatch.setattr(original_joint,'make_network',lambda *args:FixedNetwork())
    monkeypatch.setattr(torch.optim,'AdamW',NoUpdateOptimizer)
    monkeypatch.setattr(module,'reference_bindings',lambda:[Binding(NoUpdateOptimizer,'__init__','torch_optimizer')])


def arguments(tmp_path):
    training,query=frames()
    repo=Path(__file__).resolve().parents[1]
    spec=yaml.safe_load((repo/'configs/strong_component_regularization/SPEC.yaml').read_text())
    paths=[__file__,module.__file__,original_component.__file__,original_joint.__file__]
    identity=dict(source_directory=str(tmp_path.resolve()),split_seed=271828,fold=0,trial_id='OLD_EMA')
    return training,query,dict(identity=identity,settings=spec['training']['tap_time_len'],
        mechanisms=spec['mechanisms'],source_hashes={str(Path(p).resolve()):sha(p) for p in paths})


def test_original_component_selector_refit_and_independent_cold_state_are_preserved(tmp_path,fixture_backend):
    training,query,kwargs=arguments(tmp_path);out=tmp_path/'model'
    original_fit=ComponentRegressor.fit;original_train=ComponentRegressor._train
    original_save=ComponentRegressor.save;original_initialize=original_joint.JointRegressor._initialize
    prediction,receipt=fit_component(out,training,query,**kwargs)
    assert ComponentRegressor.fit is original_fit and ComponentRegressor._train is original_train
    assert ComponentRegressor.save is original_save and original_joint.JointRegressor._initialize is original_initialize
    assert receipt['initializations']==2 and receipt['native_counts']['torch_optimizer']==2
    assert receipt['model_metadata']['selected_epoch']==1 and receipt['checkpoints']['refit']['trace']['selected_epoch']==1
    assert receipt['checkpoints']['selection']['trace']['stopped_epoch']==26
    for phase in ('selection','refit'):
        saved=torch.load(out/(phase+'.pt'),weights_only=True)
        torch.testing.assert_close(saved['state']['value'],torch.tensor(.1),rtol=0,atol=1e-7)
    first=json.loads((out/'native/call-0001/start.json').read_text())['partition']
    second=json.loads((out/'native/call-0002/start.json').read_text())['partition']
    assert set(first['training_ids'])|set(first['calibration_ids'])==set(training.sample_id)
    assert second['training_ids']==training.sample_id.tolist() and second['calibration_ids']==[]
    np.testing.assert_array_equal(np.load(out/'predictions.npz')['prediction'][:,0],prediction)
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH','')
    code='''import json,sys,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from bf_tap_r2.ema_span_confirmation_models import audit_component
print(json.dumps(audit_component(sys.argv[1],sys.argv[2])))
'''
    cold=subprocess.run([sys.executable,'-c',code,str(out),sha(out/'complete.json')],env=env,
        capture_output=True,text=True,check=True)
    report=json.loads(cold.stdout)
    assert report['status']=='passed' and report['retained_states']==2 and report['cold_pid']!=os.getpid()
    assert all(all(v==0 for v in a['differences'].values()) for a in report['audits'].values())
    with pytest.raises(FileExistsError):audit_component(out,sha(out/'complete.json'))
    with pytest.raises(FileExistsError):fit_component(out,training,query,**kwargs)


def test_outer_labels_cannot_enter_component_prediction_query(tmp_path,fixture_backend):
    training,query,kwargs=arguments(tmp_path);query['tap_time_len']=99
    with pytest.raises(ValueError,match='outer partition/query'):
        fit_component(tmp_path/'failed',training,query,**kwargs)
    assert (tmp_path/'failed/failure.json').exists()
    assert not (tmp_path/'failed/native').exists()


def test_original_component_failure_restores_all_wrapped_methods(tmp_path,fixture_backend,monkeypatch):
    training,query,kwargs=arguments(tmp_path);save=ComponentRegressor.save
    def fail(self,frame,y):raise RuntimeError('original initialization failure')
    monkeypatch.setattr(original_joint.JointRegressor,'_initialize',fail)
    with pytest.raises(RuntimeError,match='original initialization failure'):
        fit_component(tmp_path/'failed',training,query,**kwargs)
    assert original_joint.JointRegressor._initialize is fail and ComponentRegressor.save is save
    assert (tmp_path/'failed/failure.json').exists()


def test_component_cold_process_and_checkpoint_identity_are_bound(tmp_path,fixture_backend):
    training,query,kwargs=arguments(tmp_path);out=tmp_path/'model'
    fit_component(out,training,query,**kwargs);receipt=sha(out/'complete.json')
    with pytest.raises(ValueError,match='independent process'):audit_component(out,receipt)
    with (out/'selection.pt').open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='checkpoint changed'):audit_component(out,receipt)
    assert not (out/'cold-complete.json').exists()
