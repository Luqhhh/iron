from copy import deepcopy

import numpy as np
import pytest
import torch

from test_round2_v12 import sample
from bf_tap_r2.component_regularization import ComponentRegressor, clone_state
from bf_tap_r2.data import TARGETS
from bf_tap_r2.sam_ema import SAMEMARegressor, sam_ema_step
from bf_tap_r2.sam_ema_audit import verify_sam_ema_saved
from bf_tap_r2.v3_4_bags import group_safe_inner_folds

RECIPE = {'backbone':'tabm','frequency':.01}
MECH = {'ema_beta':.99,'sam_rho':.05,'sam_epsilon':1e-12}


def test_average_uses_restored_adamw_update_and_failure_preserves_average():
    model = torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad(): model.weight.fill_(2.)
    optimizer = torch.optim.SGD(model.parameters(),lr=.1)
    ema = clone_state(model)
    sam_ema_step(model,optimizer,lambda:.5*model.weight.square().sum(),ema,dict(MECH,ema_beta=.5))
    assert model.weight.item()==pytest.approx(1.795,abs=1e-12)
    assert ema['weight'].item()==pytest.approx(1.8975,abs=1e-12)
    before = clone_state(model); average = deepcopy(ema); calls = 0

    def fail():
        nonlocal calls
        calls += 1
        if calls==2: raise RuntimeError('second SAM loss failed')
        return model.weight.square().sum()

    with pytest.raises(RuntimeError): sam_ema_step(model,optimizer,fail,ema,MECH)
    assert torch.equal(model.weight,before['weight'])
    assert torch.equal(ema['weight'],average['weight'])


@pytest.mark.parametrize('outputs',[1,2])
@pytest.mark.parametrize('arm,mechanisms',[
    ('EMA',dict(MECH,sam_rho=0.)),
    ('SAM',dict(MECH,ema_beta=0.)),
])
def test_limiting_recipe_reproduces_native_trajectory_epoch_state_and_rng(outputs,arm,mechanisms):
    frame,settings = sample(); y = frame[['tap_time_len'] if outputs==1 else list(TARGETS)].to_numpy()
    native = ComponentRegressor(RECIPE,settings,arm,mechanisms).fit(frame,y)
    rng = torch.get_rng_state()
    composed = SAMEMARegressor(RECIPE,settings,'SAM_EMA',mechanisms).fit(frame,y)
    assert torch.equal(rng,torch.get_rng_state())
    query = frame.drop(columns=list(TARGETS))
    np.testing.assert_array_equal(native.predict(query),composed.predict(query))
    assert native.metadata_['selected_epoch']==composed.metadata_['selected_epoch']
    for key,value in native.model_.state_dict().items(): assert torch.equal(value,composed.model_.state_dict()[key])
    for phase,trace in composed.traces.items():
        assert trace['selected_epoch']==native.traces[phase]['selected_epoch']
        assert trace['stopped_epoch']==native.traces[phase]['stopped_epoch']
        assert trace['gradient_evaluations']==2*trace['updates']
        assert trace['ema_updates']==trace['updates']
        for a,b in zip(trace['history'],native.traces[phase]['history']):
            for key in ('training_eval_mae','training_eval_mse','mean_batch_training_loss'):
                assert a[key]==b[key]
            if phase=='selection': assert a['validation_mae']==b['validation_mae']


def test_independent_saved_audit_checks_training_preprocessing_selection_and_update_order(tmp_path):
    frame,settings = sample(); y = frame[['tap_time_len']].to_numpy()
    model = SAMEMARegressor(RECIPE,settings,'SAM_EMA',MECH,tmp_path).fit(frame,y)
    inner = np.asarray(group_safe_inner_folds(frame,seed=settings['inner_seed'])['fold'])
    fitting = frame.loc[inner!=0].reset_index(drop=True); calibration = frame.loc[inner==0]
    selector = verify_sam_ema_saved(tmp_path/'selection.pt',fitting,fitting[['tap_time_len']].to_numpy(),
        'SAM_EMA',settings,MECH,calibration)
    cold = verify_sam_ema_saved(tmp_path/'refit.pt',frame,y,'SAM_EMA',settings,MECH,
        expected_epoch=selector.saved['trace']['selected_epoch'])
    query = frame.drop(columns=list(TARGETS))
    np.testing.assert_array_equal(cold.predict(query),model.predict(query))
    with pytest.raises(ValueError,match='targets'): cold.predict(frame)
    saved = torch.load(tmp_path/'refit.pt',weights_only=True)
    saved['trace']['history'][0]['ema_updates'] += 1
    torch.save(saved,tmp_path/'tampered.pt')
    with pytest.raises(ValueError,match='EMA'): verify_sam_ema_saved(tmp_path/'tampered.pt',frame,y,'SAM_EMA',settings,MECH)
    with pytest.raises(FileExistsError): model.save(tmp_path/'refit.pt',model.traces['refit'])
