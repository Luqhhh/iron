import copy,math
import numpy as np
import pytest
from bf_tap_r2.cycle_swa_protocol import epoch_lr,cycle_epochs,verify_cycle_selection,verify_cycle_window

def test_cycle_lr_warmup_endpoints_and_reset():
    assert epoch_lr(80)==.001 and epoch_lr(81)==.001
    assert epoch_lr(100)==.0001 and epoch_lr(101)==.001
    assert epoch_lr(120)==.0001
    assert epoch_lr(90)==pytest.approx(.0001+.0009*(1+math.cos(math.pi*9/19))/2)
    with pytest.raises(ValueError):epoch_lr(0)

def test_sparse_cycle_windows_exclude_warmup_and_keep_latest_five():
    assert cycle_epochs(100)==[100]
    assert cycle_epochs(140)==[100,120,140]
    assert cycle_epochs(240)==[160,180,200,220,240]
    with pytest.raises(ValueError):cycle_epochs(139)

def test_complete_cycle_selection_preserves_earliest_min_delta_winner():
    history=[{'epoch':e,'learning_rate':epoch_lr(e)} for e in range(1,241)]
    for e in range(100,241,20):history[e-1]['validation_mae']=.2
    history[119]['validation_mae']=.2-.000005
    history[139]['validation_mae']=.19
    trace={'history':history,'selected_epoch':140,'stopped_epoch':240}
    assert verify_cycle_selection(trace,{'min_delta':.00001},.19)==140
    bad=copy.deepcopy(trace);bad['history'][100]['learning_rate']=.0001
    with pytest.raises(ValueError,match='rate'):verify_cycle_selection(bad,{'min_delta':.00001},.19)
    bad=copy.deepcopy(trace);bad['history'][78]['validation_mae']=.18
    with pytest.raises(ValueError,match='cycle'):verify_cycle_selection(bad,{'min_delta':.00001},.19)

def test_sparse_window_independent_mean_and_future_epoch_rejection():
    torch=pytest.importorskip('torch')
    saved={'trace':{'selected_epoch':140},'state':{'weight':torch.tensor([2.])}}
    witness={'window':5,'warmup_epochs':80,'cycle_length':20,'selected_epoch':140,'epochs':[100,120,140],'states':[{'weight':torch.tensor([v])} for v in [1.,2.,3.]]}
    assert verify_cycle_window(saved,witness)==0
    witness['epochs'][-1]=160
    with pytest.raises(ValueError,match='window'):verify_cycle_window(saved,witness)
