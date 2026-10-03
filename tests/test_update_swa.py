import numpy as np
import pytest

def api():
    from bf_tap_r2 import update_swa_audit as a
    return a

def payload(torch):
    states=[{"w":torch.tensor([float(i)])} for i in range(1,71)]
    trace={"history":[{"epoch":i,"updates":7,"cumulative_updates":7*i} for i in range(1,11)],"selected_epoch":10}
    saved={"state":{"w":torch.tensor([35.5])},"trace":trace}
    witness={"updates":list(range(1,71)),"states":states,"window_updates":70,"selected_epoch":10,"selected_update":70}
    return saved,witness

def test_update_window_independent_mean():
    torch=pytest.importorskip("torch");s,w=payload(torch)
    assert api().verify_update_window(s,w)==0
    w["states"][0]["w"]+=1
    with pytest.raises(AssertionError):api().verify_update_window(s,w)

def test_update_number_and_available_prefix():
    torch=pytest.importorskip("torch");s,w=payload(torch)
    w["updates"][0]=0
    with pytest.raises(ValueError):api().verify_update_window(s,w)
    s,w=payload(torch);s["trace"]["selected_epoch"]=1;s["state"]["w"]=torch.tensor([4.])
    w.update(updates=list(range(1,8)),states=w["states"][:7],selected_epoch=1,selected_update=7)
    assert api().verify_update_window(s,w)==0

def test_candidate_control_and_scientific_gate():
    from bf_tap_r2.update_swa_protocol import expected_mechanisms,CANDIDATE,CONTROL,decide
    assert expected_mechanisms(CANDIDATE)=={"uniform_update_window":70}
    assert expected_mechanisms(CONTROL)=={"uniform_epoch_window":10}
    assert not decide([.001,.002],[.002,.003],False)["selected_for_confirmation"]
    assert decide([.004,.005],[.001,.002],False)["selected_for_confirmation"]==[CANDIDATE]

def test_fixed_update_count_and_lr_objective():
    torch=pytest.importorskip("torch")
    from bf_tap_r2.update_swa_audit import verify_update_trace
    settings={"batch_size":256,"learning_rate":.001}
    trace={"fit_rows":1762,"history":[{"epoch":1,"updates":7,"cumulative_updates":7,"learning_rate":.001,"training_objective":"joint_standardized_MSE_per_head"}]}
    verify_update_trace(trace,settings)
    trace["history"][0]["cumulative_updates"]=8
    with pytest.raises(ValueError):verify_update_trace(trace,settings)

def test_window_rejects_nan_and_changing_integer_buffer():
    torch=pytest.importorskip("torch")
    from bf_tap_r2.tabm_swa_window import EpochWindow
    w=EpochWindow(70);w.append(1,{"w":torch.tensor([1.]),"i":torch.tensor(1)})
    with pytest.raises(ValueError):w.append(2,{"w":torch.tensor([float('nan')]),"i":torch.tensor(1)})
    with pytest.raises(ValueError):w.append(2,{"w":torch.tensor([2.]),"i":torch.tensor(2)})
