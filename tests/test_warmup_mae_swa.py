import importlib,importlib.util
import pytest

def model_module():
    name="bf_tap_r2.warmup_mae_swa_model"
    assert importlib.util.find_spec(name) is not None,"Scheduled MAE SWA not implemented"
    return importlib.import_module(name)

def test_fixed_boundary_uses_original_mse_then_time_mae_per_head():
    torch=pytest.importorskip("torch");m=model_module()
    pred=torch.tensor([[[2.,3.],[-2.,-3.]]]);y=torch.zeros((1,2))
    assert m.scheduled_joint_loss(pred,y,40).item()==pytest.approx(6.5)
    assert m.scheduled_joint_loss(pred,y,41).item()==pytest.approx(3.5)
    from bf_tap_r2.v12_joint import joint_loss
    torch.testing.assert_close(m.scheduled_joint_loss(pred,y,1),joint_loss(pred,y),rtol=0,atol=0)

def test_tail_bounds_time_gradient_and_warmup_preserves_mse_gradient():
    torch=pytest.importorskip("torch");m=model_module()
    for epoch,expected in [(40,[[1.5,15.]]),(41,[[.25,.25]])]:
        pred=torch.tensor([[[2.,3.],[2.,30.]]],requires_grad=True)
        m.scheduled_joint_loss(pred,torch.zeros((1,2)),epoch).backward()
        torch.testing.assert_close(pred.grad[:,:,0],torch.ones((1,2)))
        torch.testing.assert_close(pred.grad[:,:,1],torch.tensor(expected))

def test_schedule_rejects_invalid_epoch_shape_and_nonfinite():
    torch=pytest.importorskip("torch");m=model_module()
    for epoch in [0,-1,41.5,True]:
        with pytest.raises(ValueError):m.scheduled_joint_loss(torch.ones((2,3,2)),torch.ones((2,2)),epoch)
    with pytest.raises(ValueError):m.scheduled_joint_loss(torch.ones((2,3,1)),torch.ones((2,1)),41)
    with pytest.raises(ValueError):m.scheduled_joint_loss(torch.full((2,3,2),float("nan")),torch.ones((2,2)),1)

def test_protocol_keeps_uniform10_mse_control_and_fixed_schedule_identity():
    name="bf_tap_r2.warmup_mae_swa_protocol"
    assert importlib.util.find_spec(name) is not None,"Warmup SWA protocol not implemented"
    m=importlib.import_module(name)
    assert m.CANDIDATE=="SWA_TIME_MAE_WARMUP40" and m.CONTROL=="SWA_MSE_CONTROL"
    assert m.expected_mechanisms(m.CONTROL)=={"uniform_epoch_window":10}
    assert m.expected_mechanisms(m.CANDIDATE)=={"uniform_epoch_window":10,"time_loss":"mse_then_mae","iron_loss":"mse","loss_coefficients":[.5,.5],"mse_warmup_epochs":40}
    with pytest.raises(ValueError):m.expected_mechanisms("SWA_TIME_MAE")

def test_complete_development_requires_positive_paired_mechanism():
    name="bf_tap_r2.warmup_mae_swa_protocol"
    assert importlib.util.find_spec(name) is not None,"Warmup SWA protocol not implemented"
    m=importlib.import_module(name)
    assert m.decide([.002,.003],[.004,.005],False)["selected_for_confirmation"]==[]
    assert m.decide([.004,.005],[.002,.003],False)["selected_for_confirmation"]==["SWA_TIME_MAE_WARMUP40"]
