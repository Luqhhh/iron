import importlib,importlib.util
import pytest

def model_module():
    name='bf_tap_r2.mae_swa_model'
    assert importlib.util.find_spec(name) is not None,'SWA time MAE loss not implemented'
    return importlib.import_module(name)

def test_loss_preserves_iron_mse_and_time_mae_per_head():
    torch=pytest.importorskip('torch');m=model_module()
    p=torch.tensor([[[2.,3.],[2.,3.]]]);y=torch.zeros((1,2))
    assert m.time_mae_joint_loss(p,y).item()==pytest.approx(3.5)
    # Averaging heads before loss would spuriously yield zero.
    p=torch.tensor([[[2.,3.],[-2.,-3.]]])
    assert m.time_mae_joint_loss(p,y).item()==pytest.approx(3.5)

def test_time_gradient_is_bounded_iron_gradient_remains_mse():
    torch=pytest.importorskip('torch');m=model_module()
    p=torch.tensor([[[2.,3.],[2.,30.]]],requires_grad=True)
    m.time_mae_joint_loss(p,torch.zeros((1,2))).backward()
    torch.testing.assert_close(p.grad[:,:,0],torch.ones((1,2)))
    torch.testing.assert_close(p.grad[:,:,1],torch.full((1,2),.25))

def test_loss_rejects_malformed_or_nonfinite_values():
    torch=pytest.importorskip('torch');m=model_module()
    with pytest.raises(ValueError):m.time_mae_joint_loss(torch.ones((2,3,1)),torch.ones((2,1)))
    with pytest.raises(ValueError):m.time_mae_joint_loss(torch.ones((2,3,2)),torch.ones((3,2)))
    with pytest.raises(ValueError):m.time_mae_joint_loss(torch.full((2,3,2),float('nan')),torch.ones((2,2)))

def test_protocol_reuses_original_constant_lr_joint_swa_not_cycle():
    name='bf_tap_r2.mae_swa_protocol'
    assert importlib.util.find_spec(name) is not None,'MAE SWA protocol not implemented'
    m=importlib.import_module(name)
    assert m.CANDIDATE=='SWA_TIME_MAE' and m.CONTROL=='SWA_MSE_CONTROL'
    assert m.expected_mechanisms(m.CONTROL)=={'uniform_epoch_window':10}
    assert m.expected_mechanisms(m.CANDIDATE)=={'uniform_epoch_window':10,'time_loss':'mae','iron_loss':'mse','loss_coefficients':[.5,.5]}
    with pytest.raises(ValueError):m.expected_mechanisms('SWA_CYCLE_TAIL')

def test_development_gate_requires_positive_mechanism_even_when_seed_gains_positive():
    name='bf_tap_r2.mae_swa_protocol'
    assert importlib.util.find_spec(name) is not None,'MAE SWA protocol not implemented'
    m=importlib.import_module(name)
    d=m.decide([.002,.003],[.004,.005],False)
    assert d['selected_for_confirmation']==[] and d['failure_reasons']==['nonpositive_mechanism_advantage']
    assert m.decide([.004,.005],[.002,.003],False)['selected_for_confirmation']==['SWA_TIME_MAE']