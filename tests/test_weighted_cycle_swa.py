import importlib,importlib.util
import pytest

def model_module():
    name='bf_tap_r2.weighted_cycle_swa_model'
    assert importlib.util.find_spec(name) is not None,'Weighted cycle loss not implemented'
    return importlib.import_module(name)

def test_time_weighted_loss_is_per_head_and_target_order_is_frozen():
    torch=pytest.importorskip('torch');m=model_module()
    pred=torch.tensor([[[2.,1.],[2.,1.]]]);target=torch.zeros((1,2))
    assert m.time_weighted_joint_loss(pred,target).item()==pytest.approx(1.75)
    pred=torch.tensor([[[1.,1.],[-1.,-1.]]])
    assert m.time_weighted_joint_loss(pred,target).item()==pytest.approx(1.)

def test_time_gradient_is_three_times_iron_for_equal_residual():
    torch=pytest.importorskip('torch');m=model_module();pred=torch.ones((2,3,2),requires_grad=True)
    m.time_weighted_joint_loss(pred,torch.zeros((2,2))).backward()
    torch.testing.assert_close(pred.grad[:,:,1],3*pred.grad[:,:,0])

def test_loss_rejects_wrong_output_shape():
    torch=pytest.importorskip('torch');m=model_module()
    with pytest.raises(ValueError,match='two'):m.time_weighted_joint_loss(torch.ones((2,3,1)),torch.ones((2,1)))

def test_control_is_original_cycle_recipe_candidate_has_fixed_weights():
    name='bf_tap_r2.weighted_cycle_swa_protocol'
    assert importlib.util.find_spec(name) is not None,'Weighted recipe protocol not implemented'
    p=importlib.import_module(name)
    from bf_tap_r2.cycle_swa_protocol import MECHANISMS as old
    assert p.CONTROL_MECHANISMS==old
    assert p.MECHANISMS==dict(old,loss_target_weights=[.25,.75])
    assert p.CANDIDATE=='SWA_CYCLE_TIME_WEIGHT'
    assert p.CONTROL=='SWA_CYCLE_WEIGHT_CONTROL'