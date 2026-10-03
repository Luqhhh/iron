import importlib.util
import numpy as np
import pytest

def test_expected_squared_loss_fixed_denominator_and_shared_target_weight():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.head_bootstrap_model import bootstrap_loss
    from bf_tap_r2.v12_joint import joint_loss
    pred=torch.arange(12,dtype=torch.float64).reshape(2,3,2).requires_grad_();y=torch.tensor([[1.,-1.],[2.,3.]],dtype=torch.float64)
    one=torch.ones(2,3,dtype=torch.float64)
    a=bootstrap_loss(pred,y,one);b=joint_loss(pred,y)
    assert a.item()==b.item()
    torch.testing.assert_close(torch.autograd.grad(a,pred,retain_graph=True)[0],torch.autograd.grad(b,pred)[0],rtol=0,atol=0)
    weights=torch.tensor([[0.,1.,2.],[3.,0.,1.]],dtype=torch.float64)
    loss=bootstrap_loss(pred,y,weights);err=pred-y[:,None,:]
    torch.testing.assert_close(loss,(weights[:,:,None]*err.square()).sum()/12,rtol=0,atol=0)
    expected=2*weights[:,:,None]*err/12
    torch.testing.assert_close(torch.autograd.grad(loss,pred)[0],expected,rtol=0,atol=0)

def test_zero_weights_zero_loss_and_gradient_no_renormalization():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.head_bootstrap_model import bootstrap_loss
    p=torch.ones(2,3,2,requires_grad=True);y=torch.zeros(2,2);w=torch.zeros(2,3)
    loss=bootstrap_loss(p,y,w);assert loss.item()==0
    torch.testing.assert_close(torch.autograd.grad(loss,p)[0],torch.zeros_like(p),rtol=0,atol=0)

@pytest.mark.parametrize('bad',[np.ones((2,3))*-1,np.ones((2,3))*.5,np.ones((2,2)),np.full((2,3),np.nan)])
def test_invalid_weights_rejected(bad):
    torch=pytest.importorskip('torch')
    from bf_tap_r2.head_bootstrap_model import bootstrap_loss
    with pytest.raises(ValueError):bootstrap_loss(torch.zeros(2,3,2),torch.zeros(2,2),torch.tensor(bad))

def test_poisson_certificate_replay_and_partition_order_rejection():
    from bf_tap_r2.head_bootstrap_protocol import BootstrapStream
    from bf_tap_r2.head_bootstrap_audit import verify_bootstrap
    ids=np.array(['a','b','c','d','e']);settings={'random_seed':42,'batch_size':2,'k':3}
    stream=BootstrapStream(42);order=np.random.default_rng(42);history=[]
    for epoch in range(1,3):
        before=stream.counts.copy();seq=order.permutation(len(ids))
        for start in range(0,len(seq),2):stream.draw(ids[seq[start:start+2]],3)
        history.append({'epoch':epoch,'bootstrap':stream.certificate(),'updates':3})
    trace={'history':history,'bootstrap_fit_ids_digest':__import__('bf_tap_r2.head_bootstrap_protocol',fromlist=['digest']).digest(ids.tolist())}
    verify_bootstrap(trace,ids,settings)
    with pytest.raises(ValueError):verify_bootstrap(trace,ids[::-1],settings)
    history[-1]['bootstrap']['weight_sum']+=1
    with pytest.raises(ValueError):verify_bootstrap(trace,ids,settings)

def test_bootstrap_rng_does_not_modify_global_numpy_state():
    from bf_tap_r2.head_bootstrap_protocol import BootstrapStream
    np.random.seed(17);before=np.random.get_state();s=BootstrapStream(42);w=s.draw(['a','b'],16);after=np.random.get_state()
    assert w.shape==(2,16) and w.dtype==np.int64
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    t=BootstrapStream(42);np.testing.assert_array_equal(w,t.draw(['a','b'],16))

def test_only_one_eligible_target_confirmation_and_formal_gate():
    from bf_tap_r2.head_bootstrap_protocol import decide,select_target
    a=decide([.003,.002],[.001,.001],False);b=decide([.004,.003],[.001,.001],False)
    assert select_target({'tap_time_len':a,'tap_iron':b})=='tap_iron'
    assert not decide([.003,.002,-.001,.002],[0.,0.,0.,0.],True)['formal_promoted']
