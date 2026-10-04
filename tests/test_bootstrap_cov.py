import numpy as np
import pytest


def test_identity_precision_reduces_to_original_online_bootstrap_loss_and_gradient():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.bootstrap_cov_model import bootstrap_covariance_loss
    from bf_tap_r2.head_bootstrap_model import bootstrap_loss
    p=torch.arange(12,dtype=torch.float64).reshape(2,3,2).requires_grad_();y=torch.tensor([[1.,-1.],[2.,3.]],dtype=torch.float64)
    w=torch.tensor([[0.,1.,2.],[3.,0.,1.]],dtype=torch.float64)
    a=bootstrap_covariance_loss(p,y,w,torch.eye(2,dtype=torch.float64));b=bootstrap_loss(p,y,w)
    torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(a,p,retain_graph=True)[0],torch.autograd.grad(b,p)[0],rtol=1e-14,atol=1e-14)


def test_all_one_weights_reduce_to_original_covariance_loss():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.bootstrap_cov_model import bootstrap_covariance_loss
    from bf_tap_r2.joint_cov_model import covariance_loss
    p=torch.tensor([[[2.,4.],[3.,5.]],[[7.,1.],[9.,2.]]],dtype=torch.float64,requires_grad=True);y=torch.tensor([[1.,2.],[4.,3.]],dtype=torch.float64)
    c=torch.tensor([[1.2,-.3],[-.3,.9]],dtype=torch.float64)
    a=bootstrap_covariance_loss(p,y,torch.ones(2,2,dtype=torch.float64),c);b=covariance_loss(p,y,c)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(a,p,retain_graph=True)[0],torch.autograd.grad(b,p)[0],rtol=0,atol=0)


def test_non_diagonal_metric_fixed_denominator_gradient_and_zero_batch():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.bootstrap_cov_model import bootstrap_covariance_loss
    p=torch.arange(12,dtype=torch.float64).reshape(2,3,2).requires_grad_();y=torch.tensor([[1.,2.],[3.,5.]],dtype=torch.float64)
    w=torch.tensor([[0.,1.,2.],[3.,0.,1.]],dtype=torch.float64);precision=torch.tensor([[1.4,.2],[.2,.8]],dtype=torch.float64)
    err=p-y[:,None,:];expected=(w*(1.4*err[:,:,0]**2+.4*err[:,:,0]*err[:,:,1]+.8*err[:,:,1]**2)).sum()/12
    value=bootstrap_covariance_loss(p,y,w,precision);torch.testing.assert_close(value,expected,rtol=1e-14,atol=1e-14)
    gradient=w[:,:,None]*(err@precision)/(2*3)
    torch.testing.assert_close(torch.autograd.grad(value,p)[0],gradient,rtol=1e-14,atol=1e-14)
    zero=bootstrap_covariance_loss(p,y,torch.zeros_like(w),precision);assert zero.item()==0
    torch.testing.assert_close(torch.autograd.grad(zero,p)[0],torch.zeros_like(p),rtol=0,atol=0)


@pytest.mark.parametrize('precision',[np.array([[1.,2.],[0.,1.]]),np.array([[1.,2.],[2.,1.]]),np.full((2,2),np.nan)])
def test_invalid_precision_rejected(precision):
    torch=pytest.importorskip('torch')
    from bf_tap_r2.bootstrap_cov_model import bootstrap_covariance_loss
    with pytest.raises(ValueError):bootstrap_covariance_loss(torch.zeros(2,3,2),torch.zeros(2,2),torch.ones(2,3),torch.tensor(precision))


def test_fractional_or_negative_weights_rejected():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.bootstrap_cov_model import bootstrap_covariance_loss
    for w in [torch.ones(2,3)*.5,-torch.ones(2,3)]:
        with pytest.raises(ValueError):bootstrap_covariance_loss(torch.zeros(2,3,2),torch.zeros(2,2),w,torch.eye(2))


def test_actual_fit_covariance_and_online_certificate_replay_reject_independent_tampering():
    from bf_tap_r2.bootstrap_cov_protocol import covariance_metric,BootstrapStream,digest
    from bf_tap_r2.bootstrap_cov_audit import verify_covariance,verify_bootstrap
    import copy
    ids=np.array(['a','b','c','d','e']);y=np.array([[1.,2.],[3.,1.],[4.,6.],[7.,3.],[9.,8.]])
    c,p=covariance_metric(y);trace={'covariance':c.tolist(),'precision':p.tolist(),'covariance_fit_ids_digest':digest(ids.tolist()),'bootstrap_fit_ids_digest':digest(ids.tolist()),'history':[]}
    stream=BootstrapStream(42);order=np.random.default_rng(42);settings={'random_seed':42,'batch_size':2,'k':3}
    for epoch in range(1,3):
        ix=order.permutation(len(ids))
        for start in range(0,len(ids),2):stream.draw(ids[ix[start:start+2]],3)
        trace['history'].append({'epoch':epoch,'bootstrap':stream.certificate(),'updates':3})
    verify_covariance(trace,y,ids);verify_bootstrap(trace,ids,settings)
    bad=copy.deepcopy(trace);bad['precision'][0][0]+=.1
    with pytest.raises(AssertionError):verify_covariance(bad,y,ids)
    bad=copy.deepcopy(trace);bad['history'][-1]['bootstrap']['weight_sum']+=1
    with pytest.raises(ValueError):verify_bootstrap(bad,ids,settings)
    with pytest.raises(ValueError):verify_covariance(trace,y,ids[::-1])


def test_online_control_identity_and_time_only_confirmation_gate():
    from bf_tap_r2.bootstrap_cov_protocol import CANDIDATE,CONTROL,expected_mechanisms,select_target,decide
    control=expected_mechanisms(CONTROL,'development');candidate=expected_mechanisms(CANDIDATE,'development')
    assert control=={'bootstrap_rate':1.,'bootstrap_seed_offset':1000003,'loss':'fixed_denominator_per_row_head_poisson_mse'}
    assert candidate['joint_cov_shrinkage']==.5
    time=decide([.003,.002],[.001,.001],False);iron=decide([.03,.02],[.001,.001],False)
    assert select_target({'tap_time_len':time,'tap_iron':iron})=='tap_time_len'
    assert select_target({'tap_time_len':decide([.003,.002],[.004,.004],False),'tap_iron':iron}) is None
