import importlib.util
import numpy as np
import pytest

def api():
    from bf_tap_r2.joint_cov_protocol import covariance_metric
    return covariance_metric

def test_half_shrinkage_independent_analytic_inverse():
    y=np.array([[-3.,-1.],[-1.,-2.],[1.,2.],[3.,1.]])
    c,p=api()(y)
    z=(y-y.mean(0))/y.std(0);rho=float(np.mean(z[:,0]*z[:,1]))
    expected=np.array([[1.,.5*rho],[.5*rho,1.]])
    np.testing.assert_allclose(c,expected,atol=1e-14)
    v=.5*rho
    np.testing.assert_allclose(p,np.array([[1.,-v],[-v,1.]])/(1-v*v),atol=1e-14)

def test_metric_ignores_origin_and_scale_and_bounds_collinear():
    y=np.array([[-2.,-4.],[-1.,-2.],[1.,2.],[2.,4.]])
    c,p=api()(y);d,q=api()(y*np.array([30.,2.])+[10.,-4.])
    np.testing.assert_allclose(c,d,atol=1e-14);np.testing.assert_allclose(p,q,atol=1e-14)
    assert np.linalg.eigvalsh(c).min()>=.5-1e-14
    assert np.linalg.eigvalsh(p).max()<=2+1e-14

@pytest.mark.parametrize('bad',[np.ones((3,2)),np.zeros((0,2)),np.ones((2,3)),[[1.,2.],[3.,float('nan')]]])
def test_invalid_covariance_targets_fail(bad):
    with pytest.raises(ValueError):api()(bad)

def test_uncorrelated_loss_and_gradient_match_native_mse():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.joint_cov_model import covariance_loss
    from bf_tap_r2.v12_joint import joint_loss
    y=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    c,p=api()(y);np.testing.assert_array_equal(p,np.eye(2))
    pred=torch.arange(24,dtype=torch.float64).reshape(4,3,2).requires_grad_();target=torch.tensor(y)
    a=covariance_loss(pred,target,torch.tensor(p));b=joint_loss(pred,target)
    assert a.item()==b.item()
    ga=torch.autograd.grad(a,pred,retain_graph=True)[0];gb=torch.autograd.grad(b,pred)[0]
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)

def test_correlated_loss_matches_expanded_quadratic_and_gradient():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.joint_cov_model import covariance_loss
    pred=torch.tensor([[[2.,-3.],[1.,4.]]],dtype=torch.float64,requires_grad=True);target=torch.tensor([[1.,2.]])
    p=torch.tensor([[1.2,-.2],[-.2,1.2]],dtype=torch.float64);r=pred-target[:,None,:]
    expected=.5*(1.2*r[:,:,0]**2-.4*r[:,:,0]*r[:,:,1]+1.2*r[:,:,1]**2).mean()
    actual=covariance_loss(pred,target,p);torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,pred,retain_graph=True)[0],torch.autograd.grad(expected,pred)[0])

def test_loss_rejects_nonpositive_or_wrong_precision():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.joint_cov_model import covariance_loss
    for p in [torch.ones(3,3),torch.tensor([[1.,2.],[2.,1.]]),torch.tensor([[1.,float('nan')],[0.,1.]])]:
        with pytest.raises(ValueError):covariance_loss(torch.zeros(2,3,2),torch.zeros(2,2),p)

def test_pool_selects_only_one_target_and_keeps_formal_gate():
    from bf_tap_r2.joint_cov_protocol import select_target
    rows={'tap_iron':{'mean_gain':.002,'selected_for_confirmation':['JOINT_COV_MSE']},'tap_time_len':{'mean_gain':.003,'selected_for_confirmation':['JOINT_COV_MSE']}}
    assert select_target(rows)=='tap_time_len'
    rows['tap_time_len']['selected_for_confirmation']=[]
    assert select_target(rows)=='tap_iron'
    rows['tap_iron']['selected_for_confirmation']=[]
    assert select_target(rows) is None


def test_independent_certificate_rejects_calibration_covariance_and_id_swap():
    from bf_tap_r2.joint_cov_audit import verify_covariance
    from bf_tap_r2.joint_cov_protocol import digest
    y=np.array([[-2.,1.],[-1.,-2.],[1.,2.],[2.,-1.]])
    ids=np.array(['a','b','c','d']);c,p=api()(y)
    trace={'covariance':c.tolist(),'precision':p.tolist(),'covariance_fit_ids_digest':digest(ids.tolist())}
    verify_covariance(trace,y,ids)
    with pytest.raises(ValueError):verify_covariance(trace,y,ids[::-1])
    with pytest.raises(AssertionError):verify_covariance(trace,np.column_stack([y[:,0],y[:,0]]),ids)

def test_candidate_prediction_rejects_query_labels_before_model_access():
    pytest.importorskip('torch')
    import pandas as pd
    from bf_tap_r2.joint_cov_model import CovarianceRegressor
    obj=object.__new__(CovarianceRegressor)
    with pytest.raises(ValueError,match='targets'):obj.predict(pd.DataFrame({'tap_iron':[1.]}))


def test_development_iron_uses_anchored_native_cache_and_rejects_wrong_fold(tmp_path):
    from bf_tap_r2.joint_cov_run import development_iron
    from bf_tap_r2.joint_cov_protocol import sha,digest
    import json
    d=tmp_path/'local/next-direction-20261001';d.mkdir(parents=True)
    p=d/'parent-iron-seed-42.npy';np.save(p,np.array([100.,200.,300.]))
    ids=np.array(['a','b','c']);folds=np.array([0,1,2])
    (d/'native-iron-reference-r1.json').write_text(json.dumps({'status':'passed_zero_fit_native_parent_iron_binding','current_platform_representative':'EMA_TIME_Q75','columns':{'42':{'file_sha256':sha(p),'rows':3,'fold_digest':digest(folds.tolist())}}}))
    inputs=tmp_path/'local/runs/tabm-target-metric-v1/development-r1';inputs.mkdir(parents=True);np.savez(inputs/'inputs.npz',ids=ids)
    np.testing.assert_array_equal(development_iron(tmp_path,42,ids,folds),[100.,200.,300.])
    with pytest.raises(ValueError):development_iron(tmp_path,42,ids,folds[::-1])
    with pytest.raises(AssertionError):development_iron(tmp_path,42,ids[::-1],folds)
