import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.optimization.structural import INPUT, directions,lad_coefficient,fit_correction,select_oof,apply_correction
from bf_tap.optimization.rate_model import RateModel
from bf_tap.models.baseline import FROZEN_PARAMETERS


def predictions():
    return pd.DataFrame([['a',100.,20.,10.],['b',90.,30.,0.]],columns=INPUT)


def test_structural_simultaneous_correction_and_invalid_rate_fallback():
    p=predictions();d,valid=directions(p)
    np.testing.assert_array_equal(d,[[100.,-10.],[0.,0.]])
    out=apply_correction(p,[.5,.25])
    np.testing.assert_array_equal(out.iloc[:,1:].to_numpy(),[[150.,17.5],[90.,30.]])
    pd.testing.assert_frame_equal(out.set_index('sample_id').sort_index(),apply_correction(p.iloc[::-1],[.5,.25]).set_index('sample_id').sort_index())
    np.testing.assert_array_equal(apply_correction(p,[0.,0.]).iloc[:,1:],p.iloc[:,1:3])


def test_rate_floor_is_strict_and_labels_rejected():
    p=predictions();p.pred_rate=1e-6
    assert not directions(p)[1].any()
    p['tap_iron']=100
    with pytest.raises(ContractError,match='prediction-only'):directions(p)


@pytest.mark.parametrize('alpha',[[1.01,0],[-.1,1],[np.nan,0],[.5]])
def test_invalid_coefficient_rejected(alpha):
    with pytest.raises(ContractError):apply_correction(predictions(),alpha)


def test_LAD_exact_optimum_with_signed_directions_and_clipping():
    base=np.array([10.,20.,30.,40.]);d=np.array([-4.,3.,2.,8.]);y=base+d*np.array([.2,.5,.8,.7])
    alpha=lad_coefficient(y,base,d)
    loss=lambda a:np.abs(y-base-a*d).sum()
    assert all(loss(alpha)<=loss(a)+1e-12 for a in np.linspace(0,1,101))
    assert lad_coefficient([4],[0],[2])==1.
    assert lad_coefficient([-4],[0],[2])==0.
    assert lad_coefficient([2,5],[0,0],[0,0])==0.
    assert lad_coefficient([.2,.8],[0,0],[1,1])==.2


def oof_fixture():
    t=pd.Timestamp('2024-05-01',tz='Asia/Shanghai')
    p=predictions();p['reference_time']=[t,t+pd.Timedelta(days=30)]
    p['label_available_at']=[t+pd.Timedelta(minutes=20),t+pd.Timedelta(days=32)]
    p['fold_cutoff']=t;p['train_reference_max']=t-pd.Timedelta(days=1)
    p['train_available_max']=t-pd.Timedelta(hours=1);p['history_available_max']=t
    p['tap_iron']=[180.,999.];p['tap_time_len']=[15.,999.]
    return p,t+pd.DateOffset(months=1)


def test_completion_filter_and_future_labels_do_not_change_coefficients():
    p,c=oof_fixture();alpha,used=fit_correction(p,c,minimum=1)
    assert used.sample_id.tolist()==['a'];np.testing.assert_allclose(alpha,[.8,.5])
    p.loc[1,['tap_iron','tap_time_len']]=1e9
    assert fit_correction(p,c,minimum=1)[0]==alpha


@pytest.mark.parametrize('field',['train_reference_max','train_available_max','history_available_max','fold_cutoff'])
def test_temporal_leakage_rejected(field):
    p,c=oof_fixture();p.loc[0,field]=c
    with pytest.raises(ContractError,match='temporal leakage'):select_oof(p,c,minimum=1)


def test_duplicate_and_insufficient_OOF_rejected():
    p,c=oof_fixture();p.loc[1,'sample_id']='a'
    with pytest.raises(ContractError,match='duplicate'):select_oof(p,c,minimum=1)
    p,c=oof_fixture()
    with pytest.raises(ContractError,match='insufficient'):select_oof(p,c,minimum=2)


def test_rate_model_zero_duration_roundtrip_and_schema(tmp_path):
    X=pd.DataFrame({'spout_no':['1']*16,'feature':np.arange(16,dtype=float)})
    iron=np.arange(16,dtype=float)+100;time=np.arange(16,dtype=float)
    m=RateModel().fit(X,iron,time,FROZEN_PARAMETERS)
    assert m.excluded_zero_duration==1 and m.training_rows==15
    p=m.predict(X)
    m.save(tmp_path/'rate',dict(parameters=FROZEN_PARAMETERS),pd.DataFrame({'sample_id':['base']}))
    loaded=RateModel.load(tmp_path/'rate')
    np.testing.assert_array_equal(p,loaded.predict(X))
    with pytest.raises(ContractError,match='schema'):loaded.predict(X[['feature','spout_no']])
    with pytest.raises(ContractError,match='positive-duration'):RateModel().fit(X,iron,np.zeros(16),FROZEN_PARAMETERS)
