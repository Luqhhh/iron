from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import PRED, forbid_fit
from bf_tap.optimization.inverse_rate_model import InverseRateModel
from bf_tap.optimization.residual_stack import (CANDIDATE, RATIOS, CONTEXT, CERT,
    FEATURES, feature_frame, select_training, ResidualFitBudget, ResidualModel,
    predict, acceptance, registration)


def data():
    p=pd.DataFrame([['a',100.,20.,4.,.1],['b',90.,30.,3.,0.]],columns=['sample_id',*PRED,*RATIOS])
    c=pd.DataFrame([['a','1',1000.,120.],['b','2',1200.,110.]],columns=['sample_id',*CONTEXT])
    return p,c


def test_meta_directions_use_base_and_context_aligns_by_ID():
    p,c=data();x=feature_frame(p,c.iloc[::-1])
    np.testing.assert_array_equal(x.direction_iron,[-20.,0.])
    np.testing.assert_array_equal(x.direction_time,[-10.,-30.])
    assert list(x)==FEATURES and x.spout_no.tolist()==['1','2']
    pd.testing.assert_frame_equal(x.iloc[::-1].reset_index(drop=True),feature_frame(p.iloc[::-1],c))


@pytest.mark.parametrize('invalid',['label','duplicate','ids','nan','negative','context_inf'])
def test_rejects_meta_label_leakage_and_invalid_identities(invalid):
    p,c=data()
    if invalid=='label':p['tap_iron']=1.
    if invalid=='duplicate':p.loc[1,'sample_id']='a'
    if invalid=='ids':c.loc[0,'sample_id']='unknown'
    if invalid=='nan':p.loc[0,'pred_rate']=np.nan
    if invalid=='negative':p.loc[0,'pred_inverse_rate']=-1.
    if invalid=='context_inf':c.loc[0,CONTEXT[1]]=np.inf
    with pytest.raises(ContractError):feature_frame(p,c)


def oof():
    p,_=data();fold=pd.Timestamp('2024-05-01',tz='Asia/Shanghai')
    p['reference_time']=[fold,fold+pd.Timedelta(days=30)]
    p['label_available_at']=[fold+pd.Timedelta(hours=1),fold+pd.DateOffset(months=1,days=1)]
    p['fold_cutoff']=fold;p['train_reference_max']=fold-pd.Timedelta(days=1)
    p['train_available_max']=fold;p['history_available_max']=fold
    for key,value in zip(CERT,(fold,fold-pd.Timedelta(days=1),fold,fold)):p[key]=value
    p['tap_iron']=[110.,50.];p['tap_time_len']=[25.,5.]
    return p,fold+pd.DateOffset(months=1)


def test_only_prior_available_OOF_labels_enter_residual_training():
    f,cutoff=oof();used=select_training(f,cutoff,1)
    assert used.sample_id.tolist()==['a']
    f.loc[1,'tap_iron']=1e9
    pd.testing.assert_frame_equal(used,select_training(f,cutoff,1))
    f.loc[0,'inverse_train_available_max']=cutoff
    with pytest.raises(ContractError,match='temporal'):select_training(f,cutoff,1)


def test_meta_rejects_base_in_sample_certificate_and_insufficient_data():
    f,cutoff=oof()
    with pytest.raises(ContractError,match='insufficient'):select_training(f,cutoff,500)
    f.loc[0,'train_reference_max']=f.loc[0,'fold_cutoff']
    with pytest.raises(ContractError,match='temporal'):select_training(f,cutoff,1)


def test_actual_residual_fit_target_budget_persistence_and_nofit(tmp_path):
    p,c=data();x=feature_frame(p,c);budget=ResidualFitBudget(1)
    actual=p.pred_tap_iron.to_numpy()+np.array([10.,-40.]);captured={};original=CatBoostRegressor.fit
    def capture(model,X,y,**kwargs):
        captured['y']=y.copy();captured['kwargs']=kwargs
        return original(model,X,y,**kwargs)
    with patch.object(CatBoostRegressor,'fit',capture),budget:
        model=ResidualModel().fit(x,actual,p.pred_tap_iron,'tap_iron',budget)
        with pytest.raises(ContractError,match='existing'):InverseRateModel().fit(None,None,None,None,None)
        with pytest.raises(ContractError,match='existing'):CatBoostRegressor().fit(x,actual)
    np.testing.assert_array_equal(captured['y'],[10.,-40.])
    assert 'sample_weight' not in captured['kwargs'] and budget.completed==1 and budget.forbidden==2
    model.save(tmp_path/'model',{'cutoff':'synthetic'})
    counter={'attempted_target_fits':0}
    with forbid_fit(counter):
        loaded=ResidualModel.load(tmp_path/'model')
        np.testing.assert_array_equal(model.predict_residual(x),loaded.predict_residual(x))
    assert counter['attempted_target_fits']==0
    exhausted=ResidualFitBudget(0)
    with exhausted,pytest.raises(ContractError,match='budget'):ResidualModel().fit(x,actual,p.pred_tap_iron,'tap_iron',exhausted)
    assert exhausted.completed==0


def test_prediction_adds_signed_residual_to_R2_with_only_nonnegative_clip():
    p,c=data()
    class Fake:
        def __init__(self,target,value):self.target,self.value=target,value
        def predict_residual(self,x):return np.full(len(x),self.value)
    models={'tap_iron':Fake('tap_iron',-200.),'tap_time_len':Fake('tap_time_len',5.)}
    out=predict(models,p,c)
    np.testing.assert_array_equal(out.pred_tap_iron,[0.,0.])
    np.testing.assert_array_equal(out.pred_tap_time_len,[25.,35.])


def test_all_fixed_gates_apply_against_V1():
    s={c:dict(J=e,horizons={f'H{h}':dict(mean_loss=e,iron_mean_wmape=e,time_mean_wmape=e) for h in range(1,5)}) for c,e in [('V1',.2),(CANDIDATE,.199)]}
    m={f'O2024{k:02d}_H1':dict(horizon=1,candidates={c:{'overall':{'loss':e}} for c,e in [('V1',.2),(CANDIDATE,.199)]}) for k in range(6,12)}
    for u in ('DEV_LONG','DEV_SHORT'):m[u]=dict(horizon=None,candidates={'V1':{'overall':{'loss':.2}},CANDIDATE:{'overall':{'loss':.199}}})
    r=registration();assert acceptance(m,s,r)['passed']
    bad=deepcopy(s);bad[CANDIDATE]['horizons']['H4']['iron_mean_wmape']=.2004
    g=acceptance(m,bad,r);assert not g['passed'] and not g['checks']['iron']
    bad=deepcopy(s);bad[CANDIDATE]['J']=.1996
    assert not acceptance(m,bad,r)['checks']['J_improvement']
    bad=deepcopy(m)
    for k in (6,7,9):bad[f'O2024{k:02d}_H1']['candidates'][CANDIDATE]['overall']['loss']=.201
    assert not acceptance(bad,s,r)['checks']['H1_origins']
