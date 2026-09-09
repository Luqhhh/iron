import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from bf_tap.exceptions import ContractError
from bf_tap.models.baseline import FROZEN_PARAMETERS
from bf_tap.optimization.rate_model import RateModel
from bf_tap.optimization.inverse_rate_model import InverseRateModel,InverseFitBudget
from bf_tap.optimization.inverse_oof import fit_certified_time


def test_inverse_training_target_weights_exclusion_roundtrip_and_budget(tmp_path,monkeypatch):
    X=pd.DataFrame({'spout_no':['1']*16,'feature':np.arange(16,dtype=float)})
    iron=np.arange(16,dtype=float);time=iron+20;captured={};original=CatBoostRegressor.fit
    def capture(self,X,y,**kwargs):
        captured['y']=np.array(y);captured['weights']=np.array(kwargs['sample_weight'])
        return original(self,X,y,**kwargs)
    monkeypatch.setattr(CatBoostRegressor,'fit',capture)
    with InverseFitBudget(1) as budget:
        model=InverseRateModel().fit(X,iron,time,FROZEN_PARAMETERS,budget)
        assert model.excluded_zero_iron==1 and model.training_rows==15
        np.testing.assert_array_equal(captured['y'],time[1:]/iron[1:])
        np.testing.assert_array_equal(captured['weights'],iron[1:]/iron[1:].mean())
        p=model.predict(X);model.save(tmp_path/'q',{'parameters':FROZEN_PARAMETERS},pd.DataFrame({'sample_id':['x']}))
        np.testing.assert_array_equal(p,InverseRateModel.load(tmp_path/'q').predict(X))
        with pytest.raises(ContractError,match='existing'):RateModel().fit(X,iron,time,FROZEN_PARAMETERS)
        with pytest.raises(ContractError,match='budget exceeded'):InverseRateModel().fit(X,iron,time,FROZEN_PARAMETERS,budget)
    assert budget.completed==1 and budget.attempted==2 and budget.forbidden==1


def test_zero_iron_only_rejected_without_fitting():
    X=pd.DataFrame({'spout_no':['1','1'],'feature':[0.,1.]})
    with InverseFitBudget(1) as budget:
        with pytest.raises(ContractError,match='positive-iron'):InverseRateModel().fit(X,[0,0],[10,20],FROZEN_PARAMETERS,budget)
    assert budget.completed==budget.attempted==0


def test_inverse_certificate_rejects_q_trained_after_fold():
    t=pd.Timestamp('2024-05-01',tz='Asia/Shanghai')
    oof=pd.DataFrame(dict(sample_id=['a'],pred_tap_iron=[100.],pred_tap_time_len=[20.],pred_rate=[5.],pred_inverse_rate=[.1],
        reference_time=[t],label_available_at=[t+pd.Timedelta(minutes=20)],fold_cutoff=[t],
        train_reference_max=[t-pd.Timedelta(days=1)],train_available_max=[t],history_available_max=[t],tap_iron=[100.],tap_time_len=[12.],
        inverse_fit_cutoff=[t],inverse_train_reference_max=[t-pd.Timedelta(days=1)],inverse_train_available_max=[t],inverse_history_available_max=[t]))
    beta,_=fit_certified_time(oof,t+pd.DateOffset(months=1),1);assert beta==.8
    oof.loc[0,'inverse_train_available_max']=t+pd.Timedelta(seconds=1)
    with pytest.raises(ContractError,match='inverse OOF temporal leakage'):fit_certified_time(oof,t+pd.DateOffset(months=1),1)
