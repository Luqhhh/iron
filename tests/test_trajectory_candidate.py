import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import PRED
from bf_tap.optimization.trajectory_candidate import predict,fit_time_coefficient,TimeModel
from bf_tap.optimization.residual_stack import ResidualFitBudget
from bf_tap.features.trajectory import COLUMNS


def test_frozen_iron_base_direction_and_fallback():
    parts=pd.DataFrame(dict(sample_id=['a','b','c'],pred_tap_iron=[100.,200.,300.],pred_tap_time_len=[4.,5.,6.],pred_rate=[10.,0.,1e-6]))
    original=pd.DataFrame(dict(sample_id=['c','b','a'],pred_tap_iron=[301.,202.,103.],pred_tap_time_len=[90.,80.,70.]))
    out,count=predict(parts,original,.5)
    assert count==2
    assert out.pred_tap_iron.tolist()==[103.,202.,301.]
    assert out.pred_tap_time_len.tolist()==[7.,5.,6.]
    reverse,_=predict(parts.iloc[::-1],original,.5)
    pd.testing.assert_frame_equal(out,reverse.iloc[::-1].reset_index(drop=True))
    for bad in (-.1,1.1,np.nan):
        with pytest.raises(ContractError):predict(parts,original,bad)


def test_only_earlier_available_oof_for_time_lad():
    cutoff=pd.Timestamp('2024-06-01',tz='Asia/Shanghai')
    old=cutoff-pd.Timedelta(days=20);fold=cutoff-pd.Timedelta(days=31)
    oof=pd.DataFrame(dict(sample_id=['a','b','c'],pred_tap_iron=[100.]*3,pred_tap_time_len=[5.]*3,pred_rate=[10.]*3,
        tap_iron=[100.]*3,tap_time_len=[7.,9.,100.],reference_time=[old]*3,label_available_at=[old,old,cutoff+pd.Timedelta(hours=1)],
        fold_cutoff=[fold]*3,train_reference_max=[fold-pd.Timedelta(hours=1)]*3,
        train_available_max=[fold]*3,history_available_max=[fold]*3))
    alpha,used=fit_time_coefficient(oof,cutoff,minimum=2)
    assert alpha==.4 and used.sample_id.tolist()==['a','b']
    oof.loc[0,'train_available_max']=cutoff
    with pytest.raises(ContractError,match='leakage'):fit_time_coefficient(oof,cutoff,minimum=2)


@pytest.mark.model
def test_single_target_fixed_parameter_persistence_and_budget(tmp_path):
    x=pd.DataFrame({c:np.arange(8,dtype=float) for c in COLUMNS})
    x.insert(0,'spout_no',pd.Series(['1','2']*4,dtype='string'))
    with ResidualFitBudget(1) as budget:
        model=TimeModel().fit(x,np.arange(8,dtype=float)+10,budget)
        model.save(tmp_path/'time',dict(synthetic=True))
        loaded=TimeModel.load(tmp_path/'time')
        assert np.array_equal(model.predict(x),loaded.predict(x))
        assert np.array_equal(model.predict(x),loaded.predict(x.iloc[::-1])[::-1])
        assert budget.completed==1
        with pytest.raises(ContractError,match='budget'):TimeModel().fit(x,np.arange(8,dtype=float)+10,budget)
    assert list((tmp_path/'time').glob('*.cbm'))==[tmp_path/'time'/'time.cbm']


def test_candidate_input_alignment_uses_ids_and_original_blend_arithmetic(monkeypatch):
    import bf_tap.optimization.trajectory_run as run
    samples=pd.DataFrame({'sample_id':['b','a']})
    old=pd.DataFrame(dict(sample_id=['a','b'],pred_tap_iron=[100.,200.],pred_tap_time_len=[3.,4.],pred_rate=[10.,20.]))
    monkeypatch.setattr(run,'predict_inputs',lambda *a:old)
    monkeypatch.setattr(run,'new_features',lambda *a:(samples,None))
    class Builder:
        def X(self,*a): return samples
    class Old:
        def predict_raw(self,x):return pd.DataFrame({PRED[1]:[-4.,10.]})
    class New:
        def predict(self,x):return np.array([20.,30.])
    fold=({'HR':{}},{'HR':(Old(),None)},None,None)
    # The public caller supplies META, including these unused synthetic columns.
    samples['spout_no']='1';samples['reference_time']=pd.Timestamp('2024-06-01',tz='Asia/Shanghai')
    result=run.candidate_inputs(Builder(),samples,fold,New())
    assert result.pred_tap_time_len.tolist()==[.8*30.+(1.-.8)*10.,.8*20.]
    assert result.pred_tap_iron.tolist()==[100.,200.]


def test_acceptance_requires_strict_origins_and_all_horizons():
    from bf_tap.optimization.trajectory_candidate import CANDIDATE,acceptance
    from bf_tap.optimization.trajectory_run import registration
    from bf_tap.optimization.component_export import units
    from bf_tap.optimization.validation import aggregate_grid
    reg=registration();metrics={}
    for unit,_,h,_,_ in units(reg):
        metrics[unit]={'horizon':h,'candidates':{}}
        for c,t in [('V1',.2),(CANDIDATE,.197)]:
            metrics[unit]['candidates'][c]={'overall':{'loss':(.2+t)/2,'iron':{'wmape':.2},'time':{'wmape':t}}}
    summary=aggregate_grid(metrics)
    assert acceptance(metrics,summary,reg,True,True)['passed']
    assert not acceptance(metrics,summary,reg,False,True)['passed']
    assert not acceptance(metrics,summary,reg,True,False)['passed']
    for month in (6,7,8):
        metrics[f'O2024{month:02d}_H1']['candidates'][CANDIDATE]=metrics[f'O2024{month:02d}_H1']['candidates']['V1']
    gate=acceptance(metrics,aggregate_grid(metrics),reg,True,True)
    assert not gate['checks']['H1_origins']
