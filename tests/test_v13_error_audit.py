import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError

spec=importlib.util.spec_from_file_location('v13_error',Path('scripts/optimization_v13_error_audit.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def synthetic():
    rows=[]
    for candidate in ['V1','OTHER']:
        for h in range(1,5):
            for i in range(2):
                rows.append(dict(sample_id=f's{h}-{i}',spout_no='1',reference_time=pd.Timestamp(f'2024-0{h+5}-03',tz='Asia/Shanghai'),
                   tap_iron=100.+i,tap_time_len=10.+i,pred_tap_iron=105.+i+(candidate=='OTHER'),pred_tap_time_len=12.+i,
                   candidate=candidate,origin='O202406',unit=f'O202406_H{h}',horizon=h))
        for unit in ['DEV_LONG','DEV_SHORT']:
            rows.extend({**r,'unit':unit,'horizon':np.nan} for r in list(rows) if r['candidate']==candidate and r['unit']=='O202406_H1')
    errors=pd.DataFrame(rows)
    summary={c:{'J':.5*((5.+(c=='OTHER'))/100.5+2./10.5)} for c in ['V1','OTHER']}
    return errors,summary


def test_contributions_rebuild_unweighted_J_and_delta_excluding_dev():
    e,s=synthetic();c,b=audit.budgets(e,{6:4},['V1','OTHER'],s)
    assert np.isclose(c.loc[c.candidate=='V1','J_contribution'].sum(),s['V1']['J'],atol=1e-15)
    assert (c.loc[~c.included_in_J,'J_contribution']==0).all()
    assert b['unique_grid_samples']==8 and b['grid_prediction_exposures_per_candidate']==8
    assert b['candidates']['OTHER']['Delta_J_V1']==pytest.approx(.5/100.5)


def test_repeated_origin_predictions_are_kept_and_ids_not_counted_independent():
    e,s=synthetic()
    extra=e.loc[e.unit.str.startswith('O')].copy();extra['unit']=extra.unit.str.replace('O202406','O202407');extra['origin']='O202407'
    e=pd.concat([e,extra]);c,b=audit.budgets(e,{6:4,7:4},['V1','OTHER'],s)
    assert b['unique_grid_samples']==8 and b['grid_prediction_exposures_per_candidate']==16
    assert b['candidates']['V1']['J']==pytest.approx(s['V1']['J'])

@pytest.mark.parametrize('mutation',['duplicate','inconsistent_label','inconsistent_metadata','missing_candidate','missing_unit','wrong_horizon','null','negative','nonfinite'])
def test_corrupt_archived_errors_rejected(mutation):
    e,s=synthetic()
    if mutation=='duplicate':e=pd.concat([e,e.iloc[:1]])
    elif mutation=='inconsistent_label':e.loc[e.index[-1],'tap_iron']=123
    elif mutation=='inconsistent_metadata':e.loc[e.index[-1],'spout_no']='2'
    elif mutation=='missing_candidate':e=e.loc[e.candidate=='V1']
    elif mutation=='missing_unit':e=e.loc[e.unit!='O202406_H4']
    elif mutation=='wrong_horizon':e.loc[0,'horizon']=4
    elif mutation=='null':e.loc[0,'sample_id']=None
    elif mutation=='negative':e.loc[0,'tap_iron']=-1
    elif mutation=='nonfinite':e.loc[0,'pred_tap_iron']=np.inf
    with pytest.raises(ContractError):audit.budgets(e,{6:4},['V1','OTHER'],s)


def test_printed_value_tolerance_not_used_for_original_prediction_reconstruction():
    e,s=synthetic();s['V1']['J']+=5e-11
    with pytest.raises(ContractError):audit.budgets(e,{6:4},['V1','OTHER'],s)


def test_bins_fixed_with_explicit_missing_and_right_boundary():
    b=audit.binned(pd.Series([np.nan,0,.01,.2,1]),[0,.01,.05,.2,1])
    assert b.iloc[0]=='MISSING' and b.iloc[2]!=b.iloc[1] and b.iloc[-1]=='[1.0, inf)'


def test_bias_mean_and_median_are_distinct_diagnostics():
    s=audit.statistic([10,10,10],[9,9,40])
    assert s['signed_mean_residual']>0 and s['median_residual']==-1
