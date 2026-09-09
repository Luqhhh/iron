import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.optimization.horizon_router import META,PRED,horizons,route


def samples():
    return pd.DataFrame({'sample_id':['a','b','c','d'],'spout_no':[1]*4,'reference_time':pd.to_datetime(['2024-12-01 01:44','2025-01-01 00:00','2025-02-01 00:00','2025-03-01 00:00']).tz_localize('Asia/Shanghai')})


def parts():
    return {k:pd.DataFrame({'sample_id':['a','b','c','d'],PRED[0]:np.arange(4)+v,PRED[1]:np.arange(4)+v+10}) for k,v in [('P2',1.),('R2',20.)]}


def test_calendar_not_elapsed_days_and_h4_exact():
    s=samples();p=parts();c=s.reference_time.iloc[0]
    assert horizons(s,c).tolist()==[1,2,3,4]
    result=route(s,c,p)
    np.testing.assert_array_equal(result[PRED].iloc[:3],p['P2'][PRED].iloc[:3])
    np.testing.assert_array_equal(result[PRED].iloc[3],p['R2'][PRED].iloc[3])


def test_reordering_and_target_map():
    s=samples();p=parts();c=s.reference_time.iloc[0]
    p['P2']=p['P2'].iloc[::-1]
    first=route(s,c,p).set_index('sample_id').sort_index()
    second=route(s.iloc[::-1],c,p).set_index('sample_id').sort_index()
    pd.testing.assert_frame_equal(first,second)
    mapping={'tap_iron':dict.fromkeys(range(1,5),'R2'),'tap_time_len':dict.fromkeys(range(1,5),'P2')}
    out=route(s,c,p,mapping)
    np.testing.assert_array_equal(out[PRED[0]],p['R2'][PRED[0]])


@pytest.mark.parametrize('bad',['duplicate','past','h5','extra_label','missing_id','nan','negative','unknown_map'])
def test_reject_bad_inputs(bad):
    s=samples();p=parts();c=s.reference_time.iloc[0];mapping=None
    if bad=='duplicate':s.loc[1,'sample_id']='a'
    if bad=='past':s.loc[0,'reference_time']=c-pd.Timedelta(minutes=1)
    if bad=='h5':s.loc[3,'reference_time']=pd.Timestamp('2025-04-01',tz='Asia/Shanghai')
    if bad=='extra_label':s['tap_iron']=10
    if bad=='missing_id':p['P2']=p['P2'].iloc[:3]
    if bad=='nan':p['P2'].loc[0,PRED[0]]=float('nan')
    if bad=='negative':p['P2'].loc[0,PRED[0]]=-1
    if bad=='unknown_map':mapping={'tap_iron':{1:'P2'}}
    with pytest.raises(ContractError):route(s,c,p,mapping)


def test_loo_excludes_held_origin_and_ties_choose_R2():
    from bf_tap.optimization.horizon_loo import choose
    def cell(origin,p,r):
        return {'origin_id':origin,'horizon':1,'candidates':{c:{'overall':{'iron':{'wmape':v}}} for c,v in [('P2',p),('R2',r)]}}
    m={'a':cell('A',0,100),'b':cell('B',2,1),'c':cell('C',2,1)}
    assert choose(m,'iron',1,'A')[0]=='R2'
    m['a']=cell('A',1000,0)
    assert choose(m,'iron',1,'A')[0]=='R2'
    m['b']=cell('B',1,1);m['c']=cell('C',1,1)
    assert choose(m,'iron',1,'A')[0]=='R2'


def test_failed_S1_blocks_release_and_OPT18(tmp_path):
    import json
    from bf_tap.optimization.horizon_release import release
    from bf_tap.optimization.horizon_loo import run
    source=tmp_path/'source';source.mkdir()
    (source/'acceptance.json').write_text(json.dumps({'passed':False}))
    (source/'final_status.json').write_text(json.dumps({'G0':'PASS_OPT17'}))
    out=tmp_path/'output'
    with pytest.raises(ContractError):release(source,'missing.yaml',out)
    with pytest.raises(ContractError):run(source,tmp_path/'absent',out)
    assert not out.exists()
