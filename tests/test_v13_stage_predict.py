import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.v13_common import zero_fit
from bf_tap.optimization.structural import lad_coefficient
from bf_tap.models.baseline import DualTargetBaseline, FROZEN_PARAMETERS

spec=importlib.util.spec_from_file_location('v13_stage',Path('scripts/optimization_v13_stage_predict.py'))
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


def config(tmp_path,stage_name='test_b',extra=None):
    p=tmp_path/'cold.yaml';paths={k:k+'.csv' for k in ['operation_hourly','burden_change','data_dictionary',stage_name+'_samples']}
    if extra:paths.update(extra)
    p.write_text(yaml.safe_dump({'schema_version':1,'paths':paths}));return p

@pytest.mark.parametrize('name',['test_a','test_b','test_c'])
def test_explicit_stage_metadata_selection(tmp_path,name):
    p=config(tmp_path,name);assert name+'_samples' in stage.stage_paths(p,name)

@pytest.mark.parametrize('key',['train_samples','tap_history_train','test_a_samples','labels','external_history'])
def test_noncurrent_stage_or_label_paths_rejected(tmp_path,key):
    with pytest.raises(ContractError):stage.stage_paths(config(tmp_path,extra={key:'forbidden.csv'}),'test_b')


def samples():
    return pd.DataFrame({'sample_id':['b','a','c'],'spout_no':['1','2','1'],
                         'reference_time':pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03']).tz_localize('Asia/Shanghai')})

class Stub:
    m={'training':{'fit_cutoff':'2024-12-01 01:44:00+08:00'}}
    loaded={'OR':(None,pd.DataFrame({'sample_id':['train']}))}
    history=pd.DataFrame({'sample_id':['train']})
    def predict(self,samples,*args):
        ids=samples.sample_id.tolist()
        return pd.DataFrame({'sample_id':ids,'pred_tap_iron':[100+ord(s) for s in ids],'pred_tap_time_len':[10+ord(s) for s in ids]}),pd.DataFrame({'sample_id':ids})


def test_full_reverse_chunk_subset_single_stable_and_three_columns():
    s=samples();p,_=stage.checked_predictions(Stub(),s,None,None,None,chunk_size=2)
    assert p.sample_id.tolist()==s.sample_id.tolist() and list(p)==['sample_id',*stage.PRED]

@pytest.mark.parametrize('mutation',['duplicate','future_model','training_id','labels','naive_time','missing'])
def test_illegal_metadata_rejected(mutation):
    s=samples()
    if mutation=='duplicate':s.loc[0,'sample_id']='a'
    elif mutation=='future_model':s.loc[0,'reference_time']=pd.Timestamp('2024-11-30',tz='Asia/Shanghai')
    elif mutation=='training_id':s.loc[0,'sample_id']='train'
    elif mutation=='labels':s['tap_iron']=0
    elif mutation=='naive_time':s['reference_time']=s.reference_time.dt.tz_localize(None)
    elif mutation=='missing':s.loc[0,'spout_no']=None
    with pytest.raises(ContractError):stage.validate_metadata(s,Stub())


def test_order_sensitive_predictor_cannot_pass():
    class Wrong(Stub):
        def predict(self,samples,*args):
            p,part=super().predict(samples,*args);p['pred_tap_iron']=np.arange(len(p));return p,part
    with pytest.raises(ContractError):stage.checked_predictions(Wrong(),samples(),None,None,None)


def test_fit_guards_reject_models_and_LAD_without_any_fit():
    from bf_tap.optimization import structural
    with zero_fit() as counter:
        with pytest.raises(ContractError):DualTargetBaseline(FROZEN_PARAMETERS).fit(None,None)
        with pytest.raises(ContractError):structural.lad_coefficient([1],[1],[1])
    assert counter['attempted_target_fits']==1 and counter['attempted_calibration_fits']==1
    assert counter['completed_model_fits']==counter['completed_calibration_fits']==0
