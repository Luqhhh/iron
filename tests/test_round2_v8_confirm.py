from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('rtdl_revisiting_models')
from bf_tap_r2.v8_confirm import strongest, reference_fold, candidate_fold, check_control


def test_confirmation_ranks_increment_after_v7_and_requires_complete_pool():
    spec = {'candidates':{'tap_iron':['a'],'tap_time_len':['a']},'split_seeds':[42,3407],
            'tie_preference_by_target':{'tap_iron':['a'],'tap_time_len':['a']}}
    def row(target, a35, v7):
        return {'target':target,'recipe':'a','comparisons':{
            'A35':{'seed_gains':{'42':a35,'3407':a35}},
            'V7_candidate_pool':{'seed_gains':{'42':v7,'3407':v7}}}}
    summary = {'status':'development_complete','records':[row('tap_iron',.01,.01),row('tap_time_len',.1,.001)]}
    assert strongest(summary,spec)['target']=='tap_iron'
    changed = deepcopy(summary)
    changed['records'][0]['comparisons']['A35']['seed_gains']['3407']=-.01
    assert strongest(changed,spec)['target']=='tap_time_len'
    changed['records'][1]['comparisons']['V7_candidate_pool']['seed_gains']['3407']=0.
    with pytest.raises(ValueError,match='No eligible'):
        strongest(changed,spec)
    with pytest.raises(ValueError,match='Complete frozen'):
        strongest({**summary,'records':summary['records'][:1]},spec)
    del summary['records'][0]['comparisons']['A35']['seed_gains']['3407']
    with pytest.raises(ValueError,match='seed coverage'):
        strongest(summary,spec)


def test_reference_control_refuses_wrong_shape_nonfinite_and_difference():
    assert check_control([1.,2.],[1.,2.],1e-9)==0.
    for a,b in [([1.],[1.,2.]),([np.nan],[1.]),([1.],[np.inf])]:
        with pytest.raises(ValueError,match='Invalid'):
            check_control(a,b,1e-9)
    with pytest.raises(ValueError,match='exceeds'):
        check_control([1.],[1.01],1e-9)


def test_reference_and_candidate_only_receive_training_labels(monkeypatch,tmp_path):
    import bf_tap_r2.v4_1_reference as reference_module
    import bf_tap_r2.v8_confirm as confirm_module
    frame = pd.DataFrame({'sample_id':['a','b','c','d'],'tap_iron':[1.,2.,3.,4.],
                          'tap_time_len':[10.,20.,30.,40.],'spout_no':[1,1,2,2]})
    folds = np.array([0,1,0,1])
    def query_assert(query):
        assert list(query.sample_id)==['a','c']
        assert 'tap_iron' not in query and 'tap_time_len' not in query
    class Factory:
        def __init__(self,*args,**kwargs):pass
        def fit_predict(self,train,query):
            assert list(train.sample_id)==['b','d']
            assert train.tap_iron.tolist()==[2.,4.]
            query_assert(query)
            return {'b36':{'tap_iron':np.ones(2),'tap_time_len':np.ones(2)*10},'meta':{}}
    class Model:
        def __init__(self,*args):self.metadata_={}
        def fit(self,train,y):
            assert list(train.sample_id)==['b','d']
            np.testing.assert_array_equal(y,[2.,4.])
        def predict(self,query):query_assert(query);return np.ones(2)
    monkeypatch.setattr(reference_module,'V36FixedRecipeFactory',Factory)
    monkeypatch.setattr(confirm_module,'AttentionRegressor',Model)
    b,_=reference_fold(tmp_path,frame,folds,0,1)
    p,_=candidate_fold(frame,folds,0,{'recipes':{'a':{}},'training':{}},{'target':'tap_iron','recipe':'a'})
    np.testing.assert_array_equal(b['tap_iron'],p)
