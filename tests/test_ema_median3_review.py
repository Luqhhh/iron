import importlib.util
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/ema_median3_review.py'
spec=importlib.util.spec_from_file_location('ema_median3_review',path)
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def test_median_is_fixed_member_aggregation_with_unchanged_background():
    q=np.array([100.,200.]);members=np.array([[10.,20.],[12.,22.],[100.,24.]])
    mean,median=review.columns(q,members)
    np.testing.assert_array_equal(median,[101.5,201.5])
    np.testing.assert_allclose(median-mean,.75*(np.median(members,axis=0)-members.mean(axis=0)))
    identical=np.tile([10.,20.],(3,1));a,b=review.columns(q,identical)
    np.testing.assert_array_equal(a,q);np.testing.assert_array_equal(b,q)

def test_missing_members_nonfinite_and_invalid_final_prediction_are_rejected():
    for members in [np.ones((2,2)),np.ones((3,3)),np.array([[1.,2.],[1.,2.],[np.nan,2.]])]:
        with pytest.raises(ValueError,match='members|member'):review.columns([1.,2.],members)
    with pytest.raises(ValueError,match='Invalid'):review.columns([1.],[[100.],[0.],[0.]])

def test_partition_and_source_identity_reject_cross_split_or_query_leakage():
    fit=['a','b'];query=['c'];w=dict(seed=42,fold=0,training_seed=1042,arm='EMA',
        partitions={'training':review.digest(fit),'query':review.digest(query)})
    v=dict(training_ids=fit,query_ids=query,identity=dict(source_directory=str(review.SOURCE),split_seed=42,
        fold=0,trial_id='EMA_INIT1042',fit_call_id='refit'))
    review.validate_unit(w,v,42,0,1042,fit,query)
    with pytest.raises(ValueError,match='identity'):review.validate_unit(w,v,3407,0,1042,fit,query)
    bad=deepcopy(v);bad['training_ids']=['a','c']
    with pytest.raises(ValueError,match='partition'):review.validate_unit(w,bad,42,0,1042,fit,query)
    bad=deepcopy(v);bad['identity']['source_directory']='/another/cache'
    with pytest.raises(ValueError,match='source'):review.validate_unit(w,bad,42,0,1042,fit,query)

def test_single_target_gain_has_correct_two_target_package_units():
    y=np.array([[100.,10.],[100.,10.]])
    assert review.score(y,[100,100],[11,11])-review.score(y,[100,100],[12,12])==5.
