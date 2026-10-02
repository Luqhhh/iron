import json
import math
from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.q75_separate_mae_iron_mean3 import admission, mean_three, unit_settings, validate_spec, iron_column, calibration_value


def test_equal_mean_matches_scalar_and_is_not_best_member_selection():
    a=np.array([1.,9.,5.]);b=np.array([8.,1.,6.]);c=np.array([3.,2.,1.])
    actual=mean_three(a,b,c)
    np.testing.assert_allclose(actual,[math.fsum(v)/3 for v in zip(a,b,c)],rtol=0,atol=1e-15)
    assert not np.array_equal(actual,a) and not np.array_equal(actual,b)
    np.testing.assert_array_equal(mean_three(a[::-1],b[::-1],c[::-1])[::-1],actual)


@pytest.mark.parametrize('extra',[np.array([1.]),np.array([np.nan,2.]),np.ones((2,1))])
def test_unaligned_or_nonfinite_member_rejected(extra):
    with pytest.raises(ValueError,match='Aligned'):mean_three(np.ones(2),np.ones(2),extra)


def test_training_seed_changes_only_copy_and_leaves_partition_and_stopping_settings():
    native={'training':{'random_seed':42,'max_epochs':240,'patience':30,'learning_rate':.001},'calibration':{'seed':27001}}
    settings=unit_settings({'native':native},'SEPARATE_INIT1042')
    assert settings==dict(native['training'],random_seed=1042)
    assert native['training']['random_seed']==42 and native['calibration']['seed']==27001
    assert unit_settings({'native':native},'SEPARATE_INIT2042')['random_seed']==2042
    with pytest.raises(KeyError):unit_settings({'native':native},'UNREGISTERED_SEED')


def test_positive_q75_gain_cannot_hide_loss_to_original_single_seed():
    result=admission({'42':.02,'3407':.01},{'42':.001,'3407':-.0001})
    assert result['confirmation_finalist'] is None
    assert result['failed_conditions']==['not_both_positive_vs_single_seed_SEPARATE_MAE']


def test_two_paired_gains_are_admission_not_four_seed_promotion():
    result=admission({'42':.02,'3407':.01},{'42':.001,'3407':.0001})
    assert result['confirmation_finalist']=='SEPARATE_MAE_IRON_MEAN3_A20' and not result['formal_promoted']
    with pytest.raises(ValueError,match='Complete'):admission({'42':.02},{'42':.01,'3407':.01})


def test_frozen_scope_rejects_extra_seed_weight_or_engineering_fit():
    spec=json.loads(Path('configs/q75_separate_mae_iron_mean3/SPEC.json').read_text());validate_spec(spec)
    for key,value in [('training_seeds',[42,1042,2043]),('weight',.5),('engineering_optimizer_runs',2),('optimizer_runs',42),('monitor_seconds',1800),('target_order',['tap_time_len','tap_iron'])]:
        with pytest.raises(ValueError,match='scope or budget'):validate_spec(dict(spec,**{key:value}))


def test_projection_preserves_iron_and_joint_selector_uses_both_targets():
    import pandas as pd
    prediction=np.array([[3.,90.],[7.,130.]])
    truth=pd.DataFrame({'tap_iron':[5.,5.],'tap_time_len':[100.,100.]})
    np.testing.assert_array_equal(iron_column(prediction),[3.,7.])
    assert calibration_value(prediction,truth,{'target_std':[2.,10.]})==1.5
    for bad in (prediction[:,0],np.ones((2,1)),np.array([[1.,np.nan]])):
        with pytest.raises(ValueError,match='two-target'):iron_column(bad)
