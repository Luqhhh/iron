import numpy as np
import pytest
from bf_tap_r2.v2_residual_structure import diagnostic_inputs,grouped_summary
from test_round2_v2_robust_joint import synthetic


def test_input_ratios_have_explicit_units_and_no_target_dependency():
    f=synthetic()
    f["air_volume"]=f.air_volume.abs()+1
    f["total_press_diff"]=f.total_press_diff.abs()+1
    x=diagnostic_inputs(f)
    np.testing.assert_allclose(x.oxygen_per_air_volume,f.oxygen/(60*f.air_volume))
    f['tap_iron']=np.nan;f['tap_time_len']=-1
    np.testing.assert_array_equal(x,diagnostic_inputs(f))
    f.loc[0,'air_volume']=0
    with pytest.raises(ValueError,match='positive denominators'):
        diagnostic_inputs(f)


def test_paired_opposite_errors_do_not_become_independent_bias_evidence():
    y=np.array([10.,20.,30.,40.]);p=np.column_stack([y+2,y-2])
    r=grouped_summary(np.arange(4),y,p,[1,2,1,2],2)
    assert r['weighted_absolute_group_bias_relative']==0
    assert r['same_sign_groups']==0
    assert all(g['mean_absolute_error']==2 for g in r['groups'])
    assert all(g['rows']==2 for g in r['groups'])
    assert r['spearman_undefined']


def test_signed_bias_and_constant_input():
    y=np.array([10.,20.,30.,40.]);p=np.column_stack([y+2,y+4])
    r=grouped_summary(np.ones(4),y,p,[1,1,2,2],10)
    assert len(r['groups'])==1
    assert r['groups'][0]['mean_residual']==3
    assert r['weighted_absolute_group_bias_relative']==pytest.approx(0.12)
    assert r['same_sign_groups']==1
