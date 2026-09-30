"""Regressions for seed identity, paired denominators and diagnostic interpretation."""
import numpy as np
import pytest

from bf_tap_r2.ema_evaluation_diagnostics import (
    assemble_fold_columns, fold_median_regions, gain, reduction_detail,
    sample_indices, simulated_gains, verify_files,
)


def test_each_seed_keeps_its_own_fold_predictions():
    folds = {42:np.arange(10)%5,3407:np.arange(10)[::-1]%5}
    archive = {}
    for seed,fv in folds.items():
        for fold in range(5):
            a = np.full((10,2),np.nan)
            a[fv==fold] = seed+fold
            archive[f"s{seed}-f{fold}-pred"] = a
    columns = assemble_fold_columns(archive,folds)
    np.testing.assert_array_equal(columns[42][:,0],42+folds[42])
    np.testing.assert_array_equal(columns[3407][:,1],3407+folds[3407])
    archive['s42-f0-pred'][1,0] = 0
    with pytest.raises(ValueError,match='coverage'):
        assemble_fold_columns(archive,folds)


def test_subset_recomputes_denominator_and_never_averages_original_scores():
    y=np.array([1.,1.,100.,100.]); base=y+np.array([1,1,1,1]); candidate=y.copy()
    draws=np.array([[0,1],[2,3]])
    values,denominators=simulated_gains(y,base,candidate,draws)
    np.testing.assert_array_equal(denominators,[2,200])
    np.testing.assert_allclose(values,[50,.5])
    assert values.mean()!=pytest.approx(gain(y,base,candidate))
    with pytest.raises(ValueError,match='distinct'):
        simulated_gains(y,base,candidate,np.array([[0,0]]))


def test_sampling_is_paired_without_replacement_and_respects_test_spouts():
    groups=np.repeat([1,2],10)
    a=sample_indices(20,8,30,9,groups,{1:3,2:5})
    np.testing.assert_array_equal(a,sample_indices(20,8,30,9,groups,{1:3,2:5}))
    assert all(len(set(row))==8 for row in a)
    assert ((groups[a]==1).sum(axis=1)==3).all()
    with pytest.raises(ValueError,match='insufficient'):
        sample_indices(20,11,1,9,groups,{1:11,2:0})


def test_correct_direction_can_overshoot_and_increase_mae():
    y=np.array([10.,20.,30.]);base=np.array([8.,18.,28.]);candidate=np.array([9.,23.,35.])
    detail=reduction_detail(y,base,candidate)
    assert detail['movement_toward_truth_fraction']==1
    assert detail['crosses_truth_fraction']==pytest.approx(2/3)
    assert detail['crosses_and_worsens_fraction']==pytest.approx(2/3)
    assert detail['mae_reduction']<0


def test_regions_use_only_the_other_folds_inputs():
    fv=np.repeat(np.arange(5),4);x=np.arange(20,dtype=float)
    _,thresholds=fold_median_regions(x,fv)
    x[fv==0]+=1e6
    _,changed=fold_median_regions(x,fv)
    assert thresholds['0']==changed['0']
    assert thresholds['1']!=changed['1']


def test_frozen_evidence_tamper_and_outside_path_are_rejected(tmp_path):
    path=tmp_path/'identity.json';path.write_text('old')
    import hashlib
    digest=hashlib.sha256(b'old').hexdigest()
    verify_files(tmp_path,{'identity.json':digest})
    path.write_text('new')
    with pytest.raises(ValueError,match='identity'):
        verify_files(tmp_path,{'identity.json':digest})
    with pytest.raises(ValueError,match='identity'):
        verify_files(tmp_path,{'../outside':digest})


@pytest.mark.parametrize('y,base,candidate',[
    ([0,0],[1,2],[1,2]), ([1,2],[1],[1,2]), ([1,2],[1,np.nan],[1,2]),
])
def test_invalid_wmape_inputs_fail_closed(y,base,candidate):
    with pytest.raises(ValueError):
        gain(y,base,candidate)
