import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import platform_transfer_diagnostics as d
from bf_tap_r2.data import FEATURES


def spec():
    return json.loads((Path(__file__).resolve().parents[1]/d.SPEC).read_text())


def test_unbiased_mmd_matches_direct_unequal_group_calculation():
    rng=np.random.default_rng(1)
    k,_=d.kernel(rng.normal(size=(17,4)))
    mask=np.arange(17)<5
    a,b=k[:5,:5],k[5:,5:]
    expected=(a.sum()-np.trace(a))/(5*4)+(b.sum()-np.trace(b))/(12*11)-2*k[:5,5:].mean()
    assert d.mmd_statistics(k,mask)[0]==pytest.approx(expected,abs=1e-13)
    assert d.mmd_statistics(k,~mask)[0]==pytest.approx(expected,abs=1e-13)


def test_kernel_and_statistic_are_row_permutation_equivariant():
    rng=np.random.default_rng(2);x=rng.normal(size=(25,4));order=rng.permutation(len(x))
    k,b=d.kernel(x);kp,bp=d.kernel(x[order]);mask=np.arange(len(x))<7
    np.testing.assert_allclose(kp,k[np.ix_(order,order)],rtol=0,atol=1e-14)
    assert bp==pytest.approx(b)
    assert d.mmd_statistics(k,mask)==pytest.approx(d.mmd_statistics(kp,mask[order]))


def test_joint_shift_with_identical_marginals_is_detectable():
    values=np.linspace(-2,2,30)
    # Every single-feature distribution is exactly identical, joint dependence differs.
    x=np.vstack([np.column_stack([values,values]),np.column_stack([values,-values])])
    s=spec();s['mmd_permutations']=199
    result,_,_,_=d.permutation_mmd(x,30,s)
    assert result['p_value']<=.05


def test_feature_allowlist_excludes_ids_and_both_targets():
    frame=pd.DataFrame({f:[1.,2.,3.] for f in FEATURES})
    frame['spout_no']=[1,2,1];frame['sample_id']=['a','b','c']
    frame['tap_iron']=[9.,8.,7.];frame['tap_time_len']=[1.,2.,3.]
    expected=d.features(frame)
    frame['sample_id']=['leak','leak','leak'];frame['tap_iron']=np.nan;frame['tap_time_len']=1e99
    np.testing.assert_array_equal(d.features(frame),expected)
    assert expected.shape==(3,len(FEATURES)+2)
    frame.loc[0,'spout_no']=3
    with pytest.raises(ValueError):d.features(frame)


def test_domain_crossfit_excludes_heldout_rows_and_corrects_prior():
    rng=np.random.default_rng(3);x=rng.normal(size=(100,4));x[80:,0]+=1
    s=spec();s['domain_folds']=2;s['histogram']['max_iter']=5
    result,arrays,partitions,models=d.domain_crossfit(x,80,s)
    assert len(models)==4
    for name,fold,model in models:
        part=partitions[fold];fit=np.asarray(part['fit_indices']);held=np.asarray(part['held_indices'])
        assert not set(fit)&set(held)
        assert set(fit)|set(held)==set(range(100))
        if name=='logistic':
            np.testing.assert_allclose(model[0].mean_,x[fit].mean(0))
        train=held[held<80];p=arrays[name+'_probability'][train]
        expected=p/(1-p)*np.sum(fit<80)/np.sum(fit>=80)
        np.testing.assert_allclose(arrays[name+'_raw_weight'][train],expected)
    assert all(0<r['effective_sample_size']<=80 for r in result.values())


def test_weighted_wmape_uses_weighted_denominator_and_is_scale_invariant():
    y=np.array([10.,100.]);ref=np.array([8.,104.]);pred=np.array([11.,107.]);w=np.array([5.,1.])
    expected=50*(5*(2-1)+(4-7))/(5*10+100)
    assert d.weighted_gain(y,ref,pred,w)==pytest.approx(expected)
    assert d.weighted_gain(y,ref,pred,10*w)==pytest.approx(expected)
    with pytest.raises(ValueError):d.weighted_gain(y,ref,pred,[0,1])
    with pytest.raises(ValueError):d.weighted_gain(y,ref,pred,[np.nan,1])


def test_same_query_movement_detects_reversal_and_undefined_zero_cosine():
    result=d.movement_detail(np.array([1.,-2.,3.]),np.array([-1.,2.,-3.]))
    assert result['cosine']==pytest.approx(-1.)
    assert result['sign_agreement']==0
    assert d.movement_detail(np.zeros(3),np.ones(3))['cosine'] is None
    with pytest.raises(ValueError):d.movement_detail(np.ones(2),np.ones(3))


def test_each_split_retains_its_own_prediction_coordinates():
    arrays={}
    for i in d.INITS:
        arrays[f'old_s-1_f-1_i{i}']=np.zeros(4)
        arrays[f'new_s-1_f-1_i{i}']=np.arange(1.,5.)
        for seed,sign in zip(d.SEEDS,[1.,-1.]):
            for fold in range(5):
                arrays[f'old_s{seed}_f{fold}_i{i}']=np.zeros(4)
                arrays[f'new_s{seed}_f{fold}_i{i}']=sign*np.arange(1.,5.)
    result,_=d.summarize_movements(arrays)
    assert result['42']['within_seed_fold_mean']['cosine']==pytest.approx(1.)
    assert result['3407']['within_seed_fold_mean']['cosine']==pytest.approx(-1.)


def test_frozen_scope_does_not_authorize_target_fits_or_platform_release():
    s=spec()
    assert s['new_target_estimators']==s['new_target_optimizers']==s['new_confirmation_seeds']==0
    assert s['packages']==s['desktop_writes']==s['agent_uploads']==0
    assert s['saved_refit_models']==66 and s['domain_classifier_fits']==10
    assert s['time_budget_seconds'] is None and not s['automatic_retries']
