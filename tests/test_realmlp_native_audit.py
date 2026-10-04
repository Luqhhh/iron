import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.realmlp_native_audit import last_best_epoch,row_multiset_digest,encoded_without_fit,forbid_native_fit
from bf_tap_r2.data import FEATURES
from bf_tap_r2.realmlp_native_admission import synthetic


def test_last_best_uses_native_exact_ties_and_rejects_gaps():
    h=[dict(epoch=i+1,mae=v) for i,v in enumerate([2.,1.,1.,1.+1e-12])]
    assert last_best_epoch(h)==(3,1.)
    with pytest.raises(ValueError):last_best_epoch(h[1:])
    with pytest.raises(ValueError):last_best_epoch([dict(epoch=1,mae=np.nan)])


def test_preprocessing_fingerprint_is_order_invariant_but_row_sensitive():
    a=np.arange(30,dtype=np.float32).reshape(10,3)
    assert row_multiset_digest(a)==row_multiset_digest(a[::-1])
    b=a.copy();b[0,0]=1
    assert row_multiset_digest(a)!=row_multiset_digest(b)


def test_cold_encoder_reconstruction_ignores_target_and_uses_training_medians():
    a=pd.DataFrame(np.arange(63).reshape(3,21),columns=FEATURES)
    a['spout_no']=[1,2,1];a['sample_id']=['a','b','c'];a['tap_time_len']=[900.,-20.,1.]
    x,med,cats=encoded_without_fit(a)
    np.testing.assert_array_equal(med,a[list(FEATURES)].median().to_numpy())
    np.testing.assert_array_equal(cats,[1,2]);assert x.shape==(3,23)
    a.tap_time_len=np.nan;a.sample_id='other'
    np.testing.assert_array_equal(x,encoded_without_fit(a)[0])


def test_cold_guard_blocks_optimizer_and_fitting():
    import torch
    from pytabkit import RealMLP_TD_Regressor
    with forbid_native_fit():
        with pytest.raises(ValueError,match='Cold native'):torch.optim.Adam([torch.tensor(1.,requires_grad=True)])
        with pytest.raises(ValueError,match='Cold native'):RealMLP_TD_Regressor().fit(np.zeros((2,2)),np.zeros(2))


def test_synthetic_query_is_disjoint_and_target_free():
    spec=dict(synthetic_seed=964515,synthetic_training_rows=30,synthetic_query_rows=7)
    training,query,y=synthetic(spec)
    assert not set(training.sample_id)&set(query.sample_id)
    assert 'tap_time_len' not in query and y.shape==(30,)
    np.testing.assert_array_equal(y,synthetic(spec)[2])
