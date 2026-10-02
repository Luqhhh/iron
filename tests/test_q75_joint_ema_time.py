import numpy as np
import pytest

from bf_tap_r2.q75_joint_ema_time import choose, compose


def test_affine_transfer_can_reject_invalid_without_clipping():
    with pytest.raises(ValueError, match='no clipping'):
        compose([1.], [100.], [0.])
    with pytest.raises(ValueError, match='Nonfinite'):
        compose([1.], [np.nan], [1.])


def test_two_split_matrices_cannot_be_treated_as_a_single_oof():
    with pytest.raises(ValueError, match='single-split'):
        compose(np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)))


def test_paired_gate_does_not_promote_incumbent_only_gain():
    assert not choose({'42': .01, '3407': .02}, {'42': .02, '3407': -.01})
    assert not choose({'42': 0., '3407': .02}, {'42': .02, '3407': .01})
    assert choose({'42': .01, '3407': .02}, {'42': .02, '3407': .01})
    with pytest.raises(ValueError, match='complete'):
        choose({'42': .01}, {'42': .02})


def test_transfer_preserves_identity_and_query_order():
    q75 = np.array([4., 7., 9.])
    single = np.array([2., 3., 6.])
    joint = np.array([5., 2., 7.])
    np.testing.assert_array_equal(compose(q75, single, single), q75)
    np.testing.assert_array_equal(compose(q75[::-1], single[::-1], joint[::-1])[::-1],
                                  compose(q75, single, joint))
