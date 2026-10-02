import csv
import math
from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.joint_support_audit import (
    direct_mmd, exclusive_units, kernel_matrix, local_output,
    permutation_test, quadratic_mmd, read_features, validate_kernel,
)


def test_independent_loop_formula():
    x = np.arange(18.).reshape(6, 3)
    k, _ = kernel_matrix(x)
    labels = np.array([0, 0, 0, 1, 1, 1])
    expected = (math.fsum(k[i,j] for i in range(3) for j in range(3) if i != j)/6
                + math.fsum(k[i,j] for i in range(3,6) for j in range(3,6) if i != j)/6
                - 2*math.fsum(k[i,j] for i in range(3) for j in range(3,6))/9)
    assert direct_mmd(k, labels) == pytest.approx(expected, abs=1e-14)
    assert quadratic_mmd(k, labels) == pytest.approx(expected, abs=1e-14)


def test_duplicate_and_shared_units():
    x, labels, counts = exclusive_units(np.array([[1],[1],[2],[3],[4]]), np.array([[3],[5],[6]]))
    assert x[:,0].tolist() == [1,2,4,5,6]
    assert labels.tolist() == [0,0,0,1,1]
    assert counts["train_duplicate_rows"] == 1
    assert counts["shared_unique_vectors_excluded"] == 1


def test_duplicate_reordering_invariant():
    a, b = np.array([[3],[1],[2],[1]]), np.array([[6],[5],[4]])
    one = exclusive_units(a,b)
    two = exclusive_units(a[::-1],b[::-1])
    np.testing.assert_array_equal(one[0],two[0])
    assert one[2] == two[2]


def test_permutation_reproducibility():
    x = np.random.default_rng(5).normal(size=(16,3))
    k, _ = kernel_matrix(x)
    labels = np.r_[np.zeros(8,dtype=int),np.ones(8,dtype=int)]
    one, n1 = permutation_test(k,labels,19,11)
    two, n2 = permutation_test(k,labels,19,11)
    assert one == two
    np.testing.assert_array_equal(n1,n2)
    assert .05 <= one["permutation_p"] <= 1


def test_joint_difference_with_equal_marginals():
    # Both columns have identical marginal means; the dependency differs.
    t = np.linspace(-3,3,60)
    x = np.r_[np.c_[t,t],np.c_[t,-t]]
    np.testing.assert_allclose(x[:60].mean(0),x[60:].mean(0),atol=1e-15)
    k, _ = kernel_matrix(x)
    result, _ = permutation_test(k,np.r_[np.zeros(60,dtype=int),np.ones(60,dtype=int)],99,18)
    assert result["permutation_p"] <= .05


def test_negative_unbiased_statistic_allowed():
    k = np.ones((4,4))-np.eye(4)
    k[0,1]=k[1,0]=k[2,3]=k[3,2]=.1
    labels = np.array([0,0,1,1])
    assert permutation_test(k,labels,9)[0]["mmd2_unbiased"] < 0


@pytest.mark.parametrize("kind",["nonsymmetric","diagonal","nonfinite","domain","singleton"])
def test_invalid_kernel_refused(kind):
    k = np.ones((4,4))-np.eye(4)
    labels = np.array([0,0,1,1])
    if kind=="nonsymmetric": k[0,1]=.5
    if kind=="diagonal": k[0,0]=1
    if kind=="nonfinite": k[0,0]=np.nan
    if kind=="domain": labels[:]=0
    if kind=="singleton": labels[:]=[0,1,1,1]
    with pytest.raises(ValueError): validate_kernel(k,labels)


def test_too_few_units_refused():
    with pytest.raises(ValueError): exclusive_units(np.ones((3,2)),np.zeros((3,2)))


def test_constant_dimension_safe():
    x = np.c_[np.arange(6),np.ones(6)]
    k, meta = kernel_matrix(x)
    assert meta["scale"][1] == 1
    assert np.isfinite(k).all()


@pytest.mark.parametrize("relative",["elsewhere","local","../escape"])
def test_private_output_guard(tmp_path,relative):
    with pytest.raises(ValueError): local_output(tmp_path,relative)


def test_reader_only_features(tmp_path):
    path = tmp_path/"features.csv"
    with path.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["sample_id",*FEATURES])
        writer.writerow(["R2S2_TRAIN_a",*range(21)])
        writer.writerow(["R2S2_TRAIN_b",*range(21)])
    ids, x = read_features(path,"R2S2_TRAIN_",2)
    assert ids == ["R2S2_TRAIN_a","R2S2_TRAIN_b"]
    assert x.shape == (2,21)
    with pytest.raises(ValueError): read_features(path,"WRONG_",2)
    with pytest.raises(ValueError): read_features(path,"R2S2_TRAIN_",3)


def test_reader_rejects_targets(tmp_path):
    path=tmp_path/"forbidden.csv"
    path.write_text("sample_id,tap_iron\na,1\n")
    with pytest.raises(ValueError): read_features(path,"a",1)
