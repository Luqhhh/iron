from __future__ import annotations

import numpy as np
import pytest

from oob_sampled import ARRAY_FIELDS, MODE_FULL_SAME_LEAF_FALLBACK, derive_attachment, predict, validate_attachment
from sampled_forest import CANDIDATES, PARAMETERS, SampledTimeForest, expected_draws, make_estimator


def fixture(candidate, n=48):
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(n, 9)).astype(np.float32)
    y = np.maximum(0.0, 20 + x[:, 0] * 3 + np.arange(n) % 7).astype(np.float64)
    ids = [f"i{index}" for index in range(n)]
    model = SampledTimeForest(candidate).fit(x, y, ids)
    arrays, certificate = derive_attachment(model, x)
    return x, y, model, arrays, certificate


def test_registered_parameters_are_single_change_only():
    differences = {name for name in PARAMETERS["A"] if PARAMETERS["A"][name] != PARAMETERS["B"][name]}
    assert differences == {"max_features", "max_samples"}
    assert PARAMETERS["A"]["max_features"] == 0.3333333333333333
    assert PARAMETERS["A"]["max_samples"] is None
    assert PARAMETERS["B"]["max_features"] == 0.7
    assert PARAMETERS["B"]["max_samples"] == 0.5
    assert make_estimator("A").get_params()["warm_start"] is False


@pytest.mark.parametrize("rows,half", [(888,444),(1180,590),(1490,745),(1803,902),(2091,1046),(2424,1212),(2754,1377)])
def test_expected_draws(rows, half):
    assert expected_draws("A", rows) == rows
    assert expected_draws("B", rows) == half


@pytest.mark.parametrize("candidate", tuple(CANDIDATES))
def test_actual_draw_shape_and_full_projection(candidate):
    x, _, model, arrays, certificate = fixture(candidate)
    assert set(arrays) == ARRAY_FIELDS
    assert arrays["draws"].shape == (256, expected_draws(candidate, len(x)))
    assert certificate["training_rows"] == len(x)
    assert certificate["draws_per_tree"] == expected_draws(candidate, len(x))
    assert certificate["all_N_rows_projected_per_tree"] is True
    assert validate_attachment(model, x, arrays, rederive=True)
    for tree in range(256):
        mask = arrays["leaf_trees"] == tree
        indices = np.flatnonzero(mask)
        full = [arrays["full_members"][arrays["full_offsets"][i]:arrays["full_offsets"][i+1]] for i in indices]
        assert np.array_equal(np.sort(np.concatenate(full)), np.arange(len(x)))


@pytest.mark.parametrize("candidate", tuple(CANDIDATES))
def test_oob_selected_or_full_leaf_fallback(candidate):
    x, _, _, arrays, certificate = fixture(candidate)
    for i, mode in enumerate(arrays["modes"]):
        full = arrays["full_members"][arrays["full_offsets"][i]:arrays["full_offsets"][i+1]]
        oob = arrays["oob_members"][arrays["oob_offsets"][i]:arrays["oob_offsets"][i+1]]
        selected = arrays["selected_members"][arrays["selected_offsets"][i]:arrays["selected_offsets"][i+1]]
        assert len(selected) >= 1
        if int(mode) == int(MODE_FULL_SAME_LEAF_FALLBACK):
            assert len(oob) == 0 and np.array_equal(selected, full)
        else:
            assert len(oob) >= 1 and np.array_equal(selected, oob)
    assert certificate["fallback_leaf_count"] == int(np.count_nonzero(arrays["modes"] == MODE_FULL_SAME_LEAF_FALLBACK))


@pytest.mark.parametrize("candidate", tuple(CANDIDATES))
def test_prediction_is_order_subset_and_chunk_invariant(candidate):
    x, y, model, arrays, _ = fixture(candidate)
    query = np.vstack((x[:7], x[-4:])).astype(np.float32)
    median, mean, diagnostics = predict(model, query, arrays)
    reverse = predict(model, query[::-1], arrays)
    assert np.array_equal(reverse[0][::-1], median)
    assert np.array_equal(reverse[1][::-1], mean)
    subset = np.asarray([0, 5, 10])
    selected = predict(model, query[subset], arrays)
    assert np.array_equal(selected[0], median[subset])
    assert np.array_equal(selected[1], mean[subset])
    chunks = [predict(model, query[i:i+3], arrays) for i in range(0, len(query), 3)]
    assert np.array_equal(np.concatenate([part[0] for part in chunks]), median)
    assert all(y.min() <= value <= y.max() for value in median)
    assert all(abs(item["weight_mass"] - 1.0) < 1e-12 for item in diagnostics)


def test_half_bootstrap_has_more_global_oob_on_fixture():
    _, _, _, _, a = fixture("A")
    _, _, _, _, b = fixture("B")
    assert b["global_oob"]["mean"] > a["global_oob"]["mean"]


def test_attachment_rejects_draw_padding():
    x, _, model, arrays, _ = fixture("B")
    broken = {name: value.copy() for name, value in arrays.items()}
    broken["draws"] = np.pad(broken["draws"], ((0, 0), (0, len(x) - broken["draws"].shape[1])))
    with pytest.raises(ValueError, match="N/m"):
        validate_attachment(model, x, broken, rederive=False)


def test_unknown_candidate_rejected():
    with pytest.raises(ValueError):
        expected_draws("C", 10)
    with pytest.raises(ValueError):
        SampledTimeForest("C")
