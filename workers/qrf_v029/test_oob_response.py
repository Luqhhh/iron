from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qrf_v015"))

from oob_response import (
    MODE_FULL_SAME_LEAF_FALLBACK,
    MODE_OOB,
    array_identity,
    derive_attachment,
    load_npz,
    predict,
    tree_structure_identity,
    validate_attachment,
)
from qrf_model import distribution_weights, lower_median


class Frozen:
    pass


def fixture_model(*, trees=4, rows=40, leaf=3):
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(rows, 5)).astype(np.float32)
    y = np.arange(rows, dtype=np.float64) + rng.normal(scale=.01, size=rows)
    forest = RandomForestRegressor(
        n_estimators=trees, criterion="absolute_error", min_samples_leaf=leaf,
        max_features=.7, bootstrap=True, max_samples=None, oob_score=False,
        random_state=2026, n_jobs=1,
    ).fit(x, y)
    model = Frozen()
    model.forest, model.y = forest, y
    model.ids = [f"id-{index}" for index in range(rows)]
    model.training_months = np.arange(rows) % 3
    model.leaves = []
    for tree in forest.estimators_:
        assigned = tree.apply(x)
        model.leaves.append({int(node): np.flatnonzero(assigned == node) for node in np.unique(assigned)})
    return model, x


def test_derives_from_estimators_samples_without_oob_score():
    model, x = fixture_model()
    arrays, certificate = derive_attachment(model, x)
    assert model.forest.oob_score is False
    assert arrays["draws"].shape == (4, 40)
    assert certificate["trees"] == 4


def test_draws_are_exact_public_interface():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    assert np.array_equal(arrays["draws"], np.asarray(model.forest.estimators_samples_))


def test_selected_members_are_oob_or_full_fallback():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    offsets = arrays["member_offsets"]
    for index, (tree, node, mode) in enumerate(zip(arrays["leaf_trees"], arrays["leaf_nodes"], arrays["modes"], strict=True)):
        full = model.leaves[int(tree)][int(node)]
        draw = arrays["draws"][int(tree)]
        expected = full[~np.isin(full, np.unique(draw))]
        selected = arrays["members"][offsets[index]:offsets[index + 1]]
        if len(expected):
            assert int(mode) == int(MODE_OOB)
            assert np.array_equal(selected, expected)
        else:
            assert int(mode) == int(MODE_FULL_SAME_LEAF_FALLBACK)
            assert np.array_equal(selected, full)


def test_selected_members_never_empty():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    assert (np.diff(arrays["member_offsets"]) >= 1).all()


def test_all_full_leaf_rows_partition_once_per_tree():
    model, x = fixture_model()
    derive_attachment(model, x)
    for mapping in model.leaves:
        assert np.array_equal(np.sort(np.concatenate(list(mapping.values()))), np.arange(len(model.ids)))


def test_tree_structure_is_unchanged():
    model, x = fixture_model()
    before = tree_structure_identity(model.forest)
    derive_attachment(model, x)
    assert tree_structure_identity(model.forest) == before


def test_prediction_has_equal_tree_mass_and_support():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    median, mean, diagnostics = predict(model, x[:7], arrays)
    assert len(median) == len(mean) == len(diagnostics) == 7
    assert (median >= model.y.min()).all() and (median <= model.y.max()).all()
    assert all(item["weight_mass"] == pytest.approx(1.0) for item in diagnostics)


def test_manual_first_prediction_matches():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    median, mean, _ = predict(model, x[:1], arrays)
    offsets = arrays["member_offsets"]
    lookup = {(int(tree), int(node)): arrays["members"][offsets[index]:offsets[index + 1]]
              for index, (tree, node) in enumerate(zip(arrays["leaf_trees"], arrays["leaf_nodes"], strict=True))}
    routed = [tree.apply(x[:1])[0] for tree in model.forest.estimators_]
    members = [lookup[(index, int(node))] for index, node in enumerate(routed)]
    weights = distribution_weights(members, len(model.y))
    assert median[0] == lower_median(model.y, weights, members)
    assert mean[0] == np.sum(model.y * weights)


def test_prediction_order_and_subset_invariant():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    full = predict(model, x, arrays)[:2]
    order = np.arange(len(x))[::-1]
    reverse = predict(model, x[order], arrays)[:2]
    assert np.array_equal(reverse[0], full[0][order])
    assert np.array_equal(reverse[1], full[1][order])


def test_prediction_chunk_invariant():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    full = predict(model, x, arrays)[:2]
    pieces = [predict(model, x[index:index + 7], arrays)[:2] for index in range(0, len(x), 7)]
    assert np.array_equal(np.concatenate([item[0] for item in pieces]), full[0])
    assert np.array_equal(np.concatenate([item[1] for item in pieces]), full[1])


def test_training_month_diagnostics_sum_to_one():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    diagnostics = predict(model, x[:2], arrays, model.training_months)[2]
    assert all(sum(item["training_month_weights"].values()) == pytest.approx(1.0) for item in diagnostics)


def test_attachment_roundtrip_without_pickle(tmp_path: Path):
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    path = tmp_path / "attachment.npz"
    np.savez(path, **arrays)
    loaded = load_npz(path)
    assert all(np.array_equal(loaded[name], value) for name, value in arrays.items())
    assert validate_attachment(model, x, loaded)


@pytest.mark.parametrize("field", ["members", "draws", "modes"])
def test_tampered_attachment_rejected(field):
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    bad = {name: value.copy() for name, value in arrays.items()}
    if field == "members":
        bad[field][0] = (bad[field][0] + 1) % len(model.ids)
    elif field == "draws":
        bad[field][0, 0] = (bad[field][0, 0] + 1) % len(model.ids)
    else:
        bad[field][0] = MODE_FULL_SAME_LEAF_FALLBACK if bad[field][0] == MODE_OOB else MODE_OOB
    with pytest.raises(ValueError, match="persisted OOB attachment differs"):
        validate_attachment(model, x, bad)


def test_duplicate_training_ids_rejected():
    model, x = fixture_model()
    model.ids[-1] = model.ids[0]
    with pytest.raises(ValueError, match="alignment"):
        derive_attachment(model, x)


def test_non_float32_training_rejected():
    model, x = fixture_model()
    with pytest.raises(ValueError, match="float32"):
        derive_attachment(model, x.astype(np.float64))


def test_nonfinite_query_rejected():
    model, x = fixture_model()
    arrays, _ = derive_attachment(model, x)
    query = x[:1].copy()
    query[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predict(model, query, arrays)


def test_array_identity_changes_with_contents():
    first = np.arange(5, dtype=np.int32)
    second = first.copy(); second[-1] += 1
    assert array_identity(first)["sha256"] != array_identity(second)["sha256"]


def test_corrupt_full_leaf_mapping_rejected():
    model, x = fixture_model()
    model.leaves = deepcopy(model.leaves)
    node = next(iter(model.leaves[0]))
    model.leaves[0][node] = model.leaves[0][node][1:]
    with pytest.raises(ValueError, match="mapping"):
        derive_attachment(model, x)
