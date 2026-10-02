"""Frozen identity, arithmetic and witness tests; zero real model fits."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import bf_tap_r2.laplace_leaf_partition_experiment as experiment
from bf_tap_r2.laplace_leaf_partition_experiment import (
    SPEC, describe, validate_development_units, validate_spec, verify_ledger, verify_node_statistics,
    verify_reference_identity,
)


def fake_tree():
    # Deliberately asymmetric signed residuals distinguish sign and abs targets.
    residual = np.array([-3., -1., 2., 6.])
    path = csr_matrix([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]])
    state = SimpleNamespace(node_count=3, n_node_samples=np.array([4, 2, 2]),
        value=np.array([.5, -2., 4.])[:, None, None], impurity=np.array([3., 1., 2.]))
    return SimpleNamespace(tree_=state, decision_path=lambda z: path), residual


def test_every_spec_field_is_frozen():
    spec = json.loads((Path(__file__).resolve().parents[1]/SPEC).read_text())
    validate_spec(spec)
    for key in spec:
        altered = {**spec, key: "changed"}
        with pytest.raises(ValueError):
            validate_spec(altered)


def test_reference_material_identity_is_external_to_self_declared_manifest(monkeypatch):
    monkeypatch.setattr(experiment, "sha", lambda path: "npz" if path.suffix==".npz" else "manifest")
    verify_reference_identity(Path("unused"), dict(reference_material_manifest_sha256="manifest",
                                                reference_development_oof_sha256="npz"))


@pytest.mark.parametrize("field", ["reference_material_manifest_sha256", "reference_development_oof_sha256"])
def test_changed_original_reference_identity_is_rejected(monkeypatch, field):
    monkeypatch.setattr(experiment, "sha", lambda path: "npz" if path.suffix==".npz" else "manifest")
    spec = dict(reference_material_manifest_sha256="manifest", reference_development_oof_sha256="npz")
    spec[field] = "changed"
    with pytest.raises(ValueError):
        verify_reference_identity(Path("unused"), spec)


def test_signed_residual_node_witness():
    tree, residual = fake_tree()
    verify_node_statistics(tree, np.zeros((4, 1)), residual)


@pytest.mark.parametrize("target", ["sign", "abs"])
def test_sign_or_abs_target_is_not_a_residual_witness(target):
    tree, residual = fake_tree()
    with pytest.raises(ValueError):
        verify_node_statistics(tree, np.zeros((4, 1)),
                               np.sign(residual) if target=="sign" else np.abs(residual))


@pytest.mark.parametrize("field", ["count", "value", "impurity", "nan", "shape"])
def test_changed_node_witness_is_rejected(field):
    tree, residual = fake_tree()
    if field=="count": tree.tree_.n_node_samples[1] = 3
    elif field=="value": tree.tree_.value[1, 0, 0] += 2e-12
    elif field=="impurity": tree.tree_.impurity[2] += 2e-12
    elif field=="nan": tree.tree_.impurity[0] = np.nan
    else: tree.tree_.node_count = 4
    with pytest.raises(ValueError):
        verify_node_statistics(tree, np.zeros((4, 1)), residual)


def test_only_node_statistics_have_predeclared_roundoff_tolerance():
    tree, residual = fake_tree()
    tree.tree_.value[1, 0, 0] += 5e-13
    tree.tree_.impurity[2] += 5e-13
    verify_node_statistics(tree, np.zeros((4, 1)), residual)


@pytest.mark.parametrize("residual", [np.array([1., 2.]), np.array([1., 2., np.nan, 3.])])
def test_invalid_residual_witness_rejected(residual):
    tree, _ = fake_tree()
    with pytest.raises(ValueError):
        verify_node_statistics(tree, np.zeros((4, 1)), residual)


def test_fixed_weight_gain_and_matched_control_arithmetic():
    y = np.full(10, 100.)
    q = dict(targets=np.column_stack((y, y)), spout=np.tile([1, 2], 5))
    predictions = {}
    for seed in (42, 3407):
        q[f"current-{seed}"] = np.column_stack((np.full(10, 90.), y))
        q[f"fold-{seed}"] = np.repeat(np.arange(5), 2)
        predictions[seed] = np.full(10, 100.)
    old = {"metrics": {str(seed): {"gain_vs_Q75": .25} for seed in (42, 3407)}}
    metrics = describe(q, predictions, old)
    for item in metrics.values():
        assert item["iron_wmape"] == 0
        assert item["gain_vs_Q75"] == pytest.approx(1.)
        assert item["gain_vs_original_long"] == pytest.approx(.75)
        assert all(v == pytest.approx(1.) for v in item["folds"].values())
        assert all(v == pytest.approx(1.) for v in item["spouts"].values())


@pytest.mark.parametrize("seeds", [[], [42], [3407], [42, 3407, 271828]])
def test_incomplete_or_unregistered_seed_vectors_rejected(seeds):
    with pytest.raises(ValueError):
        describe({}, {seed: np.ones(10) for seed in seeds}, {})


def test_exact_development_unit_roles():
    units = [dict(key=f"s{seed}-f{fold}-D3_L1_PARTITION", seed=seed, fold=fold)
             for seed in (42, 3407) for fold in range(5)]
    validate_development_units(units, {"split_seeds": [42, 3407]})


@pytest.mark.parametrize("change", ["seed", "fold", "duplicate", "missing", "extra"])
def test_changed_seed_fold_pool_rejected(change):
    units = [dict(key=f"s{seed}-f{fold}-D3_L1_PARTITION", seed=seed, fold=fold)
             for seed in (42, 3407) for fold in range(5)]
    if change=="seed": units[-1]["seed"] = 42
    elif change=="fold": units[-1]["fold"] = 0
    elif change=="duplicate": units[-1] = units[0]
    elif change=="missing": units.pop()
    else: units.append(units[0])
    with pytest.raises(ValueError):
        validate_development_units(units, {"split_seeds": [42, 3407]})


def ledger_fixture():
    metadata = dict(fitted_tree_count=3, fit_rows=20)
    checked = {"one": metadata}
    result = dict(models=[dict(key="one", metadata=metadata)], model_fits=1, tree_fits=3)
    entries = [dict(event="model_started", key="one", epochs=3, fit_rows=20),
               dict(event="model_completed", key="one", metadata=metadata)]
    # Mock the ledger's read boundary rather than creating another fit/artifact.
    class FakeOutput:
        def __truediv__(self, name):
            assert name=="ledger.jsonl"
            return SimpleNamespace(read_text=lambda: "\n".join(json.dumps(e) for e in entries))
    return FakeOutput(), result, checked, entries


def test_closed_model_and_tree_ledger():
    out, result, checked, _ = ledger_fixture()
    assert verify_ledger(out, result, checked, 1, 3) == 3


@pytest.mark.parametrize("change", ["started_epoch", "started_rows", "count", "budget", "extra", "end_key"])
def test_ledger_mismatch_rejected(change):
    out, result, checked, entries = ledger_fixture()
    budget = 3
    if change=="started_epoch": entries[0]["epochs"] = 2
    elif change=="started_rows": entries[0]["fit_rows"] = 19
    elif change=="count": result["tree_fits"] = 2
    elif change=="budget": budget = 2
    elif change=="extra": entries.append(entries[-1])
    else: entries[1]["key"] = "other"
    with pytest.raises(ValueError):
        verify_ledger(out, result, checked, 1, budget)
