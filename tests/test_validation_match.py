from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.slot_screen import read_json
from bf_tap_r2.validation_match import leave_one_out, rmse, row_basis


def test_leave_one_out_recovers_a_well_determined_linear_system():
    generator = np.random.default_rng(7)
    X = generator.normal(size=(20, 3))
    truth = np.array([1.0, -2.0, 0.5])
    y = X @ truth
    assert rmse(leave_one_out(X, y), y) < 1e-8


def test_leave_one_out_is_chance_level_when_labels_are_noise():
    generator = np.random.default_rng(11)
    X = generator.normal(size=(12, 322))
    y = generator.normal(size=12)
    loo = rmse(leave_one_out(X, y), y)
    assert loo > 0.5 * rmse(np.zeros_like(y), y)


def test_rmse_matches_hand_value():
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_row_basis_shapes_and_constant_column():
    parent = np.linspace(10.0, 20.0, 50)
    spouts = np.array([1, 2] * 25)
    constant = row_basis(parent, spouts, "constant")
    full = row_basis(parent, spouts, "full")
    assert constant.shape == (50, 1) and np.all(constant == 1.0)
    assert full.shape[1] > constant.shape[1]
    assert np.allclose(full[:, 0], 1.0)


def test_spec_budget_is_zero_and_points_at_frozen_receipts():
    spec = read_json(Path("configs/validation_match/SPEC.json"))
    assert spec["new_fits"] == 0 and spec["new_packages"] == 0
    assert spec["agent_uploads"] == 0 and spec["desktop_writes"] == 0
    assert spec["receipt_source"] == "configs/slot_screen/SPEC.json"
    assert spec["task_c"]["splits"] == [42, 3407]
    assert spec["task_c"]["platform_level"] == 96.3979
