import json

import numpy as np
import pytest

from bf_tap_r2.rfm_audit import audit_partition, oracle_kernel_and_gradient
from bf_tap_r2.rfm_model import fit_partition
from bf_tap_r2.rfm_protocol import file_hash
from test_rfm_model import sample, recorder


def test_independent_oracle_matches_analytic_one_dimensional_example():
    z = np.array([[0.], [2.]])
    c = np.zeros((2, 0))
    center = np.array([[1.]])
    cc = np.zeros((1, 0))
    k, g = oracle_kernel_and_gradient(z, c, center, cc, np.eye(1), 1., np.array([2.]))
    np.testing.assert_allclose(k, [[1/np.e], [1/np.e]], atol=1e-14)
    np.testing.assert_allclose(g, [[2/np.e], [-2/np.e]], atol=1e-14)


@pytest.mark.parametrize("arm", ["FIXED_KRR", "FULL_RFM"])
def test_complete_partition_audit_is_zero_fit(tmp_path, monkeypatch, arm):
    frame = sample()
    _, selection = fit_partition(frame, "tap_time_len", arm, recorder(tmp_path).scoped(("unit",)),
                                  directory=tmp_path / "models")
    from bf_tap_r2 import rfm_model
    def forbidden(*a, **k): raise AssertionError("audit attempted fitting")
    monkeypatch.setattr(rfm_model, "solve_kernel", forbidden)
    monkeypatch.setattr(rfm_model.NumericPreprocessor, "fit", forbidden)
    audit = audit_partition(tmp_path / "models", frame, "tap_time_len",
                            expected_selection_sha256=file_hash(tmp_path / "models" / "selection.json"))
    assert audit["status"] == "passed"
    assert audit["models_checked"] == (5 if arm == "FULL_RFM" else 2) + selection["selected_state"]
    assert audit["maximum_difference"] < 1e-8
    assert audit["new_solver_calls"] == 0


def test_original_data_and_row_order_binding(tmp_path):
    frame = sample()
    directory = tmp_path / "models"
    fit_partition(frame, "tap_iron", "FULL_RFM", recorder(tmp_path).scoped(("unit",)), directory=directory)
    anchor = file_hash(directory / "selection.json")
    with pytest.raises(ValueError, match="identity"):
        audit_partition(directory, frame.iloc[::-1], "tap_iron", expected_selection_sha256=anchor)
    changed = frame.copy()
    changed["tap_iron"] += 1
    with pytest.raises(ValueError, match="target mean"):
        audit_partition(directory, changed, "tap_iron", expected_selection_sha256=anchor)


def test_wrong_coefficients_fail_equations_even_with_rehashed_artifacts(tmp_path):
    frame = sample()
    directory = tmp_path / "models"
    fit_partition(frame, "tap_iron", "FIXED_KRR", recorder(tmp_path).scoped(("unit",)), directory=directory)
    state_dir = directory / "inner" / "state-0"
    with np.load(state_dir / "arrays.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays["alpha"][0] += .25
    np.savez(state_dir / "arrays.npz", **arrays)
    complete = json.loads((state_dir / "complete.json").read_text())
    complete["hashes"]["arrays.npz"] = file_hash(state_dir / "arrays.npz")
    (state_dir / "complete.json").write_text(json.dumps(complete))
    selection = json.loads((directory / "selection.json").read_text())
    selection["artifacts"]["inner"][0]["complete_sha256"] = file_hash(state_dir / "complete.json")
    selection["artifacts"]["inner"][0]["hashes"] = complete["hashes"]
    (directory / "selection.json").write_text(json.dumps(selection))
    # Even when the caller supplied a new anchor, independent equations still
    # reject a finite, shape-correct but mathematically wrong model.
    with pytest.raises(ValueError, match="solve equation"):
        audit_partition(directory, frame, "tap_iron",
                        expected_selection_sha256=file_hash(directory / "selection.json"))


def test_selector_tamper_not_accepted_as_new_best_state(tmp_path):
    frame = sample()
    directory = tmp_path / "models"
    fit_partition(frame, "tap_iron", "FULL_RFM", recorder(tmp_path).scoped(("unit",)), directory=directory)
    selection = json.loads((directory / "selection.json").read_text())
    selection["calibration_mae"][0] = 0.
    (directory / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="selector MAEs"):
        audit_partition(directory, frame, "tap_iron",
                        expected_selection_sha256=file_hash(directory / "selection.json"))
