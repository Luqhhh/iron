from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.slot_screen import (
    average_ranks,
    geometry,
    hypergeometric_tail,
    read_json,
    spearman,
)


def fields(iron, time):
    return {"pred_tap_iron": [f"{v:.17g}" for v in iron],
            "pred_tap_time_len": [f"{v:.17g}" for v in time]}


def package(iron, time):
    return {"fields": fields(iron, time),
            "values": {"pred_tap_iron": np.asarray(iron, dtype=float),
                       "pred_tap_time_len": np.asarray(time, dtype=float)}}


def test_geometry_requires_exactly_one_changed_column():
    base = package([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0])
    same = package([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0])
    with pytest.raises(ValueError):
        geometry(same, base)
    both = package([1.5, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.5])
    with pytest.raises(ValueError):
        geometry(both, base)


def test_geometry_measures_compression_and_expansion():
    base = package([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    shrink = package([1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0])
    grow = package([1.0, 2.0, 3.0, 4.0], [1.5, 3.0, 4.5, 6.0])
    flat = package([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5])
    assert geometry(shrink, base)["slope"] == pytest.approx(-0.5)
    assert geometry(grow, base)["slope"] == pytest.approx(0.5)
    assert geometry(flat, base)["slope"] == pytest.approx(0.0, abs=1e-12)
    assert geometry(flat, base)["changed_target"] == "pred_tap_time_len"
    assert geometry(flat, base)["unchanged_target"] == "pred_tap_iron"
    assert geometry(flat, base)["changed_fraction"] == 1.0
    assert geometry(flat, base)["bias_pct"] == pytest.approx(100 * 0.5 / 2.5)


def test_average_ranks_and_spearman_match_known_values():
    assert average_ranks(np.array([3.0, 1.0, 1.0, 4.0])).tolist() == [3.0, 1.5, 1.5, 4.0]
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_hypergeometric_tail_is_exact_on_a_known_table():
    assert hypergeometric_tail(0, 10, 8, 23) == pytest.approx(0.0026248485664288, rel=1e-9)
    assert hypergeometric_tail(10, 10, 8, 23) == pytest.approx(1.0)


def test_frozen_spec_receipts_are_well_formed():
    spec = read_json(Path("configs/slot_screen/SPEC.json"))
    ids = [receipt["id"] for receipt in spec["receipts"]]
    assert len(ids) == len(set(ids)) == 23
    for receipt in spec["receipts"]:
        assert receipt["parent"] in spec["parent_scores"]
        assert receipt["platform_delta"] != 0.0
        assert receipt["local_gain"] is None or isinstance(receipt["local_gain"], float)
        assert receipt["dir"].startswith("local/runs/") and receipt["parent_dir"].startswith("local/runs/")
    assert spec["new_fits"] == 0 and spec["new_packages"] == 0
    assert spec["agent_uploads"] == 0 and spec["desktop_writes"] == 0
    assert spec["slot_admission_rule"]["status"].startswith("prospective_unvalidated")
