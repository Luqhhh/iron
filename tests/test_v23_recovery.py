from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.v23_recovery import (
    FIXED_ALGORITHMS, V1, V10, V21, V22,
    _require_explicit_offsets,
    canonical_errors, reference_deltas, score_summary, scorecard,
    six_decimal_float, local_calendar_month,
)


def evidence():
    rows = []
    for origin, horizons in zip(range(6, 12), (4, 4, 4, 3, 2, 1), strict=True):
        for horizon in range(1, horizons + 1):
            unit = f"O2024{origin:02d}_H{horizon}"
            for candidate_index, candidate in enumerate(FIXED_ALGORITHMS):
                rows.append({
                    "sample_id": f"{unit}-id", "reference_time": f"2024-{origin:02d}-01 00:00:00+08:00",
                    "spout_no": "1", "tap_iron": 500.0, "tap_time_len": 100.0,
                    "pred_tap_iron": 510.00000049 + candidate_index,
                    "pred_tap_time_len": 101.00000049 + candidate_index,
                    "candidate": candidate, "origin": f"O2024{origin:02d}", "horizon": horizon,
                    "unit": unit, "week": "2024-W01",
                })
    for unit, origin in (("DEV_LONG", 7), ("DEV_SHORT", 9)):
        for candidate_index, candidate in enumerate(FIXED_ALGORITHMS):
            rows.append({
                "sample_id": f"{unit}-id", "reference_time": f"2024-{origin:02d}-01 00:00:00+08:00",
                "spout_no": "2", "tap_iron": 500.0, "tap_time_len": 100.0,
                "pred_tap_iron": 510.00000049 + candidate_index,
                "pred_tap_time_len": 101.00000049 + candidate_index,
                "candidate": candidate, "origin": f"O2024{origin:02d}", "horizon": np.nan,
                "unit": unit, "week": "2024-W01",
            })
    return pd.DataFrame(rows)


def test_six_decimal_uses_submission_text_precision():
    assert six_decimal_float(1.23456749) == 1.234567
    assert six_decimal_float(1.23456751) == 1.234568


def test_canonical_alignment_and_grid():
    errors, identity = canonical_errors(evidence())
    assert identity["grid_cells"] == 18
    assert identity["horizon_cells"] == {"1": 6, "2": 5, "3": 4, "4": 3}
    assert len(errors) == 80


def test_canonical_recomputes_serialized_errors():
    errors, _ = canonical_errors(evidence())
    row = errors.loc[errors.candidate == V1].iloc[0]
    assert row.pred_tap_iron == 510.0
    assert row.abs_error_tap_iron == 10.0


def test_scorecard_has_two_targets_per_unit_algorithm():
    errors, _ = canonical_errors(evidence())
    card = scorecard(errors)
    assert len(card) == 4 * 20 * 2
    assert set(card.target) == {"tap_iron", "tap_time_len"}


def test_summary_uses_18_cell_registered_J():
    errors, _ = canonical_errors(evidence())
    summary = score_summary(scorecard(errors))
    assert set(summary.loc[summary.scope_type == "HORIZON", "scope"]) == {"H1", "H2", "H3", "H4"}
    assert len(summary.loc[summary.scope == "J"]) == 4


def test_reference_deltas_keep_all_fixed_algorithms():
    errors, _ = canonical_errors(evidence())
    deltas = reference_deltas(score_summary(scorecard(errors)))
    assert set(deltas.algorithm) == set(FIXED_ALGORITHMS)
    assert (deltas.loc[deltas.algorithm == V1, "delta_vs_V1"] == 0).all()
    assert (deltas.loc[deltas.algorithm == V10, "delta_vs_V10"] == 0).all()
    assert set((V21, V22)) <= set(deltas.algorithm)


def test_month_grouping_uses_competition_timezone():
    values = pd.Series(["2024-06-01 01:00:00+08:00", "2024-06-30 23:00:00+08:00"])
    assert local_calendar_month(values).tolist() == ["2024-06", "2024-06"]


def test_historical_v21_replay_requires_explicit_timezone_offsets():
    _require_explicit_offsets(pd.Series(["2024-06-01 00:00:00+08:00"]), "aware")
    with pytest.raises(ContractError, match="explicit timezone offset"):
        _require_explicit_offsets(pd.Series(["2024-06-01 00:00:00"]), "naive")
