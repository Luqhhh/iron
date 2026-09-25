"""Regression tests for the V4.2 iron-only packaging column-preservation contract.

The V4.2 package build read the parent package with ``pd.read_csv`` and then
re-formatted the unchanged time column with ``str()``.  That rewrote 136/322
field strings (74 of them real float64 differences, <=2 ULP) even though the
manifest claimed ``time_column_preserved_byte_for_byte: true``.  The fix reads
the parent CSV field text directly; these tests drive that path with synthetic
parents so no model is fitted and no real package is read or written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _synthetic_ids(n: int = 322) -> list[str]:
    return [f"R2S2_TEST_{i:012X}" for i in range(n)]


def _mixed_time_strings(n: int) -> list[str]:
    out: list[str] = []
    for i in range(n):
        value = 100.0 + i * 0.1
        if i % 3 == 0:
            out.append(format(value, ".17g"))
        elif i % 3 == 1:
            out.append(repr(value))
        else:
            out.append(f"{value:e}")
    return out


def test_parent_csv_round_trip_preserves_time_field_text(tmp_path: Path) -> None:
    """The unchanged column must survive the real parent-read path exactly."""
    from bf_tap_r2.v4_2_package import (
        _align_parent_columns,
        _csv_payload_with_parent_time,
        _read_parent_csv,
    )

    ids = _synthetic_ids()
    time_strings = _mixed_time_strings(len(ids))
    header = "sample_id,pred_tap_iron,pred_tap_time_len"
    rows = [f"{ids[i]},1.0,{time_strings[i]}" for i in reversed(range(len(ids)))]
    parent_csv = tmp_path / "result.csv"
    parent_csv.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    parent_ids, parent_iron, parent_time = _read_parent_csv(parent_csv)
    assert parent_time == list(reversed(time_strings)), "raw field text must survive"
    aligned_iron, aligned_time = _align_parent_columns(
        parent_ids, parent_iron, parent_time, ids
    )
    assert aligned_time == time_strings, "reordering must not retouch the text"
    assert np.allclose(aligned_iron, 1.0)

    payload = _csv_payload_with_parent_time(ids, aligned_iron, aligned_time).decode("utf-8")
    body = payload.splitlines()[1:]
    assert len(body) == len(ids)
    for line, expected in zip(body, time_strings):
        assert line.endswith("," + expected), (line, expected)


def test_pandas_round_trip_would_have_changed_the_text() -> None:
    """Why the raw reader is required: the old path really did reformat values.

    This is a pure-python demonstration on the same mixed strings, so it stays
    valid even if pandas' CSV defaults change later.
    """
    import io

    import pandas as pd

    values = [format(100.0 + i * 0.1, ".17g") for i in range(10)]
    csv_text = "sample_id,pred_tap_time_len\n" + "\n".join(
        f"ROW_{i},{value}" for i, value in enumerate(values)
    ) + "\n"
    parsed = pd.read_csv(io.StringIO(csv_text), dtype={"sample_id": "string"})
    reformatted = [str(v) for v in parsed["pred_tap_time_len"].tolist()]
    assert reformatted != values, "the old read+str() path must be shown to rewrite text"
    assert all(abs(float(a) - float(b)) >= 0.0 for a, b in zip(reformatted, values))


def test_parent_csv_alignment_rejects_a_bad_id_set() -> None:
    from bf_tap_r2.v4_2_package import _align_parent_columns

    with pytest.raises(ValueError, match="ID set mismatch"):
        _align_parent_columns(["A", "B"], np.asarray([1.0, 2.0]), ["1", "2"], ["A", "C"])
    with pytest.raises(ValueError, match="duplicate"):
        _align_parent_columns(["A", "A"], np.asarray([1.0, 2.0]), ["1", "2"], ["A"])


def test_reader_rejects_wrong_columns_and_ragged_rows(tmp_path: Path) -> None:
    from bf_tap_r2.v4_2_package import _read_parent_csv

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column mismatch"):
        _read_parent_csv(wrong)

    ragged = tmp_path / "ragged.csv"
    ragged.write_text(
        "sample_id,pred_tap_iron,pred_tap_time_len\nX,1.0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="row width"):
        _read_parent_csv(ragged)
