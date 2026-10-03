"""Nonfitting checks for the fixed release's numerical and CSV boundaries."""
import csv
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('mean3_q100', Path(__file__).resolve().parents[1]/'scripts/ema_mean3_q100.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_affine_probe_keeps_negative_component_coefficient():
    # q=1 permits a negative V7 coefficient; it is not a convex blend.
    assert module.probe([8.], [10.], [[12.], [15.], [18.]]).tolist() == [13.]


@pytest.mark.parametrize('members', [[[1.], [2.]], [[1.], [2.], [3.], [4.]]])
def test_exactly_three_members(members):
    with pytest.raises(ValueError, match='three aligned'):
        module.probe([10.], [2.], members)


def test_invalid_extrapolation_fails_without_clipping():
    with pytest.raises(ValueError, match='no clipping'):
        module.probe([1.], [20.], [[2.], [2.], [2.]])
    with pytest.raises(ValueError, match='Nonfinite'):
        module.probe([1.], [2.], [[np.nan], [2.], [2.]])


def test_unchanged_iron_strings_and_id_order_survive_serialization():
    parent = [dict(sample_id='R2S_TEST_002', pred_tap_iron='1.23000000000000001e+03'),
              dict(sample_id='R2S_TEST_001', pred_tap_iron='999.0000')]
    rows = list(csv.DictReader(io.StringIO(module.payload(parent, [2.5, 3.]).decode())))
    assert [(r['sample_id'], r['pred_tap_iron']) for r in rows] == [(r['sample_id'], r['pred_tap_iron']) for r in parent]
    assert [float(r['pred_tap_time_len']) for r in rows] == [2.5, 3.]


def test_misaligned_payload_does_not_silently_truncate():
    with pytest.raises(ValueError, match='length'):
        module.payload([dict(sample_id='x', pred_tap_iron='1')], [])
