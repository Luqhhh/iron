from copy import deepcopy
import csv
import io
import numpy as np
import pytest
from bf_tap_r2.realmlp_time_release import blend,require_quality,payload


def test_fixed_time_blend_and_raw_iron_field_preservation():
    parent=[dict(sample_id='id1',pred_tap_iron='001.230000000',pred_tap_time_len='100.0')]
    result=blend([100],[105]);assert result[0]==101
    row=list(csv.DictReader(io.StringIO(payload(parent,result).decode())))[0]
    assert row['pred_tap_iron']=='001.230000000'
    assert float(row['pred_tap_time_len'])==101
    with pytest.raises(ValueError):blend([100],[np.nan])
    with pytest.raises(ValueError):blend([0],[-1])


def test_full_fit_is_blocked_by_any_missing_quality_or_terminal_gate():
    report=dict(candidate='REALMLP_SINGLE_A20',gains={str(s):.005 for s in [42,3407,271828,314159]},paired=dict(passed=True,lcb95=.003))
    audit=dict(status='passed',four_seed_gate_passed=True);terminal=dict(status='passed',actual_supervisor_exit_code=0)
    require_quality(report,audit,terminal)
    failed=deepcopy(report);failed['gains']['314159']=-.001
    with pytest.raises(ValueError):require_quality(failed,audit,terminal)
    failed=deepcopy(report);failed['paired']['lcb95']=0
    with pytest.raises(ValueError):require_quality(failed,audit,terminal)
    with pytest.raises(ValueError):require_quality(report,audit,dict(status='passed',actual_supervisor_exit_code=1))
