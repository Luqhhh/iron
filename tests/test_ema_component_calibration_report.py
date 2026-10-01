"""Real tier API and frozen pairing decisions on retained synthetic vectors."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from bf_tap_r2.ema_component_calibration_report import build_report


def arrays(global_shift, pressure_shift):
    actual = np.column_stack((np.full(100, 500.), np.full(100, 100.)))
    return dict(actual=actual, iron=actual[:, 0], q75=actual[:, 1]+2,
        UNCORRECTED=actual[:, 1]+1, GLOBAL=actual[:, 1]+global_shift,
        PRESSURE=actual[:, 1]+pressure_shift, folds=np.arange(100)%5, spouts=np.arange(100)%2+1)


def policy():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/candidate_tiers.yaml').read_text())


def test_report_calls_actual_classifier_and_preserves_matched_gate():
    report = build_report({'42': arrays(.5, .7), '3407': arrays(1.5, .7)}, policy())
    assert report['selected_for_confirmation'] == 'PRESSURE'
    assert report['records']['3407']['matched_gains']['GLOBAL'] < 0
    assert 'GLOBAL' in report['candidate_tiers']['decisions']['tap_time_len']
    assert report['recovery_correction_fit_calls'] == report['new_model_fits'] == 0


def test_report_tie_prefers_original_global_and_rejects_incomplete_pool():
    report = build_report({'42': arrays(.5, .5), '3407': arrays(.5, .5)}, policy())
    assert report['selected_for_confirmation'] == 'GLOBAL'
    with pytest.raises(ValueError, match='split pool'): build_report({'42': arrays(.5, .5)}, policy())
