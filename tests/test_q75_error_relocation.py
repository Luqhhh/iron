import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.q75_error_relocation import check_vectors, concentration, input_regions, movement_summary, residual_summary


def test_joint_contribution_uses_separate_target_denominators_and_conserves_gain():
    y = np.array([[100., 1.], [100., 1.]])
    ref = y+[[10., 0.], [0., 1.]]
    candidate = y.copy()
    den = y.sum(axis=0)
    full = movement_summary(y, ref, candidate, np.array([True, True]), den)
    first = movement_summary(y, ref, candidate, np.array([True, False]), den)
    second = movement_summary(y, ref, candidate, np.array([False, True]), den)
    assert full['gain_global_denominator'] == pytest.approx(27.5)
    assert first['gain_global_denominator']+second['gain_global_denominator'] == pytest.approx(27.5)
    assert first['gain_region_denominator'] == pytest.approx(5.)
    assert first['gain_region_denominator'] != first['gain_global_denominator']
    summary = residual_summary(y, ref, np.array([True, True]), den)
    assert summary['score_loss_global_denominator'] == pytest.approx(27.5)


def test_top20_concentration_does_not_confuse_large_error_with_small_target():
    values = np.array([100.]+[1.]*99)
    result = concentration(values)
    assert result['top20']['share'] == pytest.approx(119/199)
    assert result['top1pct']['share'] == pytest.approx(100/199)
    assert concentration(np.zeros(100))['top20']['share'] == 0


def test_feature_regions_do_not_consult_held_inputs_or_targets_for_cutpoints():
    frame = pd.DataFrame({'spout_no': np.tile([1, 2], 20), 'x': np.arange(40.), 'tap_iron': np.arange(40.)})
    folds = np.arange(40)%5
    regions, cuts = input_regions(frame, folds, ['x'])
    changed = frame.copy()
    changed.loc[folds == 0, 'x'] += 1e6
    changed['tap_iron'] *= -1000
    _, altered = input_regions(changed, folds, ['x'])
    assert cuts['x']['0'] == altered['x']['0']
    assert cuts['x']['1'] != altered['x']['1']
    assert sum(regions[f'x:Q{q}'].sum() for q in range(1, 5)) == len(frame)
    assert regions['spout=1'].sum() == regions['spout=2'].sum() == 20


@pytest.mark.parametrize('y,p', [([[0,1]], [[1,1]]), ([[1,1]], [[-1,1]]), ([[1,1]], [[1,np.nan]]), ([[1,1]], [[1]])])
def test_invalid_targets_predictions_fail_closed(y, p):
    with pytest.raises(ValueError):
        check_vectors(y, p)
