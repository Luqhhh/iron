import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.features.trajectory import VALUES, COLUMNS, build_trajectory_features, append_trajectory
from bf_tap.features.temporal import build_temporal_features


def fixtures(values, hours=None):
    ref = pd.Timestamp('2024-06-01 12:00', tz='Asia/Shanghai')
    hours = hours if hours is not None else list(range(-len(values)+1, 1))
    times = [ref+pd.Timedelta(hours=h) for h in hours]
    events = pd.DataFrame(dict(event_time=times, available_at=times, **{v: values for v in VALUES}))
    samples = pd.DataFrame(dict(sample_id=['a'], reference_time=[ref]))
    return samples, events


def build(samples, events):
    return build_trajectory_features(samples, events, contract_value_columns=VALUES)[0]


def val(x, stat, window=6):
    return x.loc[0, f'trajectory__air_volume__{window}h__{stat}']


def test_real_times_and_boundary():
    s, e = fixtures([900., 0., 2., 6.], [-6, -3, -2, 0])
    x = build(s, e)
    assert list(x) == COLUMNS
    assert val(x, 'slope') == pytest.approx(2.)
    assert val(x, 'valid_pair_count') == 1
    assert np.isnan(val(x, 'mean_abs_step_rate'))


@pytest.mark.parametrize('values,hours,pairs,slope,rate', [
    ([2.,2.,2.], [-2,-1,0], 2, 0., 0.),
    ([0.,np.nan,2.,3.], [-3,-2,-1,0], 1, 1., np.nan),
    ([0.,3.,6.], [-6,-3,0], 0, np.nan, np.nan),
    ([0.,3.,6.], [-3,-1.5,0], 2, 2., 2.),
    ([np.nan,np.nan,np.nan], [-2,-1,0], 0, np.nan, np.nan),
])
def test_missing_constant_long_gap(values, hours, pairs, slope, rate):
    s, e = fixtures(values, hours); x = build(s, e)
    assert val(x, 'valid_pair_count') == pairs
    np.testing.assert_allclose([val(x,'slope'), val(x,'mean_abs_step_rate')], [slope,rate], equal_nan=True)


def test_future_availability_shuffle_duplicate_conflicts():
    s, e = fixtures([0.,1.,4.,3.]); original = build(s,e)
    pd.testing.assert_frame_equal(original, build(s,e.sample(frac=1,random_state=1)))
    pd.testing.assert_frame_equal(original, build(s,pd.concat([e,e.iloc[[1]]])))
    future = e.iloc[[0]].copy(); future.event_time=s.reference_time[0]+pd.Timedelta(hours=1); future.available_at=future.event_time
    pd.testing.assert_frame_equal(original, build(s,pd.concat([e,future])))
    delayed=e.iloc[[0]].copy(); delayed.event_time-=pd.Timedelta(minutes=5); delayed.available_at=s.reference_time[0]+pd.Timedelta(hours=1)
    pd.testing.assert_frame_equal(original, build(s,pd.concat([e,delayed])))
    bad=e.iloc[[1]].copy(); bad['oxygen']=99.
    with pytest.raises(ContractError,match='conflicting'):
        build(s,pd.concat([e,bad]))


def test_order_representation_and_old_columns_unchanged():
    s, a=fixtures([0.,1.,4.,2.,3.]); _, b=fixtures([0.,4.,1.,2.,3.])
    def old(e):
        return build_temporal_features(s,e,prefix='operation',event_time='event_time',available_at='available_at',
            value_columns=list(VALUES),stale_hours=24,windows_hours=[6,24])[0]
    first,second=old(a),old(b)
    pd.testing.assert_frame_equal(first,second)
    assert not build(s,a).equals(build(s,b))
    combined=append_trajectory(first,build(s,a))
    pd.testing.assert_frame_equal(combined[list(first)],first)
    with pytest.raises(ContractError):
        append_trajectory(combined,build(s,a))
