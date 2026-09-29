import copy
import numpy as np
import pytest

from bf_tap_r2.rfm_scoring import score_seed, decide_development, decide_confirmation, LCB_MULTIPLIER
from bf_tap_r2.rfm_protocol import ARMS, TARGETS


def records(seeds=(42,3407), gain=.01, control=.005, score=96.25):
    return [dict(target=t, arm=a, seed=s, gain=gain if a=='FULL_RFM' else control,
                 candidate_score=score, blend_weight=.2, other_target_unchanged=True)
            for t in TARGETS for a in ARMS for s in seeds]


def test_score_is_fixed_blended_column_not_raw_member_or_both_targets():
    y={t:np.full(10,10.) for t in TARGETS}
    current={t:np.full(10,11.) for t in TARGETS}
    historical={t:np.full(10,12.) for t in TARGETS}
    members={'FULL_RFM':np.full(10,9.), 'FIXED_KRR':np.full(10,11.)}
    before=copy.deepcopy(current)
    rows=score_seed(y,np.arange(10)%5,current,historical,members,target='tap_iron',seed=42)
    full=next(r for r in rows if r['arm']=='FULL_RFM')
    assert full['gain']==pytest.approx(2.)
    assert full['candidate_score']==pytest.approx(92.)
    assert full['historical_target_gain']==pytest.approx(7.)
    assert full['candidate_minus_historical_package']==pytest.approx(12.)
    assert full['standalone_wmape']==pytest.approx(.1)
    assert all(r['gain']==pytest.approx(2.) for r in full['folds_descriptive'])
    for t in TARGETS: np.testing.assert_array_equal(current[t],before[t])
    with pytest.raises(ValueError,match='five-fold'):
        score_seed(y,np.arange(10)%2,current,historical,members,target='tap_iron',seed=42)


def test_frozen_development_boundaries_and_control_never_selected():
    assert decide_development(records())['eligible_targets']==list(TARGETS)
    for change in [dict(gain=.009999),dict(score=96.24999),dict(control=.01)]:
        assert not decide_development(records(**change))['eligible_targets']
    r=records();next(v for v in r if v['target']=='tap_iron' and v['arm']=='FULL_RFM')['gain']=0.
    assert decide_development(r)['eligible_targets']==['tap_time_len']


def test_confirmation_uses_four_seeds_sample_sd_and_no_unauthorized_target():
    dev=records(gain=.02)
    conf=records(seeds=(7777,12011),gain=.025)
    result=decide_confirmation(dev,conf)
    expected=.0225-LCB_MULTIPLIER*np.std([.02,.02,.025,.025],ddof=1)/2
    assert result['decisions']['tap_iron']['seed_lcb95']==pytest.approx(expected)
    assert result['promoted_targets']==list(TARGETS)
    assert result['release_authorized'] is False
    for row in dev:
        if row['target']=='tap_iron' and row['arm']=='FULL_RFM': row['gain']=.001
    with pytest.raises(ValueError,match='unexpected'):
        decide_confirmation(dev,conf)
    conf=[r for r in conf if r['target']=='tap_time_len']
    conf[-1]['gain']=-.001
    assert not decide_confirmation(dev,conf)['promoted_targets']


def test_no_positive_mean_escape_for_negative_lcb():
    conf=records(seeds=(7777,12011),gain=.001)
    result=decide_confirmation(records(gain=.06),conf)
    assert not result['promoted_targets']
    assert 'paired_seed_lcb95_positive' in result['decisions']['tap_iron']['failure_reasons']


def test_missing_duplicate_nonfinite_and_wrong_endpoint_rejected():
    r=records()
    for altered in (r[:-1],r+[r[0]], [dict(v,gain=np.nan) for v in r],
                    [dict(v,blend_weight=1.) for v in r]):
        with pytest.raises(ValueError):decide_development(altered)
    assert decide_confirmation(records(gain=.001),[])['promoted_targets']==[]
