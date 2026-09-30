from copy import deepcopy

import numpy as np
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.dnnr_phase_protocol import phase_tasks, required_arms, phase_limits
from bf_tap_r2.dnnr_scoring import score_seed, decide_development, decide_confirmation
from bf_tap_r2.dnnr_phase_arithmetic import direct_decision


def gate_records(seeds=(42, 3407), arms=('KNN_FIXED', 'DNNR_FIXED', 'DNNR_LEARNED')):
    return [dict(target=t, arm=a, seed=s, gain=g, candidate_score=96.28,
                 blend_weight=.2, other_target_unchanged=True)
            for t in TARGETS for s in seeds
            for a, g in (('KNN_FIXED', 0.), ('DNNR_FIXED', .02), ('DNNR_LEARNED', .03)) if a in arms]


def test_failed_learned_candidate_consumes_no_confirmation_fit_or_metric_epoch():
    pairs = [[TARGETS[0], 'DNNR_FIXED']]
    tasks = phase_tasks('confirmation', pairs)
    assert len(tasks) == 10 and {t['target'] for t in tasks} == {TARGETS[0]}
    assert required_arms(TARGETS[0], 'confirmation', pairs) == ('KNN_FIXED', 'DNNR_FIXED')
    assert phase_limits('confirmation', pairs) == dict(pair_unit=10, estimator=40, derivative_bank=20, metric_epoch=0)
    assert phase_limits('development') == dict(pair_unit=20, estimator=120, derivative_bank=80, metric_epoch=40)


def test_learned_candidate_keeps_fixed_taylor_control_without_fitting_knn():
    pairs = [[TARGETS[1], 'DNNR_LEARNED']]
    assert required_arms(TARGETS[1], 'confirmation', pairs) == ('DNNR_FIXED', 'DNNR_LEARNED')
    assert phase_limits('confirmation', pairs) == dict(pair_unit=10, estimator=40, derivative_bank=40, metric_epoch=20)
    with pytest.raises(ValueError): required_arms(TARGETS[0], 'confirmation', pairs)
    with pytest.raises(ValueError): phase_tasks('development', pairs)
    with pytest.raises(ValueError): phase_tasks('confirmation', pairs+pairs)


def test_isolated_score_uses_current_other_column_and_parent_comparison():
    n = 50; fold = np.arange(n) % 5
    y = dict(tap_iron=np.full(n, 100.), tap_time_len=np.full(n, 20.))
    current = dict(tap_iron=np.full(n, 102.), tap_time_len=np.full(n, 21.))
    parent = dict(tap_iron=np.full(n, 104.), tap_time_len=current['tap_time_len'])
    records, _ = score_seed(y, fold, np.ones(n, int), current, parent,
                           {'DNNR_FIXED': np.full(n, 100.)}, 'tap_iron', 42)
    row = records[0]
    assert row['gain'] == pytest.approx(.2)
    assert row['candidate_score'] == pytest.approx(96.7)
    assert row['candidate_minus_parent_package'] == pytest.approx(1.2)
    assert row['other_target_unchanged'] is True
    with pytest.raises(ValueError, match='coverage'):
        score_seed(y, fold*0, np.ones(n), current, parent, {'DNNR_FIXED': y['tap_iron']}, 'tap_iron', 42)


@pytest.mark.parametrize('defect', ['negative_seed', 'small_gain', 'control', 'working_gate'])
def test_each_original_development_gate_refuses_on_both_independent_implementations(defect):
    records = gate_records()
    for row in records:
        if row['target'] != TARGETS[0] or row['arm'] != 'DNNR_FIXED': continue
        if defect == 'negative_seed' and row['seed'] == 42: row['gain'] = -.0001
        elif defect == 'small_gain': row['gain'] = .00999
        elif defect == 'control': row['gain'] = 0.
        elif defect == 'working_gate': row['candidate_score'] = 96.2499
    result = decide_development(records)
    assert not result['decisions'][TARGETS[0]]['DNNR_FIXED']['eligible']
    assert result['eligible_pairs'] == direct_decision(records, 'development')[0]


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'nan'])
def test_partial_or_invalid_gate_evidence_never_qualifies(defect):
    records = gate_records()
    if defect == 'missing': records.pop()
    elif defect == 'duplicate': records.append(deepcopy(records[0]))
    else: records[0]['gain'] = float('nan')
    for check in (decide_development, lambda r: direct_decision(r, 'development')):
        with pytest.raises(ValueError): check(records)


@pytest.mark.parametrize('derived_gains,admitted', [((.02,.02),True), ((-.001,.02),False), ((.001,.1),False)])
def test_four_seed_lcb_and_sign_are_binding_folds_are_descriptive(derived_gains, admitted):
    dev = gate_records()
    # Fixed passes; learned is inferior to its causal control and remains omitted.
    for row in dev:
        if row['arm'] == 'DNNR_LEARNED': row['gain'] = .015
        row['folds_descriptive'] = [{'gain': -.1}]*5
    selected = decide_development(dev)['eligible_pairs']
    confirm = gate_records((7777,12011), ('KNN_FIXED','DNNR_FIXED'))
    for row in confirm:
        if row['arm'] == 'DNNR_FIXED': row['gain'] = derived_gains[(7777,12011).index(row['seed'])]
    result = decide_confirmation(dev, confirm)
    assert result['promoted_pairs'] == direct_decision(dev+confirm, 'confirmation', selected)[0]
    assert bool(result['promoted_pairs']) is admitted


def test_failed_candidate_cannot_be_reintroduced_by_derived_seed_score():
    dev = gate_records()
    for row in dev:
        if row['arm'] == 'DNNR_LEARNED': row['gain'] = .001
    confirm = gate_records((7777,12011))
    with pytest.raises(ValueError, match='Unexpected'):
        decide_confirmation(dev, confirm)
