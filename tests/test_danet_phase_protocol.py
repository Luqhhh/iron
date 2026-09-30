from copy import deepcopy

import numpy as np
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.danet_phase_protocol import phase_tasks,required_arms,phase_limits
from bf_tap_r2.danet_scoring import score_seed,decide_development,decide_confirmation
from bf_tap_r2.danet_phase_arithmetic import direct_decision


def gate_records(seeds=(42,3407),targets=TARGETS):
    return [dict(target=t,arm=a,seed=s,gain=g,candidate_score=96.28,blend_weight=.2,other_target_unchanged=True)
        for t in targets for s in seeds for a,g in (('DANET_FIXED',.015),('DANET_LEARNED',.03))]


def test_only_earned_target_and_its_fixed_control_consume_confirmation():
    pairs=[[TARGETS[1],'DANET_LEARNED']]
    assert phase_limits('development')==dict(pair_unit=20,estimator=80,optimizer=80)
    assert phase_limits('confirmation',pairs)==dict(pair_unit=10,estimator=40,optimizer=40)
    assert len(phase_tasks('confirmation',pairs))==10
    assert required_arms(TARGETS[1],'confirmation',pairs)==('DANET_FIXED','DANET_LEARNED')
    with pytest.raises(ValueError):required_arms(TARGETS[0],'confirmation',pairs)
    with pytest.raises(ValueError):phase_tasks('confirmation',pairs+pairs)
    with pytest.raises(ValueError):phase_tasks('confirmation',[[TARGETS[1],'DANET_FIXED']])
    with pytest.raises(ValueError):phase_tasks('development',pairs)


def test_score_keeps_current_other_target_and_reports_parent_difference():
    n=50;fold=np.arange(n)%5;y=dict(tap_iron=np.full(n,100.),tap_time_len=np.full(n,20.))
    current=dict(tap_iron=np.full(n,102.),tap_time_len=np.full(n,21.))
    parent=dict(tap_iron=np.full(n,104.),tap_time_len=current['tap_time_len'])
    rows,_=score_seed(y,fold,np.ones(n,int),current,parent,{'DANET_LEARNED':y['tap_iron']},'tap_iron',42)
    assert rows[0]['gain']==pytest.approx(.2) and rows[0]['candidate_score']==pytest.approx(96.7)
    assert rows[0]['candidate_minus_parent_package']==pytest.approx(1.2)
    with pytest.raises(ValueError):score_seed(y,fold*0,np.ones(n),current,parent,{'DANET_LEARNED':y['tap_iron']},'tap_iron',42)


@pytest.mark.parametrize('defect',['negative_seed','small_gain','control','working_gate'])
def test_each_frozen_development_gate_is_independently_binding(defect):
    rows=gate_records()
    for row in rows:
        if row['target']!=TARGETS[0] or row['arm']!='DANET_LEARNED':continue
        if defect=='negative_seed' and row['seed']==42:row['gain']=-.0001
        elif defect=='small_gain':row['gain']=.00999
        elif defect=='control':row['gain']=.015
        elif defect=='working_gate':row['candidate_score']=96.2499
    result=decide_development(rows)
    assert not result['decisions'][TARGETS[0]]['DANET_LEARNED']['eligible']
    assert result['eligible_pairs']==direct_decision(rows,'development')[0]


@pytest.mark.parametrize('defect',['missing','duplicate','nan'])
def test_incomplete_evidence_never_earns_derived_seed(defect):
    rows=gate_records()
    if defect=='missing':rows.pop()
    elif defect=='duplicate':rows.append(deepcopy(rows[0]))
    else:rows[0]['gain']=np.nan
    for check in (decide_development,lambda r:direct_decision(r,'development')):
        with pytest.raises(ValueError):check(rows)


@pytest.mark.parametrize('derived_gains,admitted',[((.03,.03),True),((-.001,.03),False),((.001,.1),False)])
def test_four_seed_lcb_sign_and_control_are_binding_fold_counts_are_descriptive(derived_gains,admitted):
    dev=gate_records();confirm=gate_records((7777,12011))
    for row in dev:row['folds_descriptive']=[dict(gain=-1.)]*5
    for row in confirm:
        if row['arm']=='DANET_LEARNED':row['gain']=derived_gains[(7777,12011).index(row['seed'])]
    selected=decide_development(dev)['eligible_pairs'];decision=decide_confirmation(dev,confirm)
    assert decision['promoted_pairs']==direct_decision(dev+confirm,'confirmation',selected)[0]
    assert bool(decision['promoted_pairs']) is admitted


def test_unearned_target_cannot_be_reintroduced_after_seeing_confirmation_scores():
    dev=gate_records()
    for row in dev:
        if row['target']==TARGETS[0] and row['arm']=='DANET_LEARNED':row['gain']=.001
    with pytest.raises(ValueError,match='Unexpected'):decide_confirmation(dev,gate_records((7777,12011)))
