import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.fixed_point_model import FixedPointRegressor
from bf_tap_r2.joint_mae_model import JointMAERegressor, TARGET_ORDER
from bf_tap_r2.q75_separate_mae_iron import admission, model_for, target_values, unit_settings, validate_spec


def test_exact_previously_engineered_factories_and_target_shapes():
    frame=pd.DataFrame({'tap_iron':[1.,2.],'tap_time_len':[10.,20.]})
    joint=model_for('SEPARATE_MAE',{});single=model_for('SINGLE_MAE',{})
    assert type(joint) is JointMAERegressor and joint.arm=='SEPARATE_MAE'
    assert type(single) is FixedPointRegressor and single.arm=='LAPLACE_FIXED'
    assert joint.train.__func__ is JointMAERegressor.train
    assert single.train.__func__ is FixedPointRegressor.train
    np.testing.assert_array_equal(target_values(frame,'SEPARATE_MAE'),frame[list(TARGET_ORDER)].to_numpy())
    np.testing.assert_array_equal(target_values(frame,'SINGLE_MAE'),frame.tap_iron.to_numpy())
    with pytest.raises(ValueError,match='Unknown'):model_for('SHARED_MAE',{})


def test_native_settings_are_each_original_program_without_rewriting():
    settings={'SEPARATE_MAE':{'training':{'objective':'two','random_seed':42}},'SINGLE_MAE':{'training':{'objective':'one','random_seed':42}}}
    m={'native_by_arm':settings}
    for arm in settings:assert unit_settings(m,arm) is settings[arm]['training']


def gains(values):
    return dict(zip(['42','3407','271828','314159'],values))


def test_formal_vs_Q75_does_not_erase_failed_increment_over_single():
    result=admission(gains([.01,.011,.009,.01]),gains([.002,.002,.002,-.001]))
    assert result['formal_promoted'] and not result['release_eligible']
    assert 'nonpositive_seed_gain' in result['mechanism_failed_conditions']


def test_four_positive_but_nonpositive_lcb_does_not_pass():
    result=admission(gains([.1,.001,.001,.001]),gains([.01]*4))
    assert not result['formal_promoted'] and not result['release_eligible']
    assert result['failed_conditions']==['nonpositive_seed_lcb95']


def test_both_gates_must_pass_without_incomplete_or_cross_split_vectors():
    result=admission(gains([.01,.011,.009,.01]),gains([.002,.003,.002,.003]))
    assert result['release_eligible']
    with pytest.raises(ValueError,match='Four complete'):admission({'42':.01,'3407':.02},gains([.002]*4))


def test_frozen_confirmation_budget_and_observed_control_origin():
    spec=json.loads(Path('configs/q75_separate_mae_iron/SPEC.json').read_text());validate_spec(spec)
    assert spec['selection_origin'].startswith('posthoc lead')
    for key,value in [('optimizer_runs',20),('candidate','SHARED_MAE_IRON_A20'),('engineering_optimizer_runs',4),('split_seeds',[42,3407])]:
        with pytest.raises(ValueError,match='scope or budget'):validate_spec(dict(spec,**{key:value}))
