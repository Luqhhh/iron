import numpy as np
import pytest
from bf_tap_r2.fixed_bootstrap_protocol import FixedBootstrapPlan,CANDIDATE,CONTROL,expected_mechanisms,select_target
from bf_tap_r2.fixed_bootstrap_audit import verify_fixed_plan

def test_weights_are_fixed_to_id_across_shuffled_batches_and_epochs():
    ids=['a','b','c'];p=FixedBootstrapPlan(42,ids,16)
    first=p.draw(ids,16);np.testing.assert_array_equal(p.draw(['c','a'],16),first[[2,0]])
    np.testing.assert_array_equal(p.draw(ids,16),first)
    assert p.certificate()['unique_poisson_draws']==48

def test_stage_plan_only_uses_actual_fit_rows_and_rejects_unknown_id():
    p=FixedBootstrapPlan(42,['a','b'],16)
    with pytest.raises(ValueError):p.draw(['query'],16)
    with pytest.raises(ValueError):FixedBootstrapPlan(42,['a','a'],16)
    with pytest.raises(ValueError):p.draw(['a'],8)

def test_plan_reinitialization_and_mutation_cannot_change_fixed_weights():
    p=FixedBootstrapPlan(42,['a','b'],16);a=p.draw(['a','b'],16);a[:]=99
    b=p.draw(['a','b'],16);q=FixedBootstrapPlan(42,['a','b'],16)
    np.testing.assert_array_equal(b,q.draw(['a','b'],16))

def test_independent_pcg64_replay_rejects_tampered_saved_plan(tmp_path):
    ids=np.array(['a','b']);p=FixedBootstrapPlan(42,ids,16);plan=tmp_path/'fixed-weights.npz';p.save(plan)
    settings={'random_seed':42,'tabm_k':16,'batch_size':1}
    p.draw(['a'],16);p.draw(['b'],16)
    trace={'bootstrap_fit_ids_digest':p.fit_digest,'history':[{'epoch':1,'bootstrap':p.certificate()}],'bootstrap_seed':1000045}
    # Order is independently generated; build the real order certificate below.
    p=FixedBootstrapPlan(42,ids,16)
    for i in np.random.default_rng(42).permutation(2):p.draw([ids[i]],16)
    trace['history'][0]['bootstrap']=p.certificate()
    verify_fixed_plan(trace,ids,settings,plan)
    with np.load(plan,allow_pickle=False) as a:weights=a['weights'].copy()
    weights[0,0]+=1;np.savez(plan,ids=ids,weights=weights)
    with pytest.raises(ValueError):verify_fixed_plan(trace,ids,settings,plan)

def test_only_time_can_be_selected_despite_positive_descriptive_iron():
    assert select_target({'tap_time_len':{'selected_for_confirmation':[],'mean_gain':.002},'tap_iron':{'selected_for_confirmation':[CANDIDATE],'mean_gain':.003}}) is None
    assert expected_mechanisms(CANDIDATE,'development')['bootstrap_draw_policy']=='once_per_actual_fit_row_head'
    assert 'bootstrap_draw_policy' not in expected_mechanisms(CONTROL,'development')
