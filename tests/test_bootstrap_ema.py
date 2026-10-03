import numpy as np
import pytest
from bf_tap_r2.bootstrap_ema_audit import replay_ema
from bf_tap_r2.bootstrap_ema_protocol import expected_mechanisms,CANDIDATE,CONTROL,decide

def test_independent_ema_initialization_updates_and_latest_buffers(tmp_path):
    np.savez(tmp_path/'initial.npz',w=np.array([1.],np.float32),b=np.array([0.],np.float32))
    np.savez(tmp_path/'step-000001.npz',w=np.array([3.],np.float32),b=np.array([7.],np.float32))
    np.savez(tmp_path/'step-000002.npz',w=np.array([5.],np.float32),b=np.array([8.],np.float32))
    value=replay_ema(tmp_path,2,['w'],.99)
    np.testing.assert_allclose(value['w'],[1.0598],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(value['b'],[8.])

def test_missing_raw_update_is_rejected(tmp_path):
    np.savez(tmp_path/'initial.npz',w=np.array([1.],np.float32))
    with pytest.raises((ValueError,FileNotFoundError)):replay_ema(tmp_path,1,['w'],.99)

def test_invalid_state_shape_and_nonfinite_rejected(tmp_path):
    np.savez(tmp_path/'initial.npz',w=np.array([1.],np.float32))
    np.savez(tmp_path/'step-000001.npz',w=np.array([np.nan,2.],np.float32))
    with pytest.raises(ValueError):replay_ema(tmp_path,1,['w'],.99)

def test_candidate_and_paired_bootstrap_control_have_distinct_ema_identity():
    assert expected_mechanisms(CANDIDATE,'development')['ema_beta']==.99
    assert 'ema_beta' not in expected_mechanisms(CONTROL,'development')

def test_positive_gain_but_negative_bootstrap_mechanism_not_formal():
    d=decide([.01,.01],[.02,.02],False)
    assert not d['selected_for_confirmation'] and not d['formal_promoted']

def test_ema_inference_restores_raw_parameters_and_rng():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.component_regularization import clone_state,update_ema,inference_state
    model=torch.nn.Linear(1,1,bias=False);state=clone_state(model)
    with torch.no_grad():model.weight.add_(1.)
    raw=clone_state(model);rng=torch.get_rng_state().clone();update_ema(model,state,.99)
    with inference_state(model,state):assert not torch.equal(model.weight,raw['weight'])
    assert torch.equal(model.weight,raw['weight']) and torch.equal(torch.get_rng_state(),rng)
