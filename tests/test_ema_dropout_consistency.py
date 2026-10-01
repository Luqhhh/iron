import copy

import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.data import FEATURES
from bf_tap_r2.ema_dropout_consistency import PairedDropoutEMARegressor, confirmation_eligible, paired_dropout_loss
from bf_tap_r2.v12_joint import joint_loss


def test_paired_objective_and_gradients_match_analytic_regression_formula():
    a = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[2., 4.], [5., 8.]]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0., 1.]], dtype=torch.float64)
    value, penalty = paired_dropout_loss(a, b, y, .5)
    expected = .5*((a-y[:, None]).square().mean()+(b-y[:, None]).square().mean())+.5*(a-b).square().mean()
    assert torch.equal(value, expected)
    assert torch.equal(penalty, (a-b).square().mean())
    ga, gb = torch.autograd.grad(value, (a, b))
    torch.testing.assert_close(ga, ((a-y[:, None])+(a-b))/a.numel())
    torch.testing.assert_close(gb, ((b-y[:, None])+(b-a))/b.numel())
    zero, _ = paired_dropout_loss(a, b, y, 0.)
    assert torch.equal(zero, .5*(joint_loss(a, y)+joint_loss(b, y)))


def test_objective_rejects_invalid_coefficient_alignment_and_nonfinite_loss():
    a = torch.zeros(2, 3, 1); y = torch.zeros(2, 1)
    for coefficient in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError): paired_dropout_loss(a, a, y, coefficient)
    with pytest.raises(ValueError): paired_dropout_loss(a, a[:, :1], y, .5)
    with pytest.raises(ValueError): paired_dropout_loss(a+float('inf'), a, y, .5)


def test_real_training_zero_penalty_matches_explicit_two_pass_control_and_saved_inference(tmp_path, monkeypatch):
    import bf_tap_r2.ema_dropout_consistency as module
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    rng = np.random.default_rng(960502); n = 64
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i}' for i in range(n)]; frame['spout_no'] = np.arange(n)%2+1
    y = (100+3*frame[FEATURES[0]].to_numpy())[:, None]
    settings = dict(random_seed=42, inner_seed=42, width=16, blocks=1, tabm_k=2, dropout=.1,
        embedding_dim=4, n_frequencies=2, lite=True, learning_rate=.001, weight_decay=.0001,
        batch_size=32, max_epochs=3, patience=2, min_delta=1e-5)
    recipe = dict(backbone='tabm', frequency=.01)
    mechanisms = dict(ema_beta=.99, dropout_consistency_lambda=0.)
    original = module.paired_dropout_loss
    model = PairedDropoutEMARegressor(recipe, settings, 'EMA', mechanisms, tmp_path).fit(frame, y)
    assert all(t['training_forward_passes']==2*t['updates'] for t in model.traces.values())
    assert any(row['mean_prediction_consistency']>0 for row in model.traces['refit']['history'])
    expected = model.predict(frame)
    monkeypatch.setattr(module, 'paired_dropout_loss', lambda a,b,target,weight:
        (.5*(joint_loss(a,target)+joint_loss(b,target)), (a-b).square().mean()))
    control = PairedDropoutEMARegressor(recipe, settings, 'EMA', mechanisms).fit(frame, y)
    np.testing.assert_array_equal(control.predict(frame), expected)
    monkeypatch.setattr(module, 'paired_dropout_loss', original)
    loaded = ComponentRegressor.load(tmp_path/'refit.pt')
    np.testing.assert_array_equal(loaded.predict(frame), expected)
    polluted = frame.copy(); polluted['tap_time_len'] = y[:, 0]
    with pytest.raises(ValueError, match='targets'): loaded.predict(polluted)
    candidate = PairedDropoutEMARegressor(recipe, settings, 'EMA', dict(mechanisms, dropout_consistency_lambda=.5)).fit(frame, y)
    assert not np.array_equal(candidate.predict(frame), expected)
    for key in model.model_.state_dict(): assert model.model_.state_dict()[key].shape == candidate.model_.state_dict()[key].shape
    bad = copy.deepcopy(mechanisms); bad['dropout_consistency_lambda'] = .25
    with pytest.raises(ValueError, match='scope'): PairedDropoutEMARegressor(recipe, settings, 'EMA', bad)


def test_confirmation_requires_both_parent_and_matched_control_gains_in_both_complete_splits():
    assert confirmation_eligible({'42':.1,'3407':.2},{'42':.01,'3407':.02})
    assert not confirmation_eligible({'42':.1,'3407':.2},{'42':.01,'3407':0.})
    assert not confirmation_eligible({'42':-.1,'3407':.2},{'42':.01,'3407':.02})
    with pytest.raises(ValueError): confirmation_eligible({'42':.1},{'42':.1})
    with pytest.raises(ValueError): confirmation_eligible({'42':.1,'3407':float('nan')},{'42':.1,'3407':.1})
