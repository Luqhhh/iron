from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip('torch')
pytest.importorskip('tabm')
pytest.importorskip('rtdl_num_embeddings')
pytest.importorskip('rtdl_revisiting_models')
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_periodic import PeriodicRegressor
from bf_tap_r2.v7_release import save_and_cold
from bf_tap_r2.v12_joint import JointRegressor, evaluate_fold, joint_loss


def sample():
    rng = np.random.default_rng(1212)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 3 + 1
    frame['tap_iron'] = 500 + 15*frame.air_volume + rng.normal(size=len(frame))
    frame['tap_time_len'] = 100 + 3*frame.air_volume + rng.normal(size=len(frame))
    spec = yaml.safe_load(Path('configs/round2_v12/SPEC.yaml').read_text())
    settings = spec['training']
    settings.update(width=16, tabm_k=2, max_epochs=3, patience=2, embedding_dim=4, n_frequencies=4)
    return frame, settings


def test_loss_keeps_both_output_and_member_errors_and_gradients():
    p = torch.tensor([[[1., 3.], [-1., -3.]]], requires_grad=True)
    y = torch.zeros((1, 2))
    loss = joint_loss(p, y)
    assert loss.item() == 5.
    loss.backward()
    torch.testing.assert_close(p.grad, p.detach()/2)
    with pytest.raises(ValueError):
        joint_loss(p, torch.zeros(1))


@pytest.mark.parametrize('frequency', [None, .01])
def test_single_output_is_exact_v7_control(frequency):
    frame, settings = sample(); recipe = {'backbone':'tabm', 'frequency':frequency}
    old = PeriodicRegressor(recipe, settings).fit(frame, frame.tap_time_len.values)
    new = JointRegressor(recipe, settings).fit(frame, frame[['tap_time_len']].values)
    np.testing.assert_array_equal(old.predict(frame), new.predict(frame)[:, 0])
    assert old.metadata_['selected_epoch'] == new.metadata_['selected_epoch']


@pytest.mark.parametrize('frequency', [None, .01])
def test_joint_scales_repeat_cold_and_unknown_category(tmp_path, frequency):
    frame, settings = sample(); recipe = {'backbone':'tabm', 'frequency':frequency}
    y = frame[list(TARGETS)].values
    model = JointRegressor(recipe, settings).fit(frame, y)
    again = JointRegressor(recipe, settings).fit(frame, y)
    np.testing.assert_array_equal(model.predict(frame), again.predict(frame))
    inner = group_safe_inner_folds(frame, seed=42)['fold'] != 0
    for label, mask in [('inner', inner), ('outer', np.ones(len(frame), dtype=bool))]:
        np.testing.assert_allclose(model.metadata_[label+'_target_mean'], y[mask].mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.metadata_[label+'_target_std'], y[mask].std(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.metadata_[label+'_feature_means'], frame.loc[mask,list(FEATURES)].mean().values, atol=1e-14)
    query = frame.drop(columns=list(TARGETS)).copy(); query['spout_no'] = 999
    cold = save_and_cold(model, query, model.predict(query), tmp_path/'cold', 1e-6)
    assert cold['bit_identical'] and cold['training_reads_prohibited']


def test_both_outer_targets_and_features_are_excluded_from_fit():
    frame, settings = sample(); folds = np.arange(len(frame)) % 5
    recipe = {'backbone':'tabm','frequency':.01}
    pred, meta = evaluate_fold(frame, folds, recipe, settings, 0)
    changed = frame.copy()
    changed.loc[folds==0, list(TARGETS)] = np.nan
    other, other_meta = evaluate_fold(changed, folds, recipe, settings, 0)
    np.testing.assert_array_equal(pred, other); assert meta == other_meta
    changed.loc[folds==0, list(FEATURES)] *= 1e6
    _, other_meta = evaluate_fold(changed, folds, recipe, settings, 0)
    assert meta == other_meta


def test_auxiliary_output_reaches_shared_parameters():
    frame, settings = sample(); recipe = {'backbone':'tabm','frequency':.01}
    model = JointRegressor(recipe, settings)
    model._initialize(frame, frame[list(TARGETS)].values)
    model.model_.eval()
    x, c = model._inputs(frame)
    for output in range(2):
        model.model_.zero_grad(set_to_none=True)
        model.model_(x,c)[:,:,output].square().mean().backward()
        # Numeric embedding is shared by both regression channels.
        gradients = [p.grad for name,p in model.model_.named_parameters() if name.startswith('num_module.')]
        assert gradients and any(g is not None and torch.count_nonzero(g) > 0 for g in gradients)
