from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip('torch')
pytest.importorskip('tabm')
pytest.importorskip('rtdl_num_embeddings')
pytest.importorskip('rtdl_revisiting_models')

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_periodic import PeriodicRegressor
from bf_tap_r2.v10_periodic_loss import LossRegressor, evaluate_fold, select_confirmation, training_loss


def sample():
    rng = np.random.default_rng(710)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 3 + 1
    frame['tap_time_len'] = 100 + 5*np.sin(frame.air_volume) + rng.normal(size=len(frame))
    frame['tap_iron'] = frame.tap_time_len*4
    settings = yaml.safe_load(Path('configs/round2_v10/SPEC.yaml').read_text())['training']
    settings.update(width=16, tabm_k=2, max_epochs=3, patience=2, embedding_dim=4, n_frequencies=4)
    return frame, settings


def test_per_head_robust_losses_have_expected_gradients():
    # Opposite head errors must not cancel before applying the loss.
    prediction = torch.tensor([[-2., 2.], [-0.05, 0.05]], requires_grad=True)
    target = torch.zeros(2)
    mae = training_loss(prediction, target, {'loss': 'mae'})
    assert float(mae.detach()) == pytest.approx(1.025)
    mae.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[-.25, .25], [-.25, .25]]))
    prediction.grad = None
    robust = training_loss(prediction, target, {'loss': 'smooth_l1', 'beta': .1})
    assert float(robust.detach()) == pytest.approx(.98125)
    robust.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[-.25, .25], [-.125, .125]]))
    with pytest.raises(ValueError, match='beta'):
        training_loss(prediction, target, {'loss': 'smooth_l1', 'beta': 0})


def test_mse_adapter_reproduces_frozen_v7_training():
    frame, settings = sample()
    recipe = {'backbone': 'tabm', 'frequency': .01, 'loss': 'mse'}
    old = PeriodicRegressor(recipe, settings).fit(frame, frame.tap_time_len.values)
    new = LossRegressor(recipe, settings).fit(frame, frame.tap_time_len.values)
    np.testing.assert_array_equal(old.predict(frame), new.predict(frame))
    for key, value in old.metadata_.items():
        assert value == new.metadata_[key]
    for name, value in old.model_.state_dict().items():
        torch.testing.assert_close(value, new.model_.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize('loss', ['mae', 'smooth_l1'])
def test_robust_training_is_inner_only_deterministic_and_cold(tmp_path, loss):
    frame, settings = sample()
    recipe = {'backbone': 'tabm', 'frequency': .01, 'loss': loss, 'beta': .1}
    model = LossRegressor(recipe, settings).fit(frame, frame.tap_time_len.values)
    again = LossRegressor(recipe, settings).fit(frame, frame.tap_time_len.values)
    pred = model.predict(frame)
    np.testing.assert_array_equal(pred, again.predict(frame))
    mask = group_safe_inner_folds(frame, seed=42)['fold'] != 0
    np.testing.assert_allclose(model.metadata_['inner_feature_means'], frame.loc[mask, list(FEATURES)].mean())
    np.testing.assert_allclose(model.metadata_['outer_feature_means'], frame[list(FEATURES)].mean())
    np.testing.assert_allclose(pred, model.predict(frame.iloc[::-1])[::-1], atol=1e-4, rtol=0)
    np.testing.assert_allclose(pred, np.r_[model.predict(frame.iloc[:40]), model.predict(frame.iloc[40:])], atol=1e-4, rtol=0)
    np.testing.assert_allclose(pred[:1], model.predict(frame.iloc[:1]), atol=1e-4, rtol=0)
    with (tmp_path/'model.pkl').open('wb') as stream:
        pickle.dump(model, stream)
    frame.drop(columns=['sample_id', 'tap_iron', 'tap_time_len']).to_pickle(tmp_path/'query.pkl')
    code = "import pickle,pandas as pd,numpy as np;from pathlib import Path;p=Path(__import__('sys').argv[1]);m=pickle.loads((p/'model.pkl').read_bytes());np.save(p/'cold.npy',m.predict(pd.read_pickle(p/'query.pkl')))"
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True)
    np.testing.assert_array_equal(pred, np.load(tmp_path/'cold.npy'))


def test_outer_labels_do_not_enter_loss_fit_or_epoch_selection():
    frame, settings = sample()
    recipe = {'backbone': 'tabm', 'frequency': .01, 'loss': 'mae'}
    folds = np.arange(len(frame)) % 5
    pred, meta = evaluate_fold(frame, folds, recipe, settings, 0)
    changed = frame.copy()
    changed.loc[folds == 0, ['tap_iron', 'tap_time_len']] = 1e9
    other, other_meta = evaluate_fold(changed, folds, recipe, settings, 0)
    np.testing.assert_array_equal(pred, other)
    assert meta == other_meta
    changed.loc[folds == 0, list(FEATURES)] += 100
    _, new_meta = evaluate_fold(changed, folds, recipe, settings, 0)
    assert meta == new_meta


def test_confirmation_requires_increment_over_existing_candidate():
    def row(name, a, b):
        return {'recipe': name, 'comparisons': {
            'A35': {'seed_gains': {42: a[0], 3407: a[1]}},
            'V7_TIME': {'seed_gains': {42: b[0], 3407: b[1]}, 'seed_summary': {'mean': np.mean(b)}}}}
    order = ['mae', 'smooth_l1_01']
    bad = row('mae', [.03, .03], [.002, -.001])
    assert select_confirmation([bad], order) is None
    good = row('smooth_l1_01', [.01, .01], [.001, .001])
    assert select_confirmation([bad, good], order) == 'smooth_l1_01'
    tie = row('mae', [.01, .01], [.001, .001])
    assert select_confirmation([good, tie], order) == 'mae'
