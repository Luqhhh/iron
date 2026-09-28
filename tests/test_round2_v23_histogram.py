
import importlib
import math
import importlib.util
import pickle
import numpy as np
import pandas as pd
import pytest

def api():
    return importlib.import_module('bf_tap_r2.v23_histogram')

def test_distributional_predictor_exists():
    assert importlib.util.find_spec('bf_tap_r2.v23_histogram') is not None

def test_grid_is_fit_only_and_target_mass_is_normalized():
    grid = api().HistogramGrid.fit(np.array([10., 20., 30.]))
    assert grid.edges[0] == 9 and grid.edges[-1] == 31
    assert len(grid.edges) == 65
    original = grid.edges.copy()
    for arm in ('hard', 'gaussian'):
        mass = grid.encode(np.array([10., 20., 30.]), arm)
        assert mass.dtype == np.float64
        assert np.all(mass >= 0)
        np.testing.assert_allclose(mass.sum(1), 1, atol=1e-14)
    grid.encode(np.array([-1000., 1000.]), 'hard')
    np.testing.assert_array_equal(original, grid.edges)
    assert np.count_nonzero(grid.encode(np.array([20.]), 'gaussian')) > 1
    # Independent erf-based integration pins the frozen bandwidth and mass shape.
    expected_sigma = .75 * (grid.edges[-1]-grid.edges[0]) / 64
    assert grid.sigma == pytest.approx(expected_sigma)
    expected = []
    for value in [10., 20., 30.]:
        cdf = np.array([.5 * (1 + math.erf((edge-value)/(expected_sigma*math.sqrt(2))))
                        for edge in grid.edges])
        mass = np.diff(cdf)
        expected.append(mass/mass.sum())
    np.testing.assert_allclose(grid.encode(np.array([10., 20., 30.]), 'gaussian'),
                               np.array(expected), rtol=1e-11, atol=1e-14)

@pytest.mark.parametrize('values', [[], [2., 2.], [1., np.nan], [1., np.inf]])
def test_invalid_support_fails(values):
    with pytest.raises(ValueError):
        api().HistogramGrid.fit(np.asarray(values))

def test_boundary_bins_and_interpolated_median():
    grid = api().HistogramGrid.fit(np.array([10., 20.]), bins=4)
    mass = grid.encode(np.array([grid.edges[0], grid.edges[-1], -10., 40.]), 'hard')
    np.testing.assert_array_equal(mass.argmax(1), [0, 3, 0, 3])
    np.testing.assert_allclose(api().histogram_median(np.array([[.1, .2, .6, .1]]), np.arange(5.)), [2 + 1/3])
    np.testing.assert_allclose(api().histogram_median(np.array([[1., 0., 0., 0.]]), np.arange(5.)), [.5])

def test_probabilities_are_mixed_before_median():
    torch = pytest.importorskip('torch')
    masses = torch.tensor([[[.8, .1, .1], [.05, .05, .9]]], dtype=torch.float64)
    mixture = api().mixture_probabilities(masses.log()).numpy()
    np.testing.assert_allclose(mixture, [[.425, .075, .5]])
    combined = api().histogram_median(mixture, np.arange(4.))
    head_medians = api().histogram_median(masses.numpy().reshape(2, 3), np.arange(4.)).mean()
    assert abs(combined[0] - head_medians) > .2

def test_loss_is_per_head_and_has_nonzero_finite_gradient():
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[[3., -1.], [-2., 2.]]], requires_grad=True)
    target = torch.tensor([[.25, .75]])
    loss = api().histogram_loss(logits, target)
    expected = -(target[:, None] * torch.log_softmax(logits, -1)).sum(-1).mean()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0

def test_no_fallback_control_promotion():
    records = []
    for target in ('tap_iron', 'tap_time_len'):
        for arm in ('hard', 'gaussian'):
            gains = [.008, .014] if arm == 'gaussian' else [.006, .006]
            records.append({'target': target, 'arm': arm, 'seed_gains': {'42': gains[0], '3407': gains[1]}})
    assert api().select_finalist(records) == 'tap_iron'
    records[-1]['seed_gains']['3407'] = -.001
    records[1]['seed_gains']['3407'] = .001
    assert api().select_finalist(records) is None
    with pytest.raises(ValueError):
        api().select_finalist(records[:-1])

def test_inner_support_isolation_initialization_and_cold_query_order():
    torch = pytest.importorskip('torch')
    pytest.importorskip('tabm')
    pytest.importorskip('rtdl_num_embeddings')
    from bf_tap_r2.data import FEATURES
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    from bf_tap_r2.v7_periodic import digest
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f's{i}' for i in range(len(frame))]
    frame['spout_no'] = 1
    settings = {'random_seed': 42, 'inner_seed': 42, 'width': 16, 'blocks': 1,
                'tabm_k': 2, 'dropout': 0., 'embedding_dim': 4, 'n_frequencies': 4,
                'lite': True, 'learning_rate': .001, 'weight_decay': .0001,
                'batch_size': 64, 'max_epochs': 3, 'patience': 2, 'min_delta': 1e-5}
    y = 20 + frame.iloc[:, 0].to_numpy()
    inner = group_safe_inner_folds(frame, seed=42)['fold'] != 0
    y[~inner] += 500
    model = api().HistogramRegressor('gaussian', settings).fit(frame, y)
    assert model.metadata_['inner_support'][1] < 30
    assert model.metadata_['outer_support'][1] > 500
    assert model.metadata_['fit_ids_digest'] == digest(frame.sample_id.tolist())
    assert model.metadata_['optimizer_runs'] == 2
    a = api().HistogramRegressor('hard', settings)
    b = api().HistogramRegressor('gaussian', settings)
    a._initialize(frame, y)
    b._initialize(frame, y)
    for x, z in zip(a.model_.parameters(), b.model_.parameters()):
        assert torch.equal(x, z)
    prediction = model.predict(frame.iloc[:12])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(prediction, restored.predict(frame.iloc[:12]))
    np.testing.assert_allclose(prediction[::-1], restored.predict(frame.iloc[:12][::-1]), atol=1e-5)

def test_scoring_uses_matching_seed_and_isolated_column():
    module = importlib.import_module('bf_tap_r2.v23_run')
    frame = pd.DataFrame({'tap_iron': [100., 100.], 'tap_time_len': [20., 20.]})
    folds = {42: np.array([0, 1]), 3407: np.array([1, 0])}
    refs = {42: {'tap_iron': np.array([101., 101.]), 'tap_time_len': np.array([21., 21.])},
            3407: {'tap_iron': np.array([102., 102.]), 'tap_time_len': np.array([22., 22.])}}
    preds = {42: np.array([100., 100.]), 3407: np.array([110., 110.])}
    rows = module.score_predictions(frame, folds, refs, refs, preds, 'tap_iron')
    assert rows['42']['gain'] == pytest.approx(.1)
    assert rows['3407']['gain'] == pytest.approx(-.8)
    assert rows['42']['other_column_equal']
    assert rows['3407']['other_column_equal']
