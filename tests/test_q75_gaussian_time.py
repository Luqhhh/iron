import numpy as np
import pytest

from bf_tap_r2.q75_gaussian_time import compose, reference_order, selected_epoch


def test_selector_rejects_extra_epochs_after_patience_and_early_stop():
    settings = dict(patience=2, max_epochs=8, min_delta_standardized_mae=.01)
    trace = [dict(epoch=i, calibration_standardized_mae=v) for i, v in enumerate([1., .7, .695, .698], 1)]
    assert selected_epoch(trace, settings) == (2, .7)
    with pytest.raises(ValueError, match='history'):
        selected_epoch(trace + [dict(epoch=5, calibration_standardized_mae=.6)], settings)
    with pytest.raises(ValueError, match='stopped'):
        selected_epoch(trace[:2], settings)


def test_reference_reordering_rejects_duplicates_or_missing_rows():
    np.testing.assert_array_equal(reference_order(['a', 'b'], ['b', 'a']), [1, 0])
    for ids in (['b', 'b'], ['a', 'c'], ['a']):
        with pytest.raises(ValueError, match='coverage'):
            reference_order(['a', 'b'], ids)


def test_composition_rejects_cross_seed_or_invalid_vectors_without_clipping():
    with pytest.raises(ValueError, match='aligned split'):
        compose(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match='Nonfinite'):
        compose([1.], [np.nan])
    with pytest.raises(ValueError, match='no clipping'):
        compose([1.], [-10.])
