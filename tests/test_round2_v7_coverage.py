from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("tabm")

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v3_6_networks import V36NetworkRegressor
from bf_tap_r2.v3_6_sampler import load_v36_config, build_N_trials
from bf_tap_r2.v7_coverage import CoverageRegressor


def test_original_training_loop_and_expanded_coverage():
    import torch
    from pathlib import Path
    torch.set_num_threads(1)
    rng = np.random.default_rng(77)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame["sample_id"] = [f"synthetic-{i:04d}" for i in range(len(frame))]
    frame["spout_no"] = np.arange(len(frame)) % 2 + 1
    y = 100 + 2 * frame.air_volume.values + rng.normal(size=len(frame))
    trial = deepcopy(next(t for t in build_N_trials(load_v36_config(Path.cwd()))
                          if t["trial_id"] == "v36-s1-N-0048"))
    trial["parameters"].update(k=2, n_blocks=2, d_block=16, max_epochs=4, early_stopping_patience=2)
    original = V36NetworkRegressor(trial).fit(frame, y)
    epoch = original.best_epoch_
    replay = CoverageRegressor(trial).fit_fixed(frame, y, epochs=epoch, all_rows=False)
    np.testing.assert_array_equal(original.predict(frame), replay.predict(frame))
    refit = CoverageRegressor(trial).fit_fixed(frame, y, epochs=epoch, all_rows=True)
    assert replay.fit_meta_["gradient_rows"] == 80
    assert refit.fit_meta_["gradient_rows"] == 100
    assert replay.fit_meta_["epochs"] == refit.fit_meta_["epochs"] == epoch
    assert replay.params == refit.params == original.params
    np.testing.assert_array_equal(replay.preprocessor_.means_, refit.preprocessor_.means_)
    assert np.max(np.abs(refit.predict(frame) - replay.predict(frame))) > 0
    np.testing.assert_allclose(refit.predict(frame), refit.predict(frame.iloc[::-1])[::-1], rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="Epoch count"):
        CoverageRegressor(trial).fit_fixed(frame, y, epochs=0, all_rows=True)
