import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats.mstats import hdquantiles
from bf_tap_r2.audit import write_json
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.kernel_regressor import KernelRegressor
from bf_tap_r2.marginal_estimators import hd_median


@pytest.mark.parametrize("target", TARGETS)
def test_kernel_engineering(target, tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(60, len(FEATURES))), columns=FEATURES)
    frame["spout_no"] = np.arange(60) % 2 + 1
    frame["sample_id"] = [f"synthetic-{i}" for i in range(60)]
    scale = 500 if target == "tap_iron" else 125
    y = scale + frame[FEATURES[0]].to_numpy() * scale * .02
    spec = yaml.safe_load(Path("configs/round2_v0_3/experiment.yaml").read_text())["model"]
    model = KernelRegressor(spec).fit(frame.iloc[:40], y[:40])
    np.testing.assert_allclose(model.preprocessor_.named_transformers_["numeric"].mean_, frame.iloc[:40][list(FEATURES)].mean())
    assert model.median_ == np.median(y[:40])
    valid = frame.iloc[40:]
    pred = model.predict(valid)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(pred, restored.predict(valid))
    shuffled = valid.sample(frac=1, random_state=3407)
    aligned = pd.Series(restored.predict(shuffled), index=shuffled.sample_id).loc[valid.sample_id].to_numpy()
    np.testing.assert_allclose(pred, aligned, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pred, np.concatenate([restored.predict(valid.iloc[:7]), restored.predict(valid.iloc[7:])]), rtol=0, atol=1e-12)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    if os.environ.get("R2_V03_ENGINEERING"):
        directory = Path(os.environ["R2_V03_ENGINEERING"])
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / f"{target}.json", {"target": target, "fits": 1, "status": "PASS", "fit_report": model.fit_report_})


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1, 2]], np.ma.array([1, 2], mask=[True, False])])
def test_hd_invalid(values):
    with pytest.raises(ValueError):
        hd_median(values)


def test_hd_reference():
    values = np.array([1., 2., 4., 8., 16.])
    assert hd_median(values) == float(hdquantiles(values, prob=[.5])[0])
