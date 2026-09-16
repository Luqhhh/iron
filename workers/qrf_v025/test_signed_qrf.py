import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qrf_v015"))
sys.path.insert(0, str(HERE))

from signed_qrf import PARENT_PROTOCOL, PROTOCOL, SignedQRF


def test_signed_qrf_retains_negative_response_and_subset_invariance(monkeypatch):
    import signed_qrf
    parameters = dict(signed_qrf.PARAMETERS)
    parameters.update(n_estimators=5, n_jobs=1, min_samples_leaf=1, max_features=1.0)
    monkeypatch.setattr(signed_qrf, "PARAMETERS", parameters)
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.asarray([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], dtype=float)
    model = SignedQRF().fit(x, y, [f"i{i}" for i in range(len(x))])
    median, mean, _ = model.predict(x)
    indices = np.asarray([8, 2, 5])
    sub_median, sub_mean, _ = model.predict(x[indices])
    assert np.array_equal(sub_median, median[indices])
    assert np.array_equal(sub_mean, mean[indices])
    assert (median < 0).any() and (median > 0).any()
    assert PROTOCOL == "QRF_FULLTRAIN_SIGNED_RESPONSE_v025"
    assert PARENT_PROTOCOL == "QRF_FULLTRAIN_LEAF_v1"

