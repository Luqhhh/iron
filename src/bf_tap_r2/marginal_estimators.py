"""Fixed, untrimmed Harrell–Davis median; no label-dependent selection."""
import numpy as np
from scipy.stats.mstats import hdquantiles


def hd_median(values):
    if np.ma.getmaskarray(values).any():
        raise ValueError("Masked labels")
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Expected nonempty finite one-dimensional labels")
    result = hdquantiles(y, prob=[0.5])[0]
    if np.ma.is_masked(result) or not np.isfinite(result):
        raise ValueError("Invalid HD estimate")
    return float(result)
