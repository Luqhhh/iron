from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "qrf_v015"))

from pooled_oob_reference import pooled_lower_median
from qrf_model import distribution_weights, lower_median


def test_equal_tree_mass_equivalence_exhaustive_small_synthetic_grid():
    rng = np.random.default_rng(31)
    for rows in range(2, 9):
        response = np.round(rng.normal(size=rows) * 5 + 50, 6)
        for size in range(1, rows + 1):
            for trees in range(1, 5):
                for _ in range(25):
                    selected = [
                        rng.choice(rows, size=size, replace=False).astype(np.int64)
                        for _ in range(trees)
                    ]
                    value, counts = pooled_lower_median(response, selected)
                    weights = distribution_weights(selected, rows)
                    assert value == lower_median(response, weights, selected)
                    concatenated = np.concatenate(selected)
                    assert np.array_equal(counts, np.bincount(concatenated, minlength=rows))
                    assert int(counts.sum()) == size * trees
