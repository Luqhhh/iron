"""Shape and leakage-guard tests for the batch-0 linear probe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.next_phase_probe import build_design, ratio_features

PAIRWISE = len(FEATURES) * (len(FEATURES) - 1) // 2
EXPECTED_WIDTH = {
    "raw": len(FEATURES) + 2 + 1,
    "four": len(FEATURES) + 2 + 4 + 1,
    "degree2": len(FEATURES) + 2 + 4 + PAIRWISE + 1,
}


def frame(rows: int, seed: int, spouts=(1, 2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(size=rows) + 10.0 for name in FEATURES}
    data["sample_id"] = [f"R2S2_TRAIN_{i:012X}" for i in range(rows)]
    data["spout_no"] = [spouts[i % len(spouts)] for i in range(rows)]
    return pd.DataFrame(data)


@pytest.mark.parametrize("kind", sorted(EXPECTED_WIDTH))
def test_design_width(kind):
    train = frame(40, 1)
    assert build_design(train, train, kind).shape == (40, EXPECTED_WIDTH[kind])


def test_ratio_features_match_definition():
    train = frame(8, 2)
    ratios = ratio_features(train)
    assert np.allclose(
        ratios["oxygen_per_air_volume"], train["oxygen"] / (60.0 * train["air_volume"])
    )
    assert np.allclose(
        ratios["pressure_per_air_volume"], train["total_press_diff"] / train["air_volume"]
    )
    assert np.allclose(
        ratios["thermal_difference"], train["hot_air_temp"] - train["furnace_throat_temp"]
    )
    assert np.allclose(
        ratios["upper_pressure_fraction"], train["upper_press_diff"] / train["total_press_diff"]
    )


def test_scaling_is_frozen_on_the_training_part():
    """A validation frame must not shift the centre it is measured against."""
    train = frame(40, 3)
    held_out = frame(40, 4)
    width = len(FEATURES)

    # Scored against its own statistics the base block is centred by construction.
    assert np.allclose(build_design(train, train, "raw")[:, :width].mean(axis=0), 0.0, atol=1e-12)
    # Scored against a different training part it is off-centre, which is what
    # freezing means: the held-out frame never contributes to its own scaling.
    assert np.abs(build_design(train, held_out, "raw")[:, :width].mean(axis=0)).max() > 0.1


def test_spout_one_hot_is_explicit():
    train = frame(6, 6, spouts=(1,))
    design = build_design(train, train, "raw")
    columns = design[:, len(FEATURES) : len(FEATURES) + 2]
    assert np.allclose(columns, np.array([1.0, 0.0]))
