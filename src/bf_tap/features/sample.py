from __future__ import annotations

import numpy as np
import pandas as pd


def build_sample_features(samples: pd.DataFrame, missing_token: str = "__MISSING__") -> pd.DataFrame:
    reference = samples["reference_time"]
    seconds = reference.dt.hour * 3600 + reference.dt.minute * 60 + reference.dt.second
    angle = 2.0 * np.pi * seconds / 86400.0
    return pd.DataFrame(
        {
            "spout_no": samples["spout_no"].astype("string").fillna(missing_token),
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
            "weekday": reference.dt.weekday.astype(float),
        },
        index=samples.index,
    )
