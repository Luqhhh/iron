from __future__ import annotations

import pandas as pd

from .exceptions import ContractError
from .models.baseline import DualTargetBaseline


def predict_with_ids(
    model: DualTargetBaseline, samples: pd.DataFrame, X: pd.DataFrame
) -> pd.DataFrame:
    if not samples.index.equals(X.index) or "sample_id" not in samples:
        raise ContractError("prediction metadata and features are not aligned")
    predicted = model.predict(X)
    return pd.DataFrame(
        {
            "sample_id": samples["sample_id"].astype("string").to_numpy(),
            "pred_tap_iron": predicted["pred_tap_iron"].to_numpy(),
            "pred_tap_time_len": predicted["pred_tap_time_len"].to_numpy(),
        }
    )
