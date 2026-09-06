from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..exceptions import ContractError


@dataclass
class MedianControls:
    min_group_count: int = 20

    def fit(self, samples: pd.DataFrame) -> "MedianControls":
        targets = ["tap_iron", "tap_time_len"]
        if samples[targets].isna().any().any() or len(samples) == 0:
            raise ContractError("controls require non-empty complete targets")
        self.global_ = samples[targets].median().to_dict()
        grouped = samples.groupby(samples["spout_no"].astype(str), sort=True)[targets]
        counts = grouped.size()
        medians = grouped.median()
        self.by_spout_ = {
            str(group): medians.loc[group].to_dict()
            for group in medians.index
            if counts.loc[group] >= self.min_group_count
        }
        return self

    def predict(self, samples: pd.DataFrame, control: str) -> pd.DataFrame:
        if not hasattr(self, "global_"):
            raise ContractError("controls are not fitted")
        result = pd.DataFrame({"sample_id": samples["sample_id"].astype("string")})
        for target in ("tap_iron", "tap_time_len"):
            if control == "B0":
                values = np.full(len(samples), self.global_[target])
            elif control == "B1":
                values = [
                    self.by_spout_.get(str(spout), self.global_)[target]
                    for spout in samples["spout_no"]
                ]
            else:
                raise ContractError(f"unknown control: {control}")
            result[f"pred_{target}"] = values
        return result
