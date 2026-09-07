from __future__ import annotations

import pandas as pd

from ..exceptions import ContractError


def add_process_change_features(
    frame: pd.DataFrame,
    config: dict,
    *,
    baseline_value_columns: list[str],
) -> pd.DataFrame:
    """Add the pre-registered signed changes using existing as-of aggregates."""
    values = list(config["value_columns"])
    if values != list(baseline_value_columns):
        raise ContractError(
            "process-change value columns differ from the frozen baseline operation columns"
        )
    output = frame.copy()
    for value in values:
        latest = f"operation__{value}__latest"
        mean_6h = f"operation__{value}__6h__mean"
        mean_24h = f"operation__{value}__24h__mean"
        missing = {latest, mean_6h, mean_24h} - set(output)
        if missing:
            raise ContractError(
                f"process-change inputs missing for {value}: {sorted(missing)}"
            )
        additions = {
            f"process_change__{value}__latest_minus_6h_mean": output[latest]
            - output[mean_6h],
            f"process_change__{value}__latest_minus_24h_mean": output[latest]
            - output[mean_24h],
            f"process_change__{value}__mean_6h_minus_24h_mean": output[mean_6h]
            - output[mean_24h],
        }
        collision = set(additions) & set(output)
        if collision:
            raise ContractError(
                f"process-change feature collision: {sorted(collision)}"
            )
        for name, series in additions.items():
            output[name] = series
    return output
