from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import stable_digest
from ..availability import freeze_history_origin
from ..exceptions import ContractError
from ..schema import validate_history


def build_history_features(
    samples: pd.DataFrame,
    history: pd.DataFrame,
    *,
    fit_cutoff: pd.Timestamp,
    available_at: str = "available_at",
    last_k: tuple[int, ...] = (3, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "sample_id",
        "spout_no",
        "reference_time",
        "tap_iron",
        "tap_time_len",
        available_at,
    }
    missing = required - set(history.columns)
    if missing:
        raise ContractError(f"history missing columns: {sorted(missing)}")
    validate_history(history, available_at=available_at)
    origin = freeze_history_origin(history, fit_cutoff, available_at=available_at)
    origin = origin.sort_values([available_at, "reference_time", "sample_id"], kind="mergesort")
    rows: list[dict[str, float]] = []
    audits: list[dict[str, object]] = []
    for sample in samples.itertuples(index=False):
        sample_id = getattr(sample, "original_sample_id", sample.sample_id)
        history_origin = getattr(sample, "history_origin", sample.reference_time)
        if history_origin > sample.reference_time:
            raise ContractError("history_origin cannot follow sample reference_time")
        visible = origin.loc[
            (origin[available_at] <= history_origin)
            & (origin["sample_id"].astype(str) != str(sample_id))
        ]
        row: dict[str, float] = {}
        for group_name, group in (
            ("all", visible),
            ("spout", visible.loc[visible["spout_no"].astype(str) == str(sample.spout_no)]),
        ):
            latest = group.iloc[-1] if len(group) else None
            if latest is None:
                age = np.nan
            else:
                age = (sample.reference_time - latest[available_at]).total_seconds() / 60.0
            row[f"history__{group_name}__latest_available_age_minutes"] = age
            for target in ("tap_iron", "tap_time_len"):
                row[f"history__{group_name}__{target}__latest"] = (
                    np.nan if latest is None else float(latest[target])
                )
                for k in last_k:
                    values = group[target].tail(k)
                    row[f"history__{group_name}__{target}__last{k}_mean"] = (
                        float(values.mean()) if len(values) else np.nan
                    )
                    row[f"history__{group_name}__{target}__last{k}_count"] = float(len(values))
        rows.append(row)
        audits.append(
            {
                "reference_time": sample.reference_time,
                "max_available_at": visible[available_at].max() if len(visible) else pd.NaT,
                "history__visible_count": len(visible),
                "history__origin_count": len(origin),
                "original_sample_id": str(sample_id),
                "view_id": str(getattr(sample, "view_id", sample.sample_id)),
                "history_origin": history_origin,
                "history_identity_sha256": stable_digest(
                    visible[["sample_id", available_at]].astype(str).to_dict("records")
                ),
            }
        )
    return pd.DataFrame(rows, index=samples.index), pd.DataFrame(audits, index=samples.index)
