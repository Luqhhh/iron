from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .exceptions import ContractError


@dataclass(frozen=True)
class TemporalSplit:
    id: str
    kind: str
    fit_cutoff: pd.Timestamp
    train_start: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


def split_from_config(value: dict[str, str]) -> TemporalSplit:
    return TemporalSplit(
        id=value["id"],
        kind=value["kind"],
        fit_cutoff=pd.Timestamp(value["fit_cutoff"]),
        train_start=pd.Timestamp(value["train_start"]),
        eval_start=pd.Timestamp(value["eval_start"]),
        eval_end=pd.Timestamp(value["eval_end"]),
    )


def select_split(
    samples: pd.DataFrame,
    split: TemporalSplit,
    *,
    label_available_at: str = "label_available_at",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    required = {"reference_time", label_available_at}
    missing = required - set(samples.columns)
    if missing:
        raise ContractError(f"split input missing columns: {sorted(missing)}")
    reference = samples["reference_time"]
    in_train_time = (reference >= split.train_start) & (reference < split.fit_cutoff)
    label_ready = samples[label_available_at] <= split.fit_cutoff
    train = samples.loc[in_train_time & label_ready].copy()
    evaluation = samples.loc[
        (reference >= split.eval_start) & (reference < split.eval_end)
    ].copy()
    evidence = {
        "candidate_train_rows": int(in_train_time.sum()),
        "eligible_train_rows": len(train),
        "excluded_label_unavailable_rows": int((in_train_time & ~label_ready).sum()),
        "evaluation_rows": len(evaluation),
    }
    return train, evaluation, evidence
