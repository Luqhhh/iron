from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .exceptions import ContractError
from .protection import ProtectionPolicy


@dataclass(frozen=True)
class TemporalSplit:
    id: str
    kind: str
    fit_cutoff: pd.Timestamp
    train_start: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


def split_from_config(value: dict[str, str]) -> TemporalSplit:
    required = {"id", "kind", "fit_cutoff", "train_start", "eval_start", "eval_end"}
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        raise ContractError(
            f"fold keys invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if value["kind"] not in {"development", "protected_report"}:
        raise ContractError(f"unsupported fold kind: {value['kind']}")
    if not isinstance(value["id"], str) or not value["id"].strip():
        raise ContractError("fold id must be non-empty")
    timestamps = {
        name: pd.Timestamp(value[name])
        for name in ("fit_cutoff", "train_start", "eval_start", "eval_end")
    }
    if any(timestamp.tzinfo is None for timestamp in timestamps.values()):
        raise ContractError("all fold boundaries must be timezone-aware")
    if not (
        timestamps["train_start"]
        < timestamps["fit_cutoff"]
        <= timestamps["eval_start"]
        < timestamps["eval_end"]
    ):
        raise ContractError("fold temporal order must satisfy train_start < fit_cutoff <= eval_start < eval_end")
    return TemporalSplit(
        id=value["id"],
        kind=value["kind"],
        fit_cutoff=timestamps["fit_cutoff"],
        train_start=timestamps["train_start"],
        eval_start=timestamps["eval_start"],
        eval_end=timestamps["eval_end"],
    )


def validate_split_definitions(
    values: list[dict[str, str]], protection: ProtectionPolicy
) -> list[TemporalSplit]:
    if not isinstance(values, list) or not values:
        raise ContractError("fold configuration must be a non-empty list")
    folds = [split_from_config(value) for value in values]
    ids = [fold.id for fold in folds]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate fold id")
    for fold in folds:
        for name in ("fit_cutoff", "train_start", "eval_start", "eval_end"):
            timestamp = getattr(fold, name)
            converted = timestamp.tz_convert(protection.timezone)
            if converted.utcoffset() != timestamp.utcoffset():
                raise ContractError(f"{fold.id}.{name} must use timezone {protection.timezone}")
        if fold.kind == "development":
            if fold.eval_end > protection.development_label_end_exclusive:
                raise ContractError(
                    f"development fold {fold.id} crosses protected label boundary"
                )
        elif not (
            fold.eval_start == protection.protected_start
            and fold.eval_end == protection.protected_end
        ):
            raise ContractError(
                f"protected_report fold {fold.id} must use the registered protected interval"
            )
    return folds


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
    overlap = set(train["sample_id"].astype(str)) & set(evaluation["sample_id"].astype(str))
    if overlap:
        raise ContractError(f"train/evaluation sample_id overlap: {sorted(overlap)[:5]}")
    if train["sample_id"].duplicated().any() or evaluation["sample_id"].duplicated().any():
        raise ContractError("duplicate sample_id within split partition")
    if train.empty or evaluation.empty:
        raise ContractError(f"{split.id}: empty train or evaluation partition")
    evidence = {
        "candidate_train_rows": int(in_train_time.sum()),
        "eligible_train_rows": len(train),
        "excluded_label_unavailable_rows": int((in_train_time & ~label_ready).sum()),
        "evaluation_rows": len(evaluation),
    }
    return train, evaluation, evidence
