"""Stable ID-aligned stratified folds, with dependent groups taking priority."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import FEATURES


def make_folds(frame: pd.DataFrame, seed: int, n_splits: int = 5,
               groups: pd.Series | None = None) -> pd.DataFrame:
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Unique non-null sample IDs required")
    if groups is None:
        # Exact feature-vector duplicates remain together even across spouts.
        groups = pd.util.hash_pandas_object(frame[list(FEATURES)], index=False).astype(str)
    if len(groups) != len(frame) or groups.isna().any():
        raise ValueError("Invalid group assignments")
    work = frame[["sample_id", "spout_no"]].copy()
    work["group_id"] = np.asarray(groups)
    work = work.sort_values("sample_id").reset_index(drop=True)
    grouped = work.group_id.duplicated().any()
    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if grouped else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    folds = np.full(len(work), -1, dtype=int)
    for fold, (_, valid) in enumerate(splitter.split(work, work.spout_no, work.group_id if grouped else None)):
        folds[valid] = fold
    work["fold"] = folds
    work["seed"] = seed
    if work.groupby("group_id").fold.nunique().max() != 1 or (folds < 0).any():
        raise ValueError("Group leakage or incomplete fold coverage")
    return work
