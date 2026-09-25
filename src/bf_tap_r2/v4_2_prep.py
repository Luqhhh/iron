"""V4.2 shared preparation: target scaling, numeric encoding, split and hash identity.

The V4.2 task book fixes a single training convention shared by all three model
lines.  This module implements the parts that are identical for R (retrieval),
S (symbolic regression) and N (differentiable trees):

* target scaling by ``m = mean(abs(y))`` of the current training subset;
* train-internal numeric encoding (standardisation for the ``raw`` recipes, an
  eight-segment piecewise-linear encoding for the ``ple`` recipes) reusing the
  already-corrected :class:`bf_tap_r2.v3_6_networks.NumericPreprocessor`
  coordinate-system rules;
* the group-safe inner ``U``/``H`` split used for early stopping;
* exact duplicate-group identity used to forbid a training query from
  retrieving itself or a row with identical features.

Nothing in this module reads the outer evaluation fold, the test snapshot
labels, or any unprotected historical table.  Every fitted object here is
fitted on a caller-supplied training frame only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import FEATURES
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor

__all__ = [
    "FEATURES",
    "PLE_SEGMENTS",
    "TARGET_SCALE_KIND",
    "TargetScaler",
    "build_encoder",
    "encode_frame",
    "exclusion_group_keys",
    "fit_row_id_hash",
    "group_hash",
    "inner_uh_split",
    "sha256_hex",
    "transform_hash",
]

#: Eight segments, as pre-registered by the V4.2 task book.
PLE_SEGMENTS = 8

#: The task book fixes ``m = mean(abs(y))`` as the per-fit target scale.
TARGET_SCALE_KIND = "train_subset_mean_abs"

_NUMERIC_STRUCTURES = {"raw": "raw_mlp", "ple": "ple_mlp"}


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fit_row_id_hash(sample_ids: Sequence[Any]) -> str:
    """Order-sensitive hash of the exact fit rows (their ``sample_id`` values)."""
    digest = hashlib.sha256()
    for value in sample_ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def group_hash(group_ids: Sequence[Any]) -> str:
    """Order-sensitive hash of the row/duplicate-group identity of the fit rows."""
    digest = hashlib.sha256()
    for value in group_ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def transform_hash(payload: Mapping[str, Any]) -> str:
    """Deterministic hash of the input/target transformation identity."""
    import json

    return sha256_hex(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    )


@dataclass(frozen=True)
class TargetScaler:
    """``m = mean(abs(y))`` scaling of the current training subset."""

    scale: float

    @classmethod
    def fit(cls, target: Sequence[float] | np.ndarray) -> "TargetScaler":
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if not len(y):
            raise ValueError("TargetScaler requires a nonempty target")
        if not np.isfinite(y).all():
            raise ValueError("TargetScaler requires finite targets")
        scale = float(np.mean(np.abs(y)))
        if not scale > 0.0:
            raise ValueError("TargetScaler requires a positive mean absolute target")
        return cls(scale=scale)

    def transform(self, target: Sequence[float] | np.ndarray) -> np.ndarray:
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        return y / self.scale

    def inverse(self, scaled: Sequence[float] | np.ndarray) -> np.ndarray:
        z = np.asarray(scaled, dtype=np.float64).reshape(-1)
        return z * self.scale

    def metadata(self) -> dict[str, Any]:
        return {"kind": TARGET_SCALE_KIND, "scale": float(self.scale)}


def build_encoder(numeric_encoding: str, *, n_bins: int = PLE_SEGMENTS) -> NumericPreprocessor:
    """Create the pre-registered train-internal numeric encoder for a recipe."""
    key = str(numeric_encoding)
    if key not in _NUMERIC_STRUCTURES:
        raise ValueError(f"Unknown V4.2 numeric encoding: {numeric_encoding!r}")
    return NumericPreprocessor(structure=_NUMERIC_STRUCTURES[key], n_bins=int(n_bins))


def encode_frame(
    frame: pd.DataFrame, encoder: NumericPreprocessor
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(numeric_block, one_hot_spout)`` in the frozen coordinates.

    ``raw`` recipes return the train-internal standardised numerical block;
    ``ple`` recipes return the eight-segment piecewise-linear block of the raw
    values with the same bin edges.  ``spout_no`` always becomes a one-hot
    block whose index ``0`` is reserved for unseen categories.
    """
    if not isinstance(encoder, NumericPreprocessor):
        raise TypeError("encode_frame requires a fitted NumericPreprocessor")
    return encoder.transform_mlp(frame)


def exclusion_group_keys(frame: pd.DataFrame) -> np.ndarray:
    """Exact duplicate identity of "identical features" used by the retrieval ban.

    Two rows share a key when every numerical feature *and* ``spout_no`` are
    byte-identical.  Keys are derived per row from the values alone, so they are
    directly comparable across two different frames (for example an inner
    early-stopping part and its support part).  Every row is its own key group
    member, so a training query can never retrieve itself.  This identity is
    deliberately stricter than the frozen fold grouping, which hashes the 21
    numeric features only.
    """
    columns = [*FEATURES, "spout_no"]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"exclusion groups require columns: {missing}")
    numeric = np.ascontiguousarray(frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64))
    spout = frame["spout_no"].to_numpy(dtype=np.int64)
    out = np.empty(len(frame), dtype=object)
    for index in range(len(frame)):
        digest = hashlib.sha256()
        digest.update(numeric[index].tobytes())
        digest.update(b"\x00")
        digest.update(str(int(spout[index])).encode("ascii"))
        out[index] = digest.hexdigest()
    if len(set(out.tolist())) == 0 and len(frame):
        raise ValueError("Exclusion-group construction produced no keys")
    return out


def inner_uh_split(
    frame: pd.DataFrame, *, n_splits: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Group-safe inner split: fold ``0`` is the ``H`` early-stop/selection part."""
    folded = group_safe_inner_folds(frame, n_splits=int(n_splits), seed=int(seed))
    fold = np.asarray(folded["fold"], dtype=int)
    if set(np.unique(fold).tolist()) != set(range(int(n_splits))):
        raise ValueError("Inner U/H split did not cover every fold")
    holdout = fold == 0
    if not holdout.any() or holdout.all():
        raise ValueError("Inner U/H split is degenerate")
    return {
        "fold": fold,
        "group_id": np.asarray(folded["group_id"], dtype=object),
        "u_index": np.flatnonzero(~holdout),
        "h_index": np.flatnonzero(holdout),
        "group_hash": str(folded["group_hash"]),
        "inner_fold_hash": str(folded["inner_fold_hash"]),
        "n_splits": int(n_splits),
        "seed": int(seed),
    }
