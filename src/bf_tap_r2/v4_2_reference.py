"""V4.2 reference separation: B_replay versus B_fit.

Two different V3.6 references exist and are never conflated:

* ``B_replay`` -- the frozen :class:`~bf_tap_r2.v4_1_reference.V36DevelopmentReference`
  development vector.  Its historical mean package score is ``96.20376256899247``.
  It is used for regression tests and diagnostics only.
* ``B_fit`` -- :class:`~bf_tap_r2.v4_1_reference.V36FixedRecipeFactory` refitted
  **inside the current training subset**.  This is the reference for this
  round's formal relative increment.

A new model is always fitted on the same outer training part ``T`` as ``B_fit``
and evaluated on the same outer evaluation part ``V``.  A difference against
``B_replay`` is never reported as a gain over ``B_fit``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import TARGETS
from .v4_1_reference import V36DevelopmentReference, V36FixedRecipeFactory

__all__ = ["B_REPLAY_HISTORICAL_MEAN", "V42References", "baseline_cache_identity"]

#: Recorded historical mean of the frozen development replay (diagnostic only).
B_REPLAY_HISTORICAL_MEAN = 96.20376256899247


def baseline_cache_identity(
    train: pd.DataFrame, *, seed: int, fold: int, source_hash: str
) -> dict[str, Any]:
    """Identity of one cached B_fit outer fit.

    The key includes the exact training IDs, the exact training group identity,
    the model recipe identity, the preprocessing identity and the source hash,
    so a cache entry can never be reused across a different split or recipe.
    """
    import hashlib

    digest = hashlib.sha256()
    for value in train["sample_id"].astype(str).tolist():
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    group_digest = hashlib.sha256()
    for value in train["spout_no"].astype(int).tolist():
        group_digest.update(str(int(value)).encode("ascii"))
        group_digest.update(b"\x00")
    return {
        "seed": int(seed),
        "fold": int(fold),
        "n_train": int(len(train)),
        "fit_row_id_hash": digest.hexdigest(),
        "train_spout_hash": group_digest.hexdigest(),
        "model_recipe": "V36_fixed_recipe_a_frozen_deployment_weights_v1",
        "preprocessing": "frozen_v36_recipe_unchanged",
        "source_hash": str(source_hash),
    }


class V42References:
    """Loads and checks both V3.6 references for the V4.2 round."""

    def __init__(
        self,
        root: Path | str,
        train: pd.DataFrame,
        *,
        workers: int = 16,
        verify_replay: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.train = train.copy(deep=True)
        self.replay = V36DevelopmentReference(
            self.root, self.train, verify_scores=bool(verify_replay)
        )
        self.factory = V36FixedRecipeFactory(self.root, workers=int(workers))
        self._source_hash = self._resolve_source_hash()

    def _resolve_source_hash(self) -> str:
        from .v4_1_reference import V36_LEDGER_DEFAULT, V36_SUMMARY_DEFAULT, _sha256_file

        summary = self.root / V36_SUMMARY_DEFAULT
        ledger = self.root / V36_LEDGER_DEFAULT
        return f"summary:{_sha256_file(summary)}|ledger:{_sha256_file(ledger)}"

    @property
    def source_hash(self) -> str:
        return str(self._source_hash)

    def replay_diagnostics(self) -> dict[str, Any]:
        """B_replay development diagnostics; never used as a formal reference."""
        from .metrics import wmape

        payload: dict[str, Any] = {
            "reference": "B_replay",
            "identity": "DESCRIPTIVE_FROZEN_V36_DEVELOPMENT_REPLAY_NOT_NEW_OUTER",
            "historical_recorded_mean_package_score": B_REPLAY_HISTORICAL_MEAN,
            "per_seed": {},
        }
        scores: list[float] = []
        for seed in (42, 3407):
            wmape_by_target = {
                target: float(wmape(self.train[target], self.replay.baseline(int(seed), target)))
                for target in TARGETS
            }
            score = float(100.0 - 50.0 * sum(wmape_by_target.values()))
            scores.append(score)
            payload["per_seed"][str(seed)] = {
                "target_wmape": wmape_by_target,
                "package_score": score,
            }
        payload["mean_package_score"] = float(np.mean(scores))
        payload["replay_gap_to_historical"] = float(
            np.mean(scores) - B_REPLAY_HISTORICAL_MEAN
        )
        payload["note"] = (
            "B_fit scores need not equal the historical replay value; the replay gap is "
            "recorded side by side and never patched."
        )
        return payload

    def fit_baseline(
        self, train: pd.DataFrame, query: pd.DataFrame
    ) -> dict[str, Any]:
        """The formal B_fit reference: fixed V3.6 recipe refit on ``train`` only."""
        bundle = self.factory.fit_predict(train.reset_index(drop=True), query.reset_index(drop=True))
        for target in TARGETS:
            values = np.asarray(bundle["b36"][target], dtype=float)
            if values.shape != (len(query),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid B_fit prediction for {target}")
        return bundle
