"""Fixed-recipe V36 backend adapter for the V4.1-r2 nested residual core.

The r2 taskbook does not ship a trainable V36 adapter.  This module connects the
recovered A-frozen factory and the frozen V3.6 selected experts to the generic
backend protocol used by ``nested_residual_core.NestedResidualBank``.

Protocol declaration
--------------------
This is a **fixed-recipe** adapter, not a nested top-level V36 selection run:

* A_dev is rebuilt on the received training rows with the frozen V34_A
  deployment recipe (``A_frozen_deployment_weights_v1``).
* The four released V36 experts are rebuilt on the received training rows.
* The released V36 top-level weights are reused as a fixed recipe.

It therefore does not re-select V36 members/weights on the query labels.  Query
labels are never passed to ``fit`` or ``predict_state``.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import TARGETS
from .v3_6_models import V36Regressor
from .v4_1_reference import FrozenAReferenceFactory

__all__ = ["FixedV36Backend"]

V36_SUMMARY_DEFAULT = "local/runs/round2-v3.6-loss-training-and-numeric-encoding/v36-summary.json"
V36_LEDGER_DEFAULT = (
    "local/runs/round2-v3.6-loss-training-and-numeric-encoding/fixed-r2-final/fit_ledger.jsonl"
)
IRON_GAP_ENDPOINT = "v36-s1-D-0029"
TIME_GAP_ENDPOINT = "v36-s1-O-0057"


def _load_trials(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete" and isinstance(event.get("trial"), dict):
            out[str(event["trial_id"])] = event["trial"]
    if not out:
        raise ValueError(f"No complete V36 trials in {path}")
    return out


class FixedV36Backend:
    """Eager ``fit`` / lazy ``predict_state`` fixed V36 wrapper.

    ``fit`` records the authorized training frame only.  ``predict_state`` fits
    all components strictly on that recorded frame and predicts the query frame;
    query targets are never an input.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        workers: int = 16,
        summary_path: str = V36_SUMMARY_DEFAULT,
        ledger_path: str = V36_LEDGER_DEFAULT,
    ) -> None:
        self.root = Path(root).resolve()
        self.workers = max(1, int(workers))
        private = (self.root / "local/runs").resolve()
        self.summary_path = (self.root / summary_path).resolve()
        self.ledger_path = (self.root / ledger_path).resolve()
        for path in (self.summary_path, self.ledger_path):
            if not path.is_relative_to(private):
                raise ValueError("V36 adapter artifacts must stay beneath local/runs")

        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        comp = summary["complete_development_composition"]
        if comp.get("reference") != "A_development_replay":
            raise ValueError("Unexpected V36 development-composition reference")

        self.selected_experts: dict[str, list[str]] = {}
        self.frozen_weights: dict[str, np.ndarray] = {}
        for target in TARGETS:
            record = comp["targets"][target]
            names = [str(name) for name in record["selected_experts"]]
            weights = np.asarray(record["weights"], dtype=float)
            if not names or len(set(names)) != len(names):
                raise ValueError(f"Invalid V36 expert list for {target}")
            if weights.shape != (1 + len(names),) or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError(f"Invalid frozen V36 weights for {target}")
            if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-9):
                raise ValueError(f"Frozen V36 weights do not sum to one for {target}")
            self.selected_experts[target] = names
            self.frozen_weights[target] = weights

        if IRON_GAP_ENDPOINT not in self.selected_experts["tap_iron"]:
            raise ValueError("V36 iron gap endpoint is not in the released expert set")
        if TIME_GAP_ENDPOINT not in self.selected_experts["tap_time_len"]:
            raise ValueError("V36 time gap endpoint is not in the released expert set")

        trials = _load_trials(self.ledger_path)
        selected_trials: dict[str, dict] = {}
        for target in TARGETS:
            for name in self.selected_experts[target]:
                if name not in trials:
                    raise KeyError(f"Missing frozen V36 trial specification: {name}")
                selected_trials[name] = trials[name]
        self.trials = selected_trials
        self.a_factory = FrozenAReferenceFactory(self.root, workers=self.workers)
        self._train: pd.DataFrame | None = None
        self._train_y: pd.DataFrame | None = None
        self._groups: np.ndarray | None = None
        self.last_fit_meta_: dict[str, Any] = {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, *, groups: Any = None) -> "FixedV36Backend":
        if not isinstance(x, pd.DataFrame) or not x.index.is_unique or len(x) == 0:
            raise ValueError("X must be a nonempty DataFrame with unique sample IDs as index")
        if "spout_no" not in x.columns:
            raise ValueError("X must contain spout_no")
        if any(target in x.columns for target in TARGETS):
            raise ValueError("X must not contain true target columns")
        if not isinstance(y, pd.DataFrame) or tuple(y.columns) != TARGETS or not y.index.equals(x.index):
            raise ValueError("Y schema/index mismatch")
        if not np.isfinite(y.to_numpy(dtype=float)).all() or (y.to_numpy() < 0).any():
            raise ValueError("Invalid target values")
        self._train = x.copy(deep=True)
        self._train_y = y.copy(deep=True)
        self._groups = None if groups is None else np.asarray(groups)
        self.last_fit_meta_ = {"n_train": int(len(x)), "recipe": "fixed_v36_frozen_weights_v1"}
        return self

    def _fit_v36_experts(self, train_frame: pd.DataFrame, query: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {target: {} for target in TARGETS}
        for target in TARGETS:
            y = self._train_y[target].to_numpy(dtype=float)  # type: ignore[union-attr]
            for name in self.selected_experts[target]:
                trial = deepcopy(self.trials[name])
                model = V36Regressor(trial)
                model.fit(train_frame.reset_index(drop=True), y)
                prediction = np.asarray(model.predict(query.reset_index(drop=True)), dtype=float)
                if prediction.shape != (len(query),) or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid V36 expert prediction {name}")
                out[target][name] = prediction
        return out

    def predict_state(self, x_query: pd.DataFrame) -> pd.DataFrame:
        if self._train is None or self._train_y is None:
            raise RuntimeError("FixedV36Backend.fit must be called before predict_state")
        if not isinstance(x_query, pd.DataFrame) or not x_query.index.is_unique or len(x_query) == 0:
            raise ValueError("Query must be a nonempty DataFrame with unique IDs")
        if "spout_no" not in x_query.columns:
            raise ValueError("Query must contain spout_no")

        train_frame = self._train.copy(deep=True)
        train_frame.insert(0, "sample_id", train_frame.index.astype(str))
        for target in TARGETS:
            train_frame[target] = self._train_y[target].to_numpy(dtype=float)
        query_frame = x_query.copy(deep=True)
        query_frame.insert(0, "sample_id", query_frame.index.astype(str))

        a_bundle = self.a_factory.fit_predict(train_frame, query_frame)
        experts = self._fit_v36_experts(train_frame, query_frame)

        predictions: dict[str, np.ndarray] = {}
        for target in TARGETS:
            weights = self.frozen_weights[target]
            names = self.selected_experts[target]
            values = weights[0] * np.asarray(a_bundle["predictions"][target], dtype=float)
            for offset, name in enumerate(names, start=1):
                values = values + weights[offset] * experts[target][name]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite fixed V36 prediction for {target}")
            predictions[target] = np.maximum(values, 0.0)

        l1 = a_bundle.get("l1")
        if not isinstance(l1, dict) or any(target not in l1 for target in TARGETS):
            raise ValueError("A-frozen factory did not expose its L1 predictions")
        l1_iron = np.asarray(l1["tap_iron"], dtype=float)
        l1_time = np.asarray(l1["tap_time_len"], dtype=float)

        state = pd.DataFrame(
            {
                "pred_iron": predictions["tap_iron"],
                "pred_time": predictions["tap_time_len"],
                "gap_iron": l1_iron - experts["tap_iron"][IRON_GAP_ENDPOINT],
                "gap_time": l1_time - experts["tap_time_len"][TIME_GAP_ENDPOINT],
                "spout_no": x_query["spout_no"].to_numpy(),
            },
            index=x_query.index,
        )
        if set(state.columns) != {"pred_iron", "pred_time", "gap_iron", "gap_time", "spout_no"}:
            raise AssertionError("Unexpected V36 state columns")
        if not np.isfinite(state[["pred_iron", "pred_time", "gap_iron", "gap_time"]].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite V36 state")
        if (state[["pred_iron", "pred_time"]].to_numpy(dtype=float) < 0).any():
            raise ValueError("Negative V36 baseline prediction after clip")
        return state
