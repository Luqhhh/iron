"""Versioned V4.1 strong-baseline factory.

The complete historical B_star process is not present in this checkout.  The
taskbook allows the reproducible V34_A process to be used as the development
anchor.  This module reconstructs that process with the frozen deployment
weights recorded by the V3.4 recovery artefacts and refits every constituent
model on the supplied training frame only.

The recovered L1 base-model catalogue is read from the private V3.1/V3.2/V3.3
ledgers.  No query frame is ever passed to ``fit``; query frames are only used
after each constituent has been fitted on ``train``.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from .data import TARGETS
from .v3_4_models import V34Regressor
from .v3_4_sampler import sample_v34

__all__ = ["FrozenAReferenceFactory", "BaselinePrediction"]


def _load_l1_recovery(root: Path):
    script = root / "local/runs/round2-v3.4-ebm-and-constrained-composition/run_l1_oof.py"
    if not script.is_file():
        raise FileNotFoundError(
            "The recovered V3.4 L1 training script is required by the A-frozen "
            f"factory: {script}"
        )
    spec = importlib.util.spec_from_file_location("bf_tap_v41_l1_recovery", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load L1 recovery module from {script}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BaselinePrediction(dict):
    """Dictionary bundle with convenience validation for baseline predictions."""

    def validate(self, n_rows: int) -> "BaselinePrediction":
        for target in TARGETS:
            array = np.asarray(self["predictions"][target], dtype=float)
            if array.shape != (int(n_rows),) or not np.isfinite(array).all():
                raise ValueError(f"Invalid baseline prediction for {target}")
        return self


class FrozenAReferenceFactory:
    """Refit the recovered V34_A member/weight recipe on an arbitrary training frame."""

    def __init__(self, root: Path | str, *, workers: int = 16,
                 recovered_script_root: Path | str | None = None) -> None:
        self.root = Path(root).resolve()
        self.workers = max(1, int(workers))
        recovery_root = Path(recovered_script_root).resolve() if recovered_script_root else self.root
        self._recovery = _load_l1_recovery(recovery_root)
        self._trials = {str(t["trial_id"]): deepcopy(t) for t in sample_v34(self.root)}
        prepared = self.root / "local/runs/round2-v3.1-directed-search/prepared-packages-r1/manifest.json"
        release = self.root / "local/runs/round2-v3.4-ebm-and-constrained-composition/release-candidates-r1/01_V34_A_MECHANICAL_CONSTRAINED/manifest.json"
        if not prepared.is_file() or not release.is_file():
            raise FileNotFoundError("Missing recovered V3.1/V3.4 weight manifests")
        prepared_data = json.loads(prepared.read_text(encoding="utf-8"))
        release_data = json.loads(release.read_text(encoding="utf-8"))
        self._l1_weights = prepared_data["packages"]["01_v31_s3_s1_full"]
        self._target_members = {str(k): tuple(v) for k, v in release_data["target_weights"].items()}
        self._target_weights = {str(k): tuple(float(x) for x in v) for k, v in release_data["target_weight_values"].items()}
        self._expert_ids = {
            target: tuple(name for name in members if name != "L1")
            for target, members in self._target_members.items()
        }
        # Pre-declare the two disagreement sides used by E3.
        self._l1_pool = {
            "tap_iron": ("L0_iron", "v31_iron_0018", "AJM1_iron", "v31_iron_0012", "j1_42"),
            "tap_time_len": ("v31_time_0050", "v31_time_0031", "v31_time_0039", "v3_0021", "v31_time_0002"),
        }
        self._ebm_expert_ids = {
            "tap_iron": tuple(name for name in self._expert_ids["tap_iron"] if "ebm_boundary" in name),
            "tap_time_len": tuple(name for name in self._expert_ids["tap_time_len"] if "ebm_boundary" in name),
        }
        if len(self._target_members) != 2 or set(self._target_members) != set(TARGETS):
            raise ValueError("Recovered A manifest does not contain both targets")
        for target in TARGETS:
            if len(self._target_members[target]) != len(self._target_weights[target]):
                raise ValueError("A member/weight manifest length mismatch")

    def _fit_base_store(self, train: pd.DataFrame, query: pd.DataFrame) -> dict[str, np.ndarray]:
        base_names = list(self._recovery.BASE_NAMES)
        train = train.reset_index(drop=True)
        query = query.reset_index(drop=True)
        with ProcessPoolExecutor(
            max_workers=min(self.workers, len(base_names)),
            initializer=self._recovery.init_worker,
            initargs=(train, query, np.zeros(len(train))),
        ) as executor:
            futures = {
                executor.submit(self._recovery.worker, {"name": name, "mode": "full"}): name
                for name in base_names
            }
            store: dict[str, np.ndarray] = {}
            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                array = np.asarray(result["pred"], dtype=float)
                if result["target"] is None:
                    if array.ndim != 2 or array.shape != (len(query), 2):
                        raise ValueError(f"Joint base model {name} returned {array.shape}")
                elif array.shape != (len(query),) or not np.isfinite(array).all():
                    raise ValueError(f"Base model {name} returned an invalid prediction")
                store[name] = array
        return store

    def fit_predict(self, train: pd.DataFrame, query: pd.DataFrame) -> BaselinePrediction:
        """Fit all A constituents on ``train`` and predict ``query`` only."""
        if len(train) == 0 or len(query) == 0:
            raise ValueError("A-frozen factory requires nonempty train and query frames")
        base_store = self._fit_base_store(train, query)
        recovery = self._recovery
        composition = recovery.comp(base_store)
        all_store = {**base_store, **composition}
        l1_predictions: dict[str, np.ndarray] = {}
        for target in TARGETS:
            names = self._l1_pool[target]
            weights = np.asarray([self._l1_weights["iron_weights" if target == "tap_iron" else "time_weights"][name]
                                  for name in names], dtype=float)
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError(f"Invalid frozen L1 weights for {target}")
            weights = weights / weights.sum()
            columns = np.column_stack([recovery.get(all_store, name, target) for name in names])
            l1_predictions[target] = columns @ weights

        expert_predictions: dict[str, dict[str, np.ndarray]] = {target: {} for target in TARGETS}
        for target in TARGETS:
            y = np.asarray(train[target], dtype=float)
            for trial_id in self._expert_ids[target]:
                if trial_id not in self._trials:
                    raise KeyError(f"Missing recovered expert trial: {trial_id}")
                model = V34Regressor(deepcopy(self._trials[trial_id]))
                model.fit(train.reset_index(drop=True), y)
                expert_predictions[target][trial_id] = np.asarray(model.predict(query.reset_index(drop=True)), dtype=float)

        predictions: dict[str, np.ndarray] = {}
        disagreement: dict[str, np.ndarray] = {}
        for target in TARGETS:
            members = self._target_members[target]
            weights = self._target_weights[target]
            parts = []
            for name, weight in zip(members, weights):
                part = l1_predictions[target] if name == "L1" else expert_predictions[target][name]
                parts.append(float(weight) * part)
            predictions[target] = np.sum(parts, axis=0)
            cb_names = self._l1_pool[target]
            cb_mean = np.mean(np.column_stack([
                recovery.get(all_store, name, target) for name in cb_names
            ]), axis=1)
            ebm_names = self._ebm_expert_ids[target]
            if ebm_names:
                ebm_mean = np.mean(np.column_stack([
                    expert_predictions[target][name] for name in ebm_names
                ]), axis=1)
            else:
                ebm_mean = np.zeros(len(query), dtype=float)
            disagreement[target] = cb_mean - ebm_mean

        members_store: dict[str, dict[str, np.ndarray]] = {}
        for name, value in all_store.items():
            try:
                members_store[name] = {target: recovery.get(all_store, name, target) for target in TARGETS}
            except Exception:
                continue
        for target in TARGETS:
            for trial_id, value in expert_predictions[target].items():
                members_store[trial_id] = {target: value}

        bundle = BaselinePrediction(
            predictions=predictions,
            disagreement=disagreement,
            l1={target: np.asarray(l1_predictions[target], dtype=float) for target in TARGETS},
            experts={target: {name: np.asarray(value, dtype=float) for name, value in expert_predictions[target].items()}
                      for target in TARGETS},
            members=members_store,
            meta={
                "factory": "A_frozen_deployment_weights_v1",
                "n_train": int(len(train)),
                "n_query": int(len(query)),
                "expert_ids": {target: list(self._expert_ids[target]) for target in TARGETS},
                "ebm_expert_ids": {target: list(self._ebm_expert_ids[target]) for target in TARGETS},
                "l1_member_pool": {target: list(self._l1_pool[target]) for target in TARGETS},
                "disagreement_definition": "mean(CB L1 member pool) - mean(EBM boundary experts)",
            },
        )
        return bundle.validate(len(query))
