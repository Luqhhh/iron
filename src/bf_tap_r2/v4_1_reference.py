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


# ---------------------------------------------------------------------------
# V3.6 development replay and fixed-recipe B36 factory
# ---------------------------------------------------------------------------

V36_SUMMARY_DEFAULT = "local/runs/round2-v3.6-loss-training-and-numeric-encoding/v36-summary.json"
V36_LEDGER_DEFAULT = (
    "local/runs/round2-v3.6-loss-training-and-numeric-encoding/fixed-r2-final/fit_ledger.jsonl"
)
V36_DEV_CACHE_DEFAULT = (
    "local/runs/round2-v3.6-loss-training-and-numeric-encoding/complete-dev-r2-final"
)
V36_FIXED_SLOTS = {
    "tap_iron": "v36-s1-D-0029",
    "tap_time_len": "v36-s1-O-0057",
}
V36_DEVELOPMENT_SEEDS = (42, 3407)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v36_trials(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load complete trial specifications from the frozen V3.6 fit ledger."""
    ledger = Path(path)
    if not ledger.is_file():
        raise FileNotFoundError(ledger)
    out: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete" and isinstance(event.get("trial"), dict):
            out[str(event["trial_id"])] = deepcopy(event["trial"])
    if not out:
        raise ValueError(f"No complete V3.6 trials in {ledger}")
    return out


class V36DevelopmentReference:
    """Exact V3.6 complete-development replay from frozen manifest + OOF caches.

    The class loads the released composition weights and selected experts from
    ``v36-summary.json`` and rebuilds the per-seed B36 development OOF vector
    from already-recorded, order-aligned components:

    * the frozen V34_A development OOF vector (loaded through the recovered
      V3.4 artefacts);
    * the four selected V3.6 expert OOF vectors in ``complete-dev-r2-final``.

    It verifies that the reconstructed full-package score equals the manifest's
    recorded per-seed score.  It never fits a model, never selects a weight, and
    never writes a submission.  This is a development-replay identity, not a
    new independent test set.
    """

    def __init__(
        self,
        root: Path | str,
        train: pd.DataFrame,
        *,
        summary_path: Path | str = V36_SUMMARY_DEFAULT,
        ledger_path: Path | str = V36_LEDGER_DEFAULT,
        cache_dir: Path | str = V36_DEV_CACHE_DEFAULT,
        verify_scores: bool = True,
    ) -> None:
        from .metrics import wmape

        self.root = Path(root).resolve()
        self.train = train.copy(deep=True)
        self.summary_path = (self.root / summary_path).resolve() if not Path(summary_path).is_absolute() else Path(summary_path).resolve()
        self.ledger_path = (self.root / ledger_path).resolve() if not Path(ledger_path).is_absolute() else Path(ledger_path).resolve()
        self.cache_dir = (self.root / cache_dir).resolve() if not Path(cache_dir).is_absolute() else Path(cache_dir).resolve()
        if not self.summary_path.is_file() or not self.ledger_path.is_file() or not self.cache_dir.is_dir():
            raise FileNotFoundError("V3.6 summary/ledger/development cache is incomplete")
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        comp = summary.get("complete_development_composition")
        if not isinstance(comp, dict) or comp.get("reference") != "A_development_replay":
            raise ValueError("V3.6 summary is not the released A-development composition")
        self.weights: dict[str, np.ndarray] = {}
        self.selected_experts: dict[str, list[str]] = {}
        for target in TARGETS:
            record = comp.get("targets", {}).get(target)
            if not isinstance(record, dict):
                raise ValueError(f"V3.6 summary is missing target {target}")
            names = [str(name) for name in record["selected_experts"]]
            weights = np.asarray(record["weights"], dtype=float)
            if len(weights) != 1 + len(names):
                raise ValueError(f"V3.6 weight/expert mismatch for {target}")
            if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-9):
                raise ValueError(f"Invalid V3.6 weights for {target}")
            self.selected_experts[target] = names
            self.weights[target] = weights
        self.trials = load_v36_trials(self.ledger_path)
        missing = sorted({
            name for names in self.selected_experts.values() for name in names
        } - set(self.trials))
        if missing:
            raise KeyError(f"Missing V3.6 trial specifications: {missing}")
        self._expected_scores = {
            str(seed): float(comp["per_seed_candidate_package_score"][str(seed)])
            for seed in V36_DEVELOPMENT_SEEDS
        }
        self._a_dev = self._load_a_dev()
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self._source_hashes = {
            str(self.summary_path.relative_to(self.root)): _sha256_file(self.summary_path),
            str(self.ledger_path.relative_to(self.root)): _sha256_file(self.ledger_path),
        }
        for seed in V36_DEVELOPMENT_SEEDS:
            self._cache[int(seed)] = self._build_seed(int(seed), wmape=wmape if verify_scores else None)

    def _load_a_dev(self) -> dict[str, dict[str, np.ndarray]]:
        # Imported lazily to avoid a module cycle with v4_run.
        from .v4_run import _load_a_dev_reference

        return _load_a_dev_reference(self.root, self.train)

    def _expert_vector(self, seed: int, trial_id: str) -> np.ndarray:
        path = self.cache_dir / f"seed-{int(seed)}" / f"pred-{trial_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.load(path).astype(float, copy=False)
        if values.shape != (len(self.train),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid V3.6 development expert cache: {path}")
        return values

    def _build_seed(self, seed: int, wmape: Any = None) -> dict[str, np.ndarray]:
        if seed not in self._a_dev:
            raise KeyError(f"V34_A replay lacks seed {seed}")
        b36: dict[str, np.ndarray] = {}
        parent: dict[str, np.ndarray] = {}
        experts: dict[str, dict[str, np.ndarray]] = {}
        for target in TARGETS:
            names = self.selected_experts[target]
            values = self.weights[target][0] * np.asarray(self._a_dev[seed][target], dtype=float)
            experts[target] = {}
            for offset, name in enumerate(names, start=1):
                vector = self._expert_vector(seed, name)
                experts[target][name] = vector
                values = values + self.weights[target][offset] * vector
            b36[target] = np.maximum(values, 0.0)
            parent[target] = experts[target][V36_FIXED_SLOTS[target]]
        if wmape is not None:
            target_wmape = {
                target: float(wmape(self.train[target].to_numpy(dtype=float), b36[target]))
                for target in TARGETS
            }
            package_score = 100.0 - 50.0 * sum(target_wmape.values())
            if not np.isclose(package_score, self._expected_scores[str(seed)], rtol=0.0, atol=1e-9):
                raise AssertionError(
                    f"V3.6 development replay score mismatch for seed {seed}: "
                    f"reconstructed={package_score!r}, manifest={self._expected_scores[str(seed)]!r}"
                )
        return {"b36": b36, "parent": parent, "experts": experts}

    def baseline(self, seed: int, target: str) -> np.ndarray:
        return np.asarray(self._cache[int(seed)]["b36"][target], dtype=float)

    def parent(self, seed: int, target: str) -> np.ndarray:
        return np.asarray(self._cache[int(seed)]["parent"][target], dtype=float)

    def slot_expert(self, target: str) -> str:
        return V36_FIXED_SLOTS[target]

    def slot_weight(self, target: str) -> float:
        names = self.selected_experts[target]
        index = names.index(V36_FIXED_SLOTS[target])
        return float(self.weights[target][index + 1])

    def meta(self) -> dict[str, Any]:
        return {
            "reference": "V36_DEVELOPMENT_REPLAY_A_DEV_OOF_PLUS_FROZEN_EXPERTS",
            "evidence_level": "DESCRIPTIVE_FROZEN_DEVELOPMENT_REPLAY_NOT_NEW_OUTER",
            "weights_reselected": False,
            "seeds": list(V36_DEVELOPMENT_SEEDS),
            "fixed_slots": dict(V36_FIXED_SLOTS),
            "selected_experts": {target: list(names) for target, names in self.selected_experts.items()},
            "weights": {target: [float(v) for v in self.weights[target]] for target in TARGETS},
            "slot_weights": {target: self.slot_weight(target) for target in TARGETS},
            "source_hashes": dict(self._source_hashes),
            "identity_limit": (
                "NPY caches carry no embedded IDs; identity is established by the frozen "
                "train row order, fold files, manifest weights, and per-seed score replay."
            ),
        }


class V36FixedRecipeFactory:
    """Fit the frozen V3.6 composition on an arbitrary training subset.

    This is the fallback used when a requested outer split is outside the
    recorded development seeds.  It is explicitly a fixed-recipe adapter:
    V34_A is rebuilt with its frozen deployment weights (not a nested top-level
    re-selection of A), and the released V3.6 top-level weights are reused.
    Query labels are never passed to :meth:`fit_predict`.
    """

    def __init__(self, root: Path | str, *, workers: int = 16,
                 summary_path: Path | str = V36_SUMMARY_DEFAULT,
                 ledger_path: Path | str = V36_LEDGER_DEFAULT) -> None:
        self.root = Path(root).resolve()
        self.summary_path = (self.root / summary_path).resolve() if not Path(summary_path).is_absolute() else Path(summary_path).resolve()
        self.ledger_path = (self.root / ledger_path).resolve() if not Path(ledger_path).is_absolute() else Path(ledger_path).resolve()
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        comp = summary.get("complete_development_composition")
        if not isinstance(comp, dict) or comp.get("reference") != "A_development_replay":
            raise ValueError("V3.6 summary is not the released A-development composition")
        self.selected_experts: dict[str, list[str]] = {}
        self.weights: dict[str, np.ndarray] = {}
        for target in TARGETS:
            record = comp["targets"][target]
            names = [str(name) for name in record["selected_experts"]]
            weights = np.asarray(record["weights"], dtype=float)
            if len(weights) != 1 + len(names) or not np.isclose(weights.sum(), 1.0, atol=1e-9):
                raise ValueError(f"Invalid V3.6 composition for {target}")
            self.selected_experts[target] = names
            self.weights[target] = weights
        self.trials = load_v36_trials(self.ledger_path)
        missing = sorted({
            name for names in self.selected_experts.values() for name in names
        } - set(self.trials))
        if missing:
            raise KeyError(f"Missing V3.6 trial specifications: {missing}")
        self.a_factory = FrozenAReferenceFactory(self.root, workers=workers)

    def fit_predict(self, train: pd.DataFrame, query: pd.DataFrame) -> dict[str, Any]:
        from .v3_6_models import V36Regressor

        if "sample_id" not in train.columns or "sample_id" not in query.columns:
            raise ValueError("V36FixedRecipeFactory frames require sample_id columns")
        for target in TARGETS:
            if target not in train.columns:
                raise ValueError(f"V36FixedRecipeFactory train frame lacks {target}")
        a_bundle = self.a_factory.fit_predict(train, query)
        expert_predictions: dict[str, dict[str, np.ndarray]] = {target: {} for target in TARGETS}
        for target in TARGETS:
            y = np.asarray(train[target], dtype=float)
            for name in self.selected_experts[target]:
                model = V36Regressor(deepcopy(self.trials[name]))
                model.fit(train.reset_index(drop=True), y)
                values = np.asarray(model.predict(query.reset_index(drop=True)), dtype=float)
                if values.shape != (len(query),) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid fixed-recipe V3.6 expert prediction: {name}")
                expert_predictions[target][name] = values
        b36: dict[str, np.ndarray] = {}
        parent: dict[str, np.ndarray] = {}
        for target in TARGETS:
            weights = self.weights[target]
            names = self.selected_experts[target]
            values = weights[0] * np.asarray(a_bundle["predictions"][target], dtype=float)
            for offset, name in enumerate(names, start=1):
                values = values + weights[offset] * expert_predictions[target][name]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite fixed-recipe B36 prediction for {target}")
            b36[target] = np.maximum(values, 0.0)
            parent[target] = expert_predictions[target][V36_FIXED_SLOTS[target]]
        return {
            "b36": b36,
            "parent": parent,
            "a": {target: np.asarray(a_bundle["predictions"][target], dtype=float) for target in TARGETS},
            "experts": expert_predictions,
            "weights": {target: [float(v) for v in self.weights[target]] for target in TARGETS},
            "selected_experts": {target: list(names) for target, names in self.selected_experts.items()},
            "meta": {
                "factory": "V36_fixed_recipe_a_frozen_deployment_weights_v1",
                "n_train": int(len(train)),
                "n_query": int(len(query)),
                "weights_reselected": False,
            },
        }
