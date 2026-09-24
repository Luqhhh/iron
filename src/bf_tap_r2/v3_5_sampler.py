"""V3.5 deterministic, label-free sampler.

The sampler expands the pre-registered 256-slot V3.5 fixed schedule:

* ``R``: 144 regularized EBM recipes (72 per target)
* ``I``: 64 feature-expression / explicit-interaction recipes (32 per target)
* ``L``: 48 global-plus-local EBM evaluation slots (24 per target)

It reads only public configuration and recorded V3.4 development trials used as
parent/centre definitions.  It never reads target labels, test labels, or outer
validation identities.  Exact duplicate identities are kept as schedule slots
and marked ``duplicate`` so the runner does not retrain them.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .data import FEATURES, TARGETS
from .v3_1_search import canonical_json, canonical_trial_hash
from .v3_run import read_complete_records

CONFIG_VERSION = "round2-v3.5-regularized-ebm-and-composition"
LINE_COUNTS = {"R": 144, "I": 64, "L": 48}
TARGET_LINE_COUNTS = {
    "R": {"tap_iron": 72, "tap_time_len": 72},
    "I": {"tap_iron": 32, "tap_time_len": 32},
    "L": {"tap_iron": 24, "tap_time_len": 24},
}
TARGET_KEYS = {"tap_iron": "iron", "tap_time_len": "time"}
RAW_EBM_COLUMNS = tuple([*FEATURES, "spout_no"])
FOUR_DERIVED_COLUMNS = (
    "oxygen_per_air_volume",
    "pressure_per_air_volume",
    "thermal_difference",
    "upper_pressure_fraction",
)
FOUR_EBM_COLUMNS = tuple([*RAW_EBM_COLUMNS, *FOUR_DERIVED_COLUMNS])


def load_v35_config(root: Path | str) -> dict:
    root = Path(root)
    path = root / "configs/round2_v3_5/search.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != CONFIG_VERSION:
        raise ValueError("Unexpected V3.5 search configuration")
    budget = config.get("fixed_budget", {})
    for line, expected in LINE_COUNTS.items():
        if int(budget.get("lines", {}).get(line, -1)) != expected:
            raise ValueError(f"V3.5 line budget mismatch for {line}")
    if int(budget.get("total", -1)) != sum(LINE_COUNTS.values()):
        raise ValueError("V3.5 fixed total must be 256")
    return config


def _base_recipe(config: Mapping[str, Any]) -> dict:
    raw = deepcopy(config["recipe"])
    allowed = {
        "max_bins", "min_samples_leaf", "interactions", "max_interaction_bins",
        "max_leaves", "objective", "learning_rate", "outer_bags", "inner_bags",
        "max_rounds", "early_stopping_rounds", "random_state", "n_jobs",
    }
    required_meta = {"objective", "target_transform", "outer_bags", "inner_bags",
                     "max_rounds", "early_stopping_rounds", "learning_rate", "max_leaves",
                     "random_state", "n_jobs", "bag_protocol", "inner_splits", "bag_seed"}
    missing = required_meta - set(raw)
    if missing:
        raise ValueError(f"V3.5 base recipe is missing explicit keys: {sorted(missing)}")
    if raw["objective"] != "rmse":
        raise ValueError("V3.5 EBM objective is frozen to rmse")
    if raw["bag_protocol"] != "group-safe-bags-v1":
        raise ValueError("V3.5 bag protocol must be group-safe-bags-v1")
    recipe = {key: deepcopy(raw[key]) for key in allowed if key in raw}
    recipe["objective"] = "rmse"
    recipe["target_transform"] = str(raw["target_transform"])
    recipe["outer_bags"] = int(raw["outer_bags"])
    recipe["inner_bags"] = int(raw["inner_bags"])
    recipe["max_rounds"] = int(raw["max_rounds"])
    recipe["early_stopping_rounds"] = int(raw["early_stopping_rounds"])
    recipe["learning_rate"] = float(raw["learning_rate"])
    recipe["max_leaves"] = int(raw["max_leaves"])
    recipe["random_state"] = int(raw["random_state"])
    recipe["n_jobs"] = int(raw["n_jobs"])
    recipe["bag_protocol"] = str(raw["bag_protocol"])
    recipe["inner_splits"] = int(raw["inner_splits"])
    recipe["bag_seed"] = int(raw["bag_seed"])
    return recipe


def _feature_columns(feature_set: str) -> tuple[str, ...]:
    if feature_set == "raw":
        return RAW_EBM_COLUMNS
    if feature_set == "four":
        return FOUR_EBM_COLUMNS
    raise ValueError(f"Unknown V3.5 feature pack: {feature_set}")


def resolve_explicit_interactions(columns: Sequence[str],
                                  pairs: Sequence[Sequence[str]]) -> list[tuple[int, int]]:
    """Map named feature pairs to current frame indices, deduplicate, and sort.

    Explicit pairs are pairwise only.  The function is deliberately strict: a
    missing name raises instead of silently dropping a registered interaction.
    """
    index = {str(name): i for i, name in enumerate(columns)}
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Explicit interaction must be a pair, got {pair!r}")
        left, right = str(pair[0]), str(pair[1])
        if left not in index or right not in index:
            raise KeyError(f"Explicit interaction feature missing: {pair!r}")
        i, j = int(index[left]), int(index[right])
        if i == j:
            continue
        key = tuple(sorted((i, j)))
        seen.add((int(key[0]), int(key[1])))
    return [tuple(pair) for pair in sorted(seen)]


def _load_recorded_trials(root: Path) -> dict[str, dict]:
    config = load_v35_config(root)
    source = config["I_feature_and_explicit_interactions"]["parent_source"]
    paths = [
        root / str(source["refine_ledger"]),
        root / str(source.get("fallback_refine_ledger", "")),
    ]
    out: dict[str, dict] = {}
    for path in paths:
        if not path or not path.exists():
            continue
        for record in read_complete_records(path):
            trial = record.get("trial")
            if isinstance(trial, dict) and trial.get("trial_id"):
                out[str(trial["trial_id"])] = deepcopy(trial)
    if not out:
        raise FileNotFoundError(
            "V3.5 requires recorded V3.4 development trials to freeze parent EBM centres"
        )
    return out


def _select_parent_centers(root: Path) -> dict[str, list[dict | None]]:
    config = load_v35_config(root)
    spec = config["I_feature_and_explicit_interactions"]
    recorded = _load_recorded_trials(root)
    centers: dict[str, list[dict | None]] = {}
    for target in TARGETS:
        preferred = [str(v) for v in spec["parent_centers"][target]]
        selected: list[dict | None] = []
        for trial_id in preferred:
            trial = recorded.get(trial_id)
            if trial is None:
                selected.append(None)
                continue
            if str(trial.get("target")) != target:
                raise ValueError(f"V3.5 parent centre {trial_id} target mismatch")
            if str(trial.get("kind")) not in {"ebm", "ebm_boundary", "ebm_base", "ebm_regularized"}:
                raise ValueError(f"V3.5 parent centre {trial_id} is not an EBM trial")
            selected.append(deepcopy(trial))
        # Preserve the pre-registered two-slot frame even when a centre is
        # absent; the absent slots are represented as N/A below.
        while len(selected) < 2:
            selected.append(None)
        centers[target] = selected[:2]
    return centers


def build_R_trials(config: Mapping[str, Any]) -> list[dict]:
    recipe = _base_recipe(config)
    grid_spec = config["R_regularized_ebm"]
    out: list[dict] = []
    for target, key in TARGET_KEYS.items():
        grid = grid_spec[key]
        for max_bins in grid["max_bins"]:
            for min_samples_leaf in grid["min_samples_leaf"]:
                for interactions in grid["interactions"]:
                    for max_interaction_bins in grid["max_interaction_bins"]:
                        params = deepcopy(recipe)
                        params.update({
                            "max_bins": int(max_bins),
                            "min_samples_leaf": int(min_samples_leaf),
                            "interactions": int(interactions),
                            "max_interaction_bins": int(max_interaction_bins),
                        })
                        out.append({
                            "kind": "ebm_regularized",
                            "line": "R",
                            "target": target,
                            "target_transform": str(recipe["target_transform"]),
                            "feature_set": "raw",
                            "parameters": params,
                            "protocol": {
                                "bags": str(recipe["bag_protocol"]),
                                "inner_splits": int(recipe["inner_splits"]),
                                "bag_seed": int(recipe["bag_seed"]),
                            },
                            "status": "available",
                        })
    if len(out) != LINE_COUNTS["R"]:
        raise AssertionError(f"V3.5 R sampler produced {len(out)} trials")
    return out


def _explicit_pairs(config: Mapping[str, Any], strategy: str) -> list[tuple[str, str]]:
    spec = config["I_feature_and_explicit_interactions"]
    if strategy == "I2":
        return [tuple(pair) for pair in spec["explicit_pairs_I2"]]
    if strategy == "I3":
        pairs = list(spec["explicit_pairs_I2"]) + list(spec["explicit_pairs_I3_extra"])
        return [tuple(pair) for pair in pairs]
    raise ValueError(strategy)


def _interaction_for_strategy(config: Mapping[str, Any], parent: Mapping[str, Any],
                              feature_set: str, strategy: str) -> int | list[tuple[int, int]]:
    parent_params = dict(parent.get("parameters", {}))
    if strategy == "I0":
        return 0
    if strategy == "I1":
        value = parent_params.get("interactions", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            # Parent centres in the pre-registered V3.4 pool use integer
            # automatic interaction counts.  Explicit parent tuples are not
            # silently accepted as an I1 recipe.
            raise ValueError("V3.5 I1 requires an integer parent interaction count")
        return int(value)
    columns = _feature_columns(feature_set)
    return resolve_explicit_interactions(columns, _explicit_pairs(config, strategy))


def build_I_trials(config: Mapping[str, Any], centers: Mapping[str, list[dict | None]]) -> list[dict]:
    spec = config["I_feature_and_explicit_interactions"]
    out: list[dict] = []
    for target, key in TARGET_KEYS.items():
        for center_index, parent in enumerate(centers.get(target, [])):
            present = parent is not None
            for feature_set in spec["feature_packs"]:
                for multiplier in spec["leaf_multipliers"]:
                    for strategy in ("I0", "I1", "I2", "I3"):
                        if not present:
                            out.append({
                                "kind": "ebm_expression",
                                "line": "I",
                                "target": target,
                                "status": "not_applicable",
                                "reason": f"V3.5 I parent centre {center_index + 1} is unavailable",
                                "parent_index": int(center_index + 1),
                                "feature_set": str(feature_set),
                                "leaf_multiplier": int(multiplier),
                                "interaction_strategy": str(strategy),
                                "parameters": {},
                            })
                            continue
                        assert parent is not None
                        parent_params = dict(parent.get("parameters", {}))
                        if "min_samples_leaf" not in parent_params:
                            raise ValueError(f"V3.5 parent {parent.get('trial_id')} lacks min_samples_leaf")
                        leaf = max(1, int(round(float(parent_params["min_samples_leaf"]) * float(multiplier))))
                        params = deepcopy(parent_params)
                        params["min_samples_leaf"] = int(leaf)
                        params["interactions"] = _interaction_for_strategy(config, parent, str(feature_set), str(strategy))
                        # Parent target transform, bag protocol and remaining
                        # recipe are inherited exactly.
                        out.append({
                            "kind": "ebm_expression",
                            "line": "I",
                            "target": target,
                            "target_transform": str(parent["target_transform"]),
                            "feature_set": str(feature_set),
                            "status": "available",
                            "parent_index": int(center_index + 1),
                            "parent_trial_id": str(parent.get("trial_id")),
                            "leaf_multiplier": int(multiplier),
                            "interaction_strategy": str(strategy),
                            "parameters": params,
                            "protocol": {
                                "bags": "group-safe-bags-v1",
                                "inner_splits": 5,
                                "bag_seed": int(params.get("random_state", 42)),
                            },
                        })
    if len(out) != LINE_COUNTS["I"]:
        raise AssertionError(f"V3.5 I sampler produced {len(out)} trials")
    return out


def _local_trial(parent: Mapping[str, Any], min_samples_leaf: int,
                 interactions: int) -> dict:
    trial = deepcopy(dict(parent))
    params = dict(trial.get("parameters", {}))
    params["min_samples_leaf"] = int(min_samples_leaf)
    params["interactions"] = int(interactions)
    trial["parameters"] = params
    trial["kind"] = "ebm_regularized"
    return trial


def build_L_trials(config: Mapping[str, Any], centers: Mapping[str, list[dict | None]]) -> list[dict]:
    spec = config["L_global_local_ebm"]
    out: list[dict] = []
    for target, _key in TARGET_KEYS.items():
        for center_index, parent in enumerate(centers.get(target, [])):
            present = parent is not None
            for local_leaf in spec["local_min_samples_leaf"]:
                for local_interactions in spec["local_interactions"]:
                    for beta in spec["beta_values"]:
                        if not present:
                            out.append({
                                "kind": "global_spout_ebm",
                                "line": "L",
                                "target": target,
                                "status": "not_applicable",
                                "reason": f"V3.5 L parent centre {center_index + 1} is unavailable",
                                "parent_index": int(center_index + 1),
                                "local_min_samples_leaf": int(local_leaf),
                                "local_interactions": int(local_interactions),
                                "beta": float(beta),
                                "parameters": {},
                            })
                            continue
                        assert parent is not None
                        out.append({
                            "kind": "global_spout_ebm",
                            "line": "L",
                            "target": target,
                            "target_transform": str(parent["target_transform"]),
                            "status": "available",
                            "parent_index": int(center_index + 1),
                            "parent_trial_id": str(parent.get("trial_id")),
                            "local_min_samples_leaf": int(local_leaf),
                            "local_interactions": int(local_interactions),
                            "beta": float(beta),
                            "parameters": {
                                "parent_trial": deepcopy(dict(parent)),
                                "local_min_samples_leaf": int(local_leaf),
                                "local_interactions": int(interactions := local_interactions),
                                "beta": float(beta),
                                "min_spout_samples": int(spec["min_spout_samples"]),
                                "local_include_spout": bool(spec["local_include_spout"]),
                                "feature_set": str(parent.get("feature_set", "raw")),
                                "component_key": canonical_json({
                                    "target": target,
                                    "parent_trial_id": str(parent.get("trial_id")),
                                    "local_min_samples_leaf": int(local_leaf),
                                    "local_interactions": int(interactions),
                                }),
                            },
                            "protocol": {
                                "bags": "group-safe-bags-v1",
                                "inner_splits": 5,
                                "bag_seed": int(dict(parent.get("parameters", {})).get("random_state", 42)),
                            },
                        })
    if len(out) != LINE_COUNTS["L"]:
        raise AssertionError(f"V3.5 L sampler produced {len(out)} trials")
    return out


def mark_duplicate_slots(trials: list[dict]) -> list[dict]:
    """Assign IDs and mark exact duplicates without dropping schedule slots."""
    seen: dict[str, str] = {}
    out: list[dict] = []
    for index, trial in enumerate(trials):
        item = deepcopy(trial)
        line = str(item.get("line", "unknown"))
        item["trial_id"] = f"v35-s1-{line}-{index:04d}"
        if item.get("status", "available") != "available":
            out.append(item)
            continue
        key = canonical_trial_hash(item)
        if key in seen:
            item["status"] = "duplicate"
            item["duplicate_of"] = seen[key]
        else:
            seen[key] = item["trial_id"]
        out.append(item)
    return out


def sample_v35(root: Path | str) -> list[dict]:
    root = Path(root)
    config = load_v35_config(root)
    centers = _select_parent_centers(root)
    trials: list[dict] = []
    trials.extend(build_R_trials(config))
    trials.extend(build_I_trials(config, centers))
    trials.extend(build_L_trials(config, centers))
    trials = mark_duplicate_slots(trials)
    if len(trials) != sum(LINE_COUNTS.values()):
        raise AssertionError(f"V3.5 sampler produced {len(trials)} slots")
    counts = pd.Series([t["line"] for t in trials]).value_counts().to_dict()
    if counts != LINE_COUNTS:
        raise AssertionError(f"V3.5 line counts mismatch: {counts}")
    target_counts = {
        line: {
            target: sum(t["line"] == line and t.get("target") == target for t in trials)
            for target in TARGETS
        }
        for line in LINE_COUNTS
    }
    if target_counts != TARGET_LINE_COUNTS:
        raise AssertionError(f"V3.5 target counts mismatch: {target_counts}")
    return trials


def available_trials(trials: list[dict]) -> list[dict]:
    return [t for t in trials if t.get("status", "available") == "available"]


def schedule_summary(trials: list[dict]) -> dict:
    return {
        "slots": len(trials),
        "available": sum(t.get("status", "available") == "available" for t in trials),
        "not_applicable": sum(t.get("status") == "not_applicable" for t in trials),
        "duplicate": sum(t.get("status") == "duplicate" for t in trials),
        "line_counts": pd.Series([t["line"] for t in trials]).value_counts().to_dict(),
        "target_counts": pd.Series([t.get("target") for t in trials if "target" in t]).value_counts().to_dict(),
    }


def build_conditional_extension_trials(selected_trials: Sequence[dict],
                                       *,
                                       max_slots: int = 64,
                                       slots_per_direction: int = 32) -> list[dict]:
    """Build a guarded conditional extension plan.

    The caller must establish the pre-registered trigger before calling this
    helper.  The returned slots are a deterministic, pre-declared neighbourhood
    expansion; the caller is still responsible for excluding exact duplicates
    through :func:`mark_duplicate_slots`.
    """
    if int(max_slots) != 64 or int(slots_per_direction) != 32:
        raise ValueError("V3.5 conditional extension is frozen at 64 slots (32 per direction)")
    centers = [deepcopy(dict(t)) for t in selected_trials[:2]]
    out: list[dict] = []
    for center in centers:
        kind = str(center.get("kind"))
        base_params = dict(center.get("parameters", {}))
        if kind in {"ebm_regularized", "ebm_expression", "ebm", "ebm_boundary", "ebm_base"}:
            leaf = int(base_params.get("min_samples_leaf", 60))
            interactions = base_params.get("interactions", 2)
            if isinstance(interactions, list):
                # Explicit interaction recipes are extended only by their
                # frozen feature set and leaf boundary, never by silently
                # converting to automatic interactions.
                factors = (0.50, 0.65, 0.80, 0.90, 1.10, 1.25, 1.50, 2.00)
                leaf_values = sorted({max(1, int(round(leaf * factor))) for factor in factors})
                variants = [{"min_samples_leaf": int(value)} for value in leaf_values]
                while len(variants) < int(slots_per_direction):
                    variants.append({"min_samples_leaf": int(max(1, leaf + len(variants) + 1))})
            else:
                bins = int(base_params.get("max_bins", 128))
                max_ib = int(base_params.get("max_interaction_bins", 32))
                interaction_count = int(interactions)
                bins_values = sorted({max(8, bins - 64), max(8, bins - 32), max(8, bins - 16),
                                      bins + 16, bins + 32, bins + 64,
                                      min(4096, bins * 2), max(8, bins // 2)})
                leaf_values = sorted({max(1, leaf - 30), max(1, leaf - 15), max(1, leaf - 5),
                                      leaf + 5, leaf + 15, leaf + 30,
                                      max(1, leaf * 2), max(1, leaf // 2)})
                int_values = sorted({max(0, interaction_count - 2), max(0, interaction_count - 1),
                                     interaction_count + 1, interaction_count + 2,
                                     interaction_count + 4, interaction_count + 8,
                                     max(1, interaction_count // 2), max(1, interaction_count * 2)})
                ib_values = sorted({max(4, max_ib - 16), max(4, max_ib - 8),
                                    max_ib + 8, max_ib + 16,
                                    max(4, max_ib // 2), max_ib * 2,
                                    max_ib + 24, max_ib + 32})
                variants = [{"max_bins": int(v)} for v in bins_values]
                variants += [{"min_samples_leaf": int(v)} for v in leaf_values]
                variants += [{"interactions": int(v)} for v in int_values]
                variants += [{"max_interaction_bins": int(v)} for v in ib_values]
        elif kind == "global_spout_ebm":
            leaf = int(center.get("local_min_samples_leaf", base_params.get("local_min_samples_leaf", 60)))
            interactions = int(center.get("local_interactions", base_params.get("local_interactions", 0)))
            variants = [{"local_min_samples_leaf": int(max(10, leaf // 2))} for _ in range(8)]
            variants += [{"local_min_samples_leaf": int(leaf * 2)} for _ in range(8)]
            variants += [{"local_interactions": int(interactions + 1)} for _ in range(8)]
            variants += [{"local_interactions": int(interactions + 3)} for _ in range(8)]
        else:
            continue
        for variant in variants[:slots_per_direction]:
            trial = deepcopy(center)
            trial.pop("trial_id", None)
            trial["status"] = "available"
            trial["extension_of"] = str(center.get("trial_id", ""))
            trial["extension_rule"] = "predeclared_conditional_neighbourhood"
            params = dict(trial.get("parameters", {}))
            current_leaf = int(trial.get("local_min_samples_leaf", params.get("min_samples_leaf", 60)))
            current_int = int(trial.get("local_interactions", params.get("interactions", 0) or 0))
            params.update(variant)
            if "local_min_samples_leaf" in variant:
                trial["local_min_samples_leaf"] = int(variant["local_min_samples_leaf"])
            if "local_interactions" in variant:
                trial["local_interactions"] = int(variant["local_interactions"])
            trial["parameters"] = params
            out.append(trial)
    return out[: int(max_slots)]
