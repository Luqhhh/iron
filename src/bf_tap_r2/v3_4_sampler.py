"""V3.4 sampler: 96 EBM boundary recipes, 32 EBM residuals and 16 shrink experts.

The sampler is deterministic and label-free.  It reads only public
configuration plus the private V3.1/V3.2 parent ledgers and the already
recorded V3.3 full-development OOF summaries needed to freeze EBM centers and
the time-target transform.  It never inspects test labels.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .v3_1_search import canonical_json, canonical_trial_hash
from .v3_run import read_complete_records

CONFIG_VERSION = "round2-v3.4-ebm-and-constrained-composition"
LINE_COUNTS = {
    "ebm_boundary": 96,
    "ebm_residual": 32,
    "global_spout_shrink": 16,
}
BUDGET_COUNTS = {
    "ebm_boundary": {"tap_iron": 64, "tap_time_len": 32},
    "ebm_residual": {"tap_iron": 16, "tap_time_len": 16},
    "global_spout_shrink": {"tap_iron": 8, "tap_time_len": 8},
}
BOUNDARY_IRON_GRID = {
    "max_bins": [128, 256],
    "min_samples_leaf": [30, 60],
    "interactions": [10, 20, 40, 60],
    "max_interaction_bins": [32, 64],
    "target_transform": ["log1p", "mean"],
}
BOUNDARY_TIME_GRID = {
    "max_bins": [128, 256],
    "min_samples_leaf": [30, 60],
    "interactions": [10, 20, 40, 60],
    "max_interaction_bins": [32, 64],
}
RESIDUAL_COORDINATES = ("original_unit", "log_unit")
RESIDUAL_CORRECTOR_NAMES = ("ridge_10", "ridge_100", "catboost_l2_30", "catboost_l2_100")
SPOUT_PARENTS = {
    "tap_iron": ("v31-s1-expr-iron-0018", "v32-s1-iron_log_expression-0019"),
    "tap_time_len": ("v31-s1-time-0021-0050", "v32-s1-time_neighborhood-0126"),
}


def load_v34_config(root: Path | str) -> dict:
    root = Path(root)
    path = root / "configs/round2_v3_4/search.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != CONFIG_VERSION:
        raise ValueError("Unexpected V3.4 search configuration")
    budget = config.get("fixed_budget", {})
    for line, expected in LINE_COUNTS.items():
        if int(budget.get("lines", {}).get(line, -1)) != expected:
            raise ValueError(f"V3.4 line budget mismatch for {line}")
    if int(budget.get("total", -1)) != sum(LINE_COUNTS.values()):
        raise ValueError("V3.4 fixed total must be 144")
    return config


def _load_old_trials(root: Path) -> dict[str, dict]:
    paths = {
        "v31": root / "local/runs/round2-v3.1-directed-search/s1-coarse-r1/fit_ledger.jsonl",
        "v32": root / "local/runs/round2-v3.2-ensemble-and-target-search/coarse-r1/fit_ledger.jsonl",
    }
    out: dict[str, dict] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing V3.4 parent ledger {name}: {path}")
        for record in read_complete_records(path):
            trial = record.get("trial")
            if isinstance(trial, dict) and "trial_id" in trial:
                out[str(trial["trial_id"])] = deepcopy(trial)
    if not out:
        raise ValueError("No V3.4 parent trials were loaded")
    return out


def _structure_key(trial: Mapping[str, Any]) -> str:
    payload = {
        "kind": trial.get("kind", trial.get("family")),
        "target": trial.get("target"),
        "target_transform": trial.get("target_transform"),
        "parameters": trial.get("parameters", {}),
        "protocol": trial.get("protocol", {}),
    }
    return canonical_json(payload)


def _load_v33_ebm_records(root: Path) -> tuple[list[dict], dict[str, dict]]:
    summary_path = root / "local/runs/round2-v3.3-structure-search/refine-r1/refine_summary.json"
    selection_path = root / "local/runs/round2-v3.3-structure-search/refine-r1/selection.json"
    if not summary_path.exists() or not selection_path.exists():
        raise FileNotFoundError(
            "V3.4 requires the recorded V3.3 full-development refine summary and selection "
            "to freeze EBM centers without reading outer labels"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    specs: dict[str, dict] = {}
    if isinstance(selection, dict):
        for _target, rows in selection.items():
            for row in rows:
                if isinstance(row, dict) and "trial_id" in row:
                    specs[str(row["trial_id"])] = deepcopy(row)
    if not isinstance(summary, list):
        raise ValueError("Unexpected V3.3 refine summary structure")
    records = [row for row in summary if row.get("line") == "ebm"]
    if not records:
        raise ValueError("No V3.3 full-development EBM records available")
    return records, specs


def _select_ebm_centers(records: list[dict], specs: dict[str, dict], target: str,
                        max_centers: int = 2) -> list[dict]:
    candidates = [row for row in records if row.get("target") == target]
    candidates.sort(key=lambda row: (float(row.get("mean_wmape", float("inf"))), str(row.get("trial_id"))))
    selected: list[dict] = []
    seen_structures: set[str] = set()
    for row in candidates:
        trial_id = str(row["trial_id"])
        if trial_id not in specs:
            raise ValueError(f"V3.3 EBM center {trial_id} lacks a full trial specification")
        trial = deepcopy(specs[trial_id])
        key = _structure_key(trial)
        if key in seen_structures:
            continue
        seen_structures.add(key)
        trial["center_role"] = "full_dev_best" if not selected else "structurally_distinct_runner_up"
        trial["center_wmape"] = float(row.get("mean_wmape"))
        selected.append(trial)
        if len(selected) >= max_centers:
            break
    return selected


def _frozen_time_ebm_transform(records: list[dict], specs: dict[str, dict], fallback: str) -> str:
    candidates = [row for row in records if row.get("target") == "tap_time_len"]
    candidates.sort(key=lambda row: (float(row.get("mean_wmape", float("inf"))), str(row.get("trial_id"))))
    for row in candidates:
        trial_id = str(row["trial_id"])
        if trial_id in specs:
            return str(specs[trial_id]["target_transform"])
    return str(fallback)


def _base_ebm_recipe(config: Mapping[str, Any]) -> dict:
    recipe = deepcopy(config["ebm_boundary"]["base_recipe"])
    required = {
        "max_leaves", "objective", "learning_rate", "outer_bags", "inner_bags",
        "max_rounds", "early_stopping_rounds", "random_state", "n_jobs",
    }
    missing = required - set(recipe)
    if missing:
        raise ValueError(f"V3.4 EBM base recipe is missing explicit keys: {sorted(missing)}")
    if recipe["objective"] != "rmse":
        raise ValueError("V3.4 boundary EBM objective is frozen to rmse")
    return recipe


def build_ebm_boundary_trials(config: Mapping[str, Any], frozen_time_transform: str) -> list[dict]:
    recipe = _base_ebm_recipe(config)
    out: list[dict] = []
    for target, grid in (
        ("tap_iron", BOUNDARY_IRON_GRID),
        ("tap_time_len", BOUNDARY_TIME_GRID),
    ):
        transforms = grid["target_transform"] if target == "tap_iron" else [frozen_time_transform]
        for max_bins in grid["max_bins"]:
            for min_samples_leaf in grid["min_samples_leaf"]:
                for interactions in grid["interactions"]:
                    for max_interaction_bins in grid["max_interaction_bins"]:
                        for transform in transforms:
                            params = deepcopy(recipe)
                            params.update({
                                "max_bins": int(max_bins),
                                "min_samples_leaf": int(min_samples_leaf),
                                "interactions": int(interactions),
                                "max_interaction_bins": int(max_interaction_bins),
                            })
                            out.append({
                                "kind": "ebm_boundary",
                                "target": target,
                                "line": "ebm_boundary",
                                "target_transform": transform,
                                "parameters": params,
                                "protocol": {
                                    "bags": "group-safe-bags-v1",
                                    "inner_splits": 5,
                                    "bag_seed": int(recipe["random_state"]),
                                },
                                "feature_source": "raw21_plus_spout_no",
                                "parent_id": "v33-s1-ebm-0095" if target == "tap_iron" else "v33-s1-ebm-0127",
                                "status": "available",
                            })
    if len(out) != LINE_COUNTS["ebm_boundary"]:
        raise AssertionError(f"V3.4 EBM boundary sampler produced {len(out)} trials")
    if sum(t["target"] == "tap_iron" for t in out) != 64:
        raise AssertionError("V3.4 EBM boundary iron count mismatch")
    if sum(t["target"] == "tap_time_len" for t in out) != 32:
        raise AssertionError("V3.4 EBM boundary time count mismatch")
    return out


def _correctors(config: Mapping[str, Any]) -> list[dict]:
    correctors: list[dict] = []
    for item in config["ebm_residual"]["correctors"]:
        correctors.append(deepcopy(dict(item)))
    if [c["name"] for c in correctors] != list(RESIDUAL_CORRECTOR_NAMES):
        raise ValueError("V3.4 residual corrector names/order changed from the frozen plan")
    return correctors


def build_ebm_residual_trials(centers_by_target: Mapping[str, list[dict]],
                              config: Mapping[str, Any]) -> list[dict]:
    correctors = _correctors(config)
    alpha = float(config["ebm_residual"]["alpha_correction"])
    out: list[dict] = []
    for target in ("tap_iron", "tap_time_len"):
        centers = list(centers_by_target.get(target, []))
        for center_index in range(2):
            center = deepcopy(centers[center_index]) if center_index < len(centers) else None
            for coordinate in RESIDUAL_COORDINATES:
                for corrector in correctors:
                    if center is None:
                        out.append({
                            "kind": "ebm_residual",
                            "target": target,
                            "line": "ebm_residual",
                            "status": "not_applicable",
                            "reason": (
                                f"V3.3 full-development EBM center {center_index + 1} does not exist "
                                "for this target; corresponding slot is preserved as N/A"
                            ),
                            "parameters": {},
                        })
                        continue
                    base_trial = deepcopy(center)
                    base_trial["protocol"] = {
                        "bags": "group-safe-bags-v1",
                        "inner_splits": 5,
                        "bag_seed": int(base_trial.get("parameters", {}).get("random_state", 42)),
                    }
                    out.append({
                        "kind": "ebm_residual",
                        "target": target,
                        "line": "ebm_residual",
                        "status": "available",
                        "center_index": int(center_index + 1),
                        "center_trial_id": str(center.get("trial_id", "")),
                        "parameters": {
                            "base_trial": base_trial,
                            "coordinate": coordinate,
                            "corrector": {
                                "kind": str(corrector["kind"]),
                                "params": deepcopy(corrector["params"]),
                                "name": str(corrector["name"]),
                            },
                            "alpha": alpha,
                            "inner_seed": 777,
                            "inner_splits": 5,
                        },
                    })
    if len(out) != LINE_COUNTS["ebm_residual"]:
        raise AssertionError(f"V3.4 EBM residual sampler produced {len(out)} slots")
    return out


def build_global_spout_trials(parents: Mapping[str, dict], config: Mapping[str, Any]) -> list[dict]:
    spec = config["global_spout_shrink"]
    out: list[dict] = []
    for target in ("tap_iron", "tap_time_len"):
        for parent_id in SPOUT_PARENTS[target]:
            if parent_id not in parents:
                raise KeyError(f"Missing V3.4 shrink parent trial: {parent_id}")
            parent = deepcopy(parents[parent_id])
            for local_l2 in spec["local_l2_multipliers"]:
                for beta in spec["beta_values"]:
                    out.append({
                        "kind": "global_spout_shrink",
                        "target": target,
                        "line": "global_spout_shrink",
                        "status": "available",
                        "parent_id": parent_id,
                        "parameters": {
                            "parent_trial": parent,
                            "local_l2_multiplier": float(local_l2),
                            "beta": float(beta),
                            "min_spout_samples": int(spec["min_spout_samples"]),
                            "local_include_spout": bool(spec["local_include_spout"]),
                        },
                    })
    if len(out) != LINE_COUNTS["global_spout_shrink"]:
        raise AssertionError(f"V3.4 global/spout sampler produced {len(out)} trials")
    return out


def mark_duplicate_slots(trials: list[dict]) -> list[dict]:
    """Assign IDs, preserve duplicate slots, and label exact duplicates.

    The runner skips ``status=duplicate`` slots rather than retraining the same
    identity.  The slot list length remains the frozen 144 schedule.
    """
    seen: dict[str, str] = {}
    out: list[dict] = []
    for index, trial in enumerate(trials):
        item = deepcopy(trial)
        trial_id = f"v34-s1-{item['line']}-{index:04d}"
        item["trial_id"] = trial_id
        if item.get("status", "available") != "available":
            out.append(item)
            continue
        key = canonical_trial_hash(item)
        if key in seen:
            item["status"] = "duplicate"
            item["duplicate_of"] = seen[key]
        else:
            seen[key] = trial_id
        out.append(item)
    return out


def sample_v34(root: Path | str) -> list[dict]:
    root = Path(root)
    config = load_v34_config(root)
    old = _load_old_trials(root)
    records, specs = _load_v33_ebm_records(root)
    centers = {
        target: _select_ebm_centers(records, specs, target, max_centers=2)
        for target in ("tap_iron", "tap_time_len")
    }
    time_transform = _frozen_time_ebm_transform(
        records, specs, str(config["ebm_boundary"].get("frozen_time_transform_fallback", "log1p"))
    )
    trials: list[dict] = []
    trials.extend(build_ebm_boundary_trials(config, time_transform))
    trials.extend(build_ebm_residual_trials(centers, config))
    trials.extend(build_global_spout_trials(old, config))
    trials = mark_duplicate_slots(trials)
    if len(trials) != sum(LINE_COUNTS.values()):
        raise AssertionError(f"V3.4 sampler produced {len(trials)} slots")
    counts = pd.Series([t["line"] for t in trials]).value_counts().to_dict()
    if counts != LINE_COUNTS:
        raise AssertionError(f"V3.4 line counts mismatch: {counts}")
    target_counts = {
        line: {
            target: sum(t["line"] == line and t["target"] == target for t in trials)
            for target in ("tap_iron", "tap_time_len")
        }
        for line in LINE_COUNTS
    }
    if target_counts != BUDGET_COUNTS:
        raise AssertionError(f"V3.4 target counts mismatch: {target_counts}")
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
        "target_counts": pd.Series([t["target"] for t in trials if "target" in t]).value_counts().to_dict(),
    }


def build_conditional_extension_trials(selected_trials: list[dict], *,
                                       max_slots: int = 16,
                                       slots_per_center: int = 8) -> list[dict]:
    """Build the optional extension slots after the development trigger fires.

    The caller must already have established the positive two-split development
    gain.  This function changes only unscanned learning-rate / training-length
    dimensions for EBM centers, or one local regularizer for a promising shrink
    center.  It never repeats the same max_bins/interactions grid point.
    """
    if int(max_slots) != 16:
        raise ValueError("V3.4 conditional extension is frozen at 16 slots")
    if int(slots_per_center) != 8:
        raise ValueError("V3.4 conditional extension is frozen at eight slots per center")
    centers = [deepcopy(t) for t in selected_trials[:2]]
    out: list[dict] = []
    for center in centers:
        kind = str(center.get("kind"))
        base_params = dict(center.get("parameters", {}))
        variants: list[dict] = []
        if kind in {"ebm", "ebm_boundary", "ebm_base"}:
            lr = float(base_params.get("learning_rate", 0.03))
            rounds = int(base_params.get("max_rounds", 6000))
            variants = [
                {"learning_rate": max(0.005, lr * 0.50)},
                {"learning_rate": min(0.20, lr * 2.00)},
                {"learning_rate": max(0.005, lr * 0.75)},
                {"learning_rate": min(0.20, lr * 1.50)},
                {"max_rounds": min(20000, max(6000, rounds + 2000))},
                {"max_rounds": min(20000, max(6000, rounds + 6000))},
                {"learning_rate": max(0.005, lr * 0.50),
                 "max_rounds": min(20000, max(6000, rounds + 2000))},
                {"learning_rate": min(0.20, lr * 1.50),
                 "max_rounds": min(20000, max(6000, rounds + 2000))},
            ]
        elif kind == "ebm_residual":
            corrector = dict(base_params.get("corrector", {}))
            params = dict(corrector.get("params", {}))
            if str(corrector.get("kind")) == "ridge":
                alpha = float(params.get("alpha", 10.0))
                values = [alpha * factor for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0)]
                variants = [{"corrector_alpha": value} for value in values]
            else:
                l2 = float(params.get("l2_leaf_reg", 30.0))
                values = [l2 * factor for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0)]
                variants = [{"corrector_l2_leaf_reg": value} for value in values]
        elif kind == "global_spout_shrink":
            l2 = float(base_params.get("local_l2_multiplier", 1.0))
            values = [max(0.1, l2 * factor) for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0)]
            variants = [{"local_l2_multiplier": value} for value in values]
        else:
            continue
        for variant in variants[:slots_per_center]:
            trial = deepcopy(center)
            trial.pop("trial_id", None)
            trial["status"] = "available"
            trial["extension_of"] = str(center.get("trial_id", center.get("center_trial_id", "")))
            trial["extension_rule"] = "unscanned_learning_rate_or_training_length" if kind in {"ebm", "ebm_boundary", "ebm_base"} else "single_regularizer_dimension"
            params = dict(trial.get("parameters", {}))
            if "corrector_alpha" in variant:
                corrector = dict(params.get("corrector", {}))
                cp = dict(corrector.get("params", {}))
                cp["alpha"] = float(variant["corrector_alpha"])
                corrector["params"] = cp
                params["corrector"] = corrector
            elif "corrector_l2_leaf_reg" in variant:
                corrector = dict(params.get("corrector", {}))
                cp = dict(corrector.get("params", {}))
                cp["l2_leaf_reg"] = float(variant["corrector_l2_leaf_reg"])
                corrector["params"] = cp
                params["corrector"] = corrector
            else:
                params.update(variant)
            trial["parameters"] = params
            out.append(trial)
    # Exact duplicate identity is dropped, not retrained; slot count may be
    # below 16 when a variant collapses to an already frozen recipe.
    unique: list[dict] = []
    seen: set[str] = set()
    for trial in out:
        key = canonical_trial_hash(trial)
        if key in seen:
            continue
        seen.add(key)
        unique.append(trial)
    return unique[: int(max_slots)]
