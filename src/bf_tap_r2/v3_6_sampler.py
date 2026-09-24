"""V3.6 deterministic, label-free fixed-schedule sampler.

The sampler expands exactly 224 pre-registered slots:
* ``O``: 64 EBM loss / target-coordinate / L2-regularization trials
* ``D``: 96 EBM training-protocol trials
* ``N``: 64 numeric-encoding MLP / TabM trials

Only public configuration and recorded complete-development V3.4/V3.5 trials
used as parent centres are read.  Target labels are not read here.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .data import TARGETS
from .v3_1_search import canonical_trial_hash
from .v3_run import read_complete_records

CONFIG_VERSION = "round2-v3.6-loss-training-and-numeric-encoding"
LINE_COUNTS = {"O": 64, "D": 96, "N": 64}
TARGET_LINE_COUNTS = {
    "O": {"tap_iron": 32, "tap_time_len": 32},
    "D": {"tap_iron": 32, "tap_time_len": 64},
    "N": {"tap_iron": 16, "tap_time_len": 48},
}
TARGET_KEY = {"tap_iron": "iron", "tap_time_len": "time"}
EBM_PARENT_KINDS = {"ebm", "ebm_boundary", "ebm_base", "ebm_regularized", "ebm_expression"}


def load_v36_config(root: Path | str) -> dict:
    root = Path(root)
    path = root / "configs/round2_v3_6/search.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != CONFIG_VERSION:
        raise ValueError("Unexpected V3.6 search configuration")
    budget = config.get("fixed_budget", {})
    for line, expected in LINE_COUNTS.items():
        if int(budget.get("lines", {}).get(line, -1)) != expected:
            raise ValueError(f"V3.6 line budget mismatch for {line}")
    if int(budget.get("total", -1)) != sum(LINE_COUNTS.values()):
        raise ValueError("V3.6 fixed total must be 224")
    if int(budget.get("hard_max", -1)) != 256:
        raise ValueError("V3.6 hard maximum must be 256")
    return config


def _read_recorded_trial(root: Path, ledger: Path, wanted: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not ledger.exists():
        return out
    for record in read_complete_records(ledger):
        trial = record.get("trial")
        if not isinstance(trial, dict):
            continue
        trial_id = str(trial.get("trial_id", ""))
        if trial_id in wanted:
            out[trial_id] = deepcopy(trial)
    return out


def load_recorded_trials(root: Path | str) -> dict[str, dict]:
    root = Path(root)
    config = load_v36_config(root)
    wanted: set[str] = set()
    for section in ("O_loss_and_regularization", "D_training_protocol"):
        for values in config[section]["parent_centers"].values():
            wanted.update(str(v) for v in values)
    out: dict[str, dict] = {}
    sources = [
        root / "local/runs/round2-v3.4-ebm-and-constrained-composition/refine-r1/seed-42/fit_ledger.jsonl",
        root / "local/runs/round2-v3.5-regularized-ebm-and-composition/refine-r1/seed-42/fit_ledger.jsonl",
    ]
    for path in sources:
        out.update(_read_recorded_trial(root, path, wanted))
    missing = sorted(wanted - set(out))
    if missing:
        raise FileNotFoundError(f"V3.6 parent centres unavailable in recorded ledgers: {missing}")
    return out


def resolve_parent_centers(root: Path | str) -> dict[str, list[dict]]:
    config = load_v36_config(root)
    recorded = load_recorded_trials(root)
    out: dict[str, list[dict]] = {}
    for section in ("O_loss_and_regularization", "D_training_protocol"):
        spec = config[section]
        for target in TARGETS:
            key = str(target)
            rows: list[dict] = []
            for trial_id in spec["parent_centers"][key]:
                trial = deepcopy(recorded[str(trial_id)])
                if str(trial.get("target")) != target:
                    raise ValueError(f"Parent centre {trial_id} target mismatch")
                if str(trial.get("kind")) not in EBM_PARENT_KINDS:
                    raise ValueError(f"Parent centre {trial_id} is not an EBM trial")
                rows.append(trial)
            out[f"{section}:{key}"] = rows
    return out


def _base_ebm_params(parent: Mapping[str, Any]) -> dict[str, Any]:
    params = deepcopy(dict(parent.get("parameters", {})))
    # Keep only the core recipe keys.  Parent trials may carry metadata for
    # local/residual models, but the fixed O/D centres are pure EBM trials.
    allowed = {
        "max_bins", "min_samples_leaf", "interactions", "max_interaction_bins",
        "max_leaves", "objective", "learning_rate", "outer_bags", "inner_bags",
        "max_rounds", "early_stopping_rounds", "random_state", "n_jobs",
    }
    return {key: deepcopy(value) for key, value in params.items() if key in allowed}


def build_O_trials(config: Mapping[str, Any], centers: Mapping[str, list[dict]]) -> list[dict]:
    spec = config["O_loss_and_regularization"]
    out: list[dict] = []
    index = 0
    for target in TARGETS:
        for parent in centers[f"O_loss_and_regularization:{target}"]:
            for target_transform in spec["target_transforms"]:
                for objective in spec["objectives"]:
                    for reg_lambda in spec["reg_lambda"]:
                        params = _base_ebm_params(parent)
                        params.update({
                            "objective": str(objective),
                            "reg_lambda": float(reg_lambda),
                        })
                        out.append({
                            "trial_id": f"v36-s1-O-{index:04d}",
                            "line": "O",
                            "kind": "ebm_loss",
                            "target": target,
                            "target_transform": str(target_transform),
                            "feature_set": str(parent.get("feature_set", "raw")),
                            "parent_trial_id": str(parent.get("trial_id")),
                            "parameters": params,
                            "protocol": {
                                "bags": "group-safe-bags-v1",
                                "inner_splits": 5,
                                "bag_seed": int(params.get("random_state", 42)),
                            },
                            "status": "available",
                        })
                        index += 1
    if len(out) != LINE_COUNTS["O"]:
        raise AssertionError(f"V3.6 O sampler produced {len(out)} trials")
    if any(sum(t["target"] == target for t in out) != TARGET_LINE_COUNTS["O"][target] for target in TARGETS):
        raise AssertionError("V3.6 O target counts differ")
    return out


def build_D_trials(config: Mapping[str, Any], centers: Mapping[str, list[dict]]) -> list[dict]:
    spec = config["D_training_protocol"]
    modes = spec["training_modes"]
    smoothing = spec["smoothing"]
    out: list[dict] = []
    index = 0
    for target in TARGETS:
        for parent in centers[f"D_training_protocol:{target}"]:
            for mode_name, mode in modes.items():
                for smooth_name, smooth in smoothing.items():
                    for max_leaves in spec["max_leaves"]:
                        for learning_rate in spec["learning_rate"]:
                            params = _base_ebm_params(parent)
                            params.update({
                                "greedy_ratio": float(mode["greedy_ratio"]),
                                "cyclic_progress": bool(mode["cyclic_progress"]),
                                "smoothing_rounds": int(smooth["smoothing_rounds"]),
                                "interaction_smoothing_rounds": int(smooth["interaction_smoothing_rounds"]),
                                "max_leaves": int(max_leaves),
                                "learning_rate": float(learning_rate),
                            })
                            out.append({
                                "trial_id": f"v36-s1-D-{index:04d}",
                                "line": "D",
                                "kind": "ebm_training",
                                "target": target,
                                "target_transform": str(parent["target_transform"]),
                                "feature_set": str(parent.get("feature_set", "raw")),
                                "parent_trial_id": str(parent.get("trial_id")),
                                "training_mode": str(mode_name),
                                "smoothing_mode": str(smooth_name),
                                "parameters": params,
                                "protocol": {
                                    "bags": "group-safe-bags-v1",
                                    "inner_splits": 5,
                                    "bag_seed": int(params.get("random_state", 42)),
                                },
                                "status": "available",
                            })
                            index += 1
    if len(out) != LINE_COUNTS["D"]:
        raise AssertionError(f"V3.6 D sampler produced {len(out)} trials")
    if any(sum(t["target"] == target for t in out) != TARGET_LINE_COUNTS["D"][target] for target in TARGETS):
        raise AssertionError("V3.6 D target counts differ")
    return out


def build_N_trials(config: Mapping[str, Any]) -> list[dict]:
    spec = config["N_numeric_encoding_networks"]
    out: list[dict] = []
    index = 0
    capacity_spec = {
        "raw_mlp": spec["mlp_capacities"],
        "ple_mlp": spec["mlp_capacities"],
        "raw_tabm": spec["tabm_capacities"],
        "ple_tabm": spec["tabm_capacities"],
    }
    for target in TARGETS:
        capacities = [spec["iron_capacity"]] if target == "tap_iron" else list(spec["time_capacities"])
        for structure in spec["structures"]:
            for capacity_name in capacities:
                for setting_name, setting in spec["training_settings"].items():
                    capacity = deepcopy(capacity_spec[str(structure)][str(capacity_name)])
                    out.append({
                        "trial_id": f"v36-s1-N-{index:04d}",
                        "line": "N",
                        "kind": "numeric_tabm" if "tabm" in str(structure) else "numeric_mlp",
                        "target": target,
                        "target_transform": "train_mean_std",
                        "structure": str(structure),
                        "capacity_name": str(capacity_name),
                        "training_setting": str(setting_name),
                        "parameters": {
                            **deepcopy(dict(setting)),
                            **deepcopy(capacity),
                            "n_bins": int(spec["ple_n_bins"]),
                            "d_embedding": int(spec["ple_d_embedding"]),
                            "random_seed": int(spec["random_seed"]),
                            "batch_size": int(spec["batch_size"]),
                            "max_epochs": int(spec["max_epochs"]),
                            "early_stopping_patience": int(spec["early_stopping_patience"]),
                            "min_delta": float(spec["min_delta"]),
                            "inner_validation_folds": int(spec["inner_validation_folds"]),
                            "inner_validation_seed": int(spec["inner_validation_seed"]),
                        },
                        "status": "available",
                    })
                    index += 1
    if len(out) != LINE_COUNTS["N"]:
        raise AssertionError(f"V3.6 N sampler produced {len(out)} trials")
    if any(sum(t["target"] == target for t in out) != TARGET_LINE_COUNTS["N"][target] for target in TARGETS):
        raise AssertionError("V3.6 N target counts differ")
    return out


def sample_v36(root: Path | str) -> list[dict]:
    config = load_v36_config(root)
    centers = resolve_parent_centers(root)
    trials = [*build_O_trials(config, centers), *build_D_trials(config, centers), *build_N_trials(config)]
    seen: set[str] = set()
    for trial in trials:
        trial_id = str(trial["trial_id"])
        if trial_id in seen:
            raise ValueError(f"Duplicate V3.6 trial_id: {trial_id}")
        seen.add(trial_id)
    return trials


def schedule_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lines: dict[str, int] = {}
    targets: dict[str, dict[str, int]] = {}
    hashes: set[str] = set()
    duplicates = 0
    for trial in trials:
        line = str(trial.get("line"))
        target = str(trial.get("target"))
        lines[line] = lines.get(line, 0) + 1
        targets.setdefault(line, {})
        targets[line][target] = targets[line].get(target, 0) + 1
        key = canonical_trial_hash(dict(trial))
        if key in hashes:
            duplicates += 1
        hashes.add(key)
    return {
        "slots": len(trials),
        "line_counts": lines,
        "target_counts": targets,
        "duplicate_trial_hashes": duplicates,
    }


__all__ = [
    "CONFIG_VERSION",
    "LINE_COUNTS",
    "TARGET_LINE_COUNTS",
    "build_D_trials",
    "build_N_trials",
    "build_O_trials",
    "load_recorded_trials",
    "load_v36_config",
    "resolve_parent_centers",
    "sample_v36",
    "schedule_summary",
]
