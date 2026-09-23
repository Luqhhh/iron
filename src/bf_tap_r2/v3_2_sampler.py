"""V3.2 directed 256-configuration sampler.

All trials are label-free specifications.  Coarse training uses model seed 42
unless a seed-ensemble recipe is explicitly supplied elsewhere.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .v3_1_search import load_config
from .v3_run import read_complete_records

IRON_TEMPLATE_IDS = ("v31-s1-expr-iron-0018", "v31-s1-expr-iron-0012", "v3-catboost-tap_iron-0107")
TIME_CENTER_IDS = ("v31-s1-time-0021-0050", "v31-s1-time-0021-0031", "v31-s1-time-0021-0039", "v31-s1-time-0021-0002")
QUANT_PARENT_IDS = {"tap_iron": "v31-s1-expr-iron-0018", "tap_time_len": "v31-s1-time-0021-0050"}
BASE_PARENT_IDS = {"tap_iron": "v31-s1-expr-iron-0018", "tap_time_len": "v31-s1-time-0021-0050"}


def _load_v31_trials(root: Path) -> dict[str, dict]:
    v31 = read_complete_records(root / "local/runs/round2-v3.1-directed-search/s1-coarse-r1/fit_ledger.jsonl")
    v3 = read_complete_records(root / "local/runs/round2-v3-local-search/coarse-catboost-r1/fit_ledger.jsonl")
    out = {r["trial_id"]: r["trial"] for r in v31}
    out.update({r["trial_id"]: r["trial"] for r in v3})
    return out


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _mutate_catboost(parent: Mapping[str, Any], rng: np.random.Generator, *, wide: bool) -> dict:
    trial = deepcopy(dict(parent))
    params = dict(trial["parameters"])
    grow = params.get("grow_policy", "SymmetricTree")
    if wide and rng.uniform() < 0.5:
        grow = str(rng.choice(("SymmetricTree", "Depthwise", "Lossguide")))
    params["grow_policy"] = grow
    if grow == "SymmetricTree":
        params["depth"] = int(_clip(int(params.get("depth", 6)) + int(rng.choice((-1, 0, 0, 1))), 3, 8))
        params.pop("max_leaves", None); params.pop("min_data_in_leaf", None)
    elif grow == "Depthwise":
        params["depth"] = int(_clip(int(params.get("depth", 6)) + int(rng.choice((-1, 0, 1))), 3, 8))
        params["min_data_in_leaf"] = int(rng.choice((10, 20, 50, 100)))
        params.pop("max_leaves", None)
    else:
        params["max_leaves"] = int(rng.choice((15, 31, 63, 127)))
        params["min_data_in_leaf"] = int(rng.choice((10, 20, 50, 100)))
        params.pop("depth", None)
    params["learning_rate"] = _clip(float(params.get("learning_rate", 0.03)) * float(rng.uniform(0.6, 1.4)), 0.005, 0.20)
    params["l2_leaf_reg"] = _clip(float(params.get("l2_leaf_reg", 10.0)) * float(rng.uniform(0.5, 2.0)), 0.05, 300.0)
    params["random_strength"] = _clip(float(params.get("random_strength", 1.0)) * float(rng.uniform(0.7, 1.3)), 0.005, 10.0)
    params["random_seed"] = 42
    params["thread_count"] = 1
    params["allow_writing_files"] = False
    params["verbose"] = False
    params["cat_features"] = ["spout_no"]
    if params.get("boosting_type") == "Ordered" and grow != "SymmetricTree":
        params["boosting_type"] = "Plain"
    if wide and rng.uniform() < 0.3:
        params["boosting_type"] = str(rng.choice(("Plain", "Ordered"))) if grow == "SymmetricTree" else "Plain"
    # legal bootstrap parameters; inherit type unless structure switch requires cleanup
    b = str(params.get("bootstrap_type", "MVS"))
    if b in {"MVS", "Bernoulli"} and "subsample" not in params:
        params["subsample"] = 0.8
    if b == "Bayesian" and "bagging_temperature" not in params:
        params["bagging_temperature"] = 1.0
    if b == "No":
        params.pop("subsample", None); params.pop("bagging_temperature", None)
    trial["parameters"] = params
    return trial


def _iron_log_feature_trials(trials: dict[str, dict], rng: np.random.Generator) -> list[dict]:
    out = []
    for center in IRON_TEMPLATE_IDS:
        parent = trials[center]
        templates = [{"name": "parent", "trial": deepcopy(parent)}]
        for j in range(3):
            templates.append({"name": f"neighbor{j}", "trial": _mutate_catboost(parent, rng, wide=False)})
        for feature_set in ("raw", "four"):
            for target_transform, loss in (("log1p", "RMSE"), ("log1p", "MAE"), ("mean", "RMSE"), ("mean", "Huber:delta=0.05")):
                for template in templates:
                    trial = deepcopy(template["trial"])
                    trial["feature_set"] = feature_set
                    trial["target_transform"] = target_transform
                    trial["parameters"]["loss_function"] = loss
                    trial["parameters"]["random_seed"] = 42
                    trial["target"] = "tap_iron"
                    trial["family"] = "catboost"
                    trial["line"] = "iron_log_expression"
                    out.append(trial)
    return out


def _time_neighborhood_trials(trials: dict[str, dict], rng: np.random.Generator) -> list[dict]:
    out = []
    for center in TIME_CENTER_IDS:
        parent = trials[center]
        for j in range(16):
            trial = _mutate_catboost(parent, rng, wide=False)
            trial["target"] = "tap_time_len"; trial["family"] = "catboost"; trial["line"] = "time_neighborhood"
            out.append(trial)
        for j in range(4):
            trial = _mutate_catboost(parent, rng, wide=True)
            trial["target"] = "tap_time_len"; trial["family"] = "catboost"; trial["line"] = "time_wide"
            out.append(trial)
    return out


def _quantization_trials(trials: dict[str, dict]) -> list[dict]:
    out = []
    for target, parent_id in QUANT_PARENT_IDS.items():
        parent = trials[parent_id]
        for border in (32, 64, 128, 254):
            for rsm in (0.70, 0.85, 0.95, 1.00):
                trial = deepcopy(parent)
                trial["target"] = target; trial["family"] = "catboost"; trial["line"] = "quantization_rsm"
                trial["parameters"] = dict(trial["parameters"])
                trial["parameters"]["border_count"] = int(border)
                trial["parameters"]["rsm"] = float(rsm)
                trial["parameters"]["random_seed"] = 42
                out.append(trial)
    return out


def _calibration_residual_trials(trials: dict[str, dict], rng: np.random.Generator) -> list[dict]:
    out = []
    for target, parent_id in BASE_PARENT_IDS.items():
        parent = trials[parent_id]
        # 4 affine calibrators
        for lam in (0.0, 0.01, 0.1, 1.0):
            out.append({
                "family": "affine", "target": target, "feature_set": parent.get("feature_set", "raw"),
                "target_transform": "identity",
                "parameters": {"base_trial": deepcopy(parent), "lambda": float(lam), "inner_seed": 777},
                "line": "calibration_residual",
            })
        # 12 shallow residual models: depth x l2 x alpha
        base_params = dict(parent["parameters"])
        base_params["random_seed"] = 42
        for depth in (2, 3):
            for l2 in (10.0, 30.0, 100.0):
                for alpha in (0.25, 0.5):
                    residual_params = {
                        "task_type": "CPU", "loss_function": "RMSE", "depth": int(depth),
                        "iterations": 1500, "learning_rate": 0.03, "l2_leaf_reg": float(l2),
                        "random_strength": 0.3, "random_seed": 42, "thread_count": 1,
                        "cat_features": ["spout_no"], "allow_writing_files": False, "verbose": False,
                        "bootstrap_type": "MVS", "subsample": 0.8,
                    }
                    out.append({
                        "family": "residual", "target": target, "feature_set": "raw", "target_transform": "identity",
                        "parameters": {"base": deepcopy(base_params), "residual": residual_params,
                                       "alpha": float(alpha), "inner_seed": 777},
                        "line": "calibration_residual",
                    })
    return out


def _hetero_centers(root: Path) -> dict[tuple[str, str], dict]:
    records = read_complete_records(root / "local/runs/round2-v3.1-directed-search/s1-coarse-r1/fit_ledger.jsonl")
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        fam = r.get("family")
        if fam not in {"lightgbm", "xgboost", "mlp", "kernel"}:
            continue
        key = (r["target"], fam)
        if key not in best or float(r["pooled_wmape"]) < float(best[key]["pooled_wmape"]):
            best[key] = r
    return best


def _hetero_trials(root: Path, rng: np.random.Generator) -> list[dict]:
    centers = _hetero_centers(root)
    out = []
    for (target, family), record in centers.items():
        parent = record["trial"]
        for j in range(2):
            trial = deepcopy(parent)
            trial["target"] = target; trial["family"] = family; trial["line"] = "hetero_far"
            params = dict(trial["parameters"])
            params["random_state"] = 42
            if family == "lightgbm":
                params["num_leaves"] = int(_clip(int(params.get("num_leaves", 31)) + int(rng.choice((-8, 0, 8))), 8, 255))
                params["max_depth"] = int(rng.choice((-1, 4, 5, 6, 8, 10)))
                params["learning_rate"] = _clip(float(params.get("learning_rate", 0.03)) * float(rng.uniform(0.7, 1.3)), 0.005, 0.2)
                params["min_child_samples"] = int(rng.choice((5, 10, 20, 30, 50, 100)))
            elif family == "xgboost":
                params["max_depth"] = int(rng.choice((3, 4, 5, 6, 8)))
                params["min_child_weight"] = float(10 ** rng.uniform(-1, 1))
                params["learning_rate"] = _clip(float(params.get("learning_rate", 0.05)) * float(rng.uniform(0.7, 1.3)), 0.005, 0.2)
                params["reg_lambda"] = float(10 ** rng.uniform(-1, 2))
            elif family == "mlp":
                params["hidden_layer_sizes"] = ((32,), (64,), (32, 16), (64, 32))[int(rng.integers(0, 4))]
                params["alpha"] = float(10 ** rng.uniform(-4, 1))
            elif family == "kernel":
                params.pop("random_state", None)
                params.pop("random_seed", None)
                params["alpha"] = float(10 ** rng.uniform(-4, 1))
                if params.get("kernel") == "rbf":
                    params["gamma"] = float(10 ** rng.uniform(-3, 0))
            trial["parameters"] = params
            out.append(trial)
    return out


def sample_v32(root: Path | str, seed: int = 20260924) -> list[dict]:
    root = Path(root)
    load_config(root)  # validates active V3.1 search config existence only
    rng = np.random.default_rng(seed)
    trials = _load_v31_trials(root)
    out: list[dict] = []
    out.extend(_iron_log_feature_trials(trials, rng))
    out.extend(_time_neighborhood_trials(trials, rng))
    out.extend(_quantization_trials(trials))
    out.extend(_calibration_residual_trials(trials, rng))
    out.extend(_hetero_trials(root, rng))
    for i, trial in enumerate(out):
        trial["trial_id"] = f"v32-s1-{trial['line']}-{i:04d}"
    counts = {}
    for t in out:
        counts[t["target"]] = counts.get(t["target"], 0) + 1
    if counts != {"tap_iron": 136, "tap_time_len": 120}:
        raise AssertionError(f"V3.2 target counts mismatch: {counts}")
    if len(out) != 256:
        raise AssertionError(f"V3.2 sampler produced {len(out)} trials")
    return out
