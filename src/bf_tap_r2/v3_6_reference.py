"""V3.6 A-development replay and reference-object diagnostics.

The replay uses only already-recorded development OOF predictions.  It never
fits a model, reads outer labels, or creates a submission.  Its purpose is to
put every V3.6 candidate on the same development coverage as the fixed A
recipe before any new score is called an improvement over A.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import wmape
from .v3_run import load_training_frame

TARGETS = ("tap_iron", "tap_time_len")
DEVELOPMENT_SEEDS = ("42", "3407")

# A deployment recipe recorded in
# local/runs/round2-v3.4-ebm-and-constrained-composition/release-candidates-r1.
A_TARGET_WEIGHTS: dict[str, tuple[float, ...]] = {
    "tap_iron": (
        0.5000000000000001,
        0.3397038169347897,
        0.16029618306521032,
    ),
    "tap_time_len": (
        0.6192377408552066,
        0.28691289439355067,
        0.09384936475124295,
    ),
}
A_EXPERTS: dict[str, tuple[str, ...]] = {
    "tap_iron": ("v34-s1-ebm_boundary-0016", "v34-s1-ebm_boundary-0020"),
    "tap_time_len": ("v34-s1-ebm_boundary-0089", "v34-s1-global_spout_shrink-0136"),
}

V35_TARGET_WEIGHTS: dict[str, tuple[float, ...]] = {
    "tap_iron": (0.5, 0.251833631595236, 0.248166368404764),
    "tap_time_len": (0.683025830741231, 0.316974169258769),
}
V35_EXPERTS: dict[str, tuple[str, ...]] = {
    "tap_iron": ("v35-s1-R-0036", "v35-s1-R-0041"),
    "tap_time_len": ("v35-s1-I-0181",),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sample_ids(train: pd.DataFrame) -> list[str]:
    ids = train["sample_id"].astype(str).tolist()
    _require(len(ids) == len(set(ids)), "Training sample_id must be unique")
    return ids


def load_l1_development(root: Path | str, train: pd.DataFrame, target: str, seed: int) -> np.ndarray:
    """Load the recorded V3.4 L1 OOF vector and align it to ``train`` order."""
    root = Path(root)
    folder = root / "local/runs/round2-v3.4-ebm-and-constrained-composition/l1-oof-r1" / f"seed-{int(seed)}"
    parts: list[pd.DataFrame] = []
    for fold in range(5):
        path = folder / f"l1-oof-fold-{fold}-{target}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_csv(path, dtype={"sample_id": "string"}))
    frame = pd.concat(parts, ignore_index=True)
    ids = _sample_ids(train)
    if frame.sample_id.duplicated().any() or set(frame.sample_id.astype(str)) != set(ids):
        raise ValueError(f"L1 development OOF identity mismatch for seed {seed} target {target}")
    return frame.set_index("sample_id").loc[ids, target].to_numpy(dtype=float)


def _load_prediction(path: Path, n_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.load(path)
    if values.shape != (int(n_rows),):
        raise ValueError(f"Prediction shape mismatch: {path}: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite prediction vector: {path}")
    return values.astype(float, copy=False)


def load_v34_expert(root: Path | str, train: pd.DataFrame, seed: int, trial_id: str) -> np.ndarray:
    root = Path(root)
    path = (
        root
        / "local/runs/round2-v3.4-ebm-and-constrained-composition/refine-r1"
        / f"seed-{int(seed)}"
        / f"pred-{trial_id}.npy"
    )
    return _load_prediction(path, len(train))


def load_v35_refine_expert(root: Path | str, train: pd.DataFrame, seed: int, trial_id: str) -> np.ndarray:
    root = Path(root)
    path = (
        root
        / "local/runs/round2-v3.5-regularized-ebm-and-composition/refine-r1"
        / f"seed-{int(seed)}"
        / f"pred-{trial_id}.npy"
    )
    return _load_prediction(path, len(train))


def _blend(l1: np.ndarray, experts: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    matrix = np.column_stack([l1, *experts])
    weight = np.asarray(weights, dtype=float)
    if matrix.shape[1] != weight.shape[0]:
        raise ValueError("Reference weight length does not match member columns")
    return matrix @ weight


def build_reference_vectors(root: Path | str, train: pd.DataFrame) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Build A/V3.5 target predictions on both complete development seeds.

    Returns ``object -> target -> seed -> vector`` for A and V3.5.
    """
    root = Path(root)
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "A": {target: {} for target in TARGETS},
        "V35": {target: {} for target in TARGETS},
    }
    for target in TARGETS:
        for seed in DEVELOPMENT_SEEDS:
            seed_int = int(seed)
            l1 = load_l1_development(root, train, target, seed_int)
            a_experts = [load_v34_expert(root, train, seed_int, tid) for tid in A_EXPERTS[target]]
            out["A"][target][seed] = _blend(l1, a_experts, A_TARGET_WEIGHTS[target])
            v35_experts = [load_v35_refine_expert(root, train, seed_int, tid) for tid in V35_EXPERTS[target]]
            out["V35"][target][seed] = _blend(l1, v35_experts, V35_TARGET_WEIGHTS[target])
    return out


def score_reference_objects(reference: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
                            train: pd.DataFrame) -> dict[str, Any]:
    """Score A, V3.5, and the two target-swap diagnostics."""
    labels = {target: train[target].to_numpy(dtype=float) for target in TARGETS}
    rows: list[dict[str, Any]] = []
    for name in ("A", "V35"):
        for target in TARGETS:
            for seed in DEVELOPMENT_SEEDS:
                pred = np.asarray(reference[name][target][seed], dtype=float)
                rows.append({
                    "object": name,
                    "target": target,
                    "seed": seed,
                    "wmape": float(wmape(labels[target], pred)),
                })
    # A iron + V3.5 time and V3.5 iron + A time.
    swap_objects = {
        "A_iron_V35_time": {
            "tap_iron": ("A", "tap_iron"),
            "tap_time_len": ("V35", "tap_time_len"),
        },
        "V35_iron_A_time": {
            "tap_iron": ("V35", "tap_iron"),
            "tap_time_len": ("A", "tap_time_len"),
        },
    }
    for name, mapping in swap_objects.items():
        for target in TARGETS:
            source_object, source_target = mapping[target]
            for seed in DEVELOPMENT_SEEDS:
                pred = np.asarray(reference[source_object][source_target][seed], dtype=float)
                rows.append({
                    "object": name,
                    "target": target,
                    "seed": seed,
                    "wmape": float(wmape(labels[target], pred)),
                })
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "seeds": list(DEVELOPMENT_SEEDS),
        "objects": {},
        "a_dev_package_score": None,
        "v35_dev_package_score": None,
        "delta_v35_vs_a_dev": None,
    }
    for name in ("A", "V35", "A_iron_V35_time", "V35_iron_A_time"):
        target_means = {
            target: float(frame[(frame.object == name) & (frame.target == target)]["wmape"].mean())
            for target in TARGETS
        }
        package = 100.0 - 100.0 * float(np.mean(list(target_means.values())))
        per_seed = {}
        for seed in DEVELOPMENT_SEEDS:
            sub = frame[(frame.object == name) & (frame.seed == seed)]
            per_seed[seed] = float(100.0 - 100.0 * sub["wmape"].mean())
        summary["objects"][name] = {
            "target_wmape": target_means,
            "per_seed_package_score": per_seed,
            "package_score": package,
        }
    summary["a_dev_package_score"] = summary["objects"]["A"]["package_score"]
    summary["v35_dev_package_score"] = summary["objects"]["V35"]["package_score"]
    summary["delta_v35_vs_a_dev"] = summary["v35_dev_package_score"] - summary["a_dev_package_score"]
    summary["rows"] = rows
    return summary


def build_a_development_replay(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    train = load_training_frame(root)
    reference = build_reference_vectors(root, train)
    return score_reference_objects(reference, train)


def write_a_development_replay(root: Path | str, output: Path) -> dict[str, Any]:
    result = build_a_development_replay(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return result


__all__ = [
    "A_EXPERTS",
    "A_TARGET_WEIGHTS",
    "DEVELOPMENT_SEEDS",
    "TARGETS",
    "V35_EXPERTS",
    "V35_TARGET_WEIGHTS",
    "build_a_development_replay",
    "build_reference_vectors",
    "load_l1_development",
    "load_v34_expert",
    "load_v35_refine_expert",
    "score_reference_objects",
    "write_a_development_replay",
]
