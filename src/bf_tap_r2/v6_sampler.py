"""Round2 V6 frozen sampler: the iron-side medium/large numeric-encoding networks.

The V3.6 sampler fixed ``iron_capacity: small`` (``src/bf_tap_r2/v3_6_sampler.py``,
``build_N_trials``), so the iron target never received a medium or large
numeric-encoding network while the time target received three capacities.  The
V5 round then showed that this family's value is a *decorrelation* value: a
single large raw-TabM member on the time column, mixed at weight 0.2, was the
only change that ever beat the frozen release, and its three sibling training
settings turned out to be near-duplicates of it (pairwise residual correlation
0.979-0.998).

This module expands exactly 32 iron-side slots — two new capacities across the
four frozen structures and four frozen training settings — under their own
frozen identifiers.  The numeric space, losses, optimizers, capacity definitions
and training settings are read from the frozen V3.6 configuration; nothing here
redefines them, and the V3.6 sampler is left untouched.

Only public configuration is read.  No labels, no fits, no package.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .v3_6_sampler import load_v36_config, sample_v36

__all__ = [
    "V6_VERSION",
    "V6_TARGET",
    "V6_STRUCTURES",
    "V6_CAPACITIES",
    "V6_SETTINGS",
    "V6_TRIAL_COUNT",
    "v6_trial_id",
    "v6_family_key",
    "sample_v6_iron_capacity_trials",
    "verify_no_v36_collision",
    "main",
]

V6_VERSION = "round2-v6-iron-capacity-networks"
V6_TARGET = "tap_iron"

#: Frozen from the V3.6 N line; do not reorder — the order defines the trial ids.
V6_STRUCTURES: tuple[str, ...] = ("raw_mlp", "ple_mlp", "raw_tabm", "ple_tabm")
V6_CAPACITIES: tuple[str, ...] = ("medium", "large")
V6_SETTINGS: tuple[str, ...] = ("mse_adam", "mae_adam", "huber_adam", "mse_adamw")

V6_TRIAL_COUNT = len(V6_CAPACITIES) * len(V6_STRUCTURES) * len(V6_SETTINGS)

_ID_PREFIX = "v6-s1-N"


def v6_trial_id(capacity: str, structure: str, setting: str) -> str:
    """Frozen identifier for one ``(capacity, structure, setting)`` slot."""
    if capacity not in V6_CAPACITIES:
        raise ValueError(f"V6 unknown capacity: {capacity!r}")
    if structure not in V6_STRUCTURES:
        raise ValueError(f"V6 unknown structure: {structure!r}")
    if setting not in V6_SETTINGS:
        raise ValueError(f"V6 unknown training setting: {setting!r}")
    index = (
        V6_CAPACITIES.index(capacity) * len(V6_STRUCTURES) * len(V6_SETTINGS)
        + V6_STRUCTURES.index(structure) * len(V6_SETTINGS)
        + V6_SETTINGS.index(setting)
    )
    return f"{_ID_PREFIX}-{index:04d}"


def v6_family_key(trial: Mapping[str, Any]) -> str:
    """The structural family used by the mutual-diversity constraint."""
    return f"{trial['structure']}|{trial['capacity_name']}"


def sample_v6_iron_capacity_trials(root: Path | str) -> list[dict[str, Any]]:
    """Expand the 32 pre-registered iron medium/large network slots."""
    config = load_v36_config(root)
    spec = config["N_numeric_encoding_networks"]
    capacity_spec = {
        "raw_mlp": spec["mlp_capacities"],
        "ple_mlp": spec["mlp_capacities"],
        "raw_tabm": spec["tabm_capacities"],
        "ple_tabm": spec["tabm_capacities"],
    }
    out: list[dict[str, Any]] = []
    for capacity_name in V6_CAPACITIES:
        for structure in V6_STRUCTURES:
            for setting_name in V6_SETTINGS:
                setting = spec["training_settings"][setting_name]
                capacity = capacity_spec[str(structure)][str(capacity_name)]
                out.append({
                    "trial_id": v6_trial_id(capacity_name, structure, setting_name),
                    "version": V6_VERSION,
                    "line": "N",
                    "kind": "numeric_tabm" if "tabm" in str(structure) else "numeric_mlp",
                    "target": V6_TARGET,
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
    if len(out) != V6_TRIAL_COUNT:
        raise AssertionError(f"V6 sampler produced {len(out)} trials, expected {V6_TRIAL_COUNT}")
    ids = [str(t["trial_id"]) for t in out]
    if len(set(ids)) != len(ids):
        raise AssertionError("V6 sampler produced duplicate trial ids")
    return out


def verify_no_v36_collision(root: Path | str) -> None:
    """The V6 identifiers must be disjoint from every V3.6 slot."""
    v6 = {str(t["trial_id"]) for t in sample_v6_iron_capacity_trials(root)}
    v36 = {str(t["trial_id"]) for t in sample_v36(root)}
    overlap = sorted(v6 & v36)
    if overlap:
        raise ValueError(f"V6 trial ids collide with V3.6: {overlap[:5]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    verify_no_v36_collision(args.root)
    trials = sample_v6_iron_capacity_trials(args.root)
    payload = {
        "version": V6_VERSION,
        "target": V6_TARGET,
        "count": len(trials),
        "families": sorted({v6_family_key(t) for t in trials}),
        "trial_ids": [str(t["trial_id"]) for t in trials],
        "agent_uploads": 0,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
