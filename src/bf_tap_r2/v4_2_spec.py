"""Loader and validator for ``configs/round2_v4_2/SEARCH_SPEC.yaml``.

The declarative specification is the pre-registration contract for the V4.2
round.  The runner refuses to start when a hard constraint in the file has been
changed to something the task book does not allow, so a silently weakened
specification cannot be executed as if it were the frozen plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["DEFAULT_SPEC_PATH", "SearchSpec", "load_search_spec"]

DEFAULT_SPEC_PATH = "configs/round2_v4_2/SEARCH_SPEC.yaml"

#: The exact recipe set each accepted schema version may declare.  The V4.2 set
#: is unchanged from the executed screen; V4.4 adds only the two mechanism
#: recipes per line that the round pre-registers.
_EXPECTED_LINES_BY_SCHEMA: dict[str, dict[str, set[str]]] = {
    "v4.2": {
        "R": {"R0", "R1", "R2", "R3"},
        "S": {"S0", "S1", "S2", "S3"},
        "N": {"N0", "N1", "N2", "N3"},
    },
    "v4.4": {
        "N": {"N2", "N4", "N5"},
        "R": {"R2", "R4", "R5"},
    },
}


def _require(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"SEARCH_SPEC is missing required key: {name}")
    return mapping[name]


@dataclass(frozen=True)
class SearchSpec:
    """Validated V4.2 search specification."""

    path: Path
    raw: dict[str, Any]

    # -- convenient accessors -------------------------------------------
    @property
    def status(self) -> str:
        return str(self.raw["status"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.raw["screen"]["seeds"])

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.raw["screen"]["folds"])

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(str(v) for v in self.raw["data_contract"]["targets"])

    @property
    def lines(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw["lines"])

    def recipes(self, line: str) -> tuple[str, ...]:
        return tuple(str(v) for v in _require(self.lines[str(line)], "recipes"))

    @property
    def units(self) -> list[tuple[str, str, str]]:
        """Every pre-registered ``(line, recipe, target)`` unit."""
        out: list[tuple[str, str, str]] = []
        for line in self.raw["lines"]:
            for recipe in self.recipes(line):
                for target in self.targets:
                    out.append((line, recipe, target))
        return out

    @property
    def outer_recipe_slots(self) -> int:
        return len(self.units) * len(self.seeds) * len(self.folds)

    @property
    def neural_training(self) -> dict[str, Any]:
        return dict(self.raw["common_training"]["neural_starting_point"])

    def neural_train_config(self) -> dict[str, Any]:
        """Translate the pre-registered starting point into runner parameters.

        ``initialisation_seed`` is the pre-registered training seed; ``note`` is
        documentation only and is not a training parameter.
        """
        training = self.neural_training
        return {
            "optimizer": str(training["optimizer"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "batch_size": int(training["batch_size"]),
            "max_epochs": int(training["max_epochs"]),
            "early_stopping_patience": int(training["early_stopping_patience"]),
            "seed": int(training["initialisation_seed"]),
        }

    @property
    def gate(self) -> dict[str, Any]:
        return dict(self.raw["screen"]["continuation_gate"])

    @property
    def fixed_quarter_weight(self) -> float:
        return float(self.raw["screen"]["fixed_quarter_weight"])

    @property
    def selection(self) -> dict[str, Any]:
        return dict(self.raw["screen"]["selection"])

    def spec_hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.raw.get("schema_version"),
            "seeds": list(self.seeds),
            "folds": list(self.folds),
            "units": [list(u) for u in self.units],
            "gate": self.gate,
            "fixed_quarter_weight": self.fixed_quarter_weight,
        }


def load_search_spec(path: Path | str = DEFAULT_SPEC_PATH, *,
                     root: Path | str = ".") -> SearchSpec:
    """Read and validate a V4.2 declarative search specification."""
    base = Path(root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = base / target
    if not target.is_file():
        raise FileNotFoundError(f"V4.2 SEARCH_SPEC not found: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SEARCH_SPEC must be a YAML mapping")
    schema = str(raw.get("schema_version"))
    if schema not in _EXPECTED_LINES_BY_SCHEMA:
        raise ValueError(
            f"Unsupported SEARCH_SPEC schema_version: {schema!r}; "
            f"accepted: {sorted(_EXPECTED_LINES_BY_SCHEMA)}"
        )
    expected_lines = _EXPECTED_LINES_BY_SCHEMA[schema]

    data = _require(raw, "data_contract")
    if int(data["numeric_features"]) != 21:
        raise ValueError("V4.2 uses the 21 numerical V2 features")
    if [str(v) for v in data["categorical_features"]] != ["spout_no"]:
        raise ValueError("V4.2 uses spout_no as the only categorical feature")
    if int(data["train_rows"]) != 2754 or int(data["test_rows"]) != 322:
        raise ValueError("V4.2 row counts must match the V2 snapshot")
    for key in ("external_data", "pretrained_weights", "test_labels",
                "id_or_row_order_features", "per_row_prediction_editing",
                "example_data_downloads"):
        if str(data.get(key)) != "forbidden":
            raise ValueError(f"V4.2 forbids {key}")
    data.setdefault("targets", ["tap_iron", "tap_time_len"])

    lines = _require(raw, "lines")
    if set(lines) != set(expected_lines):
        raise ValueError(f"{schema} lines must be exactly {sorted(expected_lines)}")
    for line, expected in expected_lines.items():
        recipes = lines[line].get("recipes")
        if not isinstance(recipes, dict) or set(recipes) != expected:
            raise ValueError(f"Line {line} must declare exactly {sorted(expected)}")

    screen = _require(raw, "screen")
    if list(int(v) for v in screen["seeds"]) != [42, 3407]:
        raise ValueError("V4.2 pre-registers split seeds 42 and 3407")
    if list(int(v) for v in screen["folds"]) != [0, 1]:
        raise ValueError("V4.2 coarse screen uses outer folds 0 and 1")
    # The unit and slot counts are derived from the declared recipe table rather
    # than hard-coded, so adding a pre-registered recipe cannot silently reuse
    # the old budget.  For v4.2 the derivation reproduces 24 and 96 exactly.
    declared_units = sum(len(lines[name]["recipes"]) for name in expected_lines)
    declared_units *= len(data["targets"])
    if int(screen["units"]) != declared_units:
        raise ValueError(
            f"{schema} must pre-register {declared_units} target-recipe units, "
            f"the file declares {screen['units']}"
        )
    declared_slots = declared_units * len(screen["seeds"]) * len(screen["folds"])
    if int(screen["outer_recipe_slots"]) != declared_slots:
        raise ValueError(
            f"{schema} must pre-register {declared_slots} outer recipe slots, "
            f"the file declares {screen['outer_recipe_slots']}"
        )
    if float(screen["fixed_quarter_weight"]) != 0.25:
        raise ValueError("V4.2 pre-registers a fixed 0.25 quarter blend")
    if screen.get("weight_scan") != "forbidden":
        raise ValueError("V4.2 forbids scanning the blend weight at screen time")
    gate = _require(screen, "continuation_gate")
    if not bool(gate["same_fixed_path_for_both_seeds"]):
        raise ValueError("The continuation gate requires one fixed path for both seeds")
    if float(gate["mean_full_package_gain_min"]) != 0.005:
        raise ValueError("The continuation gate threshold is 0.005")
    if not bool(gate["r0_control_never_counts_as_retrieval_success"]):
        raise ValueError("R0 must never count as retrieval success")

    if "R" in lines:
        r_start = lines["R"]["starting_point"]
        if (int(r_start["repr_width"]), int(r_start["n_blocks"]), int(r_start["n_neighbors"]),
                float(r_start["dropout"]), int(r_start["ple_segments"])) != (128, 2, 32, 0.10, 8):
            raise ValueError("R fixed starting point changed")

    if "S" in lines:
        s_limits = lines["S"]["limits"]
        if (int(s_limits["max_nodes"]), int(s_limits["max_depth"]),
                int(s_limits["eval_budget_per_training_subset"])) != (24, 5, 200_000):
            raise ValueError("S complexity or budget limits changed")
        if not bool(lines["S"]["backend"]["verified"]["budget_controllable"]):
            raise ValueError("The S backend budget-control verification must be recorded")

    n_line = lines["N"]
    if int(n_line["total_trees"]) != 128 or int(n_line["depth"]) != 4:
        raise ValueError("N fixes 128 trees at depth 4")
    for name, recipe in n_line["recipes"].items():
        if int(recipe["n_layers"]) * int(recipe["trees_per_layer"]) != 128:
            raise ValueError(f"N recipe {name} must keep 128 trees in total")

    training = _require(raw, "common_training")["neural_starting_point"]
    expected = {"optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001,
                "batch_size": 256, "max_epochs": 1000, "early_stopping_patience": 50,
                "initialisation_seed": 42}
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"Common neural training value {key} changed: {training.get(key)!r}")

    return SearchSpec(path=target, raw=raw)
