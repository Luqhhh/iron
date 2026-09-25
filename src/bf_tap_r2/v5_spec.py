"""Loader and hard validator for ``configs/round2_v5/SPEC.yaml``.

The V5 specification is the pre-registration contract for the round
"error-covariance member search, composition shape, evaluation resolution and
the platform noise floor".  ``load_v5_spec`` refuses to return a specification
whose frozen constraints have been changed, so a weakened plan cannot be
executed as if it were the pre-registered one.

Only the ``status`` field may be annotated after execution; it is deliberately
not validated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import TARGETS

__all__ = [
    "DEFAULT_SPEC_PATH",
    "V5_SPEC_VERSION",
    "V5_REQUIRED_VERSION",
    "USER_REQUESTED_PLATFORM_SCORE_VERBATIM",
    "OPERATIONAL_PLATFORM_TARGET",
    "CURRENT_PLATFORM_BEST_SCORE",
    "LOCAL_WORKING_GATE",
    "V5Spec",
    "load_v5_spec",
    "validate_v5_spec",
]

DEFAULT_SPEC_PATH = "configs/round2_v5/SPEC.yaml"

V5_SPEC_VERSION = "round2-v5-error-covariance-resolution"
V5_REQUIRED_VERSION = V5_SPEC_VERSION

#: Preserved verbatim from the user request.  The repo does not silently rewrite
#: this number; the operating target is declared separately.
USER_REQUESTED_PLATFORM_SCORE_VERBATIM = 93.5
OPERATIONAL_PLATFORM_TARGET = "platform_gt_96.3"
CURRENT_PLATFORM_BEST_SCORE = 96.2734
LOCAL_WORKING_GATE = 96.25

#: Hard admissibility thresholds frozen by the plan.
RESIDUAL_CORRELATION_MAX = 0.95
SINGLE_WMAPE_RATIO_MAX = 1.25
ALPHA_BOUNDS = (0.05, 0.5)
ALPHA_GRID_POINTS = 46

#: Resolution contract.  Fold-level bounds cannot separate the historical N2
#: candidate from noise, so the split seed is the promotion unit.
MIN_SEEDS_FOR_PROMOTION = 4
PAIRED_LCB_LEVEL = 0.95
POSITIVE_CELLS_MIN = 8
POSITIVE_CELLS_TOTAL = 10

FROZEN_FOLD_SEEDS = (42, 3407, 2026)
DERIVED_FOLD_SEEDS = (7777, 12011)

EXPECTED_ROWS = {"train": 2754, "test": 322}


def _require(mapping: Mapping[str, Any], name: str, where: str = "") -> Any:
    if name not in mapping:
        prefix = f"{where}: " if where else ""
        raise ValueError(f"{prefix}V5 SPEC is missing required key: {name}")
    return mapping[name]


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-12:
            raise ValueError(f"V5 SPEC froze {name} = {expected!r}, found {actual!r}")
        return
    if actual != expected:
        raise ValueError(f"V5 SPEC froze {name} = {expected!r}, found {actual!r}")


def _require_sequence_equal(actual: Any, expected: tuple[Any, ...], name: str) -> None:
    values = tuple(actual)
    if values != tuple(expected):
        raise ValueError(f"V5 SPEC froze {name} = {tuple(expected)!r}, found {values!r}")


def _validate_targets(spec: Mapping[str, Any]) -> None:
    targets = tuple(str(v) for v in _require(spec["data_contract"], "targets", "data_contract"))
    if targets != tuple(TARGETS):
        raise ValueError(f"V5 SPEC targets must be {tuple(TARGETS)!r}, found {targets!r}")


def _validate_library(spec: Mapping[str, Any]) -> None:
    library = _require(spec, "library")
    if not bool(library.get("require_fold_alignment")):
        raise ValueError("V5 SPEC must require fold alignment")
    if not bool(library.get("forbid_trial_id_only_keys")):
        raise ValueError("V5 SPEC must forbid trial-id-only library keys")
    sources = library.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("V5 SPEC library.sources must be a non-empty list")
    names = [str(_require(s, "name", "library.sources")) for s in sources]
    if len(set(names)) != len(names):
        raise ValueError("V5 SPEC library source names must be unique")
    required = {"v36_dev_experts", "v3_refine_catboost", "v3_refine_all", "v3_refine_xgboost",
                "v2_oof_columns", "v36_coarse_diagnostic"}
    missing = required - set(names)
    if missing:
        raise ValueError(f"V5 SPEC library is missing required sources: {sorted(missing)}")
    for source in sources:
        seeds = tuple(int(v) for v in _require(source, "seeds", str(source.get("name"))))
        folds = tuple(int(v) for v in _require(source, "folds", str(source.get("name"))))
        coverage = str(_require(source, "coverage", str(source.get("name"))))
        if coverage == "full" and (seeds != (42, 3407) or folds != (0, 1, 2, 3, 4)):
            raise ValueError(
                f"V5 SPEC full-coverage source {source['name']!r} must use seeds (42, 3407) "
                f"and folds (0..4), found {seeds}/{folds}"
            )
        if coverage == "folds01" and (seeds != (42,) or folds != (0, 1)):
            raise ValueError(
                f"V5 SPEC folds01 source {source['name']!r} must use seed 42 and folds (0, 1)"
            )
    classes = library.get("packagability_classes")
    if not isinstance(classes, Mapping) or set(classes) != {"P0", "P1", "P2"}:
        raise ValueError("V5 SPEC packagability_classes must define exactly P0, P1, P2")


def _validate_admissibility(spec: Mapping[str, Any]) -> None:
    adm = _require(spec, "admissibility")
    _require_equal(adm.get("residual_correlation_max"), RESIDUAL_CORRELATION_MAX,
                   "admissibility.residual_correlation_max")
    _require_equal(adm.get("single_wmape_ratio_max"), SINGLE_WMAPE_RATIO_MAX,
                   "admissibility.single_wmape_ratio_max")
    _require_sequence_equal(_require(adm, "alpha_search_bounds"), ALPHA_BOUNDS,
                            "admissibility.alpha_search_bounds")
    _require_equal(adm.get("alpha_grid_points"), ALPHA_GRID_POINTS,
                   "admissibility.alpha_grid_points")
    if not bool(adm.get("require_alpha_interior")):
        raise ValueError("V5 SPEC must require an interior alpha optimum")


def _validate_resolution(spec: Mapping[str, Any]) -> None:
    res = _require(spec, "resolution")
    fold = _require(res, "fold_level")
    _require_equal(fold.get("positive_cells_min"), POSITIVE_CELLS_MIN, "resolution.fold_level.positive_cells_min")
    _require_equal(fold.get("positive_cells_total"), POSITIVE_CELLS_TOTAL,
                   "resolution.fold_level.positive_cells_total")
    if str(fold.get("role")) != "descriptive_only":
        raise ValueError("V5 SPEC fold-level evidence must be descriptive_only")
    seed = _require(res, "seed_level")
    _require_equal(seed.get("min_seeds_for_promotion"), MIN_SEEDS_FOR_PROMOTION,
                   "resolution.seed_level.min_seeds_for_promotion")
    _require_equal(seed.get("paired_lcb_level"), PAIRED_LCB_LEVEL,
                   "resolution.seed_level.paired_lcb_level")
    if not bool(seed.get("require_paired_lcb_gt_zero")):
        raise ValueError("V5 SPEC must require a positive paired seed-level lower bound")
    derivation = _require(res, "new_seed_derivation")
    _require_sequence_equal(_require(derivation, "frozen_seeds"), FROZEN_FOLD_SEEDS,
                            "resolution.new_seed_derivation.frozen_seeds")
    _require_sequence_equal(_require(derivation, "derived_seeds"), DERIVED_FOLD_SEEDS,
                            "resolution.new_seed_derivation.derived_seeds")
    if str(derivation.get("routine")) != "bf_tap_r2.splits.make_folds":
        raise ValueError("V5 SPEC derived seeds must come from bf_tap_r2.splits.make_folds")
    backtest = _require(res, "backtest")
    _require_equal(backtest.get("known_bad_min_rejected"), 3, "resolution.backtest.known_bad_min_rejected")
    _require_equal(backtest.get("known_bad_total"), 4, "resolution.backtest.known_bad_total")
    if len(list(backtest.get("known_bad", []))) != 4:
        raise ValueError("V5 SPEC backtest.known_bad must list four candidates")
    if not list(backtest.get("known_good_must_remain_eligible", [])):
        raise ValueError("V5 SPEC backtest must name the known-good control")


def _validate_budget(spec: Mapping[str, Any]) -> None:
    budget = _require(spec, "budget")
    _require_equal(budget.get("stage1_new_model_fits"), 0, "budget.stage1_new_model_fits")
    stage2a = _require(budget, "stage2a")
    if str(stage2a.get("family")) != "N" or str(stage2a.get("target")) != "tap_time_len":
        raise ValueError("V5 SPEC stage2a must be the N-family time-column completion")
    _require_equal(stage2a.get("max_selected_trials"), 16, "budget.stage2a.max_selected_trials")
    _require_equal(stage2a.get("extra_seed"), 2026, "budget.stage2a.extra_seed")
    _require_equal(stage2a.get("extra_seed_top_k"), 3, "budget.stage2a.extra_seed_top_k")
    stage2b = _require(budget, "stage2b")
    _require_sequence_equal(_require(stage2b, "new_split_seeds"), DERIVED_FOLD_SEEDS,
                            "budget.stage2b.new_split_seeds")
    if str(budget.get("stop_rule_on_backtest_failure")) != "stop_before_stage2":
        raise ValueError("V5 SPEC must stop before Stage 2 when the backtest fails")


def _validate_gates(spec: Mapping[str, Any]) -> None:
    gates = _require(spec, "gates")
    _require_equal(_require(gates, "stage2").get("require_local_package_min"),
                   LOCAL_WORKING_GATE, "gates.stage2.require_local_package_min")
    stage3 = _require(gates, "stage3")
    for key in ("require_unchanged_column_byte_identical", "require_parent_reproduction_check",
                "require_cold_consistency_check"):
        if not bool(stage3.get(key)):
            raise ValueError(f"V5 SPEC gates.stage3.{key} must be true")


def _validate_publication(spec: Mapping[str, Any]) -> None:
    publication = _require(spec, "publication")
    if str(publication.get("upload_actor")) != "user":
        raise ValueError("V5 SPEC upload_actor must be user")
    _require_equal(publication.get("agent_uploads"), 0, "publication.agent_uploads")
    if not bool(publication.get("no_automatic_packages")):
        raise ValueError("V5 SPEC must forbid automatic packaging")


def validate_v5_spec(raw: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a frozen V5 constraint has been changed."""
    _require_equal(raw.get("version"), V5_REQUIRED_VERSION, "version")
    targets = _require(raw, "targets")
    _require_equal(targets.get("user_requested_platform_score_verbatim"),
                   USER_REQUESTED_PLATFORM_SCORE_VERBATIM,
                   "targets.user_requested_platform_score_verbatim")
    _require_equal(targets.get("operational_platform_target"), OPERATIONAL_PLATFORM_TARGET,
                   "targets.operational_platform_target")
    best = _require(targets, "current_platform_best")
    _require_equal(best.get("score"), CURRENT_PLATFORM_BEST_SCORE, "targets.current_platform_best.score")
    _require_equal(targets.get("local_working_gate"), LOCAL_WORKING_GATE, "targets.local_working_gate")
    _require_equal(_require(raw, "data_contract").get("expected_rows"), EXPECTED_ROWS,
                   "data_contract.expected_rows")
    _validate_targets(raw)
    _validate_library(raw)
    _validate_admissibility(raw)
    _validate_resolution(raw)
    _validate_budget(raw)
    _validate_gates(raw)
    _validate_publication(raw)
    routes = _require(raw, "closed_routes")
    if not list(routes.get("do_not_reopen", [])) or not list(routes.get("unavailable_not_untried", [])):
        raise ValueError("V5 SPEC closed_routes must record both lists")


@dataclass(frozen=True)
class V5Spec:
    """Validated V5 pre-registration specification."""

    path: Path
    raw: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.raw["status"])

    @property
    def version(self) -> str:
        return str(self.raw["version"])

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(str(v) for v in self.raw["data_contract"]["targets"])

    @property
    def library_sources(self) -> list[dict[str, Any]]:
        return [dict(v) for v in self.raw["library"]["sources"]]

    @property
    def admissibility(self) -> dict[str, Any]:
        return dict(self.raw["admissibility"])

    @property
    def alpha_grid(self) -> tuple[float, ...]:
        import numpy as np

        low, high = (float(v) for v in self.raw["admissibility"]["alpha_search_bounds"])
        points = int(self.raw["admissibility"]["alpha_grid_points"])
        return tuple(float(v) for v in np.linspace(low, high, points))

    @property
    def stage2a(self) -> dict[str, Any]:
        return dict(self.raw["budget"]["stage2a"])

    @property
    def stage2b(self) -> dict[str, Any]:
        return dict(self.raw["budget"]["stage2b"])

    @property
    def gates(self) -> dict[str, Any]:
        return dict(self.raw["gates"])

    @property
    def backtest(self) -> dict[str, Any]:
        return dict(self.raw["resolution"]["backtest"])


def load_v5_spec(root: Path | str) -> V5Spec:
    """Load and hard-validate the V5 pre-registration under ``root``."""
    path = Path(root) / DEFAULT_SPEC_PATH
    if not path.is_file():
        raise FileNotFoundError(f"V5 pre-registration spec not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V5 SPEC must be a mapping")
    validate_v5_spec(raw)
    return V5Spec(path=path, raw=raw)
