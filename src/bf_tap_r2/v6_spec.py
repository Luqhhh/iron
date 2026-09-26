"""Loader and hard validator for ``configs/round2_v6/SPEC.yaml``.

Same contract as :mod:`bf_tap_r2.v5_spec`: the V6 pre-registration is frozen and
a weakened revision is refused rather than executed.  The design is derived from
the V5 outcome, so the validator also pins the *facts* the design rests on — the
winning member's signature, the sibling-duplication finding, the retracted
transfer-error claim and the untouched iron capacity policy.  If one of those
changes, the V6 design must be re-derived rather than silently reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .data import TARGETS
from .v6_sampler import V6_CAPACITIES, V6_STRUCTURES, V6_TRIAL_COUNT

__all__ = [
    "DEFAULT_SPEC_PATH",
    "V6_REQUIRED_VERSION",
    "CURRENT_PLATFORM_BEST",
    "USER_REQUESTED_PLATFORM_SCORE_VERBATIM",
    "V6Spec",
    "load_v6_spec",
    "validate_v6_spec",
]

DEFAULT_SPEC_PATH = "configs/round2_v6/SPEC.yaml"
V6_REQUIRED_VERSION = "round2-v6-iron-capacity-networks"

USER_REQUESTED_PLATFORM_SCORE_VERBATIM = 93.5
CURRENT_PLATFORM_BEST = 96.3143
FROZEN_LOCAL_WORKING_GATE = 96.25

#: Signed facts from V5 that the V6 design depends on.
WINNING_MEMBER = {
    "trial_id": "v36-s1-N-0048",
    "target": "tap_time_len",
    "structure": "raw_tabm",
    "capacity": "large",
    "residual_correlation": 0.888,
    "accuracy_ratio": 1.07,
    "release_alpha": 0.20,
}
PLATFORM_RESOLUTION = 0.0005
OBSERVED_TRANSFER_ERROR = 0.031
IRON_SIBLING_DUPLICATION = ("v36-s1-N-0048", "v36-s1-N-0049", "v36-s1-N-0050", "v36-s1-N-0051")

#: Frozen V6 signature and budget numbers.
STAGE_A = {"residual_correlation_max": 0.95, "accuracy_ratio_max": 1.25,
           "projected_nested_fold_gain_min": 0.0, "split_seed": 42}
STAGE_B = {"residual_correlation_max": 0.95, "accuracy_ratio_max": 1.15}
MUTUAL_DIVERSITY = {"max_members_per_structural_family": 1, "mutual_residual_correlation_max": 0.95}
PROMOTION = {"min_seeds": 4, "paired_lcb_level": 0.95, "fold_reference_pair": (8, 10)}
BUDGET = {"stage_a_new_fits": 0, "stage_b_max_trials_per_target": 6,
          "stage_b_max_fit_slots_per_target": 60, "stage_b_max_fit_slots_total": 120,
          "stage_c_max_candidates": 2, "stage_c_max_candidate_fit_slots": 20,
          "stage_d_max_packages": 1, "wall_clock_cap_hours": 3}
DERIVED_SEEDS = (7777, 12011)
ALPHA_BOUNDS = (0.05, 0.5)
ALPHA_GRID_POINTS = 46
STAGE_A2 = {"target": "tap_iron", "max_fit_slots": 64, "seeds": (42,), "folds": (0, 1)}
MEASURED_OFFSET_RANGE = (0.0347, 0.1008)

EXPECTED_SOURCE_TRIALS = {"s1_iron_recorded_never_complete": 12,
                          "s1_time_recorded_never_complete": 44,
                          "s2_iron_capacity_networks": V6_TRIAL_COUNT}


def _require(mapping: Mapping[str, Any], name: str, where: str = "") -> Any:
    if name not in mapping:
        prefix = f"{where}: " if where else ""
        raise ValueError(f"{prefix}V6 SPEC is missing required key: {name}")
    return mapping[name]


def _require_equal(actual: Any, expected: Any, name: str, tolerance: float = 1e-12) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > tolerance:
            raise ValueError(f"V6 SPEC froze {name} = {expected!r}, found {actual!r}")
        return
    if actual != expected:
        raise ValueError(f"V6 SPEC froze {name} = {expected!r}, found {actual!r}")


def _require_sequence_equal(actual: Any, expected: Sequence[Any], name: str) -> None:
    values = tuple(actual)
    if values != tuple(expected):
        raise ValueError(f"V6 SPEC froze {name} = {tuple(expected)!r}, found {values!r}")


def validate_v6_spec(raw: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a frozen V6 constraint has been changed."""
    _require_equal(raw.get("version"), V6_REQUIRED_VERSION, "version")

    targets = _require(raw, "targets")
    _require_equal(targets.get("user_requested_platform_score_verbatim"),
                   USER_REQUESTED_PLATFORM_SCORE_VERBATIM,
                   "targets.user_requested_platform_score_verbatim")
    best = _require(targets, "current_platform_best")
    _require_equal(best.get("score"), CURRENT_PLATFORM_BEST, "targets.current_platform_best.score")
    _require_equal(best.get("candidate"), "V5_TIME_N0048_Q20", "targets.current_platform_best.candidate")
    _require_equal(targets.get("frozen_local_working_gate"), FROZEN_LOCAL_WORKING_GATE,
                   "targets.frozen_local_working_gate")
    if str(targets.get("submission_requires")) != "frozen_gate_or_recorded_user_authorisation":
        raise ValueError("V6 SPEC must keep the frozen gate as the submission condition")

    context = _require(raw, "context_from_v5")
    winning = _require(context, "winning_member")
    for key, expected in WINNING_MEMBER.items():
        _require_equal(winning.get(key), expected, f"context_from_v5.winning_member.{key}")
    _require_equal(context.get("platform_resolution_score"), PLATFORM_RESOLUTION,
                   "context_from_v5.platform_resolution_score")
    transfer = _require(context, "transfer_error")
    _require_equal(transfer.get("observed"), OBSERVED_TRANSFER_ERROR,
                   "context_from_v5.transfer_error.observed")
    if str(transfer.get("role")) != "candidate_dependent":
        raise ValueError("V6 SPEC must record the transfer error as candidate-dependent")
    if "0.005" not in str(transfer.get("retracted", "")):
        raise ValueError("V6 SPEC must retract the constant 0.005 transfer-error claim")
    magnitude = _require(context, "local_magnitude")
    if str(magnitude.get("role")) != "sign_and_ranking_only":
        raise ValueError("V6 SPEC must restrict local evidence to sign and ranking")
    siblings = _require(context, "sibling_duplication")
    _require_sequence_equal(_require(siblings, "trials"), IRON_SIBLING_DUPLICATION,
                            "context_from_v5.sibling_duplication.trials")

    sources = _require(raw, "candidate_sources")
    by_name = {str(_require(source, "name", "candidate_sources")): source for source in sources}
    if set(by_name) != set(EXPECTED_SOURCE_TRIALS):
        raise ValueError(f"V6 SPEC candidate_sources must be exactly {sorted(EXPECTED_SOURCE_TRIALS)}")
    for name, expected_trials in EXPECTED_SOURCE_TRIALS.items():
        source = by_name[name]
        _require_equal(source.get("expected_trials"), expected_trials,
                       f"candidate_sources.{name}.expected_trials")
        if source.get("target") not in tuple(TARGETS):
            raise ValueError(f"candidate_sources.{name}.target must be a known target")
    s2 = by_name["s2_iron_capacity_networks"]
    _require_equal(s2.get("target"), "tap_iron", "candidate_sources.s2_iron_capacity_networks.target")
    _require_equal(s2.get("sampler"), "bf_tap_r2.v6_sampler",
                   "candidate_sources.s2_iron_capacity_networks.sampler")
    if bool(s2.get("zero_fit_screen")):
        raise ValueError("the new iron capacity space cannot be screened without fits")

    signature = _require(raw, "signature")
    stage_a = _require(signature, "stage_a")
    for key, expected in STAGE_A.items():
        _require_equal(stage_a.get(key), expected, f"signature.stage_a.{key}")
    _require_sequence_equal(_require(stage_a, "folds"), (0, 1), "signature.stage_a.folds")
    stage_b = _require(signature, "stage_b")
    for key, expected in STAGE_B.items():
        _require_equal(stage_b.get(key), expected, f"signature.stage_b.{key}")
    _require_sequence_equal(_require(stage_b, "split_seeds"), (42, 3407),
                            "signature.stage_b.split_seeds")
    _require_sequence_equal(_require(stage_b, "folds"), (0, 1, 2, 3, 4),
                            "signature.stage_b.folds")
    diversity = _require(signature, "mutual_diversity")
    for key, expected in MUTUAL_DIVERSITY.items():
        _require_equal(diversity.get(key), expected, f"signature.mutual_diversity.{key}")

    fusion = _require(raw, "fusion")
    _require_sequence_equal(_require(fusion, "alpha_search_bounds"), ALPHA_BOUNDS,
                            "fusion.alpha_search_bounds")
    _require_equal(fusion.get("alpha_grid_points"), ALPHA_GRID_POINTS, "fusion.alpha_grid_points")
    if not bool(fusion.get("no_alpha_scan_at_release")):
        raise ValueError("V6 SPEC must forbid an alpha scan at release")

    budget = _require(raw, "budget")
    _require_equal(budget.get("stage_a_new_fits"), BUDGET["stage_a_new_fits"], "budget.stage_a_new_fits")
    stage_a2 = _require(budget, "stage_a2_iron_capacity_screen")
    _require_equal(stage_a2.get("target"), STAGE_A2["target"],
                   "budget.stage_a2_iron_capacity_screen.target")
    _require_equal(stage_a2.get("max_fit_slots"), STAGE_A2["max_fit_slots"],
                   "budget.stage_a2_iron_capacity_screen.max_fit_slots")
    _require_sequence_equal(_require(stage_a2, "seeds"), STAGE_A2["seeds"],
                            "budget.stage_a2_iron_capacity_screen.seeds")
    _require_sequence_equal(_require(stage_a2, "folds"), STAGE_A2["folds"],
                            "budget.stage_a2_iron_capacity_screen.folds")
    if "stage A" not in str(stage_a2.get("gated_on", "")):
        raise ValueError("the new iron capacity screen must stay gated on stage A")
    reference = _require(raw, "reference")
    for key in ("v36_summary", "v36_development_cache", "v36_coarse_ledger", "v36_parent_package",
                "v5_time_family", "baseline_cache"):
        if not str(_require(reference, key, "reference")):
            raise ValueError(f"V6 SPEC reference.{key} must be a non-empty path")
    frozen_folds = _require(reference, "frozen_folds")
    if {int(k) for k in frozen_folds} != {42, 3407, 2026}:
        raise ValueError("V6 SPEC must freeze the fold vectors for seeds 42, 3407 and 2026")

    amendment = _require(raw, "amendment")
    if bool(amendment.get("thresholds_changed")):
        raise ValueError("the V6 amendment must not have changed a threshold")
    stage_b_budget = _require(budget, "stage_b")
    _require_equal(stage_b_budget.get("max_trials_per_target"), BUDGET["stage_b_max_trials_per_target"],
                   "budget.stage_b.max_trials_per_target")
    _require_equal(stage_b_budget.get("max_fit_slots_per_target"),
                   BUDGET["stage_b_max_fit_slots_per_target"],
                   "budget.stage_b.max_fit_slots_per_target")
    _require_equal(stage_b_budget.get("max_fit_slots_total"),
                   BUDGET["stage_b_max_fit_slots_total"], "budget.stage_b.max_fit_slots_total")
    if int(stage_b_budget.get("max_fit_slots_per_target", 0)) != (
            int(stage_b_budget.get("max_trials_per_target", 0)) * 2 * 5):
        raise ValueError("V6 stage B budget must equal trials x 2 seeds x 5 folds")
    stage_c = _require(budget, "stage_c")
    _require_sequence_equal(_require(stage_c, "derived_seeds"), DERIVED_SEEDS,
                            "budget.stage_c.derived_seeds")
    _require_equal(stage_c.get("max_candidates"), BUDGET["stage_c_max_candidates"],
                   "budget.stage_c.max_candidates")
    _require_equal(stage_c.get("max_candidate_fit_slots"), BUDGET["stage_c_max_candidate_fit_slots"],
                   "budget.stage_c.max_candidate_fit_slots")
    if not bool(stage_c.get("reuse_cached_baselines")):
        raise ValueError("V6 SPEC must reuse the cached derived-seed baselines")
    _require_equal(_require(budget, "stage_d").get("max_packages"), BUDGET["stage_d_max_packages"],
                   "budget.stage_d.max_packages")
    _require_equal(budget.get("wall_clock_cap_hours"), BUDGET["wall_clock_cap_hours"],
                   "budget.wall_clock_cap_hours")

    gates = _require(raw, "gates")
    promotion = _require(gates, "promotion")
    _require_equal(promotion.get("min_seeds"), PROMOTION["min_seeds"], "gates.promotion.min_seeds")
    _require_equal(promotion.get("paired_lcb_level"), PROMOTION["paired_lcb_level"],
                   "gates.promotion.paired_lcb_level")
    if not bool(promotion.get("require_paired_lcb_gt_zero")):
        raise ValueError("V6 SPEC must require a positive paired lower bound")
    if not bool(promotion.get("require_all_contributing_seeds_positive")):
        raise ValueError("V6 SPEC must require every contributing seed to be positive")
    if str(promotion.get("fold_level_role")) != "descriptive_only":
        raise ValueError("V6 SPEC must keep the fold level descriptive_only")
    _require_sequence_equal(_require(promotion, "fold_reference_pair"), PROMOTION["fold_reference_pair"],
                            "gates.promotion.fold_reference_pair")
    release = _require(gates, "release")
    for key in ("require_unchanged_column_byte_identical", "require_read_back_verification",
                "require_cold_check"):
        if not bool(release.get(key)):
            raise ValueError(f"V6 SPEC gates.release.{key} must be true")
    if not bool(release.get("single_target_replacement_only")):
        raise ValueError("V6 SPEC must release single-target replacement packages only")
    submission = _require(gates, "platform_submission")
    _require_equal(submission.get("frozen_local_working_gate"), FROZEN_LOCAL_WORKING_GATE,
                   "gates.platform_submission.frozen_local_working_gate")
    if str(submission.get("gate_rederivation")) != "PENDING_USER_DECISION":
        raise ValueError("V6 SPEC must leave any gate re-derivation to the user")
    _require_sequence_equal(_require(submission, "measured_offset_range"), MEASURED_OFFSET_RANGE,
                            "gates.platform_submission.measured_offset_range")

    stop_rules = _require(raw, "stop_rules")
    if len(list(stop_rules)) < 4:
        raise ValueError("V6 SPEC must record the stage stop rules")
    non_goals = list(_require(raw, "non_goals"))
    for required in ("seed_or_loss_diversity_inside_one_structural_family",
                     "researching_the_existing_oof_pool_for_more_members",
                     "temporal_or_lag_features"):
        if required not in non_goals:
            raise ValueError(f"V6 SPEC non_goals must include {required!r}")
    publication = _require(raw, "publication")
    if str(publication.get("upload_actor")) != "user":
        raise ValueError("V6 SPEC upload_actor must be user")
    _require_equal(publication.get("agent_uploads"), 0, "publication.agent_uploads")
    if not bool(publication.get("no_automatic_packages")):
        raise ValueError("V6 SPEC must forbid automatic packaging")


@dataclass(frozen=True)
class V6Spec:
    """Validated V6 pre-registration specification."""

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
        return tuple(TARGETS)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [dict(source) for source in self.raw["candidate_sources"]]

    @property
    def stage_a(self) -> dict[str, Any]:
        return dict(self.raw["signature"]["stage_a"])

    @property
    def stage_b(self) -> dict[str, Any]:
        return dict(self.raw["signature"]["stage_b"])

    @property
    def diversity(self) -> dict[str, Any]:
        return dict(self.raw["signature"]["mutual_diversity"])

    @property
    def promotion(self) -> dict[str, Any]:
        return dict(self.raw["gates"]["promotion"])

    @property
    def budget(self) -> dict[str, Any]:
        return dict(self.raw["budget"])

    @property
    def alpha_grid(self) -> tuple[float, ...]:
        import numpy as np

        low, high = (float(v) for v in self.raw["fusion"]["alpha_search_bounds"])
        return tuple(float(v) for v in np.linspace(low, high, int(self.raw["fusion"]["alpha_grid_points"])))

    @property
    def new_iron_structure_families(self) -> list[str]:
        return [f"{structure}|{capacity}" for capacity in V6_CAPACITIES
                for structure in V6_STRUCTURES]


def load_v6_spec(root: Path | str) -> V6Spec:
    """Load and hard-validate the V6 pre-registration under ``root``."""
    path = Path(root) / DEFAULT_SPEC_PATH
    if not path.is_file():
        raise FileNotFoundError(f"V6 pre-registration spec not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V6 SPEC must be a mapping")
    validate_v6_spec(raw)
    return V6Spec(path=path, raw=raw)
