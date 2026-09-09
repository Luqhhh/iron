from __future__ import annotations

from ...exceptions import ContractError
from .contracts import CandidateDefinition


_DEFINITIONS = (
    CandidateDefinition(
        candidate_id="M1_BLEND",
        family="blend",
        package="bf_tap.models.optimization_v02.m1_blend",
        config_relpath="configs/optimization/v0.2/m1_blend.yaml",
    ),
    CandidateDefinition(
        candidate_id="M2_RECENCY",
        family="recency",
        package="bf_tap.models.optimization_v02.m2_recency",
        config_relpath="configs/optimization/v0.2/m2_recency.yaml",
    ),
    CandidateDefinition(
        candidate_id="M3_RESIDUAL",
        family="residual",
        package="bf_tap.models.optimization_v02.m3_residual",
        config_relpath="configs/optimization/v0.2/m3_residual.yaml",
    ),
    CandidateDefinition(
        candidate_id="M4_ENSEMBLE",
        family="ensemble",
        package="bf_tap.models.optimization_v02.m4_ensemble",
        config_relpath="configs/optimization/v0.2/m4_ensemble.yaml",
    ),
)
_BY_ID = {definition.candidate_id: definition for definition in _DEFINITIONS}


def candidate_ids() -> tuple[str, ...]:
    return tuple(_BY_ID)


def get_candidate_definition(candidate_id: str) -> CandidateDefinition:
    try:
        return _BY_ID[candidate_id]
    except KeyError as exc:
        raise ContractError(f"unknown optimization candidate: {candidate_id}") from exc
