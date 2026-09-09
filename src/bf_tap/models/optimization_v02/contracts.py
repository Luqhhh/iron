from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateDefinition:
    candidate_id: str
    family: str
    package: str
    config_relpath: str
