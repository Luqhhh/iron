from __future__ import annotations

import pandas as pd

from ..exceptions import ContractError
from .config import Candidate


def select_candidate_features(
    frame: pd.DataFrame,
    candidate: Candidate,
    selection_config: dict,
) -> pd.DataFrame:
    if candidate.kind != "model":
        raise ContractError("only model candidates have feature selections")
    source_prefixes = selection_config["source_prefixes"]
    selected: list[str] = []
    for column in frame.columns:
        source = next(
            (
                name
                for name, prefixes in source_prefixes.items()
                if any(column == prefix or column.startswith(prefix) for prefix in prefixes)
            ),
            None,
        )
        if source is None:
            raise ContractError(f"unclassified baseline feature column: {column}")
        if source not in candidate.sources:
            continue
        if source == "history":
            matches = {
                component
                for component, patterns in selection_config["history_component_patterns"].items()
                if any(pattern in column for pattern in patterns)
            }
            if len(matches) != 1:
                raise ContractError(f"history feature must map to exactly one component: {column}")
            if not matches.issubset(candidate.history_components):
                continue
        selected.append(column)
    if not selected or "spout_no" not in selected:
        raise ContractError(f"{candidate.id}: selected feature set is invalid")
    return frame.loc[:, selected].copy()
