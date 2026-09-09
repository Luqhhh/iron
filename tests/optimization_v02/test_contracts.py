from importlib import import_module
from importlib.util import find_spec

import pytest

from bf_tap.exceptions import ContractError


EXPECTED = {
    "M1_BLEND": (
        "blend",
        "bf_tap.models.optimization_v02.m1_blend",
        "configs/optimization/v0.2/m1_blend.yaml",
    ),
    "M2_RECENCY": (
        "recency",
        "bf_tap.models.optimization_v02.m2_recency",
        "configs/optimization/v0.2/m2_recency.yaml",
    ),
    "M3_RESIDUAL": (
        "residual",
        "bf_tap.models.optimization_v02.m3_residual",
        "configs/optimization/v0.2/m3_residual.yaml",
    ),
    "M4_ENSEMBLE": (
        "ensemble",
        "bf_tap.models.optimization_v02.m4_ensemble",
        "configs/optimization/v0.2/m4_ensemble.yaml",
    ),
}


def _registry():
    package = find_spec("bf_tap.models.optimization_v02")
    assert package is not None, "optimization_v02 package must exist"
    return import_module("bf_tap.models.optimization_v02.registry")


def test_registry_lists_tiered_candidate_definitions():
    registry = _registry()

    assert registry.candidate_ids() == tuple(EXPECTED)
    for candidate_id, expected in EXPECTED.items():
        definition = registry.get_candidate_definition(candidate_id)
        assert (
            definition.family,
            definition.package,
            definition.config_relpath,
        ) == expected
        assert find_spec(definition.package) is not None


def test_registry_rejects_unknown_candidate():
    registry = _registry()

    with pytest.raises(ContractError, match="unknown optimization candidate"):
        registry.get_candidate_definition("M0_BASELINE")
