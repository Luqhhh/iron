"""V4.4 NODE per-depth selection: numerical acceptance checks.

These tests answer the question the V4.4 pre-registration asks, and they check it
numerically rather than inferring it from a class or recipe name:

1. binding every depth's selector to one vector reproduces the shared-selector
   forward output;
2. perturbing one depth's selector does not rewrite another depth's selector;
3. the selector logits receive a non-zero gradient at every depth;
4. serialisation round-trips the ``(trees, depth, features)`` layout;
5. the analytic entmax gradient agrees with finite differences on the
   per-depth routing path.

The tests use a small stand-in spec because they exercise the layer algebra, not
the pre-registered tree count; ``test_real_recipes_*`` pins the real N2/N4/N5
recipes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bf_tap_r2.v4_2_n_node import N_RECIPES, NodeSpec, _make_network, entmax15


@dataclass(frozen=True)
class _MiniSpec:
    """Layer-algebra test double: same attribute surface NodeSpec exposes."""

    selector: str = "per_depth"
    depth: int = 3
    n_layers: int = 1
    trees_per_layer: int = 4
    init_scale: float = 0.05
    temperature: float = 1.0
    learnable_temperature: bool = False
    data_aware_init: bool = False
    threshold_init_beta: float = 1.0
    threshold_init_cutoff: float = 1.0


def _input(rows: int, features: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, features, generator=generator, dtype=torch.float32)


def _bind_depths_to_one_vector(per_depth_layer, shared_layer) -> None:
    """Copy a shared selector into every depth slot of a per-depth layer."""
    with torch.no_grad():
        for tree in range(shared_layer.feature_logits.shape[0]):
            for depth in range(per_depth_layer.depth):
                per_depth_layer.feature_logits[tree, depth, :] = (
                    shared_layer.feature_logits[tree, :]
                )


def test_binding_every_depth_selector_reproduces_the_shared_forward() -> None:
    """Check 1: same vector at every depth == the shared-selector model."""
    rows, features = 7, 5
    shared = _MiniSpec(selector="shared", temperature=1.0)
    per_depth = _MiniSpec(selector="per_depth", temperature=1.0)

    torch.manual_seed(11)
    shared_net = _make_network(shared, features, 0)
    torch.manual_seed(11)
    per_depth_net = _make_network(per_depth, features, 0)

    _bind_depths_to_one_vector(per_depth_net.layers[0], shared_net.layers[0])
    # The selector has a different parameter count in each layout, so the two
    # constructions consume different amounts of RNG.  Copy the remaining
    # parameters explicitly instead of assuming the streams coincide.
    with torch.no_grad():
        per_depth_net.layers[0].thresholds.copy_(shared_net.layers[0].thresholds)
        per_depth_net.layers[0].leaf_weights.copy_(shared_net.layers[0].leaf_weights)
        per_depth_net.head.weight.copy_(shared_net.head.weight)
        per_depth_net.head.bias.copy_(shared_net.head.bias)

    x = _input(rows, features, seed=3)
    with torch.no_grad():
        expected = shared_net(x, torch.zeros(rows, 0))
        observed = per_depth_net(x, torch.zeros(rows, 0))
    assert torch.allclose(observed, expected, atol=1e-6), (observed - expected).abs().max()


def test_perturbing_one_depth_selector_leaves_other_depths_untouched() -> None:
    """Check 2: the depth axis is a real, independent axis."""
    spec = _MiniSpec(selector="per_depth")
    net = _make_network(spec, 5, 0)
    layer = net.layers[0]
    before = layer.selection().detach().clone()

    with torch.no_grad():
        # A single feature logit, not the whole row: entmax is shift-invariant,
        # so adding a constant to every logit would be a no-op.
        layer.feature_logits[0, 1, 0] += 5.0
    after = layer.selection().detach()

    assert not torch.allclose(after[0, 1, :], before[0, 1, :]), "edited depth did not change"
    for depth in (0, 2):
        assert torch.allclose(after[0, depth, :], before[0, depth, :]), (
            f"depth {depth} was rewritten by a change to depth 1"
        )
    for tree in range(1, layer.n_trees):
        assert torch.allclose(after[tree], before[tree]), f"tree {tree} changed"


def test_every_depth_selector_receives_a_nonzero_gradient() -> None:
    """Check 3: no depth is a dead parameter."""
    rows, features = 6, 4
    spec = _MiniSpec(selector="per_depth")
    net = _make_network(spec, features, 0)
    layer = net.layers[0]

    x = _input(rows, features, seed=5).requires_grad_(True)
    out = net(x, torch.zeros(rows, 0))
    ((out - 1.0) ** 2).mean().backward()

    grad = layer.feature_logits.grad
    assert grad is not None and grad.shape == layer.feature_logits.shape
    per_depth_norm = grad.abs().sum(dim=(0, 2))
    assert bool((per_depth_norm > 0).all()), per_depth_norm
    assert bool((grad.abs().sum(dim=(0, 2)) > 0).all())


def test_serialisation_round_trips_the_depth_axis() -> None:
    """Check 4: the (trees, depth, features) layout survives save/load."""
    rows, features = 4, 3
    spec = _MiniSpec(selector="per_depth")
    net = _make_network(spec, features, 0)
    layer = net.layers[0]
    assert layer.feature_logits.shape == (
        layer.n_trees, layer.depth, layer.n_features,
    )

    state = {k: v.clone() for k, v in net.state_dict().items()}
    restored = _make_network(spec, features, 0)
    restored.load_state_dict(state)

    assert restored.layers[0].feature_logits.shape == layer.feature_logits.shape
    x = _input(rows, features, seed=9)
    with torch.no_grad():
        assert torch.allclose(restored(x, torch.zeros(rows, 0)), net(x, torch.zeros(rows, 0)))


def test_per_depth_entmax_gradient_matches_finite_differences() -> None:
    """Check 5: the analytic backward pass is numerically verified."""
    values = torch.tensor([[0.3, -1.2, 0.7, 2.1]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.5, -0.25, 1.5, 0.75], dtype=torch.float64)
    (entmax15(values, dim=-1) * weights).sum().backward()
    analytic = values.grad.detach().numpy().reshape(-1)

    epsilon = 1e-5
    numeric = []
    base = values.detach().numpy()
    for index in range(base.shape[1]):
        delta = np.zeros_like(base)
        delta[0, index] = epsilon
        plus = entmax15(torch.tensor(base + delta, dtype=torch.float64), dim=-1)
        minus = entmax15(torch.tensor(base - delta, dtype=torch.float64), dim=-1)
        numeric.append(float(((plus - minus) * weights).sum() / (2 * epsilon)))
    assert np.allclose(analytic, numeric, atol=1e-6), (analytic, numeric)


def test_shared_selector_keeps_the_original_two_dimensional_layout() -> None:
    """The executed V4.2 structure must not be silently reshaped."""
    spec = _MiniSpec(selector="shared")
    net = _make_network(spec, 5, 0)
    layer = net.layers[0]
    assert layer.feature_logits.ndim == 2
    assert layer.feature_logits.shape == (layer.n_trees, layer.n_features)
    assert not hasattr(layer, "log_temperatures")


def test_real_recipes_register_the_mechanism_they_claim() -> None:
    """N2/N4/N5 must differ only in the declared mechanism fields."""
    assert {"N0", "N1", "N2", "N3", "N4", "N5"} <= set(N_RECIPES)
    control, depth_only, odst = (NodeSpec(r) for r in ("N2", "N4", "N5"))

    assert control.selector == "shared"
    assert not control.learnable_temperature and not control.data_aware_init

    assert depth_only.selector == "per_depth"
    assert not depth_only.learnable_temperature and not depth_only.data_aware_init

    assert odst.selector == "per_depth"
    assert odst.learnable_temperature and odst.data_aware_init

    for spec in (control, depth_only, odst):
        assert spec.total_trees == 128 and spec.depth == 4
        assert spec.n_layers == 2 and spec.trees_per_layer == 64
        assert spec.numeric_encoding == "raw"
        assert spec.temperature == 1.0

    net = _make_network(depth_only, 6, 1)
    layer = net.layers[0]
    assert layer.feature_logits.shape == (64, 4, 7)
    assert not hasattr(layer, "log_temperatures")

    odst_net = _make_network(odst, 6, 1)
    odst_layer = odst_net.layers[0]
    assert odst_layer.feature_logits.shape == (64, 4, 7)
    assert odst_layer.log_temperatures.shape == (64, 4)


def test_routing_entropy_is_finite_at_saturation() -> None:
    """Regression: saturated routing must not turn the collapse check into NaN.

    ``1 - 1e-9`` is exactly ``1.0`` in float32, so the old clamp upper bound left
    ``p == 1`` and ``(1 - p) * log(1 - p)`` evaluated to ``0 * -inf = NaN``.  A
    learnable routing scale saturates often, so the diagnostic went NaN exactly
    where it was supposed to warn about routing collapse.
    """
    from bf_tap_r2.v4_2_n_node import binary_routing_entropy

    saturated = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    entropy = binary_routing_entropy(saturated)
    assert bool(torch.isfinite(entropy).all()), entropy
    # A fully saturated response carries essentially no information.  The
    # residual is the entropy of the float32-representable clamp margin, not zero.
    assert float(entropy.mean()) < 1e-4
    # A maximally uncertain response is exactly 1 after normalisation.
    half = binary_routing_entropy(torch.full((4,), 0.5, dtype=torch.float32))
    assert abs(float(half.mean()) - 1.0) < 1e-5
    # The old formulation is what produced the NaN, so pin the failure mode.
    legacy = torch.tensor([1.0], dtype=torch.float32).clamp(1e-9, 1 - 1e-9)
    assert bool(torch.isnan(-(legacy * legacy.log() + (1 - legacy) * (1 - legacy).log())).any())


def test_data_aware_init_replaces_the_random_thresholds() -> None:
    """The ODST core must actually initialise from data, not from randn."""
    torch.manual_seed(4)
    spec = _MiniSpec(selector="per_depth", data_aware_init=True, learnable_temperature=True)
    net = _make_network(spec, 4, 0)
    layer = net.layers[0]
    torch.manual_seed(0)
    before = layer.thresholds.detach().clone()

    x = torch.randn(64, 4) * 3.0 + 1.5
    net.initialize_data_aware(x, torch.zeros(64, 0), np.random.default_rng(7))

    after = layer.thresholds.detach()
    assert not torch.allclose(after, before), "thresholds were not initialised"
    assert bool(torch.isfinite(after).all())
    # Beta(1, 1) quantiles of the data: the thresholds must lie inside the data
    # range, unlike an unconstrained 0.05 * randn draw.
    assert float(after.min()) >= float(x.min()) - 1e-4
    assert float(after.max()) <= float(x.max()) + 1e-4

    temperatures = torch.exp(layer.log_temperatures.detach())
    assert bool(torch.isfinite(temperatures).all())
    assert bool((temperatures > 0).all()), temperatures


def test_data_aware_init_is_reproducible_for_a_fixed_seed() -> None:
    """Same seed -> same initialisation; the stream is not process-local."""
    spec = _MiniSpec(selector="per_depth", data_aware_init=True, learnable_temperature=True)
    x = torch.randn(48, 4)

    def build():
        torch.manual_seed(3)
        net = _make_network(spec, 4, 0)
        net.initialize_data_aware(x, torch.zeros(48, 0), np.random.default_rng(101))
        return net

    first, second = build(), build()
    assert torch.allclose(first.layers[0].thresholds, second.layers[0].thresholds)
    assert torch.allclose(first.layers[0].log_temperatures, second.layers[0].log_temperatures)


def test_data_aware_init_requires_per_depth_selection() -> None:
    spec = _MiniSpec(selector="shared", data_aware_init=True)
    net = _make_network(spec, 4, 0)
    with pytest.raises(ValueError):
        net.layers[0].initialize_data_aware(torch.randn(8, 4), np.random.default_rng(1))


def test_invalid_mechanism_combinations_are_rejected(monkeypatch) -> None:
    """The spec must refuse a learnable scale without per-depth selection."""
    from bf_tap_r2 import v4_2_n_node

    monkeypatch.setitem(
        v4_2_n_node.N_RECIPES,
        "NX",
        {**v4_2_n_node.N_RECIPES["N2"], "recipe_id": "NX",
         "learnable_temperature": True},
    )
    with pytest.raises(ValueError):
        NodeSpec("NX")
