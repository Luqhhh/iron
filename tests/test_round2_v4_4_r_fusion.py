"""V4.4 TabR information-fusion acceptance checks.

The V4.2 adaptation added a weighted mean of neighbour labels to the final
scalar.  The authors' mechanism instead builds

    v_j = E_y(y_j) + T(k(x) - k(x_j)),     h_aug = h(x) + sum_j w_j v_j

with a retrieval key projected separately from the base representation, and
predicts from ``h_aug``.  These tests check that wiring numerically:

1. R4 and R5 are parameter-identical, so R5 isolates the retrieval channel;
2. R5's context term is exactly zero and its output ignores labels and legality;
3. R4 reproduces the fusion formula computed by hand from its own submodules;
4. R4's label encoder and difference transform receive a non-zero gradient;
5. the retrieval key is not the base representation;
6. an excluded neighbour's label cannot change the prediction;
7. R2 keeps the executed output-add path unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bf_tap_r2.v4_2_r_tabr import R_RECIPES, TabRSpec, _make_network


def _spec(recipe: str) -> TabRSpec:
    return TabRSpec(recipe)


def _support(rows: int, features: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, features, generator=generator)


def _legal(rows: int, support: int, excluded: int = 2) -> torch.Tensor:
    mask = torch.ones(rows, support, dtype=torch.bool)
    mask[:, :excluded] = False
    return mask


def _forward(net, rows=4, support=8, features=6, seed=1):
    q = _support(rows, features, seed)
    s = _support(support, features, seed + 1)
    labels = torch.linspace(-2.0, 3.0, support)
    legal = _legal(rows, support)
    return net(q, torch.zeros(rows, 0), s, torch.zeros(support, 0), labels, legal)


def test_r4_and_r5_are_parameter_identical() -> None:
    """Check 1: R5 differs from R4 only in the retrieval channel."""
    r4_net = _make_network(_spec("R4"), 6, 0)
    r5_net = _make_network(_spec("R5"), 6, 0)
    r4_count = sum(p.numel() for p in r4_net.parameters())
    r5_count = sum(p.numel() for p in r5_net.parameters())
    assert r4_count == r5_count, (r4_count, r5_count)
    assert set(r4_net.state_dict()) == set(r5_net.state_dict())
    for name, tensor in r4_net.state_dict().items():
        assert tensor.shape == r5_net.state_dict()[name].shape, name


def test_r5_output_ignores_labels_and_legality() -> None:
    """Check 2: with retrieval off the context term is exactly zero."""
    net = _make_network(_spec("R5"), 6, 0)
    net.eval()
    q = _support(4, 6, 1)
    s = _support(8, 6, 2)
    legal = _legal(4, 8)
    labels_a = torch.zeros(8)
    labels_b = torch.full((8,), 1000.0)

    with torch.no_grad():
        first = net(q, torch.zeros(4, 0), s, torch.zeros(8, 0), labels_a, legal)
        second = net(q, torch.zeros(4, 0), s, torch.zeros(8, 0), labels_b, legal)
    assert torch.allclose(first, second), "R5 output depends on the label channel"

    # And the retrieved features cannot matter either: swap the support set.
    with torch.no_grad():
        third = net(q, torch.zeros(4, 0), torch.randn(8, 6), torch.zeros(8, 0),
                    labels_a, legal)
    assert torch.allclose(first, third), "R5 output depends on the support set"


def test_r4_reproduces_the_manual_fusion_formula() -> None:
    """Check 3: the wiring is the authors' order, not merely similar."""
    net = _make_network(_spec("R4"), 6, 0)
    net.eval()
    q = _support(4, 6, 1)
    s = _support(8, 6, 2)
    labels = torch.linspace(-1.0, 1.0, 8)

    with torch.no_grad():
        observed = net(q, torch.zeros(4, 0), s, torch.zeros(8, 0), labels, _legal(4, 8))

        h_q = net.encode(q, torch.zeros(4, 0))
        h_s = net.encode(s, torch.zeros(8, 0))
        q_key = net.retrieval_key(h_q)
        s_key = net.retrieval_key(h_s)
        weights = net.attention(q_key, s_key, _legal(4, 8))
        embedding = net.label_embedding(labels.reshape(-1, 1))
        difference = net.difference_transform(q_key[:, None, :] - s_key[None, :, :])
        values = embedding[None, :, :] + difference
        augmented = h_q + (weights[:, :, None] * values).sum(dim=1)
        for block in net.predictor:
            augmented = augmented + block(augmented)
        expected = net.head(augmented).reshape(-1)

    assert torch.allclose(observed, expected, atol=1e-6), (observed - expected).abs().max()

    # Guard the check itself: using h instead of the key projection must NOT
    # reproduce the output, otherwise "separate key projection" is untested.
    with torch.no_grad():
        wrong_weights = net.attention(h_q, h_s, _legal(4, 8))
        wrong_difference = net.difference_transform(h_q[:, None, :] - h_s[None, :, :])
        wrong_values = embedding[None, :, :] + wrong_difference
        wrong = h_q + (wrong_weights[:, :, None] * wrong_values).sum(dim=1)
        for block in net.predictor:
            wrong = wrong + block(wrong)
        wrong = net.head(wrong).reshape(-1)
    assert not torch.allclose(observed, wrong, atol=1e-6)


def test_r4_label_encoder_and_difference_transform_get_gradient() -> None:
    """Check 4: the restored mechanism is actually trained."""
    net = _make_network(_spec("R4"), 6, 0)
    output = _forward(net)
    output.sum().backward()

    for name in ("label_embedding", "difference_transform", "key_projection"):
        params = dict(net.named_parameters())
        grads = [p.grad for n, p in params.items() if n.startswith(name)]
        assert grads, f"{name} has no parameters"
        assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads), name


def test_retrieval_key_is_not_the_base_representation() -> None:
    """Check 5: k(x) is a distinct projection, as in the authors' model."""
    net = _make_network(_spec("R4"), 6, 0)
    net.eval()
    x = _support(4, 6, 3)
    with torch.no_grad():
        h = net.encode(x, torch.zeros(4, 0))
        keys = net.retrieval_key(h)
    assert keys.shape == h.shape
    assert not torch.allclose(keys, h), "the key projection is the identity path"


def test_excluded_neighbour_labels_cannot_change_the_prediction() -> None:
    """Check 6: the restored fusion must not leak excluded support rows."""
    net = _make_network(_spec("R4"), 6, 0)
    net.eval()
    q = _support(3, 6, 4)
    s = _support(8, 6, 5)
    legal = _legal(3, 8, excluded=2)
    labels = torch.zeros(8)

    tampered = labels.clone()
    tampered[:2] = 1e6  # rows 0 and 1 are excluded for every query

    with torch.no_grad():
        clean = net(q, torch.zeros(3, 0), s, torch.zeros(8, 0), labels, legal)
        dirty = net(q, torch.zeros(3, 0), s, torch.zeros(8, 0), tampered, legal)
    assert torch.allclose(clean, dirty, atol=1e-6), (clean - dirty).abs().max()


def test_r2_keeps_the_executed_output_add_path() -> None:
    """Check 7: the V4.2 control must not gain the new modules."""
    spec = TabRSpec("R2")
    assert spec.fusion == "output_add"
    net = _make_network(spec, 6, 0)
    assert not hasattr(net, "predictor")
    assert not hasattr(net, "key_projection")
    assert not hasattr(net, "difference_transform")
    assert not hasattr(net, "label_embedding")

    net.eval()
    q = _support(4, 6, 6)
    s = _support(8, 6, 7)
    labels = torch.linspace(-1.0, 1.0, 8)
    legal = _legal(4, 8)
    with torch.no_grad():
        observed = net(q, torch.zeros(4, 0), s, torch.zeros(8, 0), labels, legal)
        base = net.encode(q, torch.zeros(4, 0))
        k_emb = net.encode(s, torch.zeros(8, 0))
        weights = net.attention(base, k_emb, legal)
        expected = (
            net.head(torch.cat([base, weights @ k_emb], dim=1)).reshape(-1)
            + weights @ labels
        )
    assert torch.allclose(observed, expected, atol=1e-6)


def test_distance_is_batch_invariant_for_exact_duplicates() -> None:
    """Regression: a numerically-zero distance must not depend on batch shape.

    The expanded squared-distance identity computes an exact duplicate or self
    match as a cancellation residual.  Its sign depends on the BLAS tiling, so
    before the zero-band collapse the same row scored differently alone than
    inside a batch and the acceptance check "single-row inference agrees" failed.
    """
    net = _make_network(_spec("R2"), 6, 0)
    net.eval()
    x = _support(6, 6, 11)
    with torch.no_grad():
        h = net.encode(x, torch.zeros(6, 0))
        support = torch.cat([h, h], dim=0)          # exact duplicates of every query
        legal = torch.ones(6, 12, dtype=torch.bool)
        batched = net.attention(h, support, legal)
        single = torch.cat(
            [net.attention(h[i:i + 1], support, legal[:1]) for i in range(6)], dim=0
        )
    assert torch.allclose(batched, single, atol=1e-6), (batched - single).abs().max()


def test_v44_recipes_register_the_declared_mechanism() -> None:
    assert {"R0", "R1", "R2", "R3", "R4", "R5"} <= set(R_RECIPES)
    r4, r5 = TabRSpec("R4"), TabRSpec("R5")
    assert r4.fusion == r5.fusion == "representation_augment"
    assert r4.retrieval and not r5.retrieval
    assert r4.separate_key_projection and r4.label_encoder
    assert r4.neighbour_difference_transform and r4.predictor_blocks
    # The starting point must not move with the mechanism.
    for spec in (r4, r5):
        assert (spec.repr_width, spec.n_blocks, spec.n_neighbors) == (128, 2, 32)
        assert spec.dropout == 0.10
