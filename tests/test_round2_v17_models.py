import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

import torch
import yaml

from bf_tap_r2.v12_joint import make_network
from bf_tap_r2.v17_models import dual_network, projected_task_grads


def test_asymmetric_projection_removes_only_conflicting_shared_component():
    shared = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    private = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    iron = shared[0] + 2 * shared[1] + private[0]
    time = -2 * shared[0] + shared[1] + 3 * private[1]
    diagnostic = projected_task_grads([shared], [private], iron, time)
    assert not diagnostic["conflict"]  # Shared dot product is exactly zero.
    assert torch.allclose(shared.grad, torch.tensor([-0.5, 1.5]))
    assert torch.allclose(private.grad, torch.tensor([0.5, 1.5]))

    shared = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    private = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    iron = shared[0] + private[0]
    time = -2 * shared[0] + shared[1] + 3 * private[1]
    diagnostic = projected_task_grads([shared], [private], iron, time)
    assert diagnostic["conflict"]
    assert torch.allclose(shared.grad, torch.tensor([0.5, 0.5]))
    assert torch.allclose(private.grad, torch.tensor([0.5, 1.5]))
    assert diagnostic["projection_norm"] > 0


def test_dual_periodic_shape_and_equal_capacity():
    settings = yaml.safe_load(open("configs/round2_v12/SPEC.yaml"))["training"]
    ll = dual_network(settings, 6, 2, [.01, .01])
    lh = dual_network(settings, 6, 2, [.01, .1])
    historical = make_network({"backbone": "tabm", "frequency": .01}, settings, 6, 2)
    assert sum(p.numel() for p in ll.parameters()) == sum(p.numel() for p in lh.parameters())
    assert sum(p.numel() for p in ll.parameters()) != sum(p.numel() for p in historical.parameters())
    assert ll(torch.zeros(3, 21), torch.zeros(3, 1, dtype=torch.long)).shape == (3, 16, 2)
