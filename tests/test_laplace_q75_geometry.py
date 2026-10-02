import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("laplace_geometry", Path(__file__).parents[1]/"scripts/inspect_laplace_q75_geometry.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_right_derivative_uses_absolute_direction_at_kink():
    r = MODULE.right_loss_derivative([100., 200.], [100., 210.], [110., 200.])
    assert r["right_WMAPE_derivative"] == 0
    assert r["exact_zero_residual_rows"] == 1


def test_uniformly_wrong_direction_has_convex_bound():
    r = MODULE.right_loss_derivative([100., 200.], [110., 220.], [120., 240.])
    assert r["right_WMAPE_derivative"] == pytest.approx(.1)
    assert r["every_positive_weight_nonimproving_on_this_fixed_OOF"]


def test_good_initial_direction_not_excluded():
    r = MODULE.right_loss_derivative([100., 200.], [110., 220.], [100., 200.])
    assert r["right_WMAPE_derivative"] == pytest.approx(-.1)
    assert not r["every_positive_weight_nonimproving_on_this_fixed_OOF"]


def test_bound_holds_with_piecewise_absolute_errors():
    y, parent, candidate = np.array([100., 200., 300.]), np.array([110., 200., 290.]), np.array([105., 250., 280.])
    derivative = MODULE.right_loss_derivative(y, parent, candidate)["right_WMAPE_derivative"]
    base = np.abs(y-parent).sum()/y.sum()
    for alpha in (.001, .1, .2, .9, 1., 2.):
        actual_loss = np.abs(y-((1-alpha)*parent+alpha*candidate)).sum()/y.sum()
        assert actual_loss >= base+alpha*derivative-1e-14


@pytest.mark.parametrize("y,p,c", [([], [], []), ([0.], [1.], [2.]), ([1.], [np.nan], [2.]), ([1.], [2., 3.], [4.])])
def test_invalid_arrays(y, p, c):
    with pytest.raises(ValueError):
        MODULE.right_loss_derivative(y, p, c)
