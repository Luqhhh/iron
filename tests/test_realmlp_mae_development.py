import numpy as np
import pytest
from bf_tap_r2.realmlp_mae_development import eligible,blend


def test_new_confirmation_requires_two_complete_positive_development_splits():
    assert eligible({'42':.001,'3407':.002})
    assert not eligible({'42':.001,'3407':-.002})
    assert not eligible({'42':.001,'3407':0.})
    with pytest.raises(ValueError):eligible({'42':.001})
    with pytest.raises(ValueError):eligible({'42':.001,'3407':float('nan')})


def test_fixed_mae_blend_never_selects_weight_or_adds_clipping():
    np.testing.assert_array_equal(blend([100,120],[105,115]),[101,119])
    with pytest.raises(ValueError):blend([100],[float('inf')])
    with pytest.raises(ValueError):blend([1],[-100])
    with pytest.raises(ValueError):blend([1,2],[1])
