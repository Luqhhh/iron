import numpy as np
import pytest
from bf_tap_r2.realmlp_time_confirmation import columns,gate,ALL_SEEDS


def test_confirmation_uses_actual_three_member_q100_reference():
    current,candidate=columns([100,120],[90,110],[[91,111],[92,112],[93,113]],[97,116])
    np.testing.assert_array_equal(current,[102,122])
    np.testing.assert_allclose(candidate,[101,120.8],rtol=0,atol=1e-12)
    with pytest.raises(ValueError):columns([100],[90],[[91],[92]],[97])
    with pytest.raises(ValueError):columns([1],[100],[[1],[1],[1]],[1])


def test_four_seed_gate_cannot_use_mean_only_or_incomplete_results():
    assert gate({str(s):.004 for s in ALL_SEEDS})['passed']
    assert not gate(dict(zip(map(str,ALL_SEEDS),[.01,.01,.01,-.0001])))['passed']
    assert not gate(dict(zip(map(str,ALL_SEEDS),[.0001,.0001,.0001,.05])))['passed']
    with pytest.raises(ValueError):gate({'42':.004,'3407':.008})
    with pytest.raises(ValueError):gate({str(s):float('nan') for s in ALL_SEEDS})
