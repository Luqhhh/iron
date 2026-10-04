import numpy as np
import pytest
from bf_tap_r2.realmlp_development_audit import arrays,scalar_gain,scalar_wmape


def test_scalar_gain_uses_paired_rows_and_target_denominator():
    y=np.array([10.,20.,30.]);base=np.array([11.,24.,29.]);candidate=np.array([10.,20.,30.])
    assert scalar_gain(y,base,candidate)==5
    assert scalar_gain(y,base,candidate,np.array([0,2]))==2.5
    assert scalar_gain(y,candidate,base)==-5
    assert scalar_wmape(y,base,[0,1,2])==.1
    with pytest.raises(ValueError):scalar_gain(np.zeros(2),np.zeros(2),np.ones(2))


def test_archive_is_closed_before_scalar_iteration(tmp_path):
    p=tmp_path/'predictions.npz';np.savez_compressed(p,y=np.array([2.,3.]),base=np.array([3.,4.]),candidate=np.array([2.,3.]))
    a=arrays(p);p.unlink()
    assert isinstance(a,dict)
    assert scalar_gain(a['y'],a['base'],a['candidate'])==20
