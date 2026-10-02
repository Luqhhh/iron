"""Identity and invalid-extrapolation regressions for cached SWA diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
    path=Path(__file__).resolve().parents[1]/"scripts/swa_delta/diagnose.py"
    spec=importlib.util.spec_from_file_location("swa_delta",path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def test_delta_zero_is_exact_parent_and_no_mutation():
    m=module();p=np.array([10.,20.]);s=np.array([12.,19.]);b=np.array([11.,21.])
    np.testing.assert_array_equal(m.endpoint(p,s,b,0),p)
    np.testing.assert_allclose(m.endpoint(p,s,b,.2),[10.2,19.6])
    np.testing.assert_array_equal(p,[10.,20.])

def test_delta_refuses_negative_extrapolation_and_nonfinite():
    m=module()
    with pytest.raises(ValueError,match="negative"):
        m.endpoint(np.array([1.]),np.array([0.]),np.array([10.]),.2)
    with pytest.raises(ValueError,match="finite"):
        m.endpoint(np.array([1.]),np.array([np.nan]),np.array([1.]),.1)

def test_cached_oof_rejects_duplicate_queries(tmp_path):
    m=module();ids=np.array(["a","b","c"])
    for fold,q in enumerate([[0,1],[1,2]]):
        d=tmp_path/f"s42-f{fold}";d.mkdir()
        np.savez(d/"partitions.npz",query=np.array(q))
        np.savez(d/"predictions.npz",ids=ids[q],BASE=np.ones(2),SWA_WINDOW10=np.ones(2))
    with pytest.raises(ValueError,match="coverage"):
        m.oof(tmp_path,42,ids,np.array([0,1,1]),fold_count=2)

def test_endpoint_gain_is_target_column_contribution():
    m=module()
    assert m.gain(np.array([10.,10.]),np.array([8.,8.]),np.array([9.,9.]))==5.
def test_verifier_import_does_not_write_into_immutable_bundle(tmp_path):
    m=module()
    script=tmp_path/"verifier.py";script.write_text("answer = 42\n")
    before={p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    loaded=m.import_readonly(script)
    assert loaded.answer==42
    after={p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    assert after==before