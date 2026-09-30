"""Resource and freeze admission use artificial fixtures, not the full-size probe."""
import numpy as np
import pytest
from bf_tap_r2.t2g_preflight import synthetic_data,resource_admission,reserve_probe,learning_check
from bf_tap_r2.t2g_freeze import verify_engineering_fields

def test_generation_order():
    f=synthetic_data()
    assert len(f)==2755
    assert f.air_volume.iloc[0]==pytest.approx(.8224981059170352)
    assert f.tap_time_len.iloc[0]==pytest.approx(11.51570917762179)
    assert f.tap_time_len.iloc[2204]==pytest.approx(8.919379329550656)
    assert f.spout_no.iloc[:4].tolist()==[1,2,1,2]
    assert f.sample_id.str.startswith("SYNTH_").all()

def test_resource_boundaries():
    # Each arm460s: 20*(460+460)/4*1.5+300 =7200.
    r=resource_admission(460.,460.,1536.,1536.,7168.)
    assert r["status"]=="passed" and r["projected_seconds"]==7200.
    assert resource_admission(460.001,460.,1536.,1536.,7168.)["status"]=="failed"
    assert resource_admission(460.,460.,1536.01,1536.,8000.)["status"]=="failed"
    assert resource_admission(460.,460.,1536.,1536.,7167.99)["status"]=="failed"

def test_probe_once(tmp_path):
    reserve_probe(tmp_path,"a"*64)
    with pytest.raises(FileExistsError): reserve_probe(tmp_path,"a"*64)

def test_changed_source_rejected():
    m={"kind":"t2g-engineering-v1","identity":"T2G_GRAPH_V1","sources":{"x.py":"a"*64},"runtime":{"python":"3.12-test"}}
    verify_engineering_fields(m,{"x.py":"a"*64},{"python":"3.12-test"})
    with pytest.raises(ValueError): verify_engineering_fields(m,{"x.py":"b"*64},{"python":"3.12-test"})
    with pytest.raises(ValueError): verify_engineering_fields(m,{"x.py":"a"*64},{"python":"3.12-different"})

def test_both_arms_learn():
    assert learning_check(np.array([1.,3.]),np.array([4.,5.]),np.array([4.,5.]))["status"]=="passed"
    assert learning_check(np.array([1.,3.]),np.array([4.,5.]),np.array([0.,0.]))["status"]=="failed"
    with pytest.raises(ValueError): learning_check(np.array([1.,3.]),np.array([4.,5.]),np.array([np.nan,5.]))

def test_formal_manifest_cannot_move_workspace(tmp_path,monkeypatch):
    import json
    from bf_tap_r2 import t2g_freeze as freeze
    from bf_tap_r2.t2g_model import file_hash
    monkeypatch.setattr(freeze,"source_snapshot",lambda:{})
    monkeypatch.setattr(freeze,"runtime",lambda:{})
    engineering={"kind":"t2g-engineering-v1","identity":"T2G_GRAPH_V1","sources":{},"runtime":{},
        "workspace":"/foreign","output":str(tmp_path)}
    (tmp_path/"engineering.json").write_text(json.dumps(engineering))
    (tmp_path/"g0.json").write_text('{"status":"passed"}')
    formal={**engineering,"kind":"t2g-formal-v1","engineering_sha256":file_hash(tmp_path/"engineering.json"),
        "g0_status":"passed","g0_path":str(tmp_path/"g0.json"),"g0_sha256":file_hash(tmp_path/"g0.json")}
    (tmp_path/"formal.json").write_text(json.dumps(formal))
    (tmp_path/"formal.complete.json").write_text(json.dumps({"sha256":file_hash(tmp_path/"formal.json")}))
    with pytest.raises(ValueError): freeze.verify_manifest(tmp_path/"formal.json")
