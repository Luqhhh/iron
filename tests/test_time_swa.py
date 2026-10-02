"""Target-selection and lifecycle tests; no fit/optimizer used."""
import importlib
import numpy as np
import pytest

def test_time_selector_does_not_follow_iron_error():
    torch=pytest.importorskip("torch")
    m=importlib.import_module("bf_tap_r2.time_swa_model")
    target=torch.zeros((2,2))
    epoch_a=torch.tensor([[100.,.1],[100.,.1]])
    epoch_b=torch.tensor([[0.,.2],[0.,.2]])
    assert m.selection_mae(epoch_a,target)<m.selection_mae(epoch_b,target)
    assert epoch_a.abs().mean()>epoch_b.abs().mean()
    with pytest.raises(ValueError,match="finite"):
        m.selection_mae(torch.tensor([[0.,float("nan")]]),torch.zeros((1,2)))

def test_time_selector_gate_retains_four_seed_requirements():
    m=importlib.import_module("bf_tap_r2.time_swa_protocol")
    assert m.decide([.01,.02],[.005,.015],False)["selected_for_confirmation"]==["SWA_TIME_SELECT"]
    d=m.decide([.01,.02,-.001,.003],[0.,0.,0.,0.],True)
    assert not d["formal_promoted"]
    assert "nonpositive_seed_gain" in d["failure_reasons"]
    with pytest.raises(ValueError,match="Incomplete"):
        m.decide([.01],[.005],False)

def test_time_selector_admission_does_not_restart_existing_directory(tmp_path):
    m=importlib.import_module("bf_tap_r2.time_swa_run")
    run=tmp_path/"local/runs/swa-time-select-v1";run.mkdir(parents=True)
    with pytest.raises(ValueError,match="Existing"):
        m.admit(tmp_path)
    assert list(run.iterdir())==[]
def test_cold_audit_rejects_checkpoint_scored_with_wrong_target():
    m=importlib.import_module("bf_tap_r2.time_swa_audit")
    trace={"history":[{"epoch":1,"validation_mae":.3},{"epoch":2,"validation_mae":.1},{"epoch":3,"validation_mae":.2}],
           "selected_epoch":2,"stopped_epoch":3}
    settings={"patience":25,"min_delta":1e-5,"max_epochs":240}
    assert m.verify_selection(trace,settings,.1)==2
    with pytest.raises(ValueError,match="Checkpoint"):
        m.verify_selection(trace,settings,.2)

def test_cold_audit_entry_verifies_source_before_reading_inputs(tmp_path,monkeypatch):
    pytest.importorskip("torch")
    import json
    m=importlib.import_module("bf_tap_r2.time_swa_audit")
    directory=tmp_path/"local/runs/time-test/engineering-r1";directory.mkdir(parents=True)
    (directory/"manifest.json").write_text(json.dumps({"source_hashes":{}}))
    called=[]
    def frozen(root,files):
        called.append(root);raise RuntimeError("source check reached")
    monkeypatch.setattr(m,"verify_tree",frozen)
    with pytest.raises(RuntimeError,match="source check reached"):
        m.audit(directory)
    assert called==[tmp_path]