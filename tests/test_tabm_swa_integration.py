import importlib
from pathlib import Path
import pytest

def test_missing_current_time_reference_fails_before_directory_or_label_access(tmp_path):
 m=importlib.import_module("bf_tap_r2.tabm_swa_run")
 target=tmp_path/"local/runs/not-created"
 with pytest.raises(FileNotFoundError,match="Q75 time reference"):m.preflight(tmp_path,target)
 assert not target.exists()

def test_new_fit_budgets_exclude_reused_controls():
 m=importlib.import_module("bf_tap_r2.tabm_swa_protocol")
 assert m.BUDGETS=={"engineering":{"state":2,"optimizer":2},"development":{"state":20,"optimizer":20},"confirmation":{"state":40,"optimizer":40}}
 assert m.decision([.01,.02],[.009,.019],confirmation=False)["selected_for_confirmation"]==["SWA_WINDOW10"]

def test_cold_witness_recalculation_rejects_wrong_epoch_or_average():
 torch=pytest.importorskip("torch");m=importlib.import_module("bf_tap_r2.tabm_swa_audit")
 states=[{"w":torch.tensor([float(i),i+.1])} for i in range(3,13)]
 witness={"window":10,"selected_epoch":12,"epochs":list(range(3,13)),"states":states}
 payload={"trace":{"selected_epoch":12},"state":{"w":torch.stack([s["w"] for s in states]).mean(0)}}
 assert m.verify_window(payload,witness)<5e-7
 witness["epochs"][0]=2
 with pytest.raises(ValueError):m.verify_window(payload,witness)
 witness["epochs"][0]=3;payload["state"]["w"]+=.01
 with pytest.raises(AssertionError):m.verify_window(payload,witness)
