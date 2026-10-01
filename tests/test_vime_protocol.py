import importlib
import json
from pathlib import Path
import pytest
from bf_tap_r2.vime_protocol import BUDGETS,decision,Ledger

def test_new_candidate_only_budget_and_nonpositive_mechanism_gate(tmp_path):
 assert BUDGETS["development"]=={"state":20,"optimizer":40}
 assert BUDGETS["confirmation"]=={"state":40,"optimizer":60}
 assert not decision([.01,.02],[.02,.03],confirmation=False)["selected_for_confirmation"]
 assert decision([.02,.03],[.01,.02],confirmation=False)["selected_for_confirmation"]==["VIME_REG"]

def test_control_reuse_refuses_missing_and_tampered_cold_identity(tmp_path):
 m=importlib.import_module("bf_tap_r2.vime_run")
 d=tmp_path/"local/runs/scarf-reg-v1/development-r1";d.mkdir(parents=True)
 (d/"supplemental-audit-r1.json").write_text(json.dumps({"status":"failed","units_sha256":{}}))
 with pytest.raises(ValueError):m.control_cache(tmp_path,"development","s42-f0")
 with pytest.raises(ValueError):m.control_cache(tmp_path,"confirmation","s7777-f0")

def test_new_audit_and_runner_import_without_training():
 m=importlib.import_module("bf_tap_r2.vime_run")
 assert m.independent_decision([-.02,-.03],[0,0],False)["release_authorized"] is False
