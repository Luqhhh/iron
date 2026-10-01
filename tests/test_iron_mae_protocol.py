import importlib
import pytest
def test_fresh_pair_budget_and_gates(tmp_path):
 m=importlib.import_module("bf_tap_r2.iron_mae_protocol")
 assert m.BUDGETS["development"]=={"state":40,"optimizer":40}
 assert m.BUDGETS["engineering"]=={"state":4,"optimizer":4}
 assert m.decision([.02,.03],[.01,.01],confirmation=False)["selected_for_confirmation"]==["COMPACT_MAE"]
 assert not m.decision([.02,-.01],[0,0],confirmation=False)["selected_for_confirmation"]
 l=m.Ledger(tmp_path);l.reserve("development","state","u")
 with pytest.raises(ValueError):l.reserve("development","state","u")
