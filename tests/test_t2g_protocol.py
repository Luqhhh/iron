import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from bf_tap_r2.t2g_protocol import UnitKey,reserve,inspect_ledger,expected_units

def test_concurrent_duplicate(tmp_path):
    key=UnitKey("development","tap_iron","LEARNED_GRAPH",42,0)
    def claim(_):
        try: reserve(tmp_path,key,"selector","a"*64); return True
        except ValueError: return False
    with ThreadPoolExecutor(2) as p: assert sorted(p.map(claim,range(2)))==[False,True]
    assert inspect_ledger(tmp_path)["optimizer_starts"]==1

def test_crash_consumes(tmp_path):
    key=UnitKey("development","tap_iron","LEARNED_GRAPH",42,0)
    reserve(tmp_path,key,"selector","a"*64)
    with pytest.raises(ValueError): reserve(tmp_path,key,"selector","a"*64)
    reserve(tmp_path,key,"refit","a"*64)
    assert inspect_ledger(tmp_path)["outer_fits"]==1
    assert inspect_ledger(tmp_path)["optimizer_starts"]==2
    with pytest.raises(ValueError): reserve(tmp_path,key,"refit","a"*64)

def test_phase_budget(tmp_path):
    units=expected_units("development",["tap_iron","tap_time_len"])
    assert len(units)==40
    for key in units:
        reserve(tmp_path,key,"selector","a"*64); reserve(tmp_path,key,"refit","a"*64)
    assert inspect_ledger(tmp_path)["outer_fits"]==40
    assert inspect_ledger(tmp_path)["optimizer_starts"]==80
    assert len(expected_units("confirmation",["tap_time_len"]))==20
    with pytest.raises(ValueError): reserve(tmp_path,UnitKey("confirmation","tap_time_len","LEARNED_GRAPH",7777,0),"selector","a"*64)

def test_refit_without_selector_or_manifest_swap(tmp_path):
    key=UnitKey("development","tap_iron","LEARNED_GRAPH",42,0)
    with pytest.raises(ValueError): reserve(tmp_path,key,"refit","a"*64)
    reserve(tmp_path,key,"selector","a"*64)
    with pytest.raises(ValueError): reserve(tmp_path,key,"refit","b"*64)

def test_confirmation_target_policy_enforced(tmp_path):
    from bf_tap_r2.t2g_protocol import initialize_ledger
    initialize_ledger(tmp_path,"confirmation",["tap_time_len"],"a"*64)
    with pytest.raises(ValueError):
        reserve(tmp_path,UnitKey("confirmation","tap_iron","LEARNED_GRAPH",7777,0),"selector","a"*64)
    for key in expected_units("confirmation",["tap_time_len"]):
        reserve(tmp_path,key,"selector","a"*64)
        reserve(tmp_path,key,"refit","a"*64)
    assert inspect_ledger(tmp_path)["optimizer_starts"]==40
