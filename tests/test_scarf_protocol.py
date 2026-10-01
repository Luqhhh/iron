import importlib
import numpy as np
import pytest

def mod(): return importlib.import_module("bf_tap_r2.scarf_protocol")

def test_ledger_failed_reservation_cannot_retry_or_exceed_budget(tmp_path):
    m=mod(); ledger=m.Ledger(tmp_path,{"development":{"state":1,"optimizer":1}})
    ledger.reserve("development","state","u")
    with pytest.raises(ValueError): ledger.reserve("development","state","u")
    with pytest.raises(ValueError): ledger.reserve("development","state","v")
    ledger.reserve("development","optimizer","u:supervised")
    ledger.complete("development","optimizer","u:supervised")
    assert ledger.counts("development")["reserved"]=={"state":1,"optimizer":1}
    assert ledger.counts("development")["completed"]=={"state":0,"optimizer":1}

def test_no_absolute_score_veto_and_negative_exploration_retained():
    m=mod()
    d=m.decision([.0001,.0002],[.00005,.0001],confirmation=False)
    assert d["selected_for_confirmation"]==["SCARF_REG"]
    d=m.decision([-.001,.0003],[-.002,.0001],confirmation=False)
    assert not d["selected_for_confirmation"]
    assert d["classification"]=="exploration_only_not_release_authorized"
    assert not m.decision([.01,.011,.012,.013],[0,0,0,0],confirmation=True)["release_authorized"]
    with pytest.raises(ValueError): m.decision([.01],[0],confirmation=False)

def test_seed_lcb_and_row_paired_bootstrap():
    m=mod(); d=m.decision([.01,.011,.012,.013],[0,0,0,0],confirmation=True)
    assert d["formal_promoted"] and d["seed_lcb95"]>0
    y=np.ones(20)*10; baseline=np.ones(20)*11; member=np.ones(20)*10
    b=m.bootstrap(y,[baseline,baseline],[member,member],np.arange(20)%2,reps=30)
    assert b["sampling_only_not_platform_forecast"]
    assert b["uniform"]["median"]==pytest.approx(5.)

def test_private_path_and_hash_tamper(tmp_path):
    m=mod(); f=tmp_path/"ok"; f.write_text("a")
    assert m.child(tmp_path,"ok")==f
    with pytest.raises(ValueError): m.child(tmp_path,"../out")
    with pytest.raises(ValueError): m.verify_tree(tmp_path,{"ok":"bad"})


def test_frozen_run_identity_blocks_tampering_and_second_launch(tmp_path):
    m=importlib.import_module("bf_tap_r2.scarf_run")
    (tmp_path/"src").mkdir(); (tmp_path/"src/a.py").write_text("a")
    from bf_tap_r2.scarf_protocol import sha
    frozen={"source_hashes":{"src/a.py":sha(tmp_path/"src/a.py")},"runtime":m.runtime(),"reference":{"frozen_files":{}}}
    m.verify_frozen(tmp_path,frozen)
    (tmp_path/"src/a.py").write_text("b")
    with pytest.raises(ValueError): m.verify_frozen(tmp_path,frozen)

def test_independent_decision_has_no_control_promotion():
    m=importlib.import_module("bf_tap_r2.scarf_run")
    d=m.independent_decision([-.001,.0003],[-.002,.0001],False)
    assert not d["selected_for_confirmation"] and not d["formal_promoted"]
    assert d["classification"]=="exploration_only_not_release_authorized"
