"""Controller lifecycle tests replace fits/audits, retaining actual durable state."""
import json
from pathlib import Path
import pytest
from scripts import run_t2g_frozen as control
from scripts import monitor_t2g_once as monitor

@pytest.fixture
def environment(tmp_path,monkeypatch):
    engineering=tmp_path/"engineering"; engineering.mkdir()
    manifest=engineering/"engineering.json"; manifest.write_text("{}")
    root=engineering/"controller-r1"
    monkeypatch.setattr(control,"verify_manifest",lambda p:{"kind":"t2g-engineering-v1","output":str(engineering)})
    monkeypatch.setattr(control,"run_preflight",lambda p,o:{"status":"passed"})
    return manifest,tmp_path/"native",tmp_path/"overlay",root

def test_absent_reference_blocks_formal(environment,monkeypatch):
    monkeypatch.setattr(control,"verify_reference",lambda *a:(_ for _ in ()).throw(FileNotFoundError("not shared")))
    def forbidden(*a): raise AssertionError("Formal fitting started")
    monkeypatch.setattr(control,"run_phase",forbidden)
    r=control.run_controller(*environment)
    assert r["status"]=="waiting_reference"
    assert monitor.read_progress(environment[-1])["status"]=="waiting_reference"

def test_g0_failure_terminal(environment,monkeypatch):
    monkeypatch.setattr(control,"run_preflight",lambda *a:{"status":"failed"})
    r=control.run_controller(*environment)
    assert r["status"]=="failed_g0"
    with pytest.raises(ValueError): control.run_controller(*environment)

def test_controller_lock(environment):
    import fcntl
    root=environment[-1]; root.mkdir()
    with (root/"controller.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert control.run_controller(*environment)["status"]=="already_running"

def test_selected_targets_only(environment,monkeypatch):
    calls=[]
    monkeypatch.setattr(control,"verify_reference",lambda *a:object())
    monkeypatch.setattr(control,"attach_reference",lambda m,r,o:o)
    monkeypatch.setattr(control,"run_phase",lambda m,p,o,t:(calls.append((p,list(t))),o/"complete.json")[1])
    monkeypatch.setattr(control,"audit_fresh",lambda *a:None)
    monkeypatch.setattr(control,"decide",lambda d,c,r:{"selected_for_confirmation":["tap_time_len"],"confirmed":[]})
    result=control.run_controller(*environment)
    assert result["status"]=="completed"
    assert calls==[("development",["tap_iron","tap_time_len"]),("confirmation",["tap_time_len"])]

def test_empty_selection_no_confirmation(environment,monkeypatch):
    calls=[]
    monkeypatch.setattr(control,"verify_reference",lambda *a:object())
    monkeypatch.setattr(control,"attach_reference",lambda m,r,o:o)
    monkeypatch.setattr(control,"run_phase",lambda m,p,o,t:(calls.append(p),o/"complete.json")[1])
    monkeypatch.setattr(control,"audit_fresh",lambda *a:None)
    monkeypatch.setattr(control,"decide",lambda *a:{"selected_for_confirmation":[],"confirmed":[]})
    control.run_controller(*environment)
    assert calls==["development"]

def test_healthy_monitor_quiet(tmp_path,capsys):
    tmp_path.joinpath("events.jsonl").write_text(json.dumps({"status":"running_development"})+"\n")
    monitor.main(["--root",str(tmp_path)])
    assert capsys.readouterr().out==""

def test_terminal_notice_once(tmp_path):
    state={"status":"failed","message":"one failure"}
    assert monitor.claim_notification(tmp_path,state)
    assert not monitor.claim_notification(tmp_path,state)

def test_waiting_reference_resumes_without_second_probe(environment,monkeypatch):
    monkeypatch.setattr(control,"verify_reference",lambda *a:(_ for _ in ()).throw(FileNotFoundError("pending")))
    control.run_controller(*environment)
    base=environment[0].parent
    g0=base/"g0-r1";g0.mkdir()
    control.write_new(g0/"preflight.json",{"status":"passed"})
    control.write_new(g0/"preflight.complete.json",{"sha256":control.file_hash(g0/"preflight.json")})
    monkeypatch.setattr(control,"run_preflight",lambda *a:(_ for _ in ()).throw(AssertionError("probe repeated")))
    assert control.run_controller(*environment)["status"]=="waiting_reference"

def test_claimed_training_cannot_restart(environment,monkeypatch):
    root=environment[-1];root.mkdir()
    control.event(root,"running_development")
    monkeypatch.setattr(control,"run_preflight",lambda *a:(_ for _ in ()).throw(AssertionError("probe repeated")))
    with pytest.raises(ValueError):control.run_controller(*environment)

def test_actual_absent_reference_waits(environment):
    assert control.run_controller(*environment)["status"]=="waiting_reference"
