import copy
import json
import subprocess
import pytest
from bf_tap_r2 import rfm_controller as controller
from bf_tap_r2.rfm_arithmetic import direct_decision
from bf_tap_r2.rfm_scoring import decide_development,decide_confirmation
from bf_tap_r2.rfm_protocol import write_new
from test_rfm_scoring import records


@pytest.mark.parametrize('eligible',[[],['tap_time_len']])
def test_serial_controller_earned_confirmation_only(eligible):
    calls=[]
    def invoke(name,module,args):
        calls.append(name)
        if name.endswith('-run'):return {'complete_sha256':'c'*64}
        if name.endswith('-audit'):return {'audit_sha256':'a'*64}
        return {'arithmetic_sha256':'b'*64,'selected_targets':eligible}
    result=controller.sequence(invoke)
    assert len(calls)==(6 if eligible else 3)
    assert calls[:3]==['development-run','development-audit','development-arithmetic']
    assert ('confirmation_skipped' in result)==(not eligible)


@pytest.mark.parametrize('fail_at',range(6))
def test_each_subprocess_failure_stops_all_successors(fail_at):
    calls=[]
    def invoke(name,module,args):
        calls.append(name)
        if len(calls)-1==fail_at:raise subprocess.CalledProcessError(1,['synthetic'])
        return dict(complete_sha256='c'*64,audit_sha256='a'*64,arithmetic_sha256='b'*64,selected_targets=['tap_iron'])
    with pytest.raises(subprocess.CalledProcessError):controller.sequence(invoke)
    assert len(calls)==fail_at+1


def test_controller_reentry_and_monitor_no_writes(tmp_path,monkeypatch):
    root=tmp_path/'local/control';root.mkdir(parents=True)
    monkeypatch.setattr(controller,'verify_manifest',lambda *args:{'workspace':str(tmp_path)})
    monkeypatch.setattr(controller,'verify_admission',lambda *args:None)
    monkeypatch.setattr(controller,'sequence',lambda invoke:{'confirmation_skipped':'no_development_finalist'})
    assert controller.monitor(root)['status']=='not_started'
    result=controller.controller(root/'manifest.json','m','p')
    assert result['status']=='completed'
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert controller.monitor(root)==result
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):controller.controller(root/'manifest.json','m','p')


def test_independent_four_seed_arithmetic_matches_frozen_boundaries():
    for kwargs in ({},{'gain':.009999},{'score':96.24999},{'control':.01}):
        dev=records(**kwargs)
        selected,_=direct_decision(dev,'development')
        assert selected==decide_development(dev)['eligible_targets']
    dev=records(gain=.02);conf=records(seeds=(7777,12011),gain=.025)
    selected,details=direct_decision(dev+conf,'confirmation',['tap_iron','tap_time_len'])
    expected=decide_confirmation(dev,conf)
    assert selected==expected['promoted_targets']
    assert details['tap_iron']['seed_lcb95']==pytest.approx(expected['decisions']['tap_iron']['seed_lcb95'])
    conf[-1]['gain']=-.01
    assert direct_decision(dev+conf,'confirmation',['tap_iron','tap_time_len'])[0]==['tap_iron']
