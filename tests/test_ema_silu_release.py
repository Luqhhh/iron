"""No-fit guards for the fixed full-data SiLU release and raw parent column."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bf_tap_r2 import ema_silu_release as release


@pytest.mark.parametrize('field,value',[
    ('candidate','SILU_A20'),('training_seeds',[42,1042]),('optimizer_runs',4),
    ('replacement_weight',.2),('new_confirmation_seeds',2),('packages',2),
    ('agent_uploads',1),('time_budget_seconds',3600),('automatic_retries',True),
    ('formal_promoted',True),('engineering_new_fits',1)])
def test_reject_scope_drift(field,value):
    spec=release.read(release.WORK/release.SPEC);release.validate_scope(spec)
    changed=deepcopy(spec);changed[field]=value
    with pytest.raises(ValueError):release.validate_scope(changed)


def test_only_parent_plus_full_mean_replacement_and_no_clipping():
    old=np.asarray([[8.,10.],[10.,12.],[12.,14.]])
    new=old+[2.,-3.]
    np.testing.assert_array_equal(release.prediction([100.,200.],old,new),[102.,197.])
    for bad in [new[:2],np.full((3,2),np.nan)]:
        with pytest.raises(ValueError):release.prediction([100.,200.],old,bad)
    with pytest.raises(ValueError,match='clipping'):
        release.prediction([1.,1.],old,new-100)


def test_raw_iron_fields_round_trip_and_strict_zip(tmp_path):
    ids=[f'R2S_TEST_{i:06d}' for i in range(322)]
    forms=['1.00000000000000000001e2','00100.00','100.000']
    rows=[dict(sample_id=s,pred_tap_iron=forms[i%3],pred_tap_time_len='50.0') for i,s in enumerate(ids)]
    values=np.linspace(60.,100.,322)
    release.package(tmp_path,release.payload(rows,values),ids)
    actual,_=release.package_rows(tmp_path/release.ZIP_NAME,ids)
    assert [r['pred_tap_iron'] for r in actual]==[r['pred_tap_iron'] for r in rows]
    np.testing.assert_array_equal([float(r['pred_tap_time_len']) for r in actual],values)
    with pytest.raises(ValueError):release.package_rows(tmp_path/release.ZIP_NAME,ids[::-1])
    with pytest.raises(FileExistsError):release.package(tmp_path,release.payload(rows,values),ids)
    with pytest.raises(ValueError):release.payload(rows,values[:-1])


@pytest.mark.parametrize('stage',['worker','cold'])
def test_delegates_to_unchanged_scientific_worker_and_restores(monkeypatch,tmp_path,stage):
    monkeypatch.setattr(release,'RUN',tmp_path)
    original_run=release.science.RUN;original_frames=release.science.unit_frames;calls=[]
    def observe(seed,fold,init):
        assert release.science.RUN==tmp_path
        assert release.science.unit_frames is release.full_unit
        calls.append((seed,fold,init))
    monkeypatch.setattr(release.science,stage,observe)
    release.native(stage,1042)
    assert calls==[(-1,-1,1042)]
    assert release.science.RUN==original_run and release.science.unit_frames is original_frames
    for bad_stage,bad_init in [('worker',0),('refit',42)]:
        with pytest.raises(ValueError):release.native(bad_stage,bad_init)
    with pytest.raises(ValueError):release.full_unit(42,0)


def test_task_budget_and_failure_stops_before_further_work(monkeypatch,tmp_path):
    roster=release.tasks()
    assert len(roster)==9
    assert sum(s=='worker' for s,i in roster)==3 and sum(s=='cold' for s,i in roster)==3
    assert roster[0]==('reuse',-1) and roster[-2:]==[('build',-1),('verify',-1)]
    monkeypatch.setattr(release,'RUN',tmp_path)
    release.write(tmp_path/'manifest.json',{'files':{}})
    monkeypatch.setattr(release,'context',lambda:{'files':{}})
    launches=[]
    def launch(*args,**kwargs):
        launches.append(args[0]);return SimpleNamespace(pid=1001,returncode=None)
    monkeypatch.setattr(release.subprocess,'Popen',launch)
    monkeypatch.setattr(release.os,'wait4',lambda *_:(1001,2<<8,SimpleNamespace(ru_maxrss=500*1024)))
    with pytest.raises(RuntimeError,match='child failed'):release.controller()
    assert len(launches)==1
    assert release.read(tmp_path/'execution/terminal.json')['status']=='failed'
    with pytest.raises(ValueError):release.require_events(1)
    with pytest.raises(ValueError):release.prepare('not-read-because-existing-run')


def test_delivery_requires_actual_terminal_before_touching_desktop(monkeypatch,tmp_path):
    monkeypatch.setattr(release,'RUN',tmp_path)
    desktop=tmp_path/'desktop';source=tmp_path/'source.zip';source.write_bytes(b'preserved')
    monkeypatch.setattr(release,'context',lambda:{'spec':{'desktop_directory':str(desktop)}})
    (tmp_path/'execution').mkdir()
    release.write(tmp_path/'execution/final-reconciliation.json',{'status':'failed'})
    release.write(tmp_path/'release.json',{'zip':str(source)})
    with pytest.raises(ValueError):release.deliver()
    assert not desktop.exists()
