import copy
from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2 import rfm_freeze as freeze
from bf_tap_r2.rfm_protocol import file_hash, write_new
from test_rfm_model import sample


def reference():
    frame=sample(30)
    folds={s:np.arange(30)%5 for s in (42,3407,7777,12011)}
    columns={s:{t:frame[t].to_numpy() for t in ('tap_iron','tap_time_len')} for s in folds}
    return frame,folds,columns,copy.deepcopy(columns),dict(status='passed',new_reference_fits=0,hashes={})


def test_private_path_rejects_external_and_symlink_destinations(tmp_path):
    (tmp_path/'local').mkdir()
    external=tmp_path/'elsewhere';external.mkdir()
    (tmp_path/'local'/'link').symlink_to(external,target_is_directory=True)
    for path in (external/'run',tmp_path/'local',tmp_path/'local'/'link'/'run'):
        with pytest.raises(ValueError,match='private'):freeze.private_path(tmp_path,path)
    assert freeze.private_path(tmp_path,tmp_path/'local'/'run')==tmp_path/'local'/'run'


def test_row_bounds_and_reference_arithmetic_identities():
    data=reference(); before=freeze.reference_identity(*data)
    data[2][42]['tap_iron'][0]+=1
    assert freeze.reference_identity(*data)!=before
    frame=sample(2756);folds={s:np.arange(2756)%5 for s in (42,3407,7777,12011)}
    with pytest.raises(ValueError,match='row bounds'):
        freeze.reference_identity(frame,folds,{}, {},data[-1])
    with pytest.raises(ValueError,match='four'):
        freeze.reference_identity(data[0],{42:data[1][42]},*data[2:])


def test_manifest_requires_complete_tested_committed_sources_and_external_anchor(tmp_path,monkeypatch):
    workspace=tmp_path/'workspace';workspace.mkdir()
    for name in (*freeze.REQUIRED_FILES,'uv.lock','pyproject.toml','src/bf_tap_r2/rfm_freeze.py'):
        path=workspace/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic\n')
    monkeypatch.setattr(freeze,'__file__',str(workspace/'src/bf_tap_r2/rfm_freeze.py'))
    monkeypatch.setattr(freeze,'runtime_snapshot',lambda:{'synthetic_runtime':1})
    monkeypatch.setattr(freeze,'load_reference_cache',lambda root:reference())
    monkeypatch.setattr(freeze.subprocess,'check_output',lambda args,**kw:'test-commit\n' if 'rev-parse' in args else '')
    receipt=tmp_path/'tests.json'
    write_new(receipt,dict(status='passed',exit_code=0,full_suite=True,
        source_hashes=freeze.source_snapshot(workspace),runtime=freeze.runtime_snapshot()))
    kwargs=dict(tests_path=receipt,tests_sha256=file_hash(receipt))
    out=workspace/'local'/'control'/'manifest.json'
    anchor=freeze.prepare_manifest(workspace,tmp_path,out,**kwargs)
    assert freeze.verify_manifest(out,anchor['sha256'])['release_authorized'] is False
    with pytest.raises(FileExistsError):freeze.prepare_manifest(workspace,tmp_path,out,**kwargs)
    with pytest.raises(ValueError,match='external'):freeze.verify_manifest(out,'0'*64)
    changed=workspace/'src/bf_tap_r2/rfm_freeze.py';changed.write_text('# changed\n')
    with pytest.raises(ValueError,match='source'):freeze.verify_manifest(out,anchor['sha256'])
    with pytest.raises(ValueError,match='exact sources'):
        freeze.prepare_manifest(workspace,tmp_path,out.with_name('other.json'),**kwargs)


def test_thread_guard_requires_every_variable(monkeypatch):
    for key in freeze.THREAD_VARS:monkeypatch.setenv(key,'1')
    freeze.require_threads()
    monkeypatch.setenv('MKL_NUM_THREADS','2')
    with pytest.raises(ValueError,match='thread'):freeze.require_threads()
