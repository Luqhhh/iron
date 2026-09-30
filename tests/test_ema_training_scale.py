"""Scientific partition and execution scope regressions, without model fitting."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.ema_training_scale import (
    assert_isolated, event, nested_subsets, require_memory, require_serial, verify_files,
)


def frame():
    rows=24
    data={name:np.arange(rows,dtype=float) for name in FEATURES}
    data.update(sample_id=[f's{i}' for i in range(rows)],spout_no=np.repeat([1,2],12),
                tap_time_len=np.arange(rows,dtype=float)+100,tap_iron=np.arange(rows,dtype=float)+500)
    result=pd.DataFrame(data)
    result.loc[1,list(FEATURES)]=result.loc[0,list(FEATURES)].to_numpy()
    result.loc[1,'spout_no']=2  # Numerical duplicate groups stay together across spouts.
    return result


def test_nested_scale_subsets_keep_groups_and_do_not_use_targets_or_row_order():
    f=frame();sets=nested_subsets(f)
    previous=set()
    for fraction,ids in sets.items():
        assert previous<=set(ids)
        assert ('s0' in ids)==('s1' in ids)
        assert set(f.set_index('sample_id').loc[ids,'spout_no'])=={1,2}
        assert ids==f.loc[f.sample_id.isin(ids),'sample_id'].tolist()
        previous=set(ids)
    assert sets['1.0']==f.sample_id.tolist()
    f['tap_time_len']=-1e9;f['tap_iron']=np.nan
    assert nested_subsets(f)==sets
    reordered=nested_subsets(f.iloc[::-1])
    assert all(set(reordered[k])==set(v) for k,v in sets.items())


def test_fixed_eval_ids_duplicates_and_labels_are_excluded():
    f=frame();train=f.iloc[:12];query=f.iloc[12:].drop(columns=['tap_time_len','tap_iron'])
    assert_isolated(train,query)
    with pytest.raises(ValueError,match='contains targets'):
        assert_isolated(train,f.iloc[12:])
    with pytest.raises(ValueError,match='ID overlap'):
        assert_isolated(train,train.drop(columns=['tap_time_len','tap_iron']))
    query=query.copy();query.loc[query.index[0],list(FEATURES)]=train.iloc[0][list(FEATURES)].to_numpy()
    with pytest.raises(ValueError,match='duplicate-group'):
        assert_isolated(train,query)


def test_event_ledger_is_append_only_and_retains_failures(tmp_path):
    path=tmp_path/'events.jsonl';event(path,{'event':'optimizer_started'});original=path.read_bytes()
    event(path,{'event':'optimizer_failed','reason':'synthetic failure'})
    assert path.read_bytes().startswith(original)
    assert [json.loads(line)['event'] for line in path.read_text().splitlines()]==['optimizer_started','optimizer_failed']


def test_file_tamper_and_path_escape_fail_closed(tmp_path):
    path=tmp_path/'source.py';path.write_text('original')
    expected=hashlib.sha256(path.read_bytes()).hexdigest();verify_files(tmp_path,{'source.py':expected})
    path.write_text('different')
    with pytest.raises(ValueError,match='identity'):
        verify_files(tmp_path,{'source.py':expected})
    with pytest.raises(ValueError,match='identity'):
        verify_files(tmp_path,{'../outside':expected})


def test_serial_dependency_requires_bound_success_and_actual_process_exit(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    terminal=tmp_path/'completion-event.json';terminal.write_text(json.dumps({'status':'completed'}))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    audit=tmp_path/'terminal-verification.json';audit.write_text(json.dumps({'status':'passed',
        'terminal_sha256':digest(terminal),'manifest_sha256':digest(manifest)}))
    spec={'previous_release_root':str(tmp_path),'previous_service':'previous.service'}
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'MainPID=0\nExecMainStatus=0\nActiveState=inactive\nSubState=dead\n')
    assert require_serial(spec)['service']=='previous.service'
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'MainPID=42\nExecMainStatus=0\nActiveState=active\n')
    with pytest.raises(ValueError,match='still active'):
        require_serial(spec)
    terminal.write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='Successful'):
        require_serial(spec)


def test_memory_admission_has_no_time_based_gate(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(available=4096*2**20))
    assert require_memory({'max_worker_rss_mib':1536})==4096
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(available=2048*2**20))
    with pytest.raises(ValueError,match='available memory'):
        require_memory({'max_worker_rss_mib':1536})


def test_unregistered_scale_and_duplicate_ids_are_refused():
    with pytest.raises(ValueError,match='Frozen fractions'):
        nested_subsets(frame(),fractions=(.2,1.))
    f=frame();f.loc[1,'sample_id']=f.loc[0,'sample_id']
    with pytest.raises(ValueError,match='Unique'):
        nested_subsets(f)
