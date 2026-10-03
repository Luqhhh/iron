from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[1]/'scripts/audit_ema_nested_residual_terminal.py'
spec=importlib.util.spec_from_file_location('nested_terminal_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def terminal():
    return dict(status='passed',dependencies_unchanged=True,events=[dict(task=n,exit_code=0,
        peak_rss_mib=600.,completed_ns=i+1) for i,n in enumerate(audit.expected_tasks())])


def test_actual_terminal_requires_every_declared_child_once():
    good=terminal();assert len(audit.verify_events(good))==121
    for bad in [dict(good,status='running'),dict(good,events=good['events'][:-1]),
                dict(good,events=good['events']+[good['events'][0]])]:
        with pytest.raises(ValueError):audit.verify_events(bad)
    bad=deepcopy(good);bad['events'][30]['exit_code']=1
    with pytest.raises(ValueError,match='exit'):audit.verify_events(bad)
    bad=deepcopy(good);bad['events'][30]['peak_rss_mib']=1537
    with pytest.raises(ValueError,match='memory'):audit.verify_events(bad)


def test_native_step_receipts_reject_role_identity_and_count_tamper():
    ident=dict(source_directory='/private/example',split_seed=42,trial_id='one')
    complete=dict(identity=ident,fit_ids=['a','b'],constructors=['selection','refit'],steps={'selection':6,'refit':4})
    fit=dict(identity=ident,fit_ids=['a','b'],time_ns=1)
    starts={role:dict(identity=ident,role=role,time_ns=i+2) for i,role in enumerate(['selection','refit'])}
    traces={'selection':dict(updates=6,history=[{'updates':2}]*3),'refit':dict(updates=4,history=[{'updates':2}]*2)}
    assert audit.verify_optimizer_receipts(complete,fit,starts,traces)==10
    bad=deepcopy(complete);bad['steps']['selection']=5
    with pytest.raises(ValueError,match='steps'):audit.verify_optimizer_receipts(bad,fit,starts,traces)
    bad=deepcopy(starts);bad['refit']['identity']['trial_id']='another'
    with pytest.raises(ValueError,match='identity'):audit.verify_optimizer_receipts(complete,fit,bad,traces)
    bad=deepcopy(starts);bad['refit']['time_ns']=1
    with pytest.raises(ValueError,match='order'):audit.verify_optimizer_receipts(complete,fit,bad,traces)


def test_audit_refuses_unfinished_run_before_loading_prediction_arrays(tmp_path,monkeypatch):
    execution=tmp_path/'execution';execution.mkdir()
    (execution/'terminal.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError,match='terminal'):audit.run(tmp_path,execution/'audit.json')
    assert not (execution/'audit.json').exists()


def test_two_target_score_keeps_correct_single_column_units():
    import numpy as np
    y=np.array([[100.,10.],[100.,10.]])
    old=audit.scalar_score(y,[100,100],[12,12]);new=audit.scalar_score(y,[100,100],[11,11])
    assert old==90 and new==95 and new-old==5


def test_final_labels_are_bound_to_original_partition_inputs():
    import pandas as pd
    frame=pd.DataFrame({'sample_id':['a','b'],'tap_iron':[100.,200.],'tap_time_len':[10.,20.]})
    plan={'training_ids':['a','b'],'query_ids':['c']};truth={}
    audit.remember_truth(truth,frame,plan)
    assert truth=={'a':(100.,10.),'b':(200.,20.)}
    bad=frame.copy();bad.loc[0,'tap_time_len']=11.
    with pytest.raises(ValueError,match='labels'):audit.remember_truth(truth,bad,plan)
    with pytest.raises(ValueError,match='identities'):audit.remember_truth({},frame,dict(plan,query_ids=['b']))
