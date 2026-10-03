"""Fault injection into terminal evidence; no model fits or predictions."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1]/'scripts/audit_de3_independent_batches_terminal.py'
spec = importlib.util.spec_from_file_location('independent_terminal', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def terminal():
    events = []
    for s in (42, 3407):
        for f in range(5):
            stages = [('reuse', -1), ('worker', 42), ('cold', 42),
                      ('worker', 104729), ('cold', 104729), ('worker', 130363), ('cold', 130363)]
            for stage, init in stages:
                events.append(dict(task=f'{len(events):03d}-{stage}-s{s}-f{f}-i{init}',
                                   exit_code=0, peak_rss_mib=800., completed_ns=10+len(events)))
    events.extend([dict(task='070-report-s42-f0-i-1', exit_code=0, peak_rss_mib=100., completed_ns=80),
                   dict(task='071-audit-s42-f0-i-1', exit_code=0, peak_rss_mib=100., completed_ns=81)])
    return dict(status='passed', dependencies_unchanged=True, events=events)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'failed', 'rss', 'nan', 'order'])
def test_rejects_unclosed_or_corrupt_actual_process_inventory(fault):
    t = terminal()
    assert len(audit.verify_events(t)) == 72
    if fault == 'missing':
        t['events'].pop()
    elif fault == 'duplicate':
        t['events'][1] = copy.deepcopy(t['events'][0])
    elif fault == 'failed':
        t['events'][30]['exit_code'] = 1
    elif fault == 'rss':
        t['events'][30]['peak_rss_mib'] = 1536.01
    elif fault == 'nan':
        t['events'][30]['peak_rss_mib'] = float('nan')
    else:
        t['events'][30]['completed_ns'] = 1
    with pytest.raises(ValueError):
        audit.verify_events(t)


def test_failed_terminal_stops_before_reading_quality_or_importing_torch(tmp_path, monkeypatch):
    import builtins
    execution = tmp_path/'execution'; execution.mkdir()
    (execution/'terminal.json').write_text(json.dumps({'status':'failed'}))
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch':
            raise AssertionError('Premature scientific import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError, match='before quality reads'):
        audit.run(tmp_path, execution/'audit.json')
    assert not (execution/'audit.json').exists()


def native():
    identity = {'source_directory':'isolated', 'split_seed':42, 'fold':0, 'trial_id':'JOINT_IBATCH_INIT42'}
    plan = dict(training_ids=['a','b','c','d'], query_ids=['q'])
    c = dict(identity=identity, pid=123, constructors=['selection','refit'],
             settings=dict(batch_size=2, min_delta=1e-5), steps=dict(selection=3,refit=4), **plan)
    start = dict(identity=identity, pid=123, time_ns=1, **plan)
    receipts = {r:dict(identity=identity, role=r, time_ns=t) for r,t in [('selection',2),('refit',3)]}
    traces = {}
    for r, n, values in [('selection',2,[.5,.4,.45]), ('refit',4,[0.,0.])]:
        h = [dict(epoch=i+1, updates=n//2, gradient_evaluations=n//2, validation_mae=v) for i,v in enumerate(values)]
        traces[r] = dict(history=h, fit_rows=n, stopped_epoch=len(h), selected_epoch=2,
            updates=sum(x['updates'] for x in h), samples_per_head_per_epoch=n,
            batch_order='independent_without_replacement', member_order_digests=['hash']*len(h),
            fit_ids_digest=audit.digest(plan['training_ids']) if r=='refit' else 'inner_hash')
    return c,start,receipts,traces,plan


@pytest.mark.parametrize('fault', ['partition', 'identity', 'step', 'epoch', 'optimizer_order'])
def test_native_ledger_rejects_partition_step_selection_or_identity_corruption(fault):
    values = native()
    assert audit.verify_native(*values) == 7
    c,start,receipts,traces,plan = copy.deepcopy(values)
    if fault == 'partition':
        c['query_ids'] = ['a']
    elif fault == 'identity':
        receipts['refit']['identity'] = {'source_directory':'wrong'}
    elif fault == 'step':
        traces['refit']['history'][0]['updates'] += 1
    elif fault == 'epoch':
        traces['selection']['selected_epoch'] = 3
    else:
        receipts['refit']['time_ns'] = 2
    with pytest.raises(ValueError):
        audit.verify_native(c,start,receipts,traces,plan)
