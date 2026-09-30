from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from bf_tap_r2.dnnr_model import ARMS, Regressor
from bf_tap_r2.dnnr_ledger import ReservationLedger, file_hash, write_new
from bf_tap_r2.dnnr_execution import execute_unit, audit_unit
from test_dnnr_model import sample

LIMITS = dict(pair_unit=1, estimator=6, metric_epoch=2, derivative_bank=4)
TASK = dict(target='tap_time_len', seed=42, fold=0)


def test_saved_complete_unit_reconciles_all_operation_reservations_without_refitting(tmp_path, monkeypatch):
    training = sample(100); query = sample(13, 57322, 200)
    y = 30+training.air_volume.to_numpy()**2+training.hot_air_press.to_numpy()**2
    ledger = ReservationLedger.create(tmp_path/'ledger', LIMITS)
    complete = execute_unit(TASK, training, y, query, tmp_path/'unit', ledger.root, ledger.policy_sha256)
    def no_fits(*_args, **_kwargs): raise AssertionError('Auditor attempted fit')
    monkeypatch.setattr(Regressor, 'fit', no_fits)
    prediction, audit = audit_unit(tmp_path/'unit', complete['complete_sha256'], TASK, training, y, query, ledger_root=ledger.root)
    assert set(prediction) == set(ARMS) and audit['saved_models'] == 6
    counts = ledger.inspect()
    assert counts['completed']['estimator'] == 6 and counts['completed']['derivative_bank'] == 4
    assert counts['completed']['metric_epoch'] == audit['counts']['metric_epoch_runs']
    assert all(v == 0 for k in ('failed', 'incomplete') for v in counts[k].values())
    before = counts.copy()
    with pytest.raises(FileExistsError):
        execute_unit(TASK, training, y, query, tmp_path/'unit', ledger.root, ledger.policy_sha256)
    assert ledger.inspect() == before
    with pytest.raises(FileExistsError):
        execute_unit(TASK, training, y, query, tmp_path/'second', ledger.root, ledger.policy_sha256)
    assert (tmp_path/'second').exists() and ledger.inspect() == before


def test_crashed_fitting_event_remains_consumed_and_cannot_retry(tmp_path):
    ledger = ReservationLedger.create(tmp_path/'ledger', {'estimator': 1})
    with pytest.raises(RuntimeError):
        with ledger.event('estimator', ('test',), {'rows': 10}):
            raise RuntimeError('synthetic crash')
    count = ledger.inspect()
    assert count['started']['estimator'] == count['failed']['estimator'] == 1
    with pytest.raises(FileExistsError):
        with ledger.event('estimator', ('test',), {'rows': 10}): pass
    with pytest.raises(ValueError, match='exhausted'):
        with ledger.event('estimator', ('different',), {'rows': 10}): pass


def test_concurrent_count_and_reserve_admits_exactly_one_event(tmp_path):
    ledger = ReservationLedger.create(tmp_path/'ledger', {'estimator': 1})
    def reserve(i):
        try:
            with ledger.event('estimator', (i,), {'rows': 10}): pass
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(2) as executor:
        assert sum(executor.map(reserve, (1, 2))) == 1
    assert ledger.inspect()['completed']['estimator'] == 1


def test_policy_and_terminal_tampering_are_detected(tmp_path):
    ledger = ReservationLedger.create(tmp_path/'ledger', {'estimator': 1})
    with ledger.event('estimator', ('test',), {'rows': 10}): pass
    terminal = next((ledger.root/'events').glob('*.complete.json'))
    data = json.loads(terminal.read_text()); data['start_sha256'] = '0'*64
    terminal.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='identity'): ledger.inspect()
    policy = ledger.root/'policy.json'
    policy.write_text(json.dumps({'version':1, 'limits':{'estimator':2}}))
    with pytest.raises(ValueError, match='policy'): ReservationLedger.open(ledger.root, ledger.policy_sha256)
