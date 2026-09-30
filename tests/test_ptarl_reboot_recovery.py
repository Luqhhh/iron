"""Reboot recovery reuses only complete, externally anchored units."""
import importlib.util
from pathlib import Path
import pytest
from bf_tap_r2.ptarl_protocol import ReservationLedger, phase_limits, write_new

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).resolve().parents[1]/'scripts/recover_ptarl_after_reboot.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def make_phase(tmp_path):
    phase = tmp_path/'development'; phase.mkdir(); (phase/'units').mkdir()
    ledger = ReservationLedger.create(phase/'ledger', phase_limits('development'))
    name = 'tap_iron-s42-f0'; unit = phase/'units'/name; unit.mkdir(); write_new(unit/'complete.json', {'name': name})
    with ledger.event('pair_unit', (name,), {}) as pair:
        for kind in ('optimizer', 'kmeans'):
            for role in ('teacher_inner', 'teacher_outer'):
                with ledger.event(kind, (name, role), {}):
                    pass
        for arm in ('CONTROL', 'PTARL_AUX'):
            for stage in ('selector', 'refit'):
                with ledger.event('optimizer', (name, arm+'_'+stage), {}):
                    pass
        pair['unit_complete_sha256'] = recovery.file_hash(unit/'complete.json')
    return phase, ledger, name


def test_complete_coverage_classifies_reuse_and_fresh_without_reading_predictions(tmp_path):
    phase, ledger, name = make_phase(tmp_path)
    partial = phase/'units/tap_time_len-s42-f0'; partial.mkdir(); (partial/'partial.pt').write_bytes(b'partial')
    reusable, fresh = recovery.partition_tasks(phase)
    assert [recovery.task_name(t) for t in reusable] == [name]
    assert len(fresh) == 19 and {recovery.task_name(t) for t in fresh} >= {'tap_time_len-s42-f0', 'tap_time_len-s3407-f4'}
    assert len(recovery.closed_events(phase, name, ledger.policy_sha256)) == 18
    assert (partial/'partial.pt').read_bytes() == b'partial'


def test_interrupted_evidence_inventory_rejects_changes_and_symlinks(tmp_path):
    phase, _, _ = make_phase(tmp_path)
    before = recovery.inventory(phase)
    recovery.verify_inventory(phase, before)
    (phase/'extra').write_bytes(b'new')
    with pytest.raises(ValueError, match='evidence changed'):
        recovery.verify_inventory(phase, before)
    (phase/'link').symlink_to(phase/'extra')
    with pytest.raises(ValueError, match='symlinks'):
        recovery.inventory(phase)


@pytest.mark.parametrize('corruption', ['missing_end', 'changed_start', 'failed_end'])
def test_partial_or_corrupted_reservations_cannot_be_reused(tmp_path, corruption):
    phase, ledger, name = make_phase(tmp_path)
    start = next((ledger.root/'events').glob('optimizer-*.started.json'))
    end = start.with_name(start.name.replace('started.json', 'complete.json'))
    if corruption == 'missing_end':
        end.unlink()  # artificial disposable fixture only
    elif corruption == 'changed_start':
        start.write_text(start.read_text()+' ')
    else:
        write_new(start.with_name(start.name.replace('started.json', 'failed.json')), {})
    with pytest.raises(ValueError, match='reservations'):
        recovery.closed_events(phase, name, ledger.policy_sha256)


def test_unknown_unit_is_not_silently_dropped(tmp_path):
    phase, _, _ = make_phase(tmp_path)
    (phase/'units/unknown').mkdir()
    with pytest.raises(ValueError, match='Unexpected'):
        recovery.partition_tasks(phase)


def test_frozen_controller_gate_prevents_confirmation_after_failed_audit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(recovery, 'verify_manifest', lambda *a: {})
    monkeypatch.setattr(recovery.subprocess, 'run', lambda command, **kw: calls.append(command) or type('Result', (), {'returncode': 1})())
    with pytest.raises(recovery.subprocess.CalledProcessError):
        recovery.continue_sequence(tmp_path, {'workspace': str(tmp_path)}, 'manifest', 'resource', {'complete_sha256': 'complete'})
    assert len(calls) == 1 and 'audit' in calls[0] and 'confirmation' not in calls[0]


def terminal(root, message='Current memory below original admitted requirement'):
    recovery.write_new(root/'completion-event.json',dict(status='failed',type='ValueError',message=message,
        packages=0,uploads=0))
    return recovery.file_hash(root/'completion-event.json')


def test_entry_refusal_reuses_closed_units_and_does_not_refit_them(tmp_path):
    phase,ledger,name=make_phase(tmp_path);sha=terminal(tmp_path)
    reused,missing,policy,counts=recovery.entry_refusal_scope(tmp_path,sha)
    assert [recovery.task_name(t) for t in reused]==[name] and len(missing)==19
    assert counts['started']==counts['completed']==dict(pair_unit=1,optimizer=6,kmeans=2)
    assert policy==ledger.policy_sha256


@pytest.mark.parametrize('defect',['partial_model','consumed_start','failed_model','other_terminal','stale_anchor'])
def test_user_retry_never_retries_failed_models_or_consumed_missing_reservations(tmp_path,defect):
    phase,ledger,name=make_phase(tmp_path)
    sha=terminal(tmp_path,'numerical failure' if defect=='other_terminal' else 'Current memory below original admitted requirement')
    if defect=='partial_model':
        missing=phase/'units/tap_time_len-s42-f0';missing.mkdir();(missing/'model.pt').write_bytes(b'artificial partial model')
    elif defect in ('consumed_start','failed_model'):
        with pytest.raises(ValueError):
            with ledger.event('pair_unit',('tap_time_len-s42-f0',),{}):raise ValueError('artificial failure')
        if defect=='consumed_start':
            next((ledger.root/'events').glob('pair_unit-*.failed.json')).unlink()  # disposable artificial fixture only
    elif defect=='stale_anchor':sha='0'*64
    with pytest.raises(ValueError):recovery.entry_refusal_scope(tmp_path,sha)
