"""Budget, reference and source-freeze safety without optimizer starts."""
import json
import numpy as np
import pandas as pd
import pytest

def test_optimizer_markers_reject_duplicate_and_unfrozen_stages(tmp_path):
 from bf_tap_r2.v46_run import FitLedger
 ledger=FitLedger(tmp_path,'NLL',42,0)
 ledger({'event':'optimizer_started','stage':'inner','arm':'NLL'})
 with pytest.raises(FileExistsError):ledger({'event':'optimizer_started','stage':'inner','arm':'NLL'})
 with pytest.raises(ValueError):ledger({'event':'optimizer_started','stage':'extra','arm':'NLL'})
 ledger({'event':'optimizer_started','stage':'outer','arm':'NLL'})
 assert len(list(tmp_path.glob('optimizer-*.json')))==2

def test_paired_scores_are_latest_reference_increment_and_fixed_a20():
 from bf_tap_r2.v46_run import score_records
 frame=pd.DataFrame({'tap_iron':[100.]*10,'tap_time_len':[20.]*10})
 folds={s:np.tile(np.arange(5),2) for s in [42,3407]}
 current={s:{'tap_iron':np.full(10,101.),'tap_time_len':np.full(10,22.)} for s in folds}
 historical={s:{'tap_iron':np.full(10,101.),'tap_time_len':np.full(10,23.)} for s in folds}
 predictions={a:{s:np.full(10,20. if a=='BETA05' else 22.) for s in folds} for a in ['NLL','BETA05']}
 records=score_records(frame,folds,current,historical,predictions)
 assert records['BETA05']['42']['gain']==pytest.approx(1.)
 assert records['BETA05']['42']['candidate_score']==pytest.approx(95.5)
 assert records['BETA05']['42']['B0_gain']==pytest.approx(3.5)
 predictions['NLL'][42][0]=np.nan
 with pytest.raises(ValueError,match='coverage'):score_records(frame,folds,current,historical,predictions)

def test_manifest_guard_rejects_changed_private_identity(tmp_path):
 from bf_tap_r2.v46_run import freeze_hashes,verify_hashes
 p=tmp_path/'evidence';p.write_bytes(b'original')
 frozen=freeze_hashes(tmp_path,[p]);verify_hashes(tmp_path,frozen)
 p.write_bytes(b'changed')
 with pytest.raises(ValueError,match='changed'):verify_hashes(tmp_path,frozen)

def test_phase_tasks_do_not_spend_confirmation_without_eligibility():
 from bf_tap_r2.v46_run import phase_tasks
 assert len(phase_tasks('development'))==20
 with pytest.raises(ValueError,match='eligible'):phase_tasks('confirmation',False)
 tasks=phase_tasks('confirmation',True)
 assert len(tasks)==20
 assert {t['seed'] for t in tasks}=={7777,12011}
 assert all(t['arm'] in ['NLL','BETA05'] and t['fold'] in range(5) for t in tasks)


def test_experiment_phase_reservation_rejects_new_directory_retry(tmp_path):
 from bf_tap_r2.v46_run import reserve_phase
 first=tmp_path/'local/runs/round2-v46-beta-nll/development-r1'
 first.parent.mkdir(parents=True)
 reservation=reserve_phase(tmp_path,first,'development','hash')
 assert json.loads(reservation.read_text())['optimizer_allocation']==40
 other=first.parent/'development-r2'
 with pytest.raises(FileExistsError):reserve_phase(tmp_path,other,'development','hash')
 # The first failure cannot free the consumed allocation.
 first.mkdir();(first/'batch-failure.json').write_text('{"status":"failed"}')
 with pytest.raises(FileExistsError):reserve_phase(tmp_path,other,'development','hash')
 assert (first/'batch-failure.json').exists()
 confirmation=reserve_phase(tmp_path,first.parent/'confirmation-r1','confirmation','hash')
 assert json.loads(confirmation.read_text())['outer_fit_allocation']==20
 with pytest.raises(ValueError):reserve_phase(tmp_path,other,'extra','hash')


def test_completed_failure_is_checked_before_any_worker_refill():
 from concurrent.futures import Future
 from bf_tap_r2.v46_run import settle_completed
 success=Future();success.set_result('ok')
 failure=Future();failure.set_exception(RuntimeError('fit_failed'))
 queued=Future();pending={success:'success',failure:'failure',queued:'queued'}
 submitted=[]
 with pytest.raises(RuntimeError,match='fit_failed'):
  settle_completed(pending,[success,failure],lambda:submitted.append('new_fit'))
 assert not submitted
 assert queued.cancelled()
 good=Future();good.set_result('complete')
 pending={good:'task'}
 assert settle_completed(pending,[good],lambda:submitted.append('new_fit'))==['task']
 assert submitted==['new_fit']
