import copy
import json

import numpy as np
import pytest

from bf_tap_r2 import rfm_preflight as preflight
from bf_tap_r2.rfm_protocol import ReservationLedger
from test_rfm_model import sample


def measured():
    return {arm:dict(arm=arm,procedures=2,solves=8 if arm=='FULL_RFM' else 2,
        updates=6 if arm=='FULL_RFM' else 0,refit_updates=3 if arm=='FULL_RFM' else 0,
        train_rows=2204,query_rows=551,seconds=100.,peak_mib=1000.,mae=1.,median_mae=2.,
        cold_difference=1e-8) for arm in ('FIXED_KRR','FULL_RFM')}


def test_resource_formula_and_frozen_admission_boundaries():
    arms=measured();decision=preflight.resource_decision(arms,5024)
    assert decision['status']=='passed' and decision['projected_development_seconds']==1800
    assert preflight.resource_decision(arms,5023)['checks']['available_memory'] is False
    for key,value,check in [('mae',2.,'synthetic_quality'),('cold_difference',1.01e-8,'cold_inference'),
                            ('peak_mib',1537.,'worker_peak'),('seconds',1000.,'development_cost')]:
        wrong=copy.deepcopy(arms);wrong['FULL_RFM'][key]=value
        assert preflight.resource_decision(wrong,100000)['checks'][check] is False
    for key,value in [('refit_updates',0),('query_rows',550),('procedures',1),('solves',7)]:
        wrong=copy.deepcopy(arms);wrong['FULL_RFM'][key]=value
        with pytest.raises(ValueError,match='frozen design'):preflight.resource_decision(wrong,100000)


def test_synthetic_generation_is_exact_frozen_formula_without_fits():
    train,y,query,truth=preflight.synthetic_data()
    rng=np.random.default_rng(56001);x=rng.normal(size=(2755,21))
    expected=10+2*np.sin(x[:,0])+x[:,1]*x[:,2]+.5*x[:,3]+.1*rng.normal(size=2755)
    assert (len(train),len(query))==(2204,551)
    np.testing.assert_array_equal(np.r_[y,truth],expected)
    assert set(train.sample_id).isdisjoint(query.sample_id)


def test_tiny_worst_case_paths_consume_exact_synthetic_schedule(tmp_path,monkeypatch):
    # Only this test substitutes small arrays; no full-size probe is executed.
    control=tmp_path/'local'/'control';root=control/'preflight';root.mkdir(parents=True)
    ledger=ReservationLedger.create(root/'ledger',preflight.LIMITS)
    frame=sample(50);train=frame.iloc[:40];query=frame.iloc[40:]
    monkeypatch.setattr(preflight,'synthetic_data',lambda:(train,train.tap_time_len.to_numpy(),query,query.tap_time_len.to_numpy()))
    monkeypatch.setattr(preflight,'verify_manifest',lambda *args:{'workspace':str(tmp_path)})
    # Avoid changing global PyTorch inter-op settings in the shared pytest process.
    import bf_tap_r2.rfm_run as run
    monkeypatch.setattr(run,'initialize_worker',lambda:None)
    for arm in preflight.ARMS:
        report=preflight.measure_arm(control/'manifest.json','test',arm,ledger.policy_sha256)
        assert report['refit_updates']==(3 if arm=='FULL_RFM' else 0)
        assert report['cold_difference']<=1e-8
        assert len(report['paths']['refit']['artifacts'])==report['refit_updates']+1
    assert preflight._verify_costs(root,ledger.policy_sha256)['completed']==preflight.LIMITS
    with pytest.raises(FileExistsError):
        preflight.measure_arm(control/'manifest.json','test','FULL_RFM',ledger.policy_sha256)


def test_existing_preflight_directory_never_reentered(tmp_path,monkeypatch):
    root=tmp_path/'local'/'control';(root/'preflight').mkdir(parents=True)
    monkeypatch.setattr(preflight,'verify_manifest',lambda *args:{'workspace':str(tmp_path)})
    with pytest.raises(FileExistsError):preflight.run_synthetic_admission(root/'manifest.json','test')
