import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.incumbent_reference_ledger import ReservationLedger,write_new,file_hash
from bf_tap_r2.incumbent_reference_phase import (resource_admission,execute_jobs,audit_new_estimators,LIMITS,TASKS,RESOURCE)


def settings():
    return dict(random_seed=42,inner_seed=42,width=16,blocks=1,tabm_k=4,dropout=.1,
        embedding_dim=4,n_frequencies=4,lite=True,learning_rate=.001,weight_decay=.0001,
        batch_size=16,max_epochs=1,patience=25,min_delta=1e-5)


def test_original_cost_projection_uses_worst_cap_and_four_workers():
    evidence=dict(cost_witnesses=[dict(seconds=100,optimizer_updates=1000,peak_rss_mib=700)]*10)
    report=resource_admission(evidence,dict(batch_size=256,max_epochs=240),10000)
    assert report['status']=='passed'
    assert report['worst_optimizer_updates_per_estimator']==4320
    assert report['projected_seconds']==20*4320*.1/4*1.5+300
    assert report['projected_peak_worker_rss_mib']==1050
    assert report['new_probe_fits']==0 and report['limits']==RESOURCE


@pytest.mark.parametrize('defect',['runtime','rss','available','nonfinite','coverage'])
def test_cost_admission_rejects_each_resource_or_evidence_failure(defect):
    evidence=dict(cost_witnesses=[dict(seconds=100,optimizer_updates=1000,peak_rss_mib=700) for _ in range(10)])
    available=10000
    if defect=='runtime':evidence['cost_witnesses'][0]['seconds']=10000
    elif defect=='rss':evidence['cost_witnesses'][0]['peak_rss_mib']=1100
    elif defect=='available':available=7000
    elif defect=='nonfinite':evidence['cost_witnesses'][0]['seconds']=float('nan')
    else:evidence['cost_witnesses'].pop()
    if defect in ('nonfinite','coverage'):
        with pytest.raises(ValueError):resource_admission(evidence,dict(batch_size=256,max_epochs=240),available)
    else:assert resource_admission(evidence,dict(batch_size=256,max_epochs=240),available)['status']=='failed'


def test_full_twenty_estimator_spawn_budget_and_zero_fit_cold_audit(tmp_path,monkeypatch):
    import bf_tap_r2.incumbent_reference_phase as phase
    from bf_tap_r2.incumbent_reference_columns import Member,SPLIT_SEEDS,TRAINING_SEEDS
    from bf_tap_r2.v7_periodic import digest
    import json
    rng=np.random.default_rng(57201);x=rng.normal(size=(30,len(FEATURES)))
    frame=pd.DataFrame(x,columns=FEATURES);frame['sample_id']=[f'reference-phase-toy-{i}' for i in range(len(x))]
    frame['spout_no']=np.arange(len(x))%2+1
    frame['tap_iron']=1000+31*x[:,0]+x[:,1]*x[:,2];frame['tap_time_len']=50+4*np.sin(x[:,3])
    folds={s:(np.arange(30)+i)%5 for i,s in enumerate(SPLIT_SEEDS)}
    ledger=ReservationLedger.create(tmp_path/'ledger',LIMITS)
    anchors=execute_jobs(frame,folds,settings(),tmp_path,ledger)
    before=ledger.inspect();assert before['started']==before['completed']==LIMITS
    assert not any(before['failed'].values()) and not any(before['incomplete'].values())
    members,reports=audit_new_estimators(frame,folds,settings(),tmp_path,ledger,anchors)
    assert {(m.split_seed,m.fold,m.training_seed) for m in members}==set(TASKS)
    assert sum(r['cold_models'] for r in reports)==40
    assert all(r['full_batch_difference']==0 and r['new_optimizer_calls']==0 for r in reports)
    assert ledger.inspect()==before
    bad=dict(anchors);name=next(iter(bad));bad[name]=dict(bad[name],key=[42,0,104729])
    with pytest.raises(ValueError,match='key differs'):audit_new_estimators(frame,folds,settings(),tmp_path,ledger,bad)
    assert ledger.inspect()==before
    # Exercise the complete overlay writer using real audited new estimators.
    # The historical-cache boundary is synthetic here; no official source or
    # original audit is replaced in CLI execution.
    j42={s:np.full(30,1000.) for s in SPLIT_SEEDS}
    parent={s:dict(tap_iron=np.full(30,1000.),tap_time_len=np.full(30,50.)) for s in SPLIT_SEEDS}
    reused=[];native=[]
    for s in SPLIT_SEEDS:
        for f in range(5):
            mask=folds[s]==f;ids=tuple(frame.loc[mask,'sample_id'])
            for t in TRAINING_SEEDS:
                m=Member(s,f,t,ids,j42[s][mask])
                if s in (42,3407):reused.append(m)
                elif t==42:native.append(m)
    write_new(tmp_path/'ledger-anchor.json',dict(policy_sha256=ledger.policy_sha256))
    write_new(tmp_path/'run.finished.json',dict(manifest_sha256='synthetic-only',anchors=anchors,counts=before))
    reference=dict(row_ids_digest=digest(frame.sample_id.tolist()),native_data_digest='synthetic-only',
        fold_digests={str(s):digest(folds[s].tolist()) for s in SPLIT_SEEDS})
    manifest=dict(output=str(tmp_path),settings=settings(),reference=reference,development_cache='synthetic-only',development_evidence={})
    monkeypatch.setattr(phase,'verify_manifest',lambda *args:manifest)
    monkeypatch.setattr(phase,'reload_context',lambda *args:(frame,folds,parent,j42,native))
    monkeypatch.setattr(phase,'audit_development',lambda *args:(reused,dict(status='synthetic_boundary_only')))
    (tmp_path/'audit.log').write_text('live audit output')
    receipt=phase.audit(tmp_path/'manifest.json','synthetic-only',file_hash(tmp_path/'run.finished.json'))
    assert receipt['sha256']==file_hash(tmp_path/'complete.json')
    report=json.loads((tmp_path/'audit.json').read_text());complete=json.loads((tmp_path/'complete.json').read_text())
    assert report['new_audit_fits']==0 and report['new_cold_models']==40
    assert report['verified_split_seeds']==list(SPLIT_SEEDS)
    assert 'audit.log' not in report['artifact_hashes']
    (tmp_path/'audit.log').write_text('live audit output\ncompletion printed')
    for name,sha in report['artifact_hashes'].items():assert file_hash(tmp_path/name)==sha
    for s in SPLIT_SEEDS:
        record=complete['iron_columns'][str(s)]
        assert file_hash(tmp_path/record['path'])==record['sha256']
        prediction=np.load(tmp_path/record['path'],allow_pickle=False)
        if s in (42,3407):np.testing.assert_array_equal(prediction,parent[s]['tap_iron'])
        assert np.isfinite(prediction).all() and (prediction>=0).all()
    assert ledger.inspect()==before
