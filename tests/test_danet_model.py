from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from bf_tap_r2.danet_model import Regressor,Settings,ARMS,state_identity
from bf_tap_r2.danet_audit import verify_saved,independent_predict
from bf_tap_r2.danet_calibration import fit_pair
from bf_tap_r2.danet_execution import execute_unit,audit_unit
from bf_tap_r2.danet_ledger import ReservationLedger,file_hash
from test_dnnr_model import sample

SMALL=replace(Settings(),layers=2,width=4,groups=2,max_epochs=3,batch_size=32,ghost_size=32)
TASK=dict(target='tap_time_len',seed=42,fold=0)


@pytest.mark.parametrize('arm',ARMS)
def test_saved_model_cold_numpy_order_and_chunks_without_refitting(tmp_path,monkeypatch,arm):
    training=sample(80);query=sample(13,57322,200);y=30+training.air_volume.to_numpy()**2
    before=torch.get_rng_state().clone();model=Regressor(arm,SMALL).fit(training,y,epochs=2)
    assert torch.equal(torch.get_rng_state(),before)
    path=tmp_path/'model.npz';sha=model.save(path);expected=model.predict(query)
    def no_fit(*args,**kwargs):raise AssertionError('Auditor attempted fitting')
    monkeypatch.setattr(Regressor,'fit',no_fit)
    audit=verify_saved(path,sha,training,y,query,expected)
    assert audit['status']=='passed' and audit['new_optimizer_runs']==0
    np.testing.assert_array_equal(Regressor.load(path,sha).predict(query),expected)
    with pytest.raises(FileExistsError):model.save(path)
    with pytest.raises(ValueError):Regressor.load(path,None)


def test_inner_selection_and_outer_stats_are_fresh_on_their_actual_partition(tmp_path):
    training=sample(100);training.loc[0,'air_volume']=1000.;y=30+training.hot_air_press.to_numpy()**2
    pair=fit_pair(training,y,'tap_time_len',settings=SMALL)
    for arm in ARMS:
        inner,outer=pair.inner[arm],pair.outer[arm]
        values=inner.trace_['calibration_predictions'];ids=pair.receipt['calibration_ids']
        target=y[training.sample_id.isin(ids)]
        assert inner.selected_epoch_==np.argmin(np.abs(values-target).mean(1))+1
        assert outer.actual_epochs_==inner.selected_epoch_
        assert outer.encoder_.fit_ids_==training.sample_id.tolist()
        assert len(inner.encoder_.fit_ids_)<len(outer.encoder_.fit_ids_)
        assert inner.model_ is not outer.model_ and inner.center_!=outer.center_
    assert pair.inner[ARMS[0]].initial_state_digest_==pair.inner[ARMS[1]].initial_state_digest_


def test_four_model_pair_audits_all_reservations_and_refuses_retry(tmp_path,monkeypatch):
    training=sample(100);query=sample(13,57322,200);y=30+training.air_volume.to_numpy()**2
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(pair_unit=1,estimator=4,optimizer=4))
    complete=execute_unit(TASK,training,y,query,tmp_path/'unit',ledger.root,ledger.policy_sha256,settings=SMALL)
    def no_fit(*args,**kwargs):raise AssertionError('Pair audit attempted fitting')
    monkeypatch.setattr(Regressor,'fit',no_fit)
    predictions,report=audit_unit(tmp_path/'unit',complete['complete_sha256'],TASK,training,y,query,settings=SMALL,ledger_root=ledger.root)
    assert set(predictions)==set(ARMS) and report['saved_models']==4 and report['new_optimizer_runs']==0
    assert ledger.inspect()['completed']==dict(pair_unit=1,estimator=4,optimizer=4)
    with pytest.raises(FileExistsError):execute_unit(TASK,training,y,query,tmp_path/'unit',ledger.root,ledger.policy_sha256,settings=SMALL)


def test_resource_upper_bound_fits_all_four_models_to_maximum_and_cannot_masquerade_as_formal(tmp_path):
    training=sample(100);query=sample(13,57322,200);y=30+training.air_volume.to_numpy()**2
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(pair_unit=1,estimator=4,optimizer=4))
    result=execute_unit(TASK,training,y,query,tmp_path/'unit',ledger.root,ledger.policy_sha256,settings=SMALL,resource_upper_bound=True)
    _,audit=audit_unit(tmp_path/'unit',result['complete_sha256'],TASK,training,y,query,settings=SMALL,resource_upper_bound=True,ledger_root=ledger.root)
    assert audit['optimizer_epochs']==4*SMALL.max_epochs
    with pytest.raises(ValueError):audit_unit(tmp_path/'unit',result['complete_sha256'],TASK,training,y,query,settings=SMALL,ledger_root=ledger.root)


def test_failed_optimizer_start_remains_consumed_and_cannot_retry(tmp_path):
    ledger=ReservationLedger.create(tmp_path/'ledger',{'optimizer':1})
    with pytest.raises(RuntimeError):
        with ledger.event('optimizer',('test',),{'epochs':3}):raise RuntimeError('synthetic optimizer failure')
    assert ledger.inspect()['failed']['optimizer']==1
    with pytest.raises(FileExistsError):
        with ledger.event('optimizer',('test',),{'epochs':3}):pass
    with pytest.raises(ValueError):
        with ledger.event('optimizer',('different',),{'epochs':3}):pass


@pytest.mark.parametrize('defect',['training_target','training_stats','fixed_mask','permutation','counter'])
def test_rehashed_tampering_still_fails_independent_training_and_state_checks(tmp_path,defect):
    training=sample(80);query=sample(13,57322,200);y=30+training.air_volume.to_numpy()**2
    model=Regressor('DANET_FIXED',SMALL).fit(training,y,epochs=2)
    path=tmp_path/'model.npz';model.save(path);expected=model.predict(query)
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
    metadata=json.loads(str(arrays.pop('metadata')))
    if defect=='training_target':arrays['y'][0]+=.1
    elif defect=='training_stats':metadata['encoder']['means'][0]+=.1
    elif defect=='fixed_mask':arrays['state::blocks.0.first.logits'][0,0]+=.1
    elif defect=='permutation':arrays['trace::permutations'][0]=arrays['trace::permutations'][0][::-1]
    else:arrays['state::blocks.0.first.normalization.bn.num_batches_tracked']+=1
    metadata['array_digests']={k:state_identity(v) for k,v in arrays.items()}
    with path.open('wb') as stream:np.savez(stream,metadata=np.array(json.dumps(metadata)),**arrays)
    with pytest.raises(ValueError):verify_saved(path,file_hash(path),training,y,query,expected)


def test_full_default_depth_saved_state_independent_inference(tmp_path):
    training=sample(80);query=sample(11,57322,200);y=30+training.air_volume.to_numpy()**2
    model=Regressor('DANET_LEARNED').fit(training,y,epochs=1)
    path=tmp_path/'full-depth.npz';sha=model.save(path)
    report=verify_saved(path,sha,training,y,query,model.predict(query))
    assert report['status']=='passed' and len(model.model_.blocks)==10


def test_labels_overlap_and_singleton_batches_fail_without_a_retry():
    training=sample(65);y=np.ones(65)
    model=Regressor('DANET_FIXED',SMALL)
    with pytest.raises(ValueError,match='singleton'):model.fit(training,y)
    with pytest.raises(ValueError,match='single-use'):model.fit(sample(80),np.ones(80))
    valid=sample(80);labelled=valid.copy();labelled['tap_time_len']=1.
    with pytest.raises(ValueError):Regressor('DANET_FIXED',SMALL).fit(labelled,np.ones(80))
    with pytest.raises(ValueError,match='Disjoint'):
        Regressor('DANET_FIXED',SMALL).fit(valid,np.ones(80),calibration=(valid.iloc[:10].copy(),np.ones(10)))
