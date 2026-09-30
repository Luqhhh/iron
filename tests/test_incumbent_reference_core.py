from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.data import FEATURES,TARGETS
from bf_tap_r2.incumbent_reference_core import (ReservedRegressor,execute_estimator,audit_estimator,RECIPE)
from bf_tap_r2.incumbent_reference_ledger import ReservationLedger,file_hash


def frame():
    rng=np.random.default_rng(57101);x=rng.normal(size=(60,len(FEATURES)))
    f=pd.DataFrame(x,columns=FEATURES);f['sample_id']=[f'incumbent-toy-{i}' for i in range(len(x))]
    f['spout_no']=np.arange(len(x))%2+1
    f['tap_iron']=1000+31*x[:,0]+x[:,1]*x[:,2]
    f['tap_time_len']=50+4*np.sin(x[:,3])
    return f


def settings():
    return dict(random_seed=104729,inner_seed=42,width=16,blocks=1,tabm_k=4,dropout=.1,
        embedding_dim=4,n_frequencies=4,lite=True,learning_rate=.001,weight_decay=.0001,
        batch_size=16,max_epochs=3,patience=25,min_delta=1e-5)


def test_reservations_do_not_change_native_rng_parameter_or_prediction_trajectory(tmp_path):
    f=frame().iloc[:50];s=settings();y=f[list(TARGETS)].to_numpy(float)
    plain=ComponentRegressor(RECIPE,s,'BASE',{}).fit(f,y)
    directory=tmp_path/'wrapped';directory.mkdir()
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(estimator=0,optimizer=2))
    wrapped=ReservedRegressor(s,directory,ledger,(7777,0,104729)).fit(f,y)
    for k,v in plain.model_.state_dict().items():torch.testing.assert_close(v,wrapped.model_.state_dict()[k],rtol=0,atol=0)
    query=frame().iloc[50:].drop(columns=list(TARGETS))
    np.testing.assert_array_equal(plain.predict(query),wrapped.predict(query))
    assert wrapped.initializations==2 and ledger.inspect()['completed']['optimizer']==2
    assert wrapped.metadata_['selected_epoch']==plain.metadata_['selected_epoch']
    with pytest.raises(ValueError,match='repeated fit'):wrapped.fit(f,y)


def test_saved_estimator_cold_audit_and_actual_optimizer_starts(tmp_path):
    f=frame();train=f.iloc[:50].reset_index(drop=True);query=f.iloc[50:].drop(columns=list(TARGETS)).reset_index(drop=True)
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(estimator=1,optimizer=2))
    key=(7777,0,104729);directory=tmp_path/'model';s=settings()
    anchor=execute_estimator(train,query,s,key,directory,ledger.root,ledger.policy_sha256)
    counts=ledger.inspect();assert counts['started']==counts['completed']==dict(estimator=1,optimizer=2)
    assert not any(counts['failed'].values()) and not any(counts['incomplete'].values())
    prediction,audit=audit_estimator(directory,train,query,s,key,anchor['complete_sha256'],ledger.root,ledger.policy_sha256)
    assert prediction.shape==(10,2) and audit['cold_models']==2 and audit['new_optimizer_calls']==0
    assert ledger.inspect()==counts
    with pytest.raises(FileExistsError):execute_estimator(train,query,s,key,directory,ledger.root,ledger.policy_sha256)


def test_bad_query_is_failed_evidence_without_optimizer_creation(tmp_path):
    f=frame();ledger=ReservationLedger.create(tmp_path/'ledger',dict(estimator=1,optimizer=2))
    with pytest.raises(ValueError,match='query labels'):
        execute_estimator(f.iloc[:50],f.iloc[50:],settings(),(7777,0,104729),tmp_path/'bad',ledger.root,ledger.policy_sha256)
    counts=ledger.inspect();assert counts['failed']['estimator']==1 and counts['started']['optimizer']==0
    assert (tmp_path/'bad').exists()


def test_initializer_failure_remains_consumed_and_cannot_retry(tmp_path):
    f=frame().iloc[:50];s=settings();directory=tmp_path/'failure';directory.mkdir()
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(estimator=0,optimizer=2))
    model=ReservedRegressor(s,directory,ledger,(7777,0,104729))
    y=f[list(TARGETS)].to_numpy(float);y[:,0]=1
    with pytest.raises(ValueError,match='Constant'):model.fit(f,y)
    assert ledger.inspect()['failed']['optimizer']==1
    assert model.active is None
    with pytest.raises(ValueError,match='repeated fit'):model.fit(f,y)
    assert ledger.inspect()['started']['optimizer']==1


def test_cold_audit_rejects_external_anchor_or_original_row_change(tmp_path):
    f=frame();train=f.iloc[:50].reset_index(drop=True);query=f.iloc[50:].drop(columns=list(TARGETS)).reset_index(drop=True)
    s=settings();key=(12011,4,130363)
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(estimator=1,optimizer=2))
    output=tmp_path/'unit';anchor=execute_estimator(train,query,s,key,output,ledger.root,ledger.policy_sha256)
    with pytest.raises(ValueError,match='anchor'):
        audit_estimator(output,train,query,s,key,'wrong',ledger.root,ledger.policy_sha256)
    altered=train.copy();altered.iloc[0,0]+=1
    with pytest.raises(ValueError,match='partition'):
        audit_estimator(output,altered,query,s,key,anchor['complete_sha256'],ledger.root,ledger.policy_sha256)
