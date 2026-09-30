from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from bf_tap_r2.ptarl_execution import execute_unit, audit_unit, inner_parts
from bf_tap_r2.ptarl_model import clean, PrototypeRegressor, latent_forward
from bf_tap_r2.ptarl_protocol import ReservationLedger, phase_tasks, phase_limits, file_hash
from bf_tap_r2.ptarl_verify import oracle, verify_bank, verify_model
from test_ptarl_model import frame, settings


def test_paired_phase_optimizer_and_kmeans_budget_is_explicit():
    assert len(phase_tasks('development'))==20
    assert phase_limits('development')==dict(pair_unit=20,optimizer=120,kmeans=40)
    assert phase_limits('confirmation',['tap_time_len'])==dict(pair_unit=10,optimizer=60,kmeans=20)
    assert phase_limits('confirmation',[])==dict(pair_unit=0,optimizer=0,kmeans=0)
    with pytest.raises(ValueError):phase_tasks('confirmation')
    with pytest.raises(ValueError):phase_tasks('development',['tap_iron'])
    with pytest.raises(ValueError):phase_tasks('confirmation',['tap_iron','tap_iron'])


def test_failed_start_consumes_budget_and_cannot_be_retried(tmp_path):
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(pair_unit=0,optimizer=1,kmeans=0))
    with pytest.raises(RuntimeError):
        with ledger.event('optimizer',('failure',),{}):raise RuntimeError('Synthetic injected failure')
    assert ledger.inspect()['failed']['optimizer']==1
    with pytest.raises(FileExistsError):
        with ledger.event('optimizer',('failure',),{}):pass
    with pytest.raises(ValueError,match='exhausted'):
        with ledger.event('optimizer',('another',),{}):pass


@pytest.fixture
def paired(tmp_path):
    f,s=frame(),settings()
    training=f.iloc[:30].reset_index(drop=True)
    query=clean(f.iloc[30:]).reset_index(drop=True)
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(pair_unit=1,optimizer=6,kmeans=2))
    task=dict(target='tap_iron',seed=42,fold=0)
    output=tmp_path/'unit'
    anchor=execute_unit(task,training,query,s,output,ledger.root,ledger.policy_sha256)
    return task,training,query,s,output,ledger,anchor


def test_complete_pair_independent_numpy_cold_audit_and_exact_accounting(paired):
    task,training,query,s,output,ledger,anchor=paired
    before=ledger.inspect()
    assert before['started']==before['completed']==dict(pair_unit=1,optimizer=6,kmeans=2)
    assert not any(before['failed'].values()) and not any(before['incomplete'].values())
    pred,report=audit_unit(output,task,training,query,s,expected_sha256=anchor['complete_sha256'],ledger_root=ledger.root)
    assert report['models_checked']==6 and report['maximum_difference']<1e-10
    assert report['new_optimizer_calls']==report['new_kmeans_calls']==0
    assert ledger.inspect()==before
    assert set(pred)=={'CONTROL','PTARL_AUX'}
    meta=json.loads((output/'complete.json').read_text())['metadata']
    assert meta['teacher_outer']['fit_rows']==30
    assert meta['teacher_inner']['fit_rows']==24
    assert meta['teacher_outer']['selected_epoch']==meta['teacher_inner']['selected_epoch']
    assert all(meta[f'{a}_refit']['selected_epoch']==meta[f'{a}_selector']['selected_epoch']
               for a in ['CONTROL','PTARL_AUX'])
    assert meta['CONTROL_selector']['prototype_initialization']==meta['PTARL_AUX_selector']['prototype_initialization']
    assert meta['CONTROL_refit']['prototype_initialization']!=meta['CONTROL_selector']['prototype_initialization']
    with pytest.raises(FileExistsError):
        execute_unit(task,training,query,s,output,ledger.root,ledger.policy_sha256)


def test_independent_oracle_rejects_labels_and_corrupt_cluster_witness(paired):
    _,training,query,_,output,_,_=paired
    teacher=torch.load(output/'teacher_outer.pt',weights_only=True)
    arm=torch.load(output/'PTARL_AUX_refit.pt',weights_only=True)
    receipt=deepcopy(arm['metadata']['prototype_initialization'])
    centers=arm['state']['initial_prototypes'].numpy()
    hidden=verify_bank(teacher,training,receipt,centers)
    assert hidden.shape==(30,16)
    receipt['cluster_labels'][0]=(receipt['cluster_labels'][0]+1)%3
    with pytest.raises(ValueError,match='membership'):
        verify_bank(teacher,training,receipt,centers)
    with pytest.raises(ValueError,match='labels'):
        oracle(teacher,training)


@pytest.mark.parametrize('mutation',['preprocessing','updates','epochs','parameters','prototypes'])
def test_independent_saved_model_audit_detects_reanchored_corruption(paired,mutation,tmp_path):
    _,training,_,s,output,_,_=paired
    fitting,_,_=inner_parts(training)
    data=torch.load(output/'PTARL_AUX_selector.pt',weights_only=True)
    if mutation=='preprocessing':data['metadata']['preprocessing']['means'][0]+=1
    if mutation=='updates':data['metadata']['auxiliary_updates']+=1
    if mutation=='epochs':data['metadata']['selected_epoch']=0
    if mutation=='parameters':data['state']['native.output.weight']=data['state']['native.output.weight'].float()
    if mutation=='prototypes':data['metadata']['prototype_initialization']['fit_ids_digest']='wrong'
    path=tmp_path/f'{mutation}.pt'
    with path.open('xb') as stream:torch.save(data,stream)
    with pytest.raises(ValueError):
        verify_model(path,fitting,fitting.tap_iron,'PTARL_AUX',s,file_hash(path),selector=True)


def test_outer_overlap_failure_is_preserved_and_does_not_start_optimizer(tmp_path):
    f,s=frame(),settings()
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(pair_unit=1,optimizer=6,kmeans=2))
    task=dict(target='tap_iron',seed=42,fold=0)
    with pytest.raises(ValueError,match='overlap'):
        execute_unit(task,f.iloc[:30],clean(f.iloc[20:]),s,tmp_path/'failed',ledger.root,ledger.policy_sha256)
    assert (tmp_path/'failed').is_dir()
    counts=ledger.inspect()
    assert counts['failed']['pair_unit']==1 and counts['started']['optimizer']==0


def test_complete_unit_rejects_changed_original_query_features(paired):
    task,training,query,s,output,ledger,anchor=paired
    query=query.copy();query.iloc[0,0]+=1
    with pytest.raises(ValueError,match='identity'):
        audit_unit(output,task,training,query,s,expected_sha256=anchor['complete_sha256'],ledger_root=ledger.root)


def test_unit_cannot_claim_a_different_ledger(paired,tmp_path):
    task,training,query,s,output,ledger,anchor=paired
    another=ReservationLedger.create(tmp_path/'another',ledger.limits)
    # Policy bytes deliberately identical; the event identity is still binding.
    assert another.policy_sha256==ledger.policy_sha256
    with pytest.raises(ValueError,match='Missing paired reservations'):
        audit_unit(output,task,training,query,s,expected_sha256=anchor['complete_sha256'],ledger_root=another.root)


def test_numpy_oracle_matches_latents_and_nontrivial_unknown_category_predictions(paired,tmp_path):
    _,_,query,_,output,_,_=paired
    data=torch.load(output/'PTARL_AUX_refit.pt',weights_only=True)
    data['state']['native.output.weight']*=1000
    data['state']['native.output.bias'][:]=torch.tensor([[-5.],[2.],[8.],[-3.]])
    path=tmp_path/'amplified.pt'
    with path.open('xb') as stream:torch.save(data,stream)
    model=PrototypeRegressor.load(path,file_hash(path))
    query=query.iloc[::-1].copy();query['spout_no']=999
    independent,hidden=oracle(data,query)
    with torch.no_grad():_,native_hidden=latent_forward(model.model_.native,*model._inputs(query))
    np.testing.assert_allclose(hidden,native_hidden.numpy(),atol=1e-12,rtol=0)
    np.testing.assert_allclose(independent,model.predict(query),atol=1e-10,rtol=0)
    assert np.ptp(independent)>0.1
