"""Conditional release boundaries and independent artifact contracts."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from test_ema_span_confirmation_models import fixture_backend
from test_ema_reference_capture import frames

ROOT=Path(__file__).resolve().parents[1]
loader=importlib.util.spec_from_file_location('ema_short_release',ROOT/'scripts/ema_short_release.py')
m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)


def cfg():return json.loads((ROOT/m.SPEC).read_text())

def confirmation():return json.loads((ROOT/'configs/ema_span_confirmation/SPEC.json').read_text())

def payload(iron='20.000000000',time='10.000'):
    ids=[f'S{i:03d}' for i in range(322)]
    return ('sample_id,pred_tap_iron,pred_tap_time_len\n'+''.join(f'{s},{iron},{time}\n' for s in ids)).encode(),ids


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_fixed_affine_time_replacement_retains_original_iron_strings():
    from bf_tap_r2.submission import validate_result
    parent,ids=payload()
    result=m.release_payload(parent,ids,np.full(322,8.),np.full(322,12.))
    rows=validate_result(result,ids)
    assert all(r['pred_tap_iron']=='20.000000000' for r in rows)
    assert all(float(r['pred_tap_time_len'])==13. for r in rows)
    assert 13. != .25*10.+.75*12.


@pytest.mark.parametrize('old,new',[([1.],[2.]),(np.ones(322),np.full(322,np.nan)),
    (np.ones(322),np.full(322,np.inf)),(np.full(322,20.),np.zeros(322))])
def test_invalid_component_or_affine_values_refused_without_clipping(old,new):
    parent,ids=payload()
    with pytest.raises(ValueError):m.release_payload(parent,ids,old,new)


def test_parent_id_order_cannot_be_changed():
    parent,ids=payload()
    with pytest.raises(ValueError):m.release_payload(parent,ids[::-1],np.ones(322),np.ones(322))


@pytest.mark.parametrize('key,value',[('replacement_weight',.5),('candidate_beta',.99),
    ('full_training_procedures',True),('new_CV',1),('packages',2),('desktop_writes',1),
    ('require_four_seed_gate',False),('maximum_runtime_seconds',600)])
def test_changed_scope_or_loosened_release_gate_refused(key,value):
    s=cfg();s[key]=value
    with pytest.raises(ValueError,match='scope'):m.validate_scope(s,confirmation())


def test_original_scientific_recipe_is_required():
    s=cfg();s['training']['max_epochs']=241
    with pytest.raises(ValueError,match='scientific'):m.validate_scope(s,confirmation())
    m.validate_scope(cfg(),confirmation())


def test_no_authorization_is_a_preparation_state_and_grants_must_be_explicit(tmp_path):
    assert m.explicit_authority(None) is None
    path=tmp_path/'grant.json'
    grant=dict(source='explicit_user_task',scope=m.AUTH_SCOPE,user_authorized=True,user_request='explicit fixture only')
    save(path,grant);assert m.explicit_authority(path)['sha256']==m.sha(path)
    for key,value in [('source','inferred_goal'),('user_request',''),('user_authorized',1)]:
        bad=deepcopy(grant);bad[key]=value;save(path,bad)
        with pytest.raises(ValueError,match='authorization'):m.explicit_authority(path)
    bad=deepcopy(grant);bad['scope']['full_training_procedures']=True;save(path,bad)
    with pytest.raises(ValueError):m.explicit_authority(path)


def test_engineering_manifest_cannot_reach_execution_or_observation(tmp_path,monkeypatch):
    manifest=dict(spec=cfg(),scientific_execution_admitted=False,authorization=None)
    monkeypatch.setattr(m,'context',lambda *a,**k:manifest)
    calls=[];monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(ValueError,match='Engineering preparation'):m.execute(tmp_path)
    with pytest.raises(ValueError,match='Engineering preparation'):m.observe(tmp_path)
    assert not calls and not (tmp_path/'activation.json').exists()


def completed_confirmation(tmp_path):
    out=tmp_path/'confirmation';out.mkdir()
    save(out/'manifest.json',{'identity':'synthetic_terminal_fixture'})
    summary={'four_seed_gate_passed':True,'paired':{'lcb95':.001}}
    save(out/'summary.json',summary)
    audit={'status':'passed','four_seed_gate_passed':True,'complete_units':10,'new_retained_states':440,
        'manifest_sha256':m.sha(out/'manifest.json'),'summary_sha256':m.sha(out/'summary.json'),
        'seed_gains':{'42':.002,'3407':.003,'271828':.002,'314159':.003}}
    save(out/'audit.json',audit)
    save(out/'completion-event.json',dict(status='completed',summary_sha256=m.sha(out/'summary.json'),audit_sha256=m.sha(out/'audit.json')))
    save(out/'process-terminal.json',{'exit_code':0})
    proof=out/'terminal-verification-r1.json'
    save(proof,dict(status='passed',manifest_sha256=m.sha(out/'manifest.json'),audit_sha256=m.sha(out/'audit.json'),
        summary_sha256=m.sha(out/'summary.json'),process_terminal_sha256=m.sha(out/'process-terminal.json'),
        completion_sha256=m.sha(out/'completion-event.json')))
    return dict(confirmation_run=str(out),confirmation_terminal_verification=str(proof))


def test_complete_four_seed_actual_terminal_is_required(tmp_path):
    s=completed_confirmation(tmp_path)
    assert len(m.confirmation_evidence(s))==6
    out=Path(s['confirmation_run']);a=json.loads((out/'audit.json').read_text());a['complete_units']=9
    save(out/'audit.json',a)
    with pytest.raises(ValueError):m.confirmation_evidence(s)


@pytest.mark.parametrize('field,value',[('four_seed_gate_passed',False),('new_retained_states',439)])
def test_audit_state_counts_and_quality_not_inferred_from_success_label(tmp_path,field,value):
    s=completed_confirmation(tmp_path);out=Path(s['confirmation_run'])
    a=json.loads((out/'audit.json').read_text());a[field]=value;save(out/'audit.json',a)
    with pytest.raises(ValueError):m.confirmation_evidence(s)


def test_failed_actual_controller_cannot_be_released(tmp_path):
    s=completed_confirmation(tmp_path);save(Path(s['confirmation_run'])/'process-terminal.json',{'exit_code':1})
    with pytest.raises(ValueError):m.confirmation_evidence(s)


def test_archive_source_hash_member_crc_and_ids_required(tmp_path):
    parent,ids=payload();path=tmp_path/'parent.zip'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('result.csv',parent)
    assert m.zip_payload(path,m.sha(path),ids)==parent
    with pytest.raises(ValueError):m.zip_payload(path,'0'*64,ids)
    with pytest.raises(ValueError):m.zip_payload(path,m.sha(path),ids[::-1])
    with zipfile.ZipFile(path,'a') as archive:archive.writestr('unrelated.txt','extra')
    with pytest.raises(ValueError,match='member'):m.zip_payload(path,m.sha(path),ids)


def test_controller_serial_order_binding_and_no_reexecution(tmp_path,monkeypatch):
    grant=tmp_path/'grant.json';save(grant,dict(source='explicit_user_task',scope=m.AUTH_SCOPE,user_authorized=True,user_request='explicit fixture only'))
    save(tmp_path/'manifest.json',{'fixture':'no numerical fits'})
    manifest=dict(spec=cfg(),authorization={'path':str(grant)},scientific_execution_admitted=True,workspace=str(ROOT))
    monkeypatch.setattr(m,'context',lambda *a,**k:manifest)
    monkeypatch.setattr(m,'confirmation_evidence',lambda *a:{})
    monkeypatch.setattr(m,'current_reference',lambda *a:None)
    monkeypatch.setattr(m,'require_memory',lambda *a:3072)
    modes=[]
    def child(args,**kwargs):
        mode=args[4];modes.append(mode)
        if mode=='warm':save(tmp_path/'warm-complete.json',{'fixture':mode})
        if mode=='build':save(tmp_path/'release.json',{'fixture':mode})
        if mode=='verify':save(tmp_path/'package-audit.json',{'fixture':mode})
        if mode in ('cold','verify'):
            assert args[5]=='--receipt'
            target='warm-complete.json' if mode=='cold' else 'release.json'
            assert args[6]==m.sha(tmp_path/target)
    monkeypatch.setattr(m.subprocess,'run',child)
    m.execute(tmp_path)
    assert modes==['warm','cold','audit','build','verify']
    assert json.loads((tmp_path/'completion-event.json').read_text())['status']=='completed'
    with pytest.raises(FileExistsError):m.execute(tmp_path)
    assert len(modes)==5


def test_failed_stage_stops_without_retry_and_preserves_activation(tmp_path,monkeypatch):
    grant=tmp_path/'grant.json';save(grant,dict(source='explicit_user_task',scope=m.AUTH_SCOPE,user_authorized=True,user_request='explicit fixture only'))
    save(tmp_path/'manifest.json',{'fixture':'no fits'})
    manifest=dict(spec=cfg(),authorization={'path':str(grant)},scientific_execution_admitted=True,workspace=str(ROOT))
    monkeypatch.setattr(m,'context',lambda *a,**k:manifest)
    monkeypatch.setattr(m,'confirmation_evidence',lambda *a:{})
    monkeypatch.setattr(m,'current_reference',lambda *a:None)
    monkeypatch.setattr(m,'require_memory',lambda *a:3072)
    calls=[]
    def fail(args,**kwargs):calls.append(args);raise RuntimeError('fixture preserved failure')
    monkeypatch.setattr(m.subprocess,'run',fail)
    with pytest.raises(RuntimeError):m.execute(tmp_path)
    assert len(calls)==1 and (tmp_path/'activation.json').exists()
    assert not (tmp_path/'completion-event.json').exists()
    assert json.loads((tmp_path/'failure.json').read_text())['automatic_retry'] is False


def test_real_original_method_bridge_uses_two_calls_and_declared_full_identity(tmp_path,monkeypatch,fixture_backend):
    from bf_tap_r2.component_regularization import ComponentRegressor
    training,query=frames();s=cfg();out=tmp_path/'synthetic-full';out.mkdir()
    query=query.copy();query['air_volume']+=100.
    save(out/'manifest.json',{'fixture':'synthetic labels; NoUpdateOptimizer; no official data'})
    source=m.sources(ROOT)
    import bf_tap_r2.ema_span_confirmation_models as bridge
    model_sources={str(ROOT/p):h for p,h in source.items()}
    model_sources.update(m.binding_sources(bridge.reference_bindings()))
    manifest=dict(spec=s,workspace=str(ROOT),sources=source,
        model_sources=model_sources,
        partition=m.partition(training,query,s['training']))
    monkeypatch.setattr(m,'activated_context',lambda *a,**k:manifest)
    monkeypatch.setattr(m,'official_frames',lambda *a:(training,query))
    monkeypatch.setattr(m,'peak',lambda *a:100.)
    original_fit=ComponentRegressor.fit;original_train=ComponentRegressor._train
    prediction=m.warm_worker(out)
    unit=out/'SHORT_SPAN_FULL';receipt=json.loads((unit/'complete.json').read_text())
    assert receipt['native_counts']['torch_optimizer']==2
    assert receipt['identity']==dict(source_directory=str(out.resolve()),split_seed=-1,fold=-1,trial_id='SHORT_SPAN_FULL')
    assert receipt['model_metadata']['selected_epoch']==1
    assert receipt['checkpoints']['refit']['trace']['selected_epoch']==1
    assert prediction.shape==(len(query),)
    assert ComponentRegressor.fit is original_fit and ComponentRegressor._train is original_train
    assert json.loads((out/'warm-complete.json').read_text())['new_CV']==0
    with pytest.raises(FileExistsError):m.warm_worker(out)
