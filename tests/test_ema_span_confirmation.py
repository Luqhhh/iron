"""Closed OOF, paired-seed gates and serial dispatch; no official data fitting."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bf_tap_r2.ema_span_confirmation import (
    DEVELOPMENT,CONFIRMATION,evaluate,validate_scope,warm_worker,execute,verify_cold_composition,
)
from bf_tap_r2.ema_span_confirmation_audit import independent_gate
import bf_tap_r2.ema_span_confirmation as module
from test_ema_reference_capture import frames

REPO=Path(__file__).resolve().parents[1]


def configuration():return json.loads((REPO/'configs/ema_span_confirmation/SPEC.json').read_text())


def evidence(gains=None):
    frame,_=frames();seeds=[*DEVELOPMENT,*CONFIRMATION]
    folds={s:np.arange(len(frame))%5 for s in seeds};vectors={}
    improvements=gains or [.01,.011,.012,.013]
    for seed,gain in zip(seeds,improvements):
        q=frame.tap_time_len.to_numpy()+2
        reduction=gain*np.abs(frame.tap_time_len.to_numpy()).sum()/50/len(frame)
        vectors[seed]=dict(q75=q,candidate=q-reduction,iron=frame.tap_iron.to_numpy()+3)
    policy=yaml.safe_load((REPO/'configs/candidate_tiers.yaml').read_text())
    return frame,folds,vectors,policy


def test_frozen_scope_preserves_original_trainer_reference_weight_and_native_budget():
    spec=configuration();original=yaml.safe_load((REPO/'configs/strong_component_regularization/SPEC.yaml').read_text())
    validate_scope(spec,original)
    for key,value in [('candidate_beta',.99),('replacement_weight',.5),('workers',True),
            ('confirmation_seeds',[1,2]),('maximum_runtime_seconds',3600),('packages',1)]:
        changed=deepcopy(spec);changed[key]=value
        with pytest.raises(ValueError,match='scope'):validate_scope(changed,original)
    changed=deepcopy(spec);changed['training']['max_epochs']+=1
    with pytest.raises(ValueError,match='trainer'):validate_scope(changed,original)
    changed=deepcopy(spec);changed['native_budget']['torch_optimizer']=101
    with pytest.raises(ValueError,match='budget'):validate_scope(changed,original)
    changed=deepcopy(spec);changed['reference_spec']['reference']['time_weights']['v36']=.21
    with pytest.raises(ValueError,match='reference'):validate_scope(changed,original)


def test_four_complete_seed_statistics_match_independent_fsum_student_t_gate():
    frame,folds,vectors,policy=evidence()
    summary=evaluate(frame,folds,vectors,policy)
    audit=independent_gate(frame,folds,vectors,summary)
    assert audit['four_seed_gate_passed'] and summary['paired']['n']==4 and summary['paired']['positive']==4
    assert summary['paired']['lcb95']>0 and summary['release_authorized'] is False
    assert audit['maximum_arithmetic_difference']<1e-9
    assert set(summary['tiers'])=={'development','confirmation_descriptive'}


@pytest.mark.parametrize('gains',[[.01,.011,-.001,.013],[.00001,.00001,.00001,.1],[0.,0.,0.,0.]])
def test_any_nonpositive_seed_or_nonpositive_seed_lcb_fails(gains):
    frame,folds,vectors,policy=evidence(gains)
    summary=evaluate(frame,folds,vectors,policy)
    assert not summary['four_seed_gate_passed']
    assert not independent_gate(frame,folds,vectors,summary)['four_seed_gate_passed']


@pytest.mark.parametrize('change',['missing-seed','missing-row','nonfinite','negative'])
def test_partial_or_invalid_oof_never_reaches_confirmation_decision(change):
    frame,folds,vectors,policy=evidence()
    if change=='missing-seed':del vectors[314159]
    if change=='missing-row':vectors[314159]['candidate']=vectors[314159]['candidate'][:-1]
    if change=='nonfinite':vectors[314159]['candidate'][0]=np.nan
    if change=='negative':vectors[314159]['candidate'][0]=-1
    with pytest.raises(ValueError):evaluate(frame,folds,vectors,policy)


def test_independent_gate_rejects_summary_gain_or_promoted_flag_tampering():
    frame,folds,vectors,policy=evidence();summary=evaluate(frame,folds,vectors,policy)
    changed=deepcopy(summary);changed['gains']['271828']+=.1
    with pytest.raises(ValueError,match='arithmetic mismatch'):independent_gate(frame,folds,vectors,changed)
    changed=deepcopy(summary);changed['four_seed_gate_passed']=False
    with pytest.raises(ValueError,match='gate mismatch'):independent_gate(frame,folds,vectors,changed)


def test_warm_worker_cannot_start_before_manifest_activation(tmp_path,monkeypatch):
    spec=configuration();frame,_=frames()
    manifest=dict(spec=spec,partitions={})
    monkeypatch.setattr(module,'manifest_context',lambda *a,**k:(manifest,frame,{271828:np.arange(len(frame))%5}))
    with pytest.raises(FileNotFoundError):warm_worker(REPO,tmp_path,271828,0)
    assert not (tmp_path/'s271828-f0').exists()


def test_warm_worker_keeps_outer_labels_out_and_runs_exact_reference_control_short_order(tmp_path,monkeypatch):
    import bf_tap_r2.v30_reference as reference
    from bf_tap_r2.ema_reference_artifacts import sha
    spec=configuration();frame,_=frames();fv=np.arange(len(frame))%5
    training,query=module.task_frames(frame,fv,0)
    expected=module.partition(training,query,spec['training'])
    changed=frame.copy();changed.loc[fv==0,['tap_iron','tap_time_len']]=-1e12
    manifest=dict(spec=spec,partitions={'s271828-f0':expected},reference_plan={},model_sources={})
    (tmp_path/'manifest.json').write_text('synthetic frozen source')
    (tmp_path/'activation.json').write_text(json.dumps({'manifest_sha256':sha(tmp_path/'manifest.json')}))
    monkeypatch.setattr(module,'manifest_context',lambda *a,**k:(manifest,changed,{271828:fv}))
    monkeypatch.setattr(module,'runtime',lambda *a,**k:None);monkeypatch.setattr(module,'require_memory',lambda *a:None)
    monkeypatch.setattr(module.resource,'getrusage',lambda *a:SimpleNamespace(ru_maxrss=1024))
    seen=[]
    def valid(tr,va):
        assert tr.sample_id.tolist()==training.sample_id.tolist()
        assert tr.tap_time_len.tolist()==training.tap_time_len.tolist()
        assert not {'tap_iron','tap_time_len'}&set(va.columns)
    class Capture:
        def __init__(self,path,**kwargs):
            valid(kwargs['training'],kwargs['query']);self.path=path;path.mkdir()
        def close(self):(self.path/'complete.json').write_text('{}')
    from contextlib import nullcontext
    monkeypatch.setattr(module,'ReferenceCapture',Capture);monkeypatch.setattr(module,'capture_original_factory',lambda *a:nullcontext())
    def baseline(root,tr,va,refspec):
        valid(tr,va);seen.append('reference');assert refspec==spec['reference_spec']
        n=len(va);return dict(tap_iron=np.full(n,500.),tap_time_len=np.full(n,96.),
            v36_iron=np.full(n,500.),v36_time=np.full(n,96.),v12_iron=np.full(n,500.),
            n_time=np.full(n,96.),v7_time=np.full(n,98.)),{}
    monkeypatch.setattr(reference,'fit_b0',baseline)
    def component(path,tr,va,**kwargs):
        valid(tr,va);name=kwargs['identity']['trial_id'];seen.append(name)
        assert kwargs['settings']==spec['training']
        assert kwargs['mechanisms']['ema_beta']==(.99 if name=='OLD_EMA' else .9801)
        path.mkdir();(path/'complete.json').write_text('{}')
        return np.full(len(va),100. if name=='OLD_EMA' else 101.),{}
    monkeypatch.setattr(module,'fit_component',component)
    warm_worker(REPO,tmp_path,271828,0)
    assert seen==['reference','OLD_EMA','SHORT_SPAN']
    with np.load(tmp_path/'s271828-f0/predictions.npz') as saved:
        np.testing.assert_array_equal(saved['q75'],np.full(len(query),97.5))
        np.testing.assert_array_equal(saved['candidate'],np.full(len(query),98.25))
        assert saved['query_ids'].tolist()==query.sample_id.tolist()
    with pytest.raises(FileExistsError):warm_worker(REPO,tmp_path,271828,0)
    assert len(seen)==3


def test_execute_does_all_warm_and_independent_cold_units_before_any_metrics(tmp_path,monkeypatch):
    frame,folds,vectors,policy=evidence();spec=configuration();spec['main_root']=str(tmp_path)
    (tmp_path/'EVIDENCE_STATUS.json').write_text(json.dumps({'round2_current_platform_best':spec['platform_reference']}))
    (tmp_path/'manifest.json').write_text('frozen synthetic manifest')
    manifest=dict(spec=spec);monkeypatch.setattr(module,'manifest_context',lambda *a,**k:(manifest,frame,folds))
    calls=[]
    def run(command,check):
        command=list(map(str,command));mode=next(x for x in ['worker','cold','audit'] if x in command)
        calls.append(mode)
        if mode=='worker':
            s=int(command[command.index('--seed')+1]);f=int(command[command.index('--fold')+1])
            d=tmp_path/f's{s}-f{f}';d.mkdir();(d/'warm-complete.json').write_text('{}')
        if mode=='audit':(tmp_path/'audit.json').write_text('{}')
    monkeypatch.setattr(module.subprocess,'run',run)
    def collected(*args):
        assert calls==['worker','cold']*10
        return vectors
    monkeypatch.setattr(module,'collect',collected)
    execute(REPO,tmp_path)
    assert calls==['worker','cold']*10+['audit']
    assert (tmp_path/'completion-event.json').exists()
    assert json.loads((tmp_path/'summary.json').read_text())['four_seed_gate_passed']


def test_execute_never_retries_a_failed_numerical_worker(tmp_path,monkeypatch):
    spec=configuration();spec['main_root']=str(tmp_path)
    (tmp_path/'EVIDENCE_STATUS.json').write_text(json.dumps({'round2_current_platform_best':spec['platform_reference']}))
    (tmp_path/'manifest.json').write_text('frozen')
    monkeypatch.setattr(module,'manifest_context',lambda *a,**k:(dict(spec=spec),None,None));calls=[]
    def fail(command,check):calls.append(command);raise RuntimeError('numerical fixture failure')
    monkeypatch.setattr(module.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='numerical fixture failure'):execute(REPO,tmp_path)
    assert len(calls)==1 and (tmp_path/'failure.json').exists() and not (tmp_path/'completion-event.json').exists()


def test_original_b36_projection_is_separate_from_affine_candidate_arithmetic(tmp_path):
    # A synthetic frozen graph tests the existing B36 projection, then the
    # unconstrained Q75 replacement. No projection is added to the candidate.
    graph=dict(slots=[dict(name='BASE',column=0,outputs=1),dict(name='EXPERT',column=0,outputs=1),
        dict(name='V12_joint',column=0,outputs=2),dict(name='V12_joint',column=1,outputs=2),
        dict(name='N0048',column=0,outputs=1),dict(name='V7_periodic',column=0,outputs=1)],
        a_coefficients={'tap_iron':[1,0,0,0,0,0],'tap_time_len':[1,0,0,0,0,0]},
        b_weights={'tap_iron':[1,0],'tap_time_len':[1,0]},b_experts={'tap_iron':['EXPERT'],'tap_time_len':['EXPERT']},
        existing_b36_nonnegative_projection=True)
    store=dict(BASE=np.array([-2.,3.]),EXPERT=np.array([5.,5.]),V12_joint=np.array([[10.,20.],[11.,21.]]),
        N0048=np.array([8.,9.]),V7_periodic=np.array([7.,8.]))
    for name,value in store.items():
        d=tmp_path/'reference'/name/'final';d.mkdir(parents=True);np.save(d/'observed.npy',value)
    for name,value in [('OLD_EMA',np.array([[7.],[8.]])),('SHORT_SPAN',np.array([[8.],[9.]]))]:
        d=tmp_path/name/'refit-witness';d.mkdir(parents=True);np.save(d/'observed.npy',value)
    b=np.maximum(store['BASE'],0);v=dict(v36_iron=b,v36_time=b,v12_iron=store['V12_joint'][:,0],
        n_time=store['N0048'],v7_time=store['V7_periodic'],old_ema=np.array([7.,8.]),short_ema=np.array([8.,9.]))
    v['tap_iron']=.5*b+.5*v['v12_iron'];v['iron']=v['tap_iron'];v['tap_time_len']=.2*b+.3*v['n_time']+.5*v['v7_time']
    v['q75']=v['tap_time_len'];v['candidate']=v['q75']+.75
    np.savez(tmp_path/'predictions.npz',**v)
    assert verify_cold_composition(tmp_path,graph)==0
    v['candidate'][0]+=1;np.savez(tmp_path/'predictions.npz',**v)
    with pytest.raises(ValueError,match='arithmetic mismatch'):verify_cold_composition(tmp_path,graph)
