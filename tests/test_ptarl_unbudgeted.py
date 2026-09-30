from copy import deepcopy
from concurrent.futures import Future
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from bf_tap_r2.data import TARGETS
from bf_tap_r2 import ptarl_unbudgeted_run as ptarl_run
from bf_tap_r2 import ptarl_unbudgeted_arithmetic as ptarl_arithmetic
from bf_tap_r2 import ptarl_unbudgeted_freeze as ptarl_freeze
from bf_tap_r2.ptarl_phase import collect_audited_phase
from bf_tap_r2.ptarl_protocol import file_hash, write_new
from bf_tap_r2.ptarl_scoring import score_seed, decide_development, decide_confirmation
from bf_tap_r2.ptarl_arithmetic import direct_decision
from bf_tap_r2.ptarl_unbudgeted_controller import sequence
from bf_tap_r2.ptarl_preflight import resource_decision, ROLES
from test_ptarl_model import frame, settings

WORKSPACE=Path(__file__).resolve().parents[1]


def records(seeds=(42,3407)):
    return [dict(target=t,arm=a,seed=s,gain=.02 if a=='PTARL_AUX' else .005,
                 candidate_score=96.3,blend_weight=.2,other_target_unchanged=True)
            for t in TARGETS for a in ('CONTROL','PTARL_AUX') for s in seeds]


def test_score_is_fixed_isolated_twenty_percent_and_keeps_other_target():
    n=25;y={t:np.linspace(10,30,n) for t in TARGETS}
    current={t:y[t]+1 for t in TARGETS};historical={t:y[t]+2 for t in TARGETS}
    folds=np.arange(n)%5
    members={'CONTROL':y['tap_time_len']+1,'PTARL_AUX':y['tap_time_len']}
    scored=score_seed(y,folds,current,historical,members,target='tap_time_len',seed=42)
    row=scored[1];expected_gain=50*.2*n/np.abs(y['tap_time_len']).sum()
    assert row['gain']==pytest.approx(expected_gain)
    assert row['candidate_score']==pytest.approx(100-50*(.8+1)*n/y['tap_time_len'].sum())
    assert row['other_target_unchanged'] and row['blend_weight']==.2
    assert row['historical_target_gain']==pytest.approx(50*1.2*n/y['tap_time_len'].sum())
    with pytest.raises(ValueError,match='five-fold'):
        score_seed(y,np.zeros(n,dtype=int),current,historical,members,target='tap_time_len',seed=42)


@pytest.mark.parametrize('failure',['one_seed','mean','control','score'])
def test_all_development_conditions_are_binding_and_match_independent_oracle(failure):
    rows=records();selected=next(r for r in rows if r['target']=='tap_time_len' and r['arm']=='PTARL_AUX' and r['seed']==42)
    if failure=='one_seed':selected['gain']=-.001
    elif failure=='mean':selected['gain']=.0001; next(r for r in rows if r['target']=='tap_time_len' and r['arm']=='PTARL_AUX' and r['seed']==3407)['gain']=.001
    elif failure=='control':
        for r in rows:
            if r['target']=='tap_time_len' and r['arm']=='CONTROL':r['gain']=.1
    else:selected['candidate_score']=96.1
    production=decide_development(rows)
    independent,_=direct_decision(rows,'development')
    assert independent==production['eligible_targets']==['tap_iron']


def test_confirmation_uses_four_seed_lcb_and_folds_are_descriptive_only():
    dev=records();confirmation=records((7777,12011))
    gains=[.02,.03,.01,.025]
    for seed,gain in zip((42,3407,7777,12011),gains,strict=True):
        for r in dev+confirmation:
            if r['arm']=='PTARL_AUX' and r['seed']==seed:r['gain']=gain;r['folds_descriptive']=[dict(gain=-1)]*5
    actual=decide_confirmation(dev,confirmation)
    selected,details=direct_decision(dev+confirmation,'confirmation',list(TARGETS))
    assert selected==actual['promoted_targets']==list(TARGETS)
    expected=np.mean(gains)-2.3533634348018264*np.std(gains,ddof=1)/2
    assert actual['decisions']['tap_iron']['seed_lcb95']==pytest.approx(expected)
    assert details['tap_iron']['seed_lcb95']==pytest.approx(expected)
    for r in confirmation:
        if r['arm']=='PTARL_AUX' and r['seed']==7777:r['gain']=-.001
    assert decide_confirmation(dev,confirmation)['promoted_targets']==[]
    assert direct_decision(dev+confirmation,'confirmation',list(TARGETS))[0]==[]
    with pytest.raises(ValueError,match='coverage'):
        decide_confirmation(dev,confirmation[:-1])


def test_independent_arithmetic_rejects_missing_duplicate_extra_and_unearned_seeds():
    dev=records()
    for bad in [dev[:-1],dev+[dev[0]],dev+[dict(dev[0],seed=999)]]:
        with pytest.raises(ValueError):direct_decision(bad,'development')
    bad=records()+records((7777,12011))
    next(r for r in bad if r['target']=='tap_iron' and r['arm']=='PTARL_AUX' and r['seed']==42)['gain']=-.01
    with pytest.raises(ValueError,match='Unqualified'):
        direct_decision(bad,'confirmation',list(TARGETS))


def test_controller_never_confirms_failed_development_or_audit():
    called=[]
    def invoke(name,module,args):
        called.append(name)
        if name.endswith('run'):return dict(complete_sha256='complete')
        if name.endswith('audit'):return dict(audit_sha256='audit')
        return dict(arithmetic_sha256='arithmetic',selected_targets=[])
    assert sequence(invoke)['confirmation_skipped']=='no_development_finalist'
    assert called==['development-run','development-audit','development-arithmetic']
    called.clear()
    def failure(name,module,args):
        called.append(name)
        if name.endswith('audit'):raise RuntimeError('Injected audit failure')
        return dict(complete_sha256='complete')
    with pytest.raises(RuntimeError):sequence(failure)
    assert called==['development-run','development-audit']


class InlinePool:
    def __init__(self,**kwargs):
        assert kwargs['max_workers']==4
        assert kwargs['mp_context'].get_start_method()=='spawn'
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def submit(self,fn,job):
        future=Future()
        try:future.set_result(fn(job))
        except BaseException as exc:future.set_exception(exc)
        return future


@pytest.fixture(scope='module')
def phase_files(tmp_path_factory):
    output=tmp_path_factory.mktemp('ptarl-complete-phase')
    f,s=frame(),settings()
    folds={seed:np.roll(np.arange(len(f))%5,i) for i,seed in enumerate((42,3407,7777,12011))}
    current={seed:{t:f[t].to_numpy()+.01 for t in TARGETS} for seed in folds}
    historical={seed:{t:f[t].to_numpy()+.02 for t in TARGETS} for seed in folds}
    manifest=dict(workspace=str(WORKSPACE),settings=s)
    monkeypatch=pytest.MonkeyPatch()
    monkeypatch.setattr(ptarl_run,'_context',lambda *_:(manifest,output))
    monkeypatch.setattr(ptarl_run,'verify_manifest',lambda *_args,**_kwargs:manifest)
    monkeypatch.setattr(ptarl_run,'reload_references',lambda _:(f,folds,current,historical,{}))
    monkeypatch.setattr(ptarl_run,'ProcessPoolExecutor',InlinePool)
    monkeypatch.setattr(ptarl_run,'require_serial_ready',lambda *_:None)
    monkeypatch.setattr(ptarl_arithmetic,'verify_manifest',lambda *_args,**_kwargs:manifest)
    monkeypatch.setattr(ptarl_arithmetic,'reload_references',lambda _:(f,folds,current,historical,{}))
    try:
        fit=ptarl_run.run_phase(output/'manifest.json','manifest','resource','development')
        audit=ptarl_run.audit_phase(output/'manifest.json','manifest','resource','development',fit['complete_sha256'])
        arithmetic=ptarl_arithmetic.check_phase(output/'manifest.json','manifest','development',audit['audit_sha256'])
        yield output,manifest,f,folds,current,historical,fit,audit,arithmetic
    finally:monkeypatch.undo()


def test_complete_twenty_unit_pipeline_cold_audit_arithmetic_and_zero_confirmation(phase_files):
    output,_,_,_,_,_,_,_,arithmetic=phase_files
    audit=json.loads((output/'development/audit.json').read_text())
    assert audit['counts']['started']==audit['counts']['completed']==dict(pair_unit=20,optimizer=120,kmeans=40)
    assert len(audit['audited_units'])==20 and sum(a['models_checked'] for a in audit['audited_units'])==120
    assert len(audit['records'])==8 and audit['new_audit_fits']==0
    assert audit['decision']['eligible_targets']==arithmetic['selected_targets']==[]
    assert all(r['other_target_unchanged'] for r in audit['records'])
    assert not (output/'confirmation').exists()


def test_incomplete_unit_pool_is_rejected_before_scoring(phase_files):
    output,m,f,folds,current,historical,_,_,_=phase_files
    complete=json.loads((output/'development/complete.json').read_text())
    anchors=deepcopy(complete['unit_anchors']);anchors.pop(next(iter(anchors)))
    with pytest.raises(ValueError,match='Incomplete'):
        collect_audited_phase(WORKSPACE,output/'development','development',f,folds,current,historical,
                              m['settings'],anchors,complete['ledger_policy_sha256'])


def test_phase_direct_arithmetic_detects_reanchored_lying_score(phase_files):
    output,_,_,_,_,_,_,audit,_=phase_files
    payload=json.loads((output/'development/audit.json').read_text())
    payload['records'][0]['gain']+=.01
    # Use a distinct phase root; failed evidence from the original stays intact.
    import shutil
    fake=output/'score-tamper';shutil.copytree(output/'development',fake/'development')
    for name in ['arithmetic.json','arithmetic-started.json']:(fake/'development'/name).unlink()
    (fake/'development/audit.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='differs from production'):
        ptarl_arithmetic.check_phase(fake/'manifest.json','manifest','development',file_hash(fake/'development/audit.json'))
    assert (fake/'development/arithmetic-failed.json').exists()


def measurement():
    return dict(train_rows=2204,query_rows=551,probe_max_epochs=32,models_checked=6,
        costs={r:dict(training_steps=224,training_seconds=2.24) for r in ROLES},
        mae={'CONTROL':.2,'PTARL_AUX':.3},median_mae=1.,peak_mib=500.,cold_difference=1e-10,kmeans_seconds=1.)


def test_resource_projection_uses_six_roles_full_240_ceiling_and_unchanged_gates():
    m=measurement();decision=resource_decision(m,4000)
    assert decision['status']=='passed'
    expected=20*(6*.01*9*240+1)/4*1.5+300
    assert decision['projected_development_seconds']==pytest.approx(expected)
    assert decision['required_available_mib']==3024
    assert resource_decision(m,3023)['checks']['available_memory'] is False
    m['costs']['PTARL_AUX_refit']['training_seconds']=2240
    assert resource_decision(m,4000)['checks']['development_cost'] is False


@pytest.mark.parametrize('failure',['shape','incomplete','step','quality','cold','peak'])
def test_resource_probe_fails_shape_budget_quality_cold_or_memory(failure):
    m=measurement()
    if failure=='shape':m['train_rows']=100
    if failure=='incomplete':m['costs'].pop('teacher_outer')
    if failure=='step':m['costs']['teacher_outer']['training_steps']=1000
    if failure in ('shape','incomplete','step'):
        with pytest.raises(ValueError):resource_decision(m,10000)
        return
    if failure=='quality':m['mae']['PTARL_AUX']=1.
    if failure=='cold':m['cold_difference']=1e-5
    if failure=='peak':m['peak_mib']=1537
    assert resource_decision(m,10000)['status']=='failed'


def test_full_spec_matches_gate_inner_resource_and_budget_implementation():
    spec=yaml.safe_load((WORKSPACE/ptarl_freeze.SPEC).read_text())
    assert ptarl_freeze.validate_spec(spec)==spec['training']
    for path,value in [(('calibration','split_seed'),27001),(('resources','projection_maximum_seconds'),100000),
                       (('future_formal_stage','development_gate','mean_gain_minimum'),.001)]:
        modified=deepcopy(spec);node=modified
        for part in path[:-1]:node=node[part]
        node[path[-1]]=value
        with pytest.raises(ValueError):ptarl_freeze.validate_spec(modified)


@pytest.mark.parametrize('training_seconds',[2240.,1e12])
def test_explicit_user_authority_accepts_cost_only_refusal_at_any_finite_duration(training_seconds):
    measured=measurement();measured['costs']['PTARL_AUX_refit']['training_seconds']=training_seconds
    decision=resource_decision(measured,10000.)
    assert decision['status']=='failed' and decision['checks']['development_cost'] is False
    limits=dict(pair_unit=1,optimizer=6,kmeans=2)
    report=dict(**decision,measurement=measured,official_fits=0,release_authorized=False,
        counts=dict(started=limits,completed=limits,failed={k:0 for k in limits},incomplete={k:0 for k in limits}))
    assert ptarl_freeze.non_time_decision(report)==dict(status='passed_non_time_requirements',maximum_runtime_seconds=None,new_fits=0)
    assert report['status']=='failed'


@pytest.mark.parametrize('defect',['cold','quality','peak','memory','incomplete','failed'])
def test_time_authority_does_not_bypass_non_time_requirements(defect):
    measured=measurement();measured['costs']['PTARL_AUX_refit']['training_seconds']=2240.
    if defect=='cold':measured['cold_difference']=1e-5
    elif defect=='quality':measured['mae']['PTARL_AUX']=1.
    elif defect=='peak':measured['peak_mib']=1537.
    decision=resource_decision(measured,1000. if defect=='memory' else 10000.)
    limits=dict(pair_unit=1,optimizer=6,kmeans=2)
    report=dict(**decision,measurement=measured,official_fits=0,release_authorized=False,
        counts=dict(started=limits,completed=dict(limits),failed={k:0 for k in limits},incomplete={k:0 for k in limits}))
    if defect in ('incomplete','failed'):report['counts'][defect]['optimizer']=1
    with pytest.raises(ValueError):ptarl_freeze.non_time_decision(report)


def test_pending_DANet_cannot_query_service_or_start_PTARL_training(tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setattr(ptarl_freeze,'DANET_ROOT',str(tmp_path))
    def no_poll(*args,**kwargs):raise AssertionError('Pending dependency polled between checkpoints')
    monkeypatch.setattr(subprocess,'check_output',no_poll)
    manifest=dict(serial_dependency=dict(root=str(tmp_path),manifest_sha256=ptarl_freeze.DANET_MANIFEST_SHA,service='iron-danet-formal-r1.service'))
    with pytest.raises(ValueError,match='terminal event'):ptarl_freeze.require_serial_ready(manifest)


def test_cold_audit_alone_never_consumes_confirmation_without_arithmetic(phase_files):
    output,_,_,_,_,_,_,audit,_=phase_files
    with pytest.raises(ValueError):ptarl_run.run_phase(output/'manifest.json','manifest','resource','confirmation',audit['audit_sha256'])
    assert not (output/'confirmation').exists()


def test_tampered_development_model_cannot_earn_confirmation_after_both_audits(phase_files):
    output,_,_,_,_,_,_,audit,arithmetic=phase_files
    import shutil
    fake=output/'changed-model';shutil.copytree(output/'development',fake/'development')
    model=next((fake/'development/units').rglob('*.pt'));model.write_bytes(b'changed model')
    with pytest.raises(ValueError,match='artifact changed'):
        ptarl_run.development_records(fake,'manifest',audit['audit_sha256'],arithmetic['arithmetic_sha256'])


def test_every_confirmation_stage_receives_independent_development_arithmetic_anchor():
    visited=[]
    def invoke(name,module,args):
        visited.append(name)
        if name in ('confirmation-run','confirmation-audit'):
            assert args[args.index('--development-arithmetic-sha256')+1]=='arithmetic'
        if name.endswith('run'):return dict(complete_sha256='complete')
        if name.endswith('audit'):return dict(audit_sha256='cold')
        return dict(arithmetic_sha256='arithmetic',selected_targets=['tap_iron'])
    sequence(invoke)
    assert len(visited)==6


def test_added_development_artifact_cannot_pass_arithmetic_or_earn_confirmation(phase_files):
    output,_,_,_,_,_,_,audit,arithmetic=phase_files
    import shutil
    fake=output/'extra-artifact';shutil.copytree(output/'development',fake/'development')
    (fake/'development/units/unrecorded.pt').write_bytes(b'extra model')
    with pytest.raises(ValueError,match='inventory changed'):
        ptarl_run.development_records(fake,'manifest',audit['audit_sha256'],arithmetic['arithmetic_sha256'])
    for name in ('arithmetic.json','arithmetic-started.json'):(fake/'development'/name).unlink()
    with pytest.raises(ValueError,match='inventory changed'):
        ptarl_arithmetic.check_phase(fake/'manifest.json','manifest','development',audit['audit_sha256'])
