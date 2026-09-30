"""Complete artificial coverage and append-only paired training provenance."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.modernnca_model import ARMS, NeighborRegressor, Settings, clean
from bf_tap_r2.modernnca_execution import execute_unit, audit_unit
from bf_tap_r2.modernnca_protocol import ReservationLedger, phase_limits, phase_tasks, task_name, write_new
from bf_tap_r2.modernnca_phase import collect_audited_phase
from bf_tap_r2.modernnca_scoring import decide_confirmation, decide_development
from bf_tap_r2.v7_periodic import file_hash


def artificial():
    rng = np.random.default_rng(64341); n = 100; x = rng.normal(size=(n,len(FEATURES)))
    f = pd.DataFrame(x,columns=FEATURES)
    f['sample_id'] = [f'synthetic-phase-{i}' for i in range(n)]
    f['spout_no'] = np.arange(n)%4+1
    f['tap_iron'] = 600+25*np.sin(x[:,0])+5*x[:,3]
    f['tap_time_len'] = 8+.4*np.sin(x[:,0])+.3*x[:,2]
    return f


@pytest.fixture(scope='module')
def phase(tmp_path_factory):
    f=artificial();root=tmp_path_factory.mktemp('modernnca-complete-phase')
    s=Settings(dim=4,frequencies=2,embedding=3,batch_size=16,max_epochs=2,patience=1)
    folds={42:np.arange(len(f))%5,3407:np.random.default_rng(876).permutation(np.arange(len(f))%5)}
    ledger=ReservationLedger.create(root/'ledger',phase_limits('development'));anchors={}
    for task in phase_tasks('development'):
        mask=folds[task['seed']]==task['fold']
        result=execute_unit(task,f.loc[~mask].reset_index(drop=True),clean(f.loc[mask].reset_index(drop=True)),
            s,root/'units'/task_name(task),ledger.root,ledger.policy_sha256)
        anchors[result['name']]=result['complete_sha256']
    current={seed:{t:f[t].to_numpy()+(.3 if t=='tap_time_len' else 5) for t in TARGETS} for seed in folds}
    parent={seed:{t:current[seed][t].copy() for t in TARGETS} for seed in folds}
    return root,f,folds,current,parent,s,ledger,anchors


def collect(phase):
    root,f,folds,current,parent,s,ledger,anchors=phase
    return collect_audited_phase(Path(__file__).resolve().parents[1],root,'development',f,folds,current,parent,s,anchors,ledger.policy_sha256)


def test_complete_twenty_pairs_have_exact_closed_80_estimator_40_optimizer_provenance(phase,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Phase cold audit attempted fit or initialization')
    monkeypatch.setattr(NeighborRegressor,'initialize',forbidden)
    monkeypatch.setattr(NeighborRegressor,'train',forbidden)
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    monkeypatch.setattr(torch.nn.Linear,'reset_parameters',forbidden)
    report=collect(phase)
    assert report['status']=='passed' and report['new_audit_fits']==0 and not report['release_authorized']
    assert report['counts']['started']==report['counts']['completed']==dict(pair_unit=20,estimator=80,optimizer=40)
    assert all(v==0 for state in ('failed','incomplete') for v in report['counts'][state].values())
    assert len(report['audited_units'])==20 and sum(r['models_checked'] for r in report['audited_units'])==80
    assert max(r['maximum_difference'] for r in report['audited_units'])<1e-8
    assert len(report['records'])==8
    assert {(r['target'],r['arm'],r['seed']) for r in report['records']}=={(t,a,s) for t in TARGETS for a in ARMS for s in (42,3407)}
    # Independent full-coverage arithmetic, within each split seed.
    root,f,folds,current,parent,settings,ledger,anchors=phase
    for r in report['records']:
        values=np.full(len(f),np.nan)
        for fold in range(5):
            path=root/'units'/task_name(dict(target=r['target'],seed=r['seed'],fold=fold))/'predictions.npz'
            with np.load(path,allow_pickle=False) as archive:values[folds[r['seed']]==fold]=archive[r['arm']]
        target=r['target'];truth=f[target].to_numpy();base=current[r['seed']][target]
        expected=50*(np.abs(truth-base).sum()-np.abs(truth-(.8*base+.2*values)).sum())/np.abs(truth).sum()
        assert abs(expected-r['gain'])<1e-12 and all(x['rows']==20 for x in r['folds_descriptive'])


def test_missing_outer_coverage_or_extra_unit_cannot_be_scored(phase,tmp_path):
    root,f,folds,current,parent,s,ledger,anchors=phase
    missing=dict(anchors);missing.pop(next(iter(missing)))
    with pytest.raises(ValueError,match='paired phase units'):
        collect_audited_phase(Path(__file__).resolve().parents[1],root,'development',f,folds,current,parent,s,missing,ledger.policy_sha256)


def test_reservation_repeat_is_consumed_without_any_new_fit(phase,monkeypatch):
    root,f,folds,_,_,s,ledger,_=phase;task=phase_tasks('development')[0];mask=folds[task['seed']]==task['fold']
    monkeypatch.setattr(NeighborRegressor,'initialize',lambda *a:pytest.fail('Repeat reached initialization'))
    with pytest.raises(FileExistsError,match='consumed'):
        execute_unit(task,f.loc[~mask].reset_index(drop=True),clean(f.loc[mask].reset_index(drop=True)),s,
            root/'units'/task_name(task),ledger.root,ledger.policy_sha256)


def test_failed_model_start_remains_failed_and_cannot_retry(tmp_path,monkeypatch):
    f=artificial();ledger=ReservationLedger.create(tmp_path/'ledger',dict(pair_unit=1,estimator=4,optimizer=2))
    s=Settings(dim=4,frequencies=2,embedding=3,batch_size=16,max_epochs=2,patience=1);task=phase_tasks('development')[0]
    def fail(*a,**kw):raise ValueError('artificial failure')
    monkeypatch.setattr(NeighborRegressor,'initialize',fail)
    with pytest.raises(ValueError,match='artificial failure'):
        execute_unit(task,f.iloc[:80].reset_index(drop=True),clean(f.iloc[80:].reset_index(drop=True)),s,
            tmp_path/'unit',ledger.root,ledger.policy_sha256)
    counts=ledger.inspect();assert counts['failed']==dict(pair_unit=1,estimator=1,optimizer=0)
    with pytest.raises(FileExistsError,match='consumed'):
        execute_unit(task,f.iloc[:80].reset_index(drop=True),clean(f.iloc[80:].reset_index(drop=True)),s,
            tmp_path/'other',ledger.root,ledger.policy_sha256)
    assert not (tmp_path/'other').exists()


def test_changed_query_or_completion_anchor_is_refused_before_model_audit(phase):
    root,f,folds,_,_,s,ledger,anchors=phase;task=phase_tasks('development')[0];name=task_name(task)
    mask=folds[task['seed']]==task['fold'];training=f.loc[~mask].reset_index(drop=True);query=clean(f.loc[mask].reset_index(drop=True))
    bad=query.copy();bad.loc[0,FEATURES[0]]+=.2
    with pytest.raises(ValueError,match='partition schema'):
        audit_unit(root/'units'/name,task,training,bad,s,expected_sha256=anchors[name],ledger_root=ledger.root)
    with pytest.raises(ValueError,match='completion'):
        audit_unit(root/'units'/name,task,training,query,s,expected_sha256='0'*64,ledger_root=ledger.root)


def rows(targets=TARGETS,seeds=(42,3407),gains=(.02,.02)):
    return [dict(target=t,arm=a,seed=s,blend_weight=.2,other_target_unchanged=True,candidate_score=96.3,
        gain=gains[i] if a=='LEARNED_ENCODER' else 0.) for t in targets for a in ARMS for i,s in enumerate(seeds)]


def test_gates_require_two_complete_positive_seeds_then_all_four_and_seed_lcb():
    development=rows();assert decide_development(development)['eligible_targets']==list(TARGETS)
    negative=rows(gains=(.08,-.001));assert not decide_development(negative)['eligible_targets']
    with pytest.raises(ValueError,match='coverage'):decide_development(development[:-1])
    passed=decide_confirmation(development,rows(seeds=(7777,12011)))
    assert passed['promoted_targets']==list(TARGETS) and not passed['release_authorized']
    failed=decide_confirmation(development,rows(seeds=(7777,12011),gains=(.2,.00001)))
    assert not failed['promoted_targets']
    assert all(not r['checks']['paired_seed_lcb95_positive'] for r in failed['decisions'].values())


def test_confirmation_task_allocation_is_only_explicit_earned_targets():
    assert phase_tasks('confirmation',[])==[]
    assert phase_limits('confirmation',['tap_time_len'])==dict(pair_unit=10,estimator=40,optimizer=20)
    with pytest.raises(ValueError):phase_tasks('confirmation')
    with pytest.raises(ValueError):phase_tasks('confirmation',['tap_time_len','tap_time_len'])
    with pytest.raises(ValueError):task_name(dict(target='tap_iron',seed=True,fold=0))
