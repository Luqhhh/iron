from pathlib import Path
import json

import numpy as np
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.rfm_execution import execute_unit, audit_unit, validate_partition, task_name, _expected_events, collect_audited_phase
from bf_tap_r2.rfm_protocol import ReservationLedger, canonical, phase_limits, phase_tasks
from test_rfm_model import sample


@pytest.mark.parametrize('arm',['FIXED_KRR','FULL_RFM'])
def test_outer_unit_cold_audit_and_budget_keys(tmp_path,arm):
    frame=sample(50);training=frame.iloc[:40].reset_index(drop=True)
    query=frame.iloc[40:].drop(columns=list(TARGETS)).reset_index(drop=True)
    ledger=ReservationLedger.create(tmp_path/'ledger',dict(outer_fit=1,procedure=2,solve=8,update=6))
    task=dict(target='tap_time_len',arm=arm,seed=42,fold=0)
    anchor=execute_unit(task,training,query,tmp_path/'unit',ledger.root,ledger.policy_sha256)
    prediction,audit=audit_unit(tmp_path/'unit',task,training,query,expected_sha256=anchor['complete_sha256'])
    assert prediction.shape==(10,) and audit['outer_cold_difference']==0
    actual=set()
    for p in (ledger.root/'events').glob('*.started.json'):
        r=json.loads(p.read_text());actual.add((r['kind'],canonical(r['key'])))
    assert actual==_expected_events(task_name(task),audit['selected_state'],arm)
    with pytest.raises(FileExistsError):
        execute_unit(task,training,query,tmp_path/'retry',ledger.root,ledger.policy_sha256)
    changed=query.iloc[::-1]
    with pytest.raises(ValueError,match='provenance'):
        audit_unit(tmp_path/'unit',task,training,changed,expected_sha256=anchor['complete_sha256'])


def test_outer_label_group_and_id_isolation(tmp_path):
    frame=sample(50);train=frame.iloc[:40];query=frame.iloc[40:].copy()
    with pytest.raises(ValueError,match='labels'):validate_partition(train,query)
    query=query.drop(columns=list(TARGETS))
    validate_partition(train,query)
    query.loc[query.index[0],'sample_id']=train.iloc[0].sample_id
    with pytest.raises(ValueError,match='IDs'):validate_partition(train,query)
    query['sample_id']=[f'query-{i}' for i in range(10)]
    from bf_tap_r2.data import FEATURES
    query.loc[query.index[0],list(FEATURES)]=train.iloc[0][list(FEATURES)].to_numpy()
    with pytest.raises(ValueError,match='group'):validate_partition(train,query)


def test_invalid_task_refused_before_any_fitting():
    for change in [dict(arm='OTHER'),dict(seed=2026),dict(fold=5),dict(target='unknown')]:
        with pytest.raises(ValueError):task_name(dict(target='tap_iron',arm='FULL_RFM',seed=42,fold=0,**{})|change)


def test_complete_phase_rebuilds_all_columns_and_rejects_unbound_receipt(tmp_path):
    # A complete tiny phase exercises real saved models, receipt accounting,
    # independent audits and tier integration without official labels.
    frame=sample(30)
    folds={42:np.arange(30)%5,3407:np.roll(np.arange(30)%5,1)}
    current={s:{t:frame[t].to_numpy()+2 for t in TARGETS} for s in folds}
    historical={s:{t:frame[t].to_numpy()+3 for t in TARGETS} for s in folds}
    ledger=ReservationLedger.create(tmp_path/'ledger',phase_limits('development'))
    anchors={}
    for task in phase_tasks('development'):
        mask=folds[task['seed']]==task['fold']; name=task_name(task)
        result=execute_unit(task,frame.loc[~mask].reset_index(drop=True),
            frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True),
            tmp_path/'units'/name,ledger.root,ledger.policy_sha256)
        anchors[name]=result['complete_sha256']
    workspace=Path(__file__).resolve().parents[1]
    def collect(identities=anchors):
        return collect_audited_phase(workspace,tmp_path,'development',frame,folds,
                                     current,historical,identities,ledger.policy_sha256)
    report=collect()
    assert report['status']=='passed' and report['new_audit_fits']==0
    assert len(report['audited_units'])==40 and len(report['records'])==8
    assert report['counts']['completed']['outer_fit']==40
    assert report['counts']['completed']['procedure']==80
    assert report['release_authorized'] is False
    assert report['tiers']['release_authorized'] is False
    assert all(set(c)=={'FULL_RFM'} for c in report['tiers']['decisions'].values())
    with pytest.raises(ValueError,match='anchors'):collect(dict(list(anchors.items())[:-1]))
    # Endpoints are not trusted merely because every model file passes audit.
    path=next((ledger.root/'events').glob('outer_fit-*.complete.json'))
    receipt=json.loads(path.read_text());receipt['result']['unit_complete_sha256']='0'*64
    path.write_bytes(canonical(receipt))
    with pytest.raises(ValueError,match='receipt'):collect()
