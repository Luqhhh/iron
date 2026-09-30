from dataclasses import asdict
import json
from pathlib import Path

import pytest
import yaml

from bf_tap_r2.dnnr_model import Encoder, Settings
from bf_tap_r2.dnnr_preflight import (PROBE, RESOURCES, LIMITS, SPEC_PATH,
    synthetic_data, resource_decision, private_path, source_hashes, freeze_probe)
from bf_tap_r2.dnnr_ledger import file_hash


def measurement(selected=1):
    return dict(train_rows=2204, query_rows=551, encoded_dimension=26,
        models_checked=7-selected, selected_metric_epochs=selected,
        mae={'KNN_FIXED':1., 'DNNR_FIXED':.5, 'DNNR_LEARNED':.4}, median_mae=2.,
        peak_mib=512., maximum_audit_difference=1e-12,
        pair_execution_seconds=100., pair_audit_seconds=20.,
        upper_bound_extra_seconds=10.*(not selected))


def test_source_spec_matches_fixed_full_size_probe_and_cost_rules():
    spec = yaml.safe_load(Path(SPEC_PATH).read_text())
    assert spec['training'] == asdict(Settings())
    assert spec['synthetic_resource_probe'] == PROBE and spec['resources'] == RESOURCES
    fitting, y, query, truth = synthetic_data()
    assert len(fitting) == 2204 and len(query) == len(truth) == 551 and len(y) == 2204
    assert set(fitting.sample_id).isdisjoint(set(query.sample_id))
    assert Encoder().fit(fitting).transform(fitting).shape == (2204,26)
    assert {'tap_iron','tap_time_len'}.isdisjoint(fitting)


@pytest.mark.parametrize('selected', [0,1])
def test_resource_gate_uses_actual_cold_audit_and_optional_full_epoch_cost(selected):
    value = measurement(selected)
    decision = resource_decision(value, 10000.)
    assert decision['status'] == 'passed'
    upper = 120.+10.*(not selected)
    assert decision['projected_development_seconds'] == 20*upper/4*1.5+300
    assert decision['required_available_mib'] == 4*512+1024


@pytest.mark.parametrize('defect', ['cost','memory','ram','quality','cold'])
def test_each_frozen_admission_gate_can_independently_refuse_probe(defect):
    value = measurement(); ram = 10000.
    if defect == 'cost': value['pair_execution_seconds'] = 1000
    elif defect == 'memory': value['peak_mib'] = 1600
    elif defect == 'ram': ram = 3000
    elif defect == 'quality': value['mae']['DNNR_LEARNED'] = 2.1
    else: value['maximum_audit_difference'] = 1e-7
    assert resource_decision(value, ram)['status'] == 'failed'


def test_incomplete_worst_case_epoch_is_not_an_admissible_cost_witness():
    value = measurement(0); value['models_checked'] = 6
    with pytest.raises(ValueError, match='coverage'): resource_decision(value,10000.)
    value = measurement(0); value['upper_bound_extra_seconds'] = 0
    with pytest.raises(ValueError, match='measurements'): resource_decision(value,10000.)


def test_private_path_rejects_shared_runs_symlink_escape_and_public_destination(tmp_path):
    workspace = tmp_path/'workspace'; (workspace/'local').mkdir(parents=True)
    elsewhere = tmp_path/'elsewhere'; elsewhere.mkdir()
    (workspace/'local'/'shared').symlink_to(elsewhere, target_is_directory=True)
    assert private_path(workspace,workspace/'local'/'new') == workspace/'local'/'new'
    for path in (workspace/'public',workspace/'local'/'shared'/'new'):
        with pytest.raises(ValueError): private_path(workspace,path)


def test_freeze_refuses_stale_engineering_snapshot_before_creating_run(tmp_path):
    workspace = Path.cwd()
    # Private isolated synthetic receipt paths, zero official source/data reads.
    folder = workspace/'local'/'research'/'synthetic-probe-freeze-fixture'
    folder.mkdir(exist_ok=True)
    import uuid
    token = uuid.uuid4().hex
    snapshot = folder/f'{token}-snapshot.json'
    snapshot.write_text(json.dumps({'missing_source.py':'0'*64}))
    receipt = folder/f'{token}-receipt.json'
    receipt.write_text(json.dumps(dict(status='passed',source_files=1,focused_tests=37,
        locked_full_tests=1347,source_snapshot=str(snapshot.relative_to(workspace)),
        source_snapshot_sha256=file_hash(snapshot))))
    output = folder/f'{token}-output'
    with pytest.raises(ValueError, match='Exact-source'):
        freeze_probe(workspace,output,receipt,file_hash(receipt))
    assert not output.exists()
