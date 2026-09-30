"""Prospective resource admission must fail closed before official evaluation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from bf_tap_r2 import danet_preflight as probe
from bf_tap_r2.danet_ledger import write_new,file_hash
from bf_tap_r2.danet_model import Settings
from bf_tap_r2.dnnr_model import Encoder

WORKSPACE=Path(__file__).resolve().parents[1]


def witness():
    return dict(pair_fit_save_seconds=600.,pair_audit_seconds=10.,maximum_worker_mib=512.,maximum_bound_fraction=.2,
        learned_mask_l1_change=2.,fixed_mask_l1_change=0.,optimizer_epochs=960,training_rows=2204,query_rows=551,
        encoded_dimensions=26,saved_models=4,mae=dict(DANET_FIXED=.8,DANET_LEARNED=.5),median_mae=1.)


def test_resource_has_complete_coverage_and_uses_all_four_maximum_epoch_fits():
    spec=probe.validate_spec(WORKSPACE);training,y,query,truth=probe.synthetic_data()
    assert spec['training']==asdict(Settings()) and len(training)==2204 and len(query)==551
    assert len(y)==len(training) and len(truth)==len(query)
    assert not set(training.sample_id)&set(query.sample_id)
    assert Encoder().fit(training).transform(training).shape==(2204,26)
    result=probe.resource_decision(witness(),4096.)
    assert result['status']=='passed' and result['projected_development_seconds']==4875.
    assert result['required_available_mib']==3072.


@pytest.mark.parametrize('key,value,failed_check',[
    ('pair_fit_save_seconds',1000.,'cost'),('maximum_worker_mib',1537.,'worker_memory'),
    ('maximum_bound_fraction',1.01,'numerical'),('learned_mask_l1_change',0.,'learned_mask_active'),
    ('fixed_mask_l1_change',.1,'fixed_mask_unchanged'),('mae',dict(DANET_FIXED=1.,DANET_LEARNED=.5),'learnability')])
def test_one_failed_resource_gate_is_binding(key,value,failed_check):
    measured=witness();measured[key]=value
    result=probe.resource_decision(measured,10000.)
    assert result['status']=='failed' and result['checks'][failed_check] is False


def test_memory_gate_uses_four_measured_workers_plus_reserve():
    result=probe.resource_decision(witness(),3071.999)
    assert result['status']=='failed' and not result['checks']['available_memory']


@pytest.mark.parametrize('key,value',[
    ('optimizer_epochs',959),('training_rows',1102),('query_rows',550),('saved_models',3),
    ('encoded_dimensions',25),('learned_mask_l1_change',np.nan),('fixed_mask_l1_change',np.inf),
    ('pair_fit_save_seconds',0.),('maximum_bound_fraction',-.1)])
def test_partial_or_invalid_witness_cannot_be_admitted(key,value):
    measured=witness();measured[key]=value
    with pytest.raises(ValueError):probe.resource_decision(measured,10000.)


@pytest.mark.parametrize('section,key,value',[
    ('training','max_epochs',239),('training','width',32),('training','layers',10),
    ('future_formal_stage','development_seeds',[42]),
    ('resources','maximum_development_seconds',7201),('monitoring','interval_seconds',60)])
def test_architecture_budget_coverage_and_monitor_contract_cannot_drift(tmp_path,section,key,value):
    spec=deepcopy(probe.validate_spec(WORKSPACE));spec[section][key]=value
    path=tmp_path/probe.SPEC;path.parent.mkdir(parents=True);path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):probe.validate_spec(tmp_path)


@pytest.fixture
def checked_workspace(tmp_path):
    root=tmp_path/'worktree';root.mkdir()
    for name in probe.REQUIRED_FILES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(WORKSPACE/name,path)
    for name in ('pyproject.toml','uv.lock'):shutil.copyfile(WORKSPACE/name,root/name)
    local=root/'local';local.mkdir();log=local/'suite.log';log.write_text('29 passed in 4.43s\n')
    receipt=dict(status='passed',exit_code=0,full_suite=True,source_changed=False,source_hashes=probe.source_hashes(root),
        runtime=probe.runtime(),passed=29,log=str(log),log_sha256=file_hash(log),summary='29 passed in 4.43s')
    path=local/'receipt.json';write_new(path,receipt)
    return root,path,receipt


def test_stale_checked_source_rejected_before_new_output_or_fitting(checked_workspace):
    root,path,receipt=checked_workspace;sha=file_hash(path);output=root/'local/not-created'
    (root/'src/bf_tap_r2/danet_network.py').write_text('# changed architecture\n')
    with pytest.raises(ValueError,match='exact-source'):probe.freeze_probe(root,output,path,sha)
    assert not output.exists()


def test_changed_test_log_and_foreign_receipts_are_rejected(checked_workspace,tmp_path):
    root,path,receipt=checked_workspace;sha=file_hash(path)
    Path(receipt['log']).write_text('29 passed in 4.43s\nextra changed evidence\n')
    with pytest.raises(ValueError,match='log changed'):
        probe.checked_receipt(root,path,sha,probe.source_hashes(root),probe.runtime())
    foreign=tmp_path/'foreign.json';shutil.copyfile(path,foreign)
    with pytest.raises(ValueError,match='private'):
        probe.checked_receipt(root,foreign,file_hash(foreign),probe.source_hashes(root),probe.runtime())


def test_shared_local_symlink_is_not_a_private_probe_destination(checked_workspace,tmp_path):
    root,_,_=checked_workspace;shared=tmp_path/'shared';shared.mkdir()
    (root/'local/runs').symlink_to(shared,target_is_directory=True)
    with pytest.raises(ValueError,match='Direct private'):
        probe.private_path(root,root/'local/runs/probe')


def test_license_changes_are_bound_by_the_full_source_snapshot(checked_workspace):
    root,_,_=checked_workspace;before=probe.source_hashes(root)
    (root/'licenses/QHoptim-MIT.txt').write_text('changed license')
    assert before!=probe.source_hashes(root)
