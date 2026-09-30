from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import bf_tap_r2.incumbent_reference_phase as phase


@pytest.mark.parametrize('defect',['scope','budget','workers','native_recipe','audit_bound','target','settings','monitor'])
def test_spec_rejects_implicit_scope_budget_model_or_audit_changes(defect):
    root=Path(phase.__file__).resolve().parents[2]
    spec=yaml.safe_load((root/phase.SPEC).read_text())
    settings=yaml.safe_load((root/'configs/strong_component_regularization/SPEC.yaml').read_text())['training']['tap_iron']
    phase.validate_spec(spec,settings)
    if defect=='scope':spec['missing_members']['split_seeds']=[42,3407,7777,12011]
    elif defect=='budget':spec['planned_budget_not_yet_reserved']['estimators']=21
    elif defect=='workers':spec['resources']['workers']=5
    elif defect=='native_recipe':spec['native_recipe']['recipe']['frequency']=.02
    elif defect=='audit_bound':spec['core_audit']['order_chunk_absolute_atol']=.001
    elif defect=='target':spec['purpose']='DE3_quality_confirmation'
    elif defect=='settings':settings['max_epochs']=241
    else:spec['monitoring']['future_interval_seconds']=30
    with pytest.raises(ValueError):phase.validate_spec(spec,settings)


@pytest.mark.parametrize('defect',['source','runtime','full_suite','exit','status'])
def test_freeze_rejects_bad_test_receipt_before_any_label_loader(monkeypatch,defect):
    root=Path(phase.__file__).resolve().parents[2]
    sources={'synthetic-marker.py':'a'*64};runtime=dict(synthetic='only')
    receipt=dict(status='passed',exit_code=0,full_suite=True,source_hashes=deepcopy(sources),runtime=deepcopy(runtime))
    if defect=='source':receipt['source_hashes']['synthetic-marker.py']='b'*64
    elif defect=='runtime':receipt['runtime']['synthetic']='changed'
    elif defect=='full_suite':receipt['full_suite']=False
    elif defect=='exit':receipt['exit_code']=1
    else:receipt['status']='failed'
    monkeypatch.setattr(phase,'source_snapshot',lambda *args:sources)
    monkeypatch.setattr(phase,'runtime_snapshot',lambda:runtime)
    monkeypatch.setattr(phase,'anchored',lambda *args:receipt)
    def forbidden(*args):raise AssertionError('No historical/official data inspection before exact-source test admission')
    monkeypatch.setattr(phase,'inspect_development',forbidden);monkeypatch.setattr(phase,'load_reference_cache',forbidden)
    with pytest.raises(ValueError,match='exact phase sources/runtime'):
        phase.prepare(root,'unused','unused','unused',root/'local/never-create-bad-freeze',root/'local/unused-receipt','unused')
    assert not (root/'local/never-create-bad-freeze').exists()
