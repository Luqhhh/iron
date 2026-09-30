from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bf_tap_r2.data import TARGETS
from bf_tap_r2.danet_model import Regressor,Settings
from bf_tap_r2.dnnr_ledger import file_hash,write_new
from bf_tap_r2.danet_phase_controller import sequence
from bf_tap_r2.danet_phase_freeze import validate_spec,prepare_manifest
from bf_tap_r2 import danet_phase_run as run
from bf_tap_r2 import danet_phase_arithmetic as arithmetic
from test_dnnr_model import sample


def test_sequence_without_finalist_stops_after_independent_development_arithmetic():
    stages=[]
    def invoke(name,module,args):
        stages.append(name)
        if name.endswith('-run'):return {'complete_sha256':'fit-anchor'}
        if name.endswith('-audit'):return {'audit_sha256':'cold-anchor'}
        return {'arithmetic_sha256':'arithmetic-anchor','selected_pairs':[]}
    result=sequence(invoke)
    assert stages==['development-run','development-audit','development-arithmetic']
    assert result['confirmation_skipped']=='no_development_finalist'


def test_sequence_forwards_external_cold_and_arithmetic_anchors_to_every_confirmation_stage():
    stages=[]
    def invoke(name,module,args):
        stages.append(name)
        if name.startswith('confirmation'):
            assert args[args.index('--development-audit-sha256')+1]=='cold-anchor'
            assert args[args.index('--development-arithmetic-sha256')+1]=='arithmetic-anchor'
        if name.endswith('-run'):return {'complete_sha256':'fit-anchor'}
        if name.endswith('-audit'):return {'audit_sha256':'cold-anchor'}
        return {'arithmetic_sha256':'arithmetic-anchor','selected_pairs':[[TARGETS[0],'DANET_LEARNED']]}
    sequence(invoke)
    assert len(stages)==6


@pytest.mark.parametrize('failed_stage',['development-run','development-audit','development-arithmetic'])
def test_failed_stage_stops_without_retry_or_confirmation(failed_stage):
    stages=[]
    def invoke(name,module,args):
        stages.append(name)
        if name==failed_stage:raise RuntimeError('synthetic failure')
        return {'complete_sha256':'fit','audit_sha256':'cold'}
    with pytest.raises(RuntimeError):sequence(invoke)
    assert stages[-1]==failed_stage and all(n.startswith('development') for n in stages)


def test_execution_spec_does_not_relax_the_preparation_model_or_gates(tmp_path):
    original=Path.cwd();folder=tmp_path/'configs/danet_abstract';folder.mkdir(parents=True)
    for name in ('SPEC.yaml','EXECUTION.yaml'):(folder/name).write_bytes((original/'configs/danet_abstract'/name).read_bytes())
    validate_spec(tmp_path)
    model=yaml.safe_load((folder/'SPEC.yaml').read_text())
    model['future_formal_stage']['development_gate']['mean_gain_minimum']=.001
    (folder/'SPEC.yaml').write_text(yaml.safe_dump(model))
    with pytest.raises(ValueError,match='gates'):validate_spec(tmp_path)


def test_complete_synthetic_phase_reconciles80_estimators_and_scores_without_new_audit_fits(tmp_path,monkeypatch):
    frame=sample(75)
    for j,t in enumerate(TARGETS):frame[t]=30+j+frame.air_volume.to_numpy()**2+frame.hot_air_press.to_numpy()
    folds={s:(np.arange(len(frame),dtype=np.int64)+i)%5 for i,s in enumerate((42,3407,7777,12011))}
    current={s:{t:frame[t].to_numpy().copy() for t in TARGETS} for s in folds}
    parent=deepcopy(current)
    for s in parent:parent[s]['tap_iron']+=.1
    reference=(frame,folds,current,parent,{})
    from dataclasses import asdict
    spec=yaml.safe_load(Path('configs/danet_abstract/EXECUTION.yaml').read_text())
    manifest=dict(workspace=str(tmp_path),settings=asdict(Settings(layers=2,width=4,groups=2,max_epochs=2,batch_size=32,ghost_size=32)),execution_spec=spec)
    path=tmp_path/'manifest.json';write_new(path,manifest);sha=file_hash(path)
    monkeypatch.setattr(run,'verify_manifest',lambda *a,**k:manifest)
    monkeypatch.setattr(arithmetic,'verify_manifest',lambda *a,**k:manifest)
    for module in (run,arithmetic):monkeypatch.setattr(module,'reload_references',lambda *_:reference)
    monkeypatch.setattr(run,'private_path',lambda workspace,root:Path(root))
    monkeypatch.setattr(run,'ProcessPoolExecutor',lambda max_workers,mp_context:ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(run.resource,'getrusage',lambda *_:SimpleNamespace(ru_maxrss=200*1024))
    (tmp_path/'configs').mkdir();(tmp_path/'configs/candidate_tiers.yaml').write_bytes(Path('configs/candidate_tiers.yaml').read_bytes())
    complete=run.run_phase(path,sha,'development')
    def no_fits(*args,**kwargs):raise AssertionError('Audit or arithmetic attempted estimator fit')
    monkeypatch.setattr(Regressor,'fit',no_fits)
    audited=run.audit_phase(path,sha,'development',complete['complete_sha256'])
    checked=arithmetic.check_phase(path,sha,'development',audited['audit_sha256'])
    report=json.loads((tmp_path/'development/audit.json').read_text())
    assert report['counts']['completed']['estimator']==80
    assert report['counts']['completed']['optimizer']==80
    assert len(report['unit_reports'])==20 and len(report['records'])==8
    assert checked['selected_pairs']==[] and report['new_audit_fits']==0
    eligible,_=run.earned_context(tmp_path,'confirmation',audited['audit_sha256'],checked['arithmetic_sha256'])
    assert eligible==[]
    with pytest.raises(ValueError,match='No qualified'):
        run.run_phase(path,sha,'confirmation',audited['audit_sha256'],checked['arithmetic_sha256'])
    assert not (tmp_path/'confirmation').exists()
    # A modified already-cold-audited prediction must not earn a derived seed.
    artifact=next((tmp_path/'development/units').rglob('prediction-*.npy'))
    artifact.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='artifact changed'):
        run.earned_context(tmp_path,'confirmation',audited['audit_sha256'],checked['arithmetic_sha256'])

