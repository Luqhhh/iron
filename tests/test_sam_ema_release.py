from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_round2_v12 import sample

WORK=Path(__file__).resolve().parents[1]
loader=importlib.util.spec_from_file_location('sam_ema_release',WORK/'scripts/sam_ema_release.py')
m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)


def spec():return json.loads((WORK/m.SPEC).read_text())


@pytest.mark.parametrize('field,value',[('full_training_procedures',True),('torch_optimizer',3),
    ('replacement_weight',.5),('new_CV',1),('packages',2),('formal_promotion',True),
    ('desktop_writes',1),('maximum_runtime_seconds',600)])
def test_standing_manual_scope_rejects_drift_and_does_not_reclassify_formal(field,value):
    s=spec();science=json.loads((WORK/s['scientific_config']).read_text());m.validate_scope(s,science)
    s[field]=value
    with pytest.raises(ValueError,match='scope'):m.validate_scope(s,science)


def test_full_recipe_must_match_the_completed_scientific_source():
    s=spec();science=json.loads((WORK/s['scientific_config']).read_text());s['training']['width']=512
    with pytest.raises(ValueError,match='recipe'):m.validate_scope(s,science)


def test_complete_contrasts_and_real_terminal_are_required(tmp_path):
    def save(name,value):(tmp_path/name).write_text(json.dumps(value))
    save('manifest.json',dict(files={}))
    save('report.json',dict(formal_promotion=False,confirmation_eligible=False,records={str(s):dict(gains={'SAM_EMA_TIME':-.02},matched_sam_gain=.01) for s in [42,3407]}))
    save('independent-score.json',dict(status='passed',report_sha256=m.sha(tmp_path/'report.json'),manifest_sha256=m.sha(tmp_path/'manifest.json'),actual_optimizer_runs=20,new_saved_states=20,old_saved_states=40))
    terminal=dict(status='passed',actual_exit_codes=[0,0],report_sha256=m.sha(tmp_path/'report.json'),manifest_sha256=m.sha(tmp_path/'manifest.json'),independent_score_sha256=m.sha(tmp_path/'independent-score.json'))
    save('terminal-verification.json',terminal)
    actual=dict(status='passed',actual_supervisor_tool_exit_code=0,terminal_sha256=m.sha(tmp_path/'terminal-verification.json'));save('actual-main-exit.json',actual)
    s=dict(scientific_development=str(tmp_path),scientific_evidence={})
    assert m.development_evidence(s)=={}
    actual['actual_supervisor_tool_exit_code']=1;save('actual-main-exit.json',actual)
    with pytest.raises(ValueError,match='actual G0'):m.development_evidence(s)
    actual['actual_supervisor_tool_exit_code']=0;save('actual-main-exit.json',actual)
    report=json.loads((tmp_path/'report.json').read_text());report['records'].pop('3407');save('report.json',report)
    with pytest.raises(ValueError):m.development_evidence(s)


def test_real_full_bridge_uses_exactly_two_native_optimizers_and_original_sam_ema_methods(tmp_path):
    from bf_tap_r2.sam_ema import SAMEMARegressor
    from bf_tap_r2.sam_ema_release_models import fit_component
    from bf_tap_r2.ema_reference_ledger import reference_bindings, binding_sources
    training,settings=sample();settings['max_epochs']=2
    query=training.iloc[-20:].drop(columns=['tap_iron','tap_time_len']).copy()
    query['sample_id']=[f'query-{i}' for i in range(len(query))];query['air_volume']+=100.
    s=spec();identity=dict(source_directory=str(tmp_path),split_seed=-1,fold=-1,trial_id='SAM_EMA_FULL')
    source=m.sources(WORK)|binding_sources(reference_bindings());fit=SAMEMARegressor.fit;train=SAMEMARegressor._train
    prediction,receipt=fit_component(tmp_path/'full',training,query,identity=identity,settings=settings,mechanisms=s['mechanisms'],source_hashes=source)
    assert receipt['native_counts']['torch_optimizer']==2
    assert all(v==0 for k,v in receipt['native_counts'].items() if k!='torch_optimizer')
    assert receipt['identity']==identity and receipt['model_metadata']['arm']=='SAM_EMA'
    assert prediction.shape==(20,)
    for trace in receipt['model_metadata']['traces'].values():
        assert trace['gradient_evaluations']==2*trace['updates'] and trace['ema_updates']==trace['updates']
    assert SAMEMARegressor.fit is fit and SAMEMARegressor._train is train
    with pytest.raises(FileExistsError):fit_component(tmp_path/'full',training,query,identity=identity,settings=settings,mechanisms=s['mechanisms'],source_hashes=source)
