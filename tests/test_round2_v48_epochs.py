import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from bf_tap_r2.v47_factorized import FactorRegressor
from bf_tap_r2.v48_factorized import LongFactorRegressor,fit_partition
from bf_tap_r2.v48_prefix import verify_prefix_files
from bf_tap_r2.v47_verify import verify_model
from test_round2_v47_factorized import frame,settings
from bf_tap_r2.data import FEATURES,TARGETS


def setup_old(tmp_path,recipe='AHOFM4',validation=False):
    data=frame(45);y=3+data[FEATURES[0]].to_numpy()+data[FEATURES[1]].to_numpy()**2
    s=settings();s.update(max_epochs=3,rank_per_order=3,batch_size=16)
    old=FactorRegressor(recipe,s).initialize(data.iloc[:30],y[:30])
    val=(data.iloc[30:],y[30:]) if validation else None
    old.train(3,val);p=tmp_path/'old.json';old.save(p)
    return data,y,s,old,p,val


def test_extended_refit_prefix_exact_and_continues(tmp_path):
    data,y,s,old,path,val=setup_old(tmp_path)
    longer=dict(s,max_epochs=7)
    new=LongFactorRegressor('AHOFM4',longer).initialize(data.iloc[:30],y[:30])
    new.train(7,prefix_path=path);p=tmp_path/'prefix.json';new.save_prefix(p)
    check=verify_prefix_files(p,path,data.iloc[30:],old.predict(data.iloc[30:]))
    assert check['prediction_difference']==0 and len(new.history_)==7
    assert new.history_[:3]==old.history_
    assert not np.array_equal(new.predict(data.iloc[30:]),old.predict(data.iloc[30:]))
    verify_model(p,data.iloc[:30],y[:30],'AHOFM4',s)


def test_selector_prefix_is_selected_state_not_raw_last_epoch(tmp_path):
    data,y,s,old,path,val=setup_old(tmp_path,validation=True)
    s.update(min_delta_standardized_mae=100.,patience=50)
    old=FactorRegressor('AHOFM4',s).initialize(data.iloc[:30],y[:30]);old.train(3,val)
    path=tmp_path/'selected.json';old.save(path)
    assert old.selected_epoch_==1 and old.stopped_epoch_==3
    new=LongFactorRegressor('AHOFM4',dict(s,max_epochs=8)).initialize(data.iloc[:30],y[:30])
    new.train(8,val,prefix_path=path);p=tmp_path/'prefix.json';new.save_prefix(p)
    verify_prefix_files(p,path,data.iloc[30:],old.predict(data.iloc[30:]))
    snap=json.loads(p.read_text())
    assert snap['metadata']['selected_epoch']==old.selected_epoch_
    assert snap['metadata']['history']==old.history_


def test_prefix_rejects_other_settings_and_parameter_tamper(tmp_path):
    data,y,s,old,path,val=setup_old(tmp_path)
    bad=dict(s,max_epochs=6,learning_rate=.002)
    new=LongFactorRegressor('AHOFM4',bad).initialize(data.iloc[:30],y[:30])
    with pytest.raises(ValueError,match='beyond epoch cap'):new.train(6,prefix_path=path)
    saved=json.loads(path.read_text());saved['state']['bias']+=1
    tamper=tmp_path/'tamper.json';tamper.write_text(json.dumps(saved))
    new=LongFactorRegressor('AHOFM4',dict(s,max_epochs=6)).initialize(data.iloc[:30],y[:30])
    with pytest.raises(ValueError,match='parameter prefix'):new.train(6,prefix_path=tamper)


def test_prefix_rejects_history_tamper_and_missing_epoch(tmp_path):
    data,y,s,old,path,val=setup_old(tmp_path)
    saved=json.loads(path.read_text());saved['metadata']['history'][0]['standardized_training_mae']+=.01
    tamper=tmp_path/'history.json';tamper.write_text(json.dumps(saved))
    new=LongFactorRegressor('AHOFM4',dict(s,max_epochs=6)).initialize(data.iloc[:30],y[:30])
    with pytest.raises(ValueError,match='trace prefix'):new.train(6,prefix_path=tamper)
    new=LongFactorRegressor('AHOFM4',dict(s,max_epochs=6)).initialize(data.iloc[:30],y[:30])
    with pytest.raises(ValueError,match='cannot reach'):new.train(2,prefix_path=path)


def test_uncapped_control_is_identical_and_no_prefix_on_new_seed(tmp_path):
    data,y,s,_,_,_=setup_old(tmp_path)
    s.update(max_epochs=12,patience=2,min_delta_standardized_mae=100)
    old=FactorRegressor('AFM2',s).initialize(data.iloc[:30],y[:30]);val=(data.iloc[30:],y[30:])
    old.train(12,val);p=tmp_path/'control.json';old.save(p)
    new=LongFactorRegressor('AFM2',dict(s,max_epochs=24)).initialize(data.iloc[:30],y[:30])
    new.train(24,val,prefix_path=p)
    assert new.stopped_epoch_==old.stopped_epoch_==3
    np.testing.assert_array_equal(new.predict(data.iloc[30:]),old.predict(data.iloc[30:]))
    fresh=LongFactorRegressor('AFM2',s).initialize(data.iloc[:30],y[:30]);fresh.train(2)
    with pytest.raises(ValueError,match='No captured'):fresh.save_prefix(tmp_path/'absent.json')


def test_frozen_schedule_only_and_gates_unchanged():
    old=yaml.safe_load(Path('configs/round2_v47/SPEC.yaml').read_text())
    new=yaml.safe_load(Path('configs/round2_v48/SPEC.yaml').read_text())
    assert new['training']['max_epochs']==2000
    assert dict(new['training'],max_epochs=400)==old['training']
    for key in ['reference','calibration','promotion','split_seeds','confirmation_seeds','candidates']:
        assert new[key]==old[key]


def test_partition_recreates_both_actual_old_models(tmp_path):
    from bf_tap_r2.v47_factorized import fit_partition as old_fit
    from bf_tap_r2.v48_prefix import verify_unit_prefix
    data=frame(40)
    for target in TARGETS:data[target]=4+data[FEATURES[0]]+data[FEATURES[1]]**2
    outer=data.iloc[:30];fitting=outer.iloc[:20];cal=outer.iloc[20:]
    query=data.iloc[30:].drop(columns=list(TARGETS));s=settings();s.update(max_epochs=2,rank_per_order=2)
    args=(fitting,cal,outer,query,TARGETS[0],'AHOFM4')
    old,p,meta,cp=old_fit(*args,s,np.full(10,4.),[0.,.5,1.])
    olddir=tmp_path/'old'/'unit';olddir.mkdir(parents=True)
    old.save(olddir/'model.json');old.calibration_model_.save(olddir/'calibration_model.json')
    np.savez(olddir/'predictions.npz',prediction=p,calibration_prediction=cp,
             query_ids=query.sample_id.to_numpy(dtype=str),calibration_ids=cal.sample_id.to_numpy(dtype=str))
    new,pred,_,_=fit_partition(*args,dict(s,max_epochs=4),np.full(10,4.),[0.,.5,1.],old_directory=olddir)
    newdir=tmp_path/'new';newdir.mkdir()
    new.save_prefix(newdir/'prefix_model.json');new.calibration_model_.save_prefix(newdir/'prefix_calibration_model.json')
    checks=verify_unit_prefix(tmp_path,{'prefix_reference':{'directory':'old'}},'unit',newdir,query,cal)
    assert checks['status']=='passed' and checks['refit']['prediction_difference']==0
    assert pred.shape==(10,)
