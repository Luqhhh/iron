"""Pin the two release formulas, isolation, and exclusive fit reservations."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.submission import validate_result

spec=importlib.util.spec_from_file_location('ptarl_ema_release',Path(__file__).parents[1]/'scripts/ptarl_ema_exploration_release.py')
release=importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def parents():
    ids=[f'synthetic-{i}' for i in range(322)]
    header='sample_id,pred_tap_iron,pred_tap_time_len\n'
    current=(header+''.join(f'{i},10.000000,30.12345000\n' for i in ids)).encode()
    v32=(header+''.join(f'{i},10.000000,20.0000\n' for i in ids)).encode()
    return current,v32,ids


def test_ptarl_retains_v32_blend_reference_and_current_iron_strings():
    current,v32,ids=parents()
    result=release.payload(release.NAMES[0],current,v32,ids,np.full(322,40.))
    rows=validate_result(result,ids)
    assert all(float(r['pred_tap_time_len'])==24. for r in rows)
    assert all(r['pred_tap_iron']=='10.000000' for r in rows)
    # Blending current EMA instead would produce32.09876, a different experiment.
    assert float(rows[0]['pred_tap_time_len']) != .8*30.12345+.2*40


def test_ema_component_replacement_preserves_scored_time_strings():
    current,v32,ids=parents()
    rows=validate_result(release.payload(release.NAMES[1],current,v32,ids,np.full(322,8.),np.full(322,4.)),ids)
    assert all(float(r['pred_tap_iron'])==12. for r in rows)
    assert all(r['pred_tap_time_len']=='30.12345000' for r in rows)


@pytest.mark.parametrize('name',release.NAMES)
@pytest.mark.parametrize('defect',['short','nan','negative'])
def test_bad_prediction_or_silent_clipping_refused(name,defect):
    current,v32,ids=parents();member=np.ones(322)
    if defect=='short':member=member[:3]
    elif defect=='nan':member[0]=np.nan
    else:member[:]=-1000.
    with pytest.raises(ValueError):release.payload(name,current,v32,ids,member,np.ones(322))


def test_mismatched_native_component_refused():
    current,v32,ids=parents()
    with pytest.raises(ValueError,match='native'):release.payload(release.NAMES[1],current,v32,ids,np.ones(322),np.ones(3))


class ExclusiveLedger:
    def __init__(self):self.events={}
    @contextmanager
    def event(self,kind,key,payload):
        if key in self.events:raise FileExistsError('consumed')
        self.events[key]=dict(status='started',payload=payload)
        result={}
        try:
            yield result
        except BaseException:
            self.events[key]['status']='failed'
            raise
        else:self.events[key].update(status='completed',result=result)


def test_reservation_keeps_failed_attempt_and_refuses_retry(monkeypatch,tmp_path):
    from bf_tap_r2.component_regularization import ComponentRegressor
    class Model:pass
    model=Model();ledger=ExclusiveLedger()
    def failing(self,frame,y,epochs,validation=None):raise RuntimeError('numerical failure')
    monkeypatch.setattr(ComponentRegressor,'_train',failing)
    release.reserve_ema_training(model,ledger,tmp_path)
    with pytest.raises(RuntimeError,match='numerical'):model._train([1,2],None,240)
    assert ledger.events['EMA_IRON','refit']['status']=='failed'
    with pytest.raises(FileExistsError):model._train([1,2],None,240)


def test_reservation_calls_unchanged_method_and_binds_saved_state(monkeypatch,tmp_path):
    from bf_tap_r2.component_regularization import ComponentRegressor
    class Model:pass
    model=Model();ledger=ExclusiveLedger();calls=[]
    def original(self,frame,y,epochs,validation=None):
        calls.append((frame,y,epochs,validation))
        (tmp_path/('selection.pt' if validation is not None else 'refit.pt')).write_bytes(b'exact-model')
        return 7
    monkeypatch.setattr(ComponentRegressor,'_train',original)
    release.reserve_ema_training(model,ledger,tmp_path)
    assert model._train([1,2],'target',240,'calibration')==7
    assert model._train([1,2,3],'target',7)==7
    assert calls==[([1,2],'target',240,'calibration'),([1,2,3],'target',7,None)]
    assert all(e['status']=='completed' for e in ledger.events.values())
    assert all(e['result']['model_sha256']==release.sha(tmp_path/'refit.pt') for e in ledger.events.values())


def test_real_ema_reservations_preserve_numerical_trajectory(tmp_path):
    import torch
    from test_round2_v12 import sample
    from bf_tap_r2.component_regularization import ComponentRegressor
    frame,settings=sample();y=frame[['tap_iron','tap_time_len']].to_numpy()
    recipe={'backbone':'tabm','frequency':.01};mechanisms={'ema_beta':.99,'sam_rho':.05,'sam_epsilon':1e-12}
    a=tmp_path/'plain';b=tmp_path/'recorded';a.mkdir();b.mkdir()
    plain=ComponentRegressor(recipe,settings,'EMA',mechanisms,a).fit(frame,y)
    recorded=ComponentRegressor(recipe,settings,'EMA',mechanisms,b);ledger=ExclusiveLedger()
    release.reserve_ema_training(recorded,ledger,b);recorded.fit(frame,y)
    query=frame.drop(columns=['tap_iron','tap_time_len'])
    np.testing.assert_array_equal(plain.predict(query),recorded.predict(query))
    assert plain.metadata_['selected_epoch']==recorded.metadata_['selected_epoch']
    for name in ('selection','refit'):
        left=torch.load(a/(name+'.pt'),weights_only=True);right=torch.load(b/(name+'.pt'),weights_only=True)
        assert all(torch.equal(left['state'][key],right['state'][key]) for key in left['state'])
    assert len(ledger.events)==2
