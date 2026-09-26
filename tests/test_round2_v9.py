from copy import deepcopy
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip('torch')
pytest.importorskip('pytabkit')
pytest.importorskip('rtdl_revisiting_models')
from bf_tap_r2.data import FEATURES
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v9_realmlp import InputEncoder, RealMLPRegressor, make_estimator


def frame():
    rng=np.random.default_rng(73)
    x=pd.DataFrame(rng.normal(size=(100,len(FEATURES))),columns=FEATURES)
    x['sample_id']=[f'synthetic-{i:04d}' for i in range(len(x))]
    x['spout_no']=np.arange(len(x))%2+1
    x['tap_iron']=100+3*x.air_volume+rng.normal(size=len(x))
    x['tap_time_len']=x.tap_iron/4
    return x


def recipe(name):
    from pytabkit import RealMLP_TD_Regressor,RealMLP_TD_S_Regressor
    r=deepcopy(yaml.safe_load(Path('configs/round2_v9/SPEC.yaml').read_text())['recipes'][name])
    r['constructor'].update(n_epochs=3,hidden_sizes=[16,16])
    cls={'realmlp_td':RealMLP_TD_Regressor,'realmlp_td_s':RealMLP_TD_S_Regressor}[name]
    r['resolved']=cls(**r['constructor']).get_config()
    return r


def test_input_encoder_ignores_ids_labels_and_unknown_spouts():
    x=frame();encoder=InputEncoder().fit(x.iloc[:80]);query=x.iloc[80:].copy()
    query.loc[:,'spout_no']=999
    query.iloc[0,0]=np.nan
    query['sample_id']='forbidden-id';query['tap_iron']=-1e8;query['tap_time_len']=1e8
    a=encoder.transform(query)
    assert a.shape==(20,len(FEATURES)+2)
    assert np.count_nonzero(a[:,-2:])==0
    assert a[0,0]==np.float32(np.median(x.iloc[:80,0]))
    np.testing.assert_array_equal(a,encoder.transform(query.drop(columns=['sample_id','tap_iron','tap_time_len'])))
    np.testing.assert_array_equal(encoder.medians_,np.median(x.iloc[:80][list(FEATURES)],axis=0))


def test_author_config_drift_is_refused():
    r=recipe('realmlp_td');r['resolved']['lr']=999
    with pytest.raises(ValueError,match='configuration changed'):
        make_estimator(r)


@pytest.mark.parametrize('name',['realmlp_td','realmlp_td_s'])
def test_native_preprocessing_is_inner_only_then_full_refit_and_cold_prediction(tmp_path,monkeypatch,name):
    from pytabkit.models.nn_models.pipeline import MedianCenterFactory
    seen=[];original=MedianCenterFactory._fit
    def observe(self,ds):
        seen.append(ds.tensors['x_cont'].detach().cpu().numpy().copy())
        return original(self,ds)
    monkeypatch.setattr(MedianCenterFactory,'_fit',observe)
    x=frame();r=recipe(name);mask=group_safe_inner_folds(x,seed=42)['fold']!=0
    model=RealMLPRegressor(r).fit(x,x.tap_iron.values)
    # TD also visits an empty feature branch; it fits no numeric parameters.
    # No-validation refit permutes rows. Verify the exact nonempty row multisets.
    nonempty=[z for z in seen if z.shape[1]>0]
    expected={int(mask.sum()):InputEncoder().fit(x.loc[mask]).transform(x.loc[mask]),
              len(x):InputEncoder().fit(x).transform(x)}
    assert list(map(len,nonempty))==[int(mask.sum()),len(x)]
    def ordered(a):return a[np.lexsort(tuple(a[:,i] for i in range(a.shape[1]-1,-1,-1)))]
    for observed in nonempty:
        np.testing.assert_array_equal(ordered(observed),ordered(expected[len(observed)]))
    pred=model.predict(x)
    again=RealMLPRegressor(r).fit(x,x.tap_iron.values)
    np.testing.assert_array_equal(pred,again.predict(x))
    np.testing.assert_allclose(pred,model.predict(x.iloc[::-1])[::-1],rtol=0,atol=1e-4)
    np.testing.assert_allclose(pred,np.r_[model.predict(x.iloc[:40]),model.predict(x.iloc[40:])],rtol=0,atol=1e-4)
    np.testing.assert_allclose(pred[:1],model.predict(x.iloc[:1]),rtol=0,atol=1e-4)
    with (tmp_path/'model.pkl').open('wb') as f:pickle.dump(model,f)
    x.drop(columns=['sample_id','tap_iron','tap_time_len']).to_pickle(tmp_path/'query.pkl')
    code="import pickle,pandas as pd,numpy as np;from pathlib import Path;p=Path(__import__('sys').argv[1]);m=pickle.loads((p/'model.pkl').read_bytes());np.save(p/'cold.npy',m.predict(pd.read_pickle(p/'query.pkl')))"
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True)
    np.testing.assert_array_equal(pred,np.load(tmp_path/'cold.npy'))
    assert model.metadata_['optimizer_runs']==2
    assert model.metadata_['fit_rows']==100 and model.metadata_['inner_fit_rows']==80
    assert model.metadata_['schedule_horizon']==3
