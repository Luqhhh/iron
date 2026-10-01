import pytest
pytest.importorskip("torch", reason="Optional neural dependency is absent from the locked baseline environment")

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v43_bart import BartRegressor
from bf_tap_r2.v44_bart import LongBartRegressor,verify_prefix


@pytest.mark.parametrize('recipe',['STUMP','BART'])
def test_long_schedule_exactly_preserves_old_trace_draws_and_predictions(recipe,tmp_path,monkeypatch):
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/round2_v43/SPEC.yaml').read_text())['training']
    cfg.update(trees=8,burn_sweeps=8,retained_draws=4,thin=2,min_leaf_rows=2)
    rng=np.random.default_rng(91);x=rng.normal(size=(60,len(FEATURES)))
    frame=pd.DataFrame(x,columns=FEATURES);frame['sample_id']=[f's{i}' for i in range(60)];frame['spout_no']=np.arange(60)%2+1
    y=10+3*x[:,0]+2*np.maximum(x[:,1],0)
    old=BartRegressor(recipe,cfg).fit(frame,y);old.save(tmp_path/'old.json')
    monkeypatch.setattr(LongBartRegressor,'PREFIX',{'burn_sweeps':8,'thin':2,'sweeps':16})
    longer=LongBartRegressor(recipe,dict(cfg,burn_sweeps=16,thin=4)).fit(frame,y)
    assert len(longer.trace_)==32 and len(longer.draws_)==len(old.draws_)==4
    assert [d['sweep'] for d in longer.draws_]==[20,24,28,32]
    payload=json.loads((tmp_path/'old.json').read_text())
    assert verify_prefix(longer,payload,frame,old.predict(frame))['max_prediction_difference']==0
    longer.save(tmp_path/'long.json');cold=LongBartRegressor.load(tmp_path/'long.json')
    assert verify_prefix(cold,payload,frame,old.predict(frame))['max_prediction_difference']==0
    assert np.array_equal(longer.predict(frame),cold.predict(frame))
    assert np.array_equal(cold.predict(frame),cold.predict(frame.iloc[::-1])[::-1])
    broken=deepcopy(payload);broken['trace'][0]['rss']+=1
    with pytest.raises(ValueError,match='prefix mismatch'): verify_prefix(cold,broken,frame,old.predict(frame))
