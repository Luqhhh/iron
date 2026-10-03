import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.ema_nested_residual import (
    choose, corrected, features, fit_head, inner_parts, predict_head, split_training,
)
from bf_tap_r2.v3_4_bags import group_safe_inner_folds


def sample():
    rng=np.random.default_rng(813)
    frame=pd.DataFrame(rng.normal(size=(150,len(FEATURES))),columns=FEATURES)
    frame['sample_id']=[f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no']=np.arange(len(frame))%3
    frame['tap_time_len']=100+frame[FEATURES[0]]*3+rng.normal(size=len(frame))
    frame['tap_iron']=500+frame[FEATURES[1]]*6
    # Exact-feature duplicates must remain together through both levels.
    frame.loc[3,list(FEATURES)]=frame.loc[0,list(FEATURES)].to_numpy()
    return frame


def test_nested_partitions_exclude_outer_ids_and_ignore_outer_labels():
    frame=sample();fv=group_safe_inner_folds(frame,seed=42)['fold']
    for fold in range(5):
        training,query=split_training(frame,fv,fold)
        changed=frame.copy();changed.loc[fv==fold,list(TARGETS)]=1e90
        other,q2=split_training(changed,fv,fold)
        pd.testing.assert_frame_equal(training,other);pd.testing.assert_frame_equal(query,q2)
        seen=[]
        for inner in range(5):
            fitting,held,mask,info=inner_parts(training,inner)
            assert not set(fitting.sample_id)&(set(held.sample_id)|set(query.sample_id))
            assert not any(t in held or t in query for t in TARGETS)
            assert len(fitting)+len(held)==len(training)
            assert set(info['fold'])==set(range(5))
            seen.extend(held.sample_id)
        assert sorted(seen)==sorted(training.sample_id) and len(seen)==len(set(seen))
    assert fv[0]==fv[3]


def test_design_includes_base_and_rejects_targets():
    frame=sample();query=frame.drop(columns=list(TARGETS));base=np.linspace(90,110,len(frame))
    x=features(query,base,[0,1,2])
    np.testing.assert_array_equal(x[:,-1],base)
    np.testing.assert_array_equal(x[:,:len(FEATURES)],frame[list(FEATURES)])
    assert x.shape==(len(frame),len(FEATURES)+4)
    with pytest.raises(ValueError,match='label-free'):features(frame,base,[0,1,2])
    with pytest.raises(ValueError,match='Misaligned'):features(query,base[:-1],[0,1,2])
    query=query.copy();query['spout_no']=99
    assert not features(query,base,[0,1,2])[:,len(FEATURES):-1].any()


@pytest.mark.parametrize('family',['RIDGE','GBM'])
def test_head_state_new_process_no_fit_and_training_scale(tmp_path,family):
    frame=sample().drop(columns=list(TARGETS));base=np.linspace(90,110,len(frame))
    x=features(frame,base,[0,1,2]);fitx=x[:120];query=x[120:].copy();query[:,0]+=100
    residual=fitx[:,0]*2-fitx[:,1]+.3
    directory=tmp_path/family;warm=fit_head(family,fitx,residual,directory)(query)
    np.save(tmp_path/'query.npy',query);np.save(tmp_path/'warm.npy',warm)
    code='''
import sys, numpy as np
from pathlib import Path
from bf_tap_r2.ema_nested_residual import setup, forbid_training, predict_head
setup()
p=Path(sys.argv[1]);family=sys.argv[2]
with forbid_training():
 got=predict_head(family,p/family,np.load(p/'query.npy'))
 np.testing.assert_allclose(got,np.load(p/'warm.npy'),rtol=0,atol=1e-10)
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path),family],check=True,env=os.environ.copy())
    if family=='RIDGE':
        with np.load(directory/'state.npz') as state:
            np.testing.assert_allclose(state['mean'],fitx.mean(0),rtol=0,atol=1e-14)
            scale=fitx.std(0);scale[scale==0]=1
            z=(fitx-fitx.mean(0))/scale;zm=z.mean(0);ym=residual.mean();zz=z-zm
            coef=np.linalg.solve(zz.T@zz+10*np.eye(z.shape[1]),zz.T@(residual-ym))
            expected=((query-fitx.mean(0))/scale-zm)@coef+ym
            np.testing.assert_allclose(warm,expected,rtol=0,atol=1e-9)
    with pytest.raises(FileExistsError):fit_head(family,fitx,residual,directory)


def test_fixed_correction_selection_and_invalid_extrapolation():
    q=np.array([100.,200.]);h=np.array([8.,-8.])
    np.testing.assert_array_equal(corrected(q,h),[101.5,198.5])
    np.testing.assert_array_equal(corrected(q,np.zeros(2)),q)
    with pytest.raises(ValueError,match='no clipping'):corrected(q,[-10000,0])
    assert choose({'RIDGE':{'42':.001,'3407':.002},'GBM':{'42':.002,'3407':.001}})=='RIDGE'
    assert choose({'RIDGE':{'42':-.001,'3407':.002},'GBM':{'42':.002,'3407':-.001}}) is None
    with pytest.raises(ValueError,match='Complete'):choose({'RIDGE':{'42':1},'GBM':{'42':1}})
