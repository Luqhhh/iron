from copy import deepcopy
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from bf_tap_r2.data import FEATURES,TARGETS
from bf_tap_r2.v47_factorized import (hat_basis,FactorBasis,FactorNetwork,FactorRegressor,
                                    elementary_newton,fit_partition)
from bf_tap_r2.v47_verify import independent_prediction,verify_model


def settings():
    return yaml.safe_load(Path('configs/round2_v47/SPEC.yaml').read_text())['training']


def frame(n=35):
    rng=np.random.default_rng(9);x=rng.normal(size=(n,len(FEATURES)))
    result=pd.DataFrame(x,columns=FEATURES)
    result['sample_id']=[f'synthetic-{i}' for i in range(n)]
    result['spout_no']=np.arange(n)%3+1
    return result


def test_newton_matches_enumeration_and_gradients():
    torch.manual_seed(82)
    phi=torch.randn(4,6,3,dtype=torch.float64,requires_grad=True)
    for degree in [1,2,3,4]:
        actual=elementary_newton(phi,degree)
        brute=sum(phi[:,list(cols),:].prod(1) for cols in itertools.combinations(range(6),degree))
        torch.testing.assert_close(actual,brute,atol=1e-12,rtol=1e-12)
        g1=torch.autograd.grad(actual.sum(),phi,retain_graph=True)[0]
        g2=torch.autograd.grad(brute.sum(),phi,retain_graph=True)[0]
        torch.testing.assert_close(g1,g2,atol=1e-12,rtol=1e-12)


def test_distinct_groups_do_not_create_self_interactions():
    phi=torch.zeros(2,22,3,dtype=torch.float64);phi[:,4]=3.
    for degree in [2,3,4]: torch.testing.assert_close(elementary_newton(phi,degree),torch.zeros(2,3,dtype=torch.float64))
    phi[:,5]=2.
    torch.testing.assert_close(elementary_newton(phi,2),torch.full((2,3),6.,dtype=torch.float64))
    for degree in [3,4]: torch.testing.assert_close(elementary_newton(phi,degree),torch.zeros(2,3,dtype=torch.float64))


def test_hat_and_train_only_basis_constants_unknowns():
    np.testing.assert_allclose(hat_basis([-2,0,.5,1,3],[0,1]),[[1,0],[1,0],[.5,.5],[0,1],[0,1]])
    train=frame();train[FEATURES[0]]=7
    train[FEATURES[1]]=np.arange(len(train))%2
    basis=FactorBasis().fit(train);snapshot=deepcopy(basis.metadata())
    b,_=basis.transform(train)
    np.testing.assert_allclose(b.mean(0),0,atol=1e-15);assert np.count_nonzero(b[:,0])==0
    query=train.iloc[:3].copy();query[FEATURES[0]]=1e9;query.spout_no=999
    q,_=basis.transform(query)
    np.testing.assert_allclose(q[:,-1],np.broadcast_to(-basis.centers_[-1],q[:,-1].shape))
    assert snapshot==basis.metadata()
    with pytest.raises(ValueError,match='targets'): basis.transform(query.assign(tap_iron=3))


def test_arm_pair_initialization_identical_and_extra_orders_active():
    s=settings();b=FactorBasis().fit(frame());basis,linear=b.transform(frame())
    a=FactorNetwork('AFM2',s,b.sizes_,b.width_);h=FactorNetwork('AHOFM4',s,b.sizes_,b.width_)
    for key,value in a.state_dict().items(): torch.testing.assert_close(value,h.state_dict()[key],atol=0,rtol=0)
    inputs=[torch.tensor(x) for x in (basis,linear)]
    assert not torch.allclose(a(*inputs),h(*inputs))
    with torch.no_grad():
        h.rank_weights['3'].zero_();h.rank_weights['4'].zero_()
    torch.testing.assert_close(a(*inputs),h(*inputs),atol=0,rtol=0)


def test_model_save_independent_predict_and_trace_tamper(tmp_path):
    s=settings();s.update(max_epochs=3,rank_per_order=3,batch_size=16)
    data=frame(45);y=2+data[FEATURES[0]].to_numpy()+data[FEATURES[1]].to_numpy()**2
    m=FactorRegressor('AHOFM4',s).initialize(data.iloc[:30],y[:30])
    m.train(3,(data.iloc[30:],y[30:]));path=tmp_path/'model.json';m.save(path)
    p=m.predict(data.iloc[30:]);cold=FactorRegressor.load(path)
    for values in [cold.predict(data.iloc[30:]),cold.predict(data.iloc[30:][::-1])[::-1],
                   np.concatenate([cold.predict(data.iloc[i:i+1]) for i in range(30,45)]),
                   independent_prediction(path,data.iloc[30:])]:
        np.testing.assert_allclose(values,p,rtol=0,atol=1e-10)
    verify_model(path,data.iloc[:30],y[:30],'AHOFM4',s,m.metadata())
    payload=json.loads(path.read_text());payload['metadata']['selected_epoch']=0
    bad=tmp_path/'tamper.json';bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='selected epoch'): verify_model(bad,data.iloc[:30],y[:30],'AHOFM4',s)
    with pytest.raises(FileExistsError): m.save(path)


def test_train_only_fresh_refit_and_label_rejection():
    s=settings();s.update(max_epochs=2,rank_per_order=2,batch_size=32)
    data=frame(35)
    for target in TARGETS: data[target]=4+data[FEATURES[0]]
    clean=data.iloc[30:].drop(columns=list(TARGETS));outer=data.iloc[:30]
    m,p,meta,cp=fit_partition(outer.iloc[:20],outer.iloc[20:],outer,clean,TARGETS[0],
                            'AFM2',s,np.full(10,4.),[0.,.5,1.])
    assert meta['refit']['selected_epoch']==meta['calibration']['selected_epoch']
    assert meta['refit']['fit_ids_digest']!=meta['calibration']['fit_ids_digest']
    assert p.shape==(5,) and cp.shape==(10,)
    with pytest.raises(ValueError,match='labels'):
        fit_partition(outer.iloc[:20],outer.iloc[20:],outer,data.iloc[30:],TARGETS[0],'AFM2',s,np.full(10,4.),[0.,1.])
    with pytest.raises(ValueError,match='overlaps'):
        fit_partition(outer.iloc[:20],outer.iloc[20:],outer,outer.iloc[:1].drop(columns=list(TARGETS)),TARGETS[0],'AFM2',s,np.full(10,4.),[0.,1.])


def test_roughness_excludes_padding_and_categories():
    s=settings();m=FactorNetwork('AHOFM4',s,[2]*len(FEATURES)+[3],16)
    assert m.roughness().item()==0
    with torch.no_grad():
        m.additive[:,-1]=1e6;m.additive[-1,:]=1e7
    assert m.roughness().item()==0


def test_frozen_gates_reject_control_and_small_gain():
    from bf_tap_r2.v47_run import select_finalists
    spec=yaml.safe_load(Path('configs/round2_v47/SPEC.yaml').read_text())
    records=[{'target':t,'recipe':r,'both_seeds_positive':True,'paired_seed_summary':{'mean':g}}
             for t in TARGETS for r,g in [('AFM2',.02),('AHOFM4',.009)]]
    assert not any(select_finalists(records,spec).values())
    for r in records:
        if r['recipe']=='AHOFM4': r['paired_seed_summary']['mean']=.021
    assert all(v=='AHOFM4' for v in select_finalists(records,spec).values())
