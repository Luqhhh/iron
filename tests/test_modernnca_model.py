from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.data import FEATURES
from bf_tap_r2.modernnca_model import (ARMS,NeighborRegressor,Settings,clean,external_pool,feature_groups,fit_pair,validate_pair)
from bf_tap_r2.modernnca_audit import audit_partition,read_state
from bf_tap_r2.v7_periodic import file_hash


def frame():
    rng=np.random.default_rng(61001);n=54
    x=rng.normal(size=(n,len(FEATURES)))
    f=pd.DataFrame(x,columns=FEATURES)
    f['sample_id']=[f'artificial-{i}' for i in range(n)];f['spout_no']=np.arange(n)%3+1
    f['tap_time_len']=6+.3*np.sin(x[:,0])+.2*x[:,2]+.1*x[:,3]**2
    f['tap_iron']=600+20*x[:,0]
    return f


def settings():
    return Settings(dim=8,frequencies=3,embedding=4,batch_size=8,max_epochs=3,patience=2)


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    f=frame();s=settings();base=tmp_path_factory.mktemp('modernnca-partitions')
    fitting=f.iloc[:30].reset_index(drop=True);calibration=f.iloc[30:42].reset_index(drop=True)
    outer=f.iloc[:42].reset_index(drop=True);query=clean(f.iloc[42:].reset_index(drop=True))
    inner,refit=fit_pair(fitting,calibration,outer,'tap_time_len',s)
    records={}
    for arm in ARMS:
        for role,model in [('inner',inner[arm]),('outer',refit[arm])]:
            p=base/f'{role}-{arm}.npz';records[role,arm]=(p,model.save(p))
    return f,fitting,calibration,outer,query,s,inner,refit,records


def test_initial_controls_match_and_only_learned_encoder_changes(fitted):
    _,_,_,_,_,_,inner,outer,_=fitted
    for models in (inner,outer):
        fixed,learned=(models[a] for a in ARMS)
        assert set(fixed.initial_state_)==set(learned.initial_state_)
        assert all(np.array_equal(fixed.initial_state_[k],learned.initial_state_[k]) for k in fixed.initial_state_)
        assert fixed.optimizer_steps_==0 and fixed.actual_epochs_==1
        assert all(np.array_equal(v.numpy(),fixed.initial_state_[k]) for k,v in fixed.model_.state_dict().items())
        assert learned.optimizer_steps_>0
        assert any(not np.array_equal(v.numpy(),learned.initial_state_[k]) for k,v in learned.model_.state_dict().items())


def test_selector_and_fresh_refit_have_their_own_training_statistics(fitted):
    _,fitting,_,whole,_,_,inner,outer,_=fitted
    for arm in ARMS:
        assert inner[arm].fit_ids_==fitting.sample_id.tolist()
        assert outer[arm].fit_ids_==whole.sample_id.tolist()
        assert np.array_equal(inner[arm].preprocessor_.means_,fitting[list(FEATURES)].mean().to_numpy())
        assert np.array_equal(outer[arm].preprocessor_.means_,whole[list(FEATURES)].mean().to_numpy())
        assert not np.array_equal(inner[arm].preprocessor_.means_,outer[arm].preprocessor_.means_)
        assert outer[arm].selected_epoch_==inner[arm].selected_epoch_
        assert outer[arm].history_==[]


def test_all_four_models_pass_independent_audit_without_initialization_or_fit(fitted,monkeypatch):
    _,fitting,calibration,whole,query,_,inner,outer,records=fitted
    def forbidden(*args,**kwargs):raise AssertionError('Cold audit attempted initialization or fitting')
    monkeypatch.setattr(NeighborRegressor,'initialize',forbidden)
    monkeypatch.setattr(NeighborRegressor,'train',forbidden)
    monkeypatch.setattr(torch.nn.Linear,'reset_parameters',forbidden)
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    for arm in ARMS:
        for role,models,training,cal in [('inner',inner,fitting,(clean(calibration),calibration.tap_time_len.to_numpy())),('outer',outer,whole,None)]:
            p,sha=records[role,arm]
            values,audit=audit_partition(p,sha,clean(training),training.tap_time_len.to_numpy(),query,cal)
            assert audit['new_estimator_fits']==audit['new_optimizer_calls']==0
            assert np.max(np.abs(values-models[arm].predict(query)))<1e-8
            assert audit['parameters_changed']==(arm=='LEARNED_ENCODER')


def test_query_labels_are_refused_unknown_port_and_identity_permutations_are_safe(fitted):
    _,_,_,_,query,_,_,outer,_=fitted
    model=outer['LEARNED_ENCODER'];baseline=model.predict(query)
    assert np.max(np.abs(model.predict(query.iloc[::-1])[::-1]-baseline))<1e-8
    assert np.max(np.abs(np.concatenate([model.predict(query.iloc[i:i+1]) for i in range(len(query))])-baseline))<1e-8
    changed=query.copy();changed['sample_id']=[f'unseen-{i}' for i in range(len(query))]
    assert np.array_equal(model.predict(changed),baseline)
    changed['spout_no']=999
    assert np.isfinite(model.predict(changed)).all()
    changed['tap_time_len']=1
    with pytest.raises(ValueError,match='target'):model.predict(changed)


def test_external_pool_excludes_all_batch_duplicate_groups_and_training_masks_the_prefix():
    f=frame().iloc[:16].copy();f.loc[1,list(FEATURES)]=f.loc[0,list(FEATURES)].to_numpy()
    groups=feature_groups(f);batch=np.array([0,2,3]);pool=external_pool(groups,batch)
    assert 0 not in pool and 1 not in pool and not set(groups[pool])&set(groups[batch])
    model=NeighborRegressor('LEARNED_ENCODER',settings()).initialize(clean(f),f.tap_time_len.to_numpy())
    # Two duplicate rows enter a training batch, but neither label can enter
    # either query's weighted neighbor response through the batch prefix.
    batch=np.array([0,1,2]);pool=external_pool(groups,batch)
    y=model.y_[batch].clone().requires_grad_();torch.manual_seed(62001)
    out=model.model_(model.x_[batch],y,model.x_[pool],model.y_[pool],training=True,groups=torch.as_tensor(groups[batch]))
    first=torch.autograd.grad(out[0],y)[0]
    assert first[0]==first[1]==0
    assert first[2]>0


@pytest.mark.parametrize('kind',['id','group','union','response'])
def test_partition_boundary_defects_are_refused_before_optimization(kind,monkeypatch):
    f=frame();fit=f.iloc[:30].copy();cal=f.iloc[30:42].copy();whole=f.iloc[:42].copy()
    if kind=='id':cal.loc[30,'sample_id']=fit.loc[0,'sample_id']
    elif kind=='group':cal.loc[30,list(FEATURES)]=fit.loc[0,list(FEATURES)].to_numpy();whole.loc[30,list(FEATURES)]=cal.loc[30,list(FEATURES)].to_numpy()
    elif kind=='union':whole.loc[0,FEATURES[0]]+=.1
    else:whole.loc[0,'tap_time_len']+=1
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *_:pytest.fail('Invalid boundary reached optimization'))
    with pytest.raises(ValueError):fit_pair(fit,cal,whole,'tap_time_len',settings())


@pytest.mark.parametrize('kind',['bank','state','learned_state','scale','epoch'])
def test_reanchored_saved_corruption_is_rejected_by_semantic_audit(fitted,tmp_path,kind):
    _,fit,cal,_,query,_,_,_,records=fitted
    p,sha=records['inner','LEARNED_ENCODER' if kind=='learned_state' else 'FIXED_ENCODER']
    with np.load(p,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    if kind=='bank':arrays['bank_y'][0]+=.1
    elif kind=='state':arrays['state::encoder.bias'][0]+=.1
    elif kind=='learned_state':arrays['state::encoder.weight'][0,0]+=1
    else:
        metadata=json.loads(str(arrays['metadata']))
        if kind=='scale':metadata['preprocessor']['means'][0]+=.1
        else:metadata['selected_epoch']=2
        arrays['metadata']=np.asarray(json.dumps(metadata,sort_keys=True))
    bad=tmp_path/f'{kind}.npz';np.savez(bad,**arrays)
    with pytest.raises(ValueError):audit_partition(bad,file_hash(bad),clean(fit),fit.tap_time_len.to_numpy(),query,(clean(cal),cal.tap_time_len.to_numpy()))


def test_empty_pool_and_one_shot_training_refuse_instead_of_falling_back(fitted,tmp_path):
    f=frame().iloc[:8];m=NeighborRegressor('LEARNED_ENCODER',replace(settings(),batch_size=8)).initialize(clean(f),f.tap_time_len.to_numpy())
    with pytest.raises(ValueError,match='neighbor pool'):m.train(1)
    with pytest.raises(ValueError,match='one-shot'):m.train(1)
    with pytest.raises(ValueError,match='completed'):m.save(tmp_path/'failed.npz')
    assert not (tmp_path/'failed.npz').exists()
    _,_,_,_,_,_,_,outer,_=fitted
    with pytest.raises(ValueError,match='one-shot'):outer['LEARNED_ENCODER'].train(1)


def test_caller_state_anchor_remains_binding(fitted,tmp_path):
    *_,records=fitted;p,sha=records['outer','LEARNED_ENCODER'];changed=tmp_path/'changed.npz'
    changed.write_bytes(p.read_bytes()+b'extra')
    with pytest.raises(ValueError,match='anchored'):read_state(changed,sha)
