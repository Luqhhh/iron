from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2.local_ridge_neighborhood import LocalRidgeNeighborhood, local_solution, independent_predict, RIDGE
from bf_tap_r2.local_ridge_run import audit_state, compare, synthetic


def data():
    rng=np.random.default_rng(43);x=rng.normal(size=(100,len(FEATURES)))
    x[:,1]=x[:,0]+1e-12*x[:,1];x[:,-1]=0
    frame=pd.DataFrame(x,columns=FEATURES)
    frame.insert(0,'sample_id',[f'id-{i:03d}' for i in range(100)])
    frame.insert(1,'spout_no',np.arange(100)%4+1)
    y=np.column_stack([100+x[:,0],20+x[:,2]])
    return frame.iloc[:80].copy(),y[:80],frame.iloc[80:].copy()


def test_ill_conditioned_slopes_match_independent_augmented_solution():
    train,y,query=data();model=LocalRidgeNeighborhood().fit(train,y)
    a=model.predict(query);b=independent_predict(model,query)
    np.testing.assert_array_equal(a['neighbors'],b['neighbors'])
    np.testing.assert_allclose(a['linear'],b['linear'],rtol=0,atol=1e-10)
    assert a['condition'].max()<1e6 and abs(a['beta']).max()<1e4
    # Both arms use exactly the same64 neighbors; no independent KNN search.
    expected=np.stack([y[indices].mean(0) for indices in a['neighbors']])
    np.testing.assert_allclose(a['knn'],expected,rtol=0,atol=1e-10)


def test_rank_zero_solution_has_unpenalized_mean_and_zero_slopes():
    x=np.zeros((64,25));y=np.column_stack([np.arange(64),np.arange(64)**2])
    pred,knn,beta,condition=local_solution(x,y,np.ones(25))
    np.testing.assert_array_equal(pred,y.mean(0));np.testing.assert_array_equal(knn,pred)
    np.testing.assert_array_equal(beta,np.zeros((25,2)));assert condition==1


def test_fixed_mean_loss_normalization():
    x=np.arange(64,dtype=float)[:,None];y=np.column_stack([x[:,0],2*x[:,0]])
    _,_,beta,_=local_solution(x,y,np.array([100.]))
    v=x.var();np.testing.assert_allclose(beta[0],[v/(v+RIDGE),2*v/(v+RIDGE)],atol=1e-12)


def test_ties_are_by_id_and_coincident_query_is_allowed():
    train,y,query=data();train[list(FEATURES)]=0;query[list(FEATURES)]=0
    train['spout_no']=1;query['spout_no']=1
    model=LocalRidgeNeighborhood().fit(train.iloc[::-1],y[::-1]);a=model.predict(query)
    assert model.ids[a['neighbors'][0]].tolist()==sorted(train.sample_id)[:64]
    np.testing.assert_allclose(a['linear'],a['knn'],atol=0)


def test_train_only_preprocessing_unknown_spout_and_query_order():
    train,y,query=data();model=LocalRidgeNeighborhood().fit(train,y)
    mean=model.mean.copy();query['spout_no']=999
    a=model.predict(query);b=model.predict(query.iloc[::-1])
    np.testing.assert_array_equal(model.mean,mean)
    assert (model.encode(query)[:,len(FEATURES)]==1).all()
    np.testing.assert_allclose(a['linear'],b['linear'][::-1],atol=0)


@pytest.mark.parametrize('bad',['label','id_overlap','nan','duplicate'])
def test_invalid_query_rejected(bad):
    train,y,query=data();model=LocalRidgeNeighborhood().fit(train,y)
    if bad=='label':query['tap_iron']=1
    elif bad=='id_overlap':query.iloc[0,query.columns.get_loc('sample_id')]=train.sample_id.iloc[0]
    elif bad=='nan':query.iloc[0,query.columns.get_loc(FEATURES[0])]=np.nan
    else:query.iloc[0,query.columns.get_loc('sample_id')]=query.sample_id.iloc[1]
    with pytest.raises(ValueError):model.predict(query)


def test_cold_audit_no_fit_and_rehashed_corruption_rejected(tmp_path,monkeypatch):
    train,y,query=data();model=LocalRidgeNeighborhood().fit(train,y);saved=model.predict(query)
    state=tmp_path/'state.npz';model.save(state)
    monkeypatch.setattr(LocalRidgeNeighborhood,'fit',lambda *a,**k: (_ for _ in ()).throw(AssertionError('fit forbidden')))
    cold=LocalRidgeNeighborhood.load(state);audit_state(cold,train,y)
    assert compare(cold,query,saved)<1e-8
    cold.y[0,0]+=.01
    with pytest.raises(AssertionError):audit_state(cold,train,y)
    cold=LocalRidgeNeighborhood.load(state);cold.mean[0]+=.01
    with pytest.raises(AssertionError):audit_state(cold,train,y)


def test_saved_recipe_corruption_and_overwrite_rejected(tmp_path):
    train,y,_=data();model=LocalRidgeNeighborhood().fit(train,y);path=tmp_path/'state.npz'
    model.save(path)
    with pytest.raises(FileExistsError):model.save(path)
    with np.load(path,allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    arrays['ridge']=np.array(.01)
    np.savez(tmp_path/'bad.npz',**arrays)
    with pytest.raises(ValueError):LocalRidgeNeighborhood.load(tmp_path/'bad.npz')


def test_synthetic_complete_size_and_duplicate_dimensions():
    train,y,query=synthetic()
    assert (len(train),len(query),y.shape)==(2203,551,(2203,2))
    assert not set(train.sample_id)&set(query.sample_id)
    assert (train[FEATURES[-1]]==0).all()
