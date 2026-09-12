from fractions import Fraction
import json
import numpy as np
import pytest
from preprocessing import Preprocessor, validate
from qrf_model import QRF, PARAMETERS, distribution_weights, lower_median
from worker import zero_fit, sha, restore


def test_equal_tree_mass_different_leaves():
    members = [np.array([0]),np.array([1,2,3])]
    w = distribution_weights(members,4)
    np.testing.assert_allclose(w,[.5,1/6,1/6,1/6],rtol=0,atol=1e-16)
    assert lower_median(np.array([1.,2.,3.,4.]),w,members)==1.
    assert np.median([1.,2.,3.,4.]) != 1.


@pytest.mark.parametrize('seed',range(10))
def test_independent_rational_brute_force(seed):
    rng=np.random.default_rng(seed)
    y=rng.integers(0,8,40).astype(float)
    members=[np.sort(rng.choice(40,size=int(rng.integers(1,41)),replace=False)) for _ in range(16)]
    w=distribution_weights(members,40)
    result=next(float(t) for t in sorted(set(y))
                if sum((Fraction(int((y[ids]<=t).sum()),len(ids)) for ids in members),Fraction())/len(members)>=Fraction(1,2))
    assert lower_median(y,w,members)==result
    assert (w>=0).all() and abs(w.sum()-1.)<1e-12


def test_not_average_of_tree_medians():
    y=np.array([0.,1.,2.,100.])
    members=[np.array([0,1,2]),np.array([3])]
    result=lower_median(y,distribution_weights(members,4),members)
    assert result==2. and result!=50.5


@pytest.fixture(scope='module')
def fitted():
    rng=np.random.default_rng(2026)
    x=rng.normal(size=(64,4)).astype(np.float32)
    y=np.maximum(x[:,0]*3+20.,0.)
    return QRF().fit(x,y,[str(i) for i in range(64)]),x


def test_fulltrain_partition(fitted):
    model,x=fitted
    assert len(model.leaves)==256 and model.forest.get_params()==PARAMETERS
    for mapping in model.leaves:
        assert np.array_equal(np.sort(np.concatenate(list(mapping.values()))),np.arange(len(x)))


@pytest.mark.parametrize('indices',[list(range(63,-1,-1)),[1,20,63],[20]])
def test_batch_exact_and_support(fitted,indices):
    model,x=fitted
    q,m,_=model.predict(x)
    a,b,_=model.predict(x[indices])
    assert np.array_equal(a,q[indices]) and np.array_equal(b,m[indices])
    assert set(q)<=set(model.y)


def test_save_trusted_restore_and_hash_rejection(fitted,tmp_path):
    import joblib
    from worker import environment,source_identity
    from qrf_model import PROTOCOL
    model,x=fitted
    pre=Preprocessor().fit(x,['1']*64,model.ids,['a','b','c','d'])
    (tmp_path/'preprocessor.json').write_text(json.dumps(pre.metadata()))
    joblib.dump(model,tmp_path/'forest.joblib')
    md=dict(environment=environment(),sources=source_identity(),parameters=PARAMETERS,protocol=PROTOCOL,
            forest_sha256=sha(tmp_path/'forest.joblib'),preprocessor_sha256=sha(tmp_path/'preprocessor.json'))
    (tmp_path/'bundle.json').write_text(json.dumps(md))
    with zero_fit():
        loaded,_,_=restore(tmp_path,sha(tmp_path/'bundle.json'))
        assert np.array_equal(loaded.predict(x)[0],model.predict(x)[0])
        with pytest.raises(ValueError,match='untrusted'): restore(tmp_path,'wrong')


def test_train_only_preprocessing_missing_unknown():
    pre=Preprocessor().fit(np.array([[1.,np.nan],[3.,np.nan]]),['1','2'],['a','b'],['x','empty'])
    value,before=pre.transform(np.array([[np.nan,np.nan],[9.,2.]]),['','new'],['c','d'],['x','empty'])
    assert value.dtype==np.float32 and pre.medians.tolist()==[2.,0.]
    assert pre.all_missing.tolist()==[False,True]
    assert value[0,0]==2 and value[0,-2]==1 and value[1,-1]==1
    assert before['missing_spout']==1 and before['unknown_spout']==1
    assert pre.vocabulary==['1','2']


@pytest.mark.parametrize('numeric,spout,ids,cols',[
    ([[np.inf]],['1'],['a'],['x']),([['bad']],['1'],['a'],['x']),
    ([[1.],[2.]],['1','2'],['a','a'],['x']),([[1.]],[],['a'],['x']),
    ([[1.]],['1'],['a'],['x','z'])])
def test_invalid_raw(numeric,spout,ids,cols):
    with pytest.raises(ValueError): validate(numeric,spout,ids,cols)


def test_float32_overflow():
    with pytest.raises(ValueError,match='overflow'):
        pre=Preprocessor().fit(np.array([[1e100]]),['1'],['a'],['x'])
        pre.transform(np.array([[1e100]]),['1'],['a'],['x'])


def test_zero_fit_rejects_every_api():
    from sklearn.ensemble import RandomForestRegressor
    with zero_fit() as count:
        for obj in (QRF(),RandomForestRegressor()):
            with pytest.raises(ValueError): obj.fit(None,None)
        with pytest.raises(ValueError): Preprocessor().fit(None,None,None,None)
    assert count=={'forest_fit_attempts':2,'preprocessor_fit_attempts':1}
