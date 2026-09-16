"""Synthetic restore and handoff checks without any model/preprocessing fit."""
import json
import numpy as np
import pytest
import inference as api
import inference_v022 as v22
from worker import zero_fit
from preprocessing import Preprocessor
from qrf_model import QRF


def test_worker_zero_fit_guards():
    with zero_fit() as counts:
        with pytest.raises(ValueError):QRF().fit(None,None,None)
        with pytest.raises(ValueError):Preprocessor().fit(None,None,None,None)
    assert counts=={'forest_fit_attempts':1,'preprocessor_fit_attempts':1}


@pytest.mark.parametrize('bad',['labels','overlap','cutoff','schema','future_reference'])
def test_worker_rejects_invalid_handoff_before_prediction(tmp_path,monkeypatch,bad):
    source=tmp_path/'input.npz';source.write_bytes(b'trusted input')
    arrays=dict(ids=np.array(['p']),numeric=np.array([[1.]]),spout=np.array(['1']),reference_ns=np.array([10]))
    info=dict(training=False,cutoff_ns=10,raw_schema_sha256='schema',ids=['p'],numeric_columns=['feature'])
    md={'input':dict(cutoff_ns=10,raw_schema_sha256='schema')}
    if bad=='labels':arrays['y']=np.array([1.])
    if bad=='overlap':info['ids']=['train']
    if bad=='cutoff':info['cutoff_ns']=9
    if bad=='schema':info['raw_schema_sha256']='other'
    if bad=='future_reference':arrays['reference_ns'][0]=9
    class Model:ids=['train']
    monkeypatch.setattr(api,'restore',lambda *args:(Model(),None,md))
    monkeypatch.setattr(api,'payload',lambda *args:(arrays,info))
    with pytest.raises(ValueError):api.inference(tmp_path,'hash',source,api.sha(source),tmp_path/'out.npz')
    assert not (tmp_path/'out.npz').exists()


def test_worker_restores_once_and_all_order_checks_without_fit(tmp_path,monkeypatch):
    source=tmp_path/'input.npz';source.write_bytes(b'trusted input')
    arrays=dict(ids=np.array(['001','010','020']),numeric=np.array([[2.],[3.],[4.]]),
                spout=np.array(['1']*3),reference_ns=np.array([10,11,12]))
    info=dict(training=False,cutoff_ns=10,raw_schema_sha256='schema',ids=arrays['ids'].tolist(),numeric_columns=['feature'])
    md=dict(input=dict(cutoff_ns=10,raw_schema_sha256='schema'),environment={},sources={},protocol='frozen',support=[1.,10.])
    calls=[]
    class Model:
        ids=['train'];training_months=np.array(['2024-11'])
        def predict(self,x,*args):return x[:,0],x[:,0]+.5,{}
    class Pre:
        def transform(self,numeric,*args):return numeric,{}
    def restore(*args):calls.append(args);return Model(),Pre(),md
    monkeypatch.setattr(api,'restore',restore)
    monkeypatch.setattr(api,'payload',lambda *args:(arrays,info))
    out=tmp_path/'out.npz'
    api.inference(tmp_path,'hash',source,api.sha(source),out)
    with np.load(out,allow_pickle=False) as saved:
        assert saved['ids'].tolist()==['001','010','020']
        assert np.array_equal(saved['mean'],[2.5,3.5,4.5])
    check=json.loads((tmp_path/'out.npz.json').read_text())
    assert len(calls)==1 and all(check['checks'].values()) and not any(check['zero_fit'].values())
    with pytest.raises(FileExistsError):api.inference(tmp_path,'hash',source,api.sha(source),out)


def test_v22_fresh_support_output_uses_certified_training_denominator(tmp_path,monkeypatch):
    source=tmp_path/'input.npz';source.write_bytes(b'trusted input')
    model_manifest=tmp_path/'model.json'
    model_manifest.write_text(json.dumps(dict(kind='V22_CERTIFIED_QRF_MODEL_v1',
        protocol='QRF_FULLTRAIN_LEAF_v1',model_path=str(tmp_path/'model'),bundle_sha256='bundle',
        training_identity_sha256='training',unique_training_rows=3,cutoff_ns=10)))
    arrays=dict(ids=np.array(['001','002']),numeric=np.array([[2.],[3.]]),
                spout=np.array(['1','2']),reference_ns=np.array([10,11]))
    info=dict(training=False,cutoff_ns=10,raw_schema_sha256='schema',
              ids=arrays['ids'].tolist(),numeric_columns=['feature'])
    bundle=dict(input=dict(cutoff_ns=10,raw_schema_sha256='schema'))
    class Model:
        ids=['a','b','c'];y=np.array([1.,2.,3.]);training_months=np.array(['x','x','x'])
        def predict(self,x,*args):
            return x[:,0],x[:,0]+1.,[{'effective_neighbors':float(v)} for v in x[:,0]]
    class Pre:
        def transform(self,numeric,*args):return numeric,{}
    monkeypatch.setattr(v22,'restore',lambda *args:(Model(),Pre(),bundle))
    monkeypatch.setattr(v22,'payload',lambda *args:(arrays,info))
    out=tmp_path/'v22.npz'
    v22.infer(model_manifest,v22.sha(model_manifest),source,v22.sha(source),out)
    with np.load(out,allow_pickle=False) as saved:
        assert saved['ids'].tolist()==['001','002']
        assert np.array_equal(saved['Q'],[2.,3.])
        assert np.array_equal(saved['effective_neighbors'],[2.,3.])
        assert np.array_equal(saved['N'],[3,3])
    report=json.loads((tmp_path/'v22.npz.json').read_text())
    assert report['unique_training_rows']==3 and all(report['checks'].values())
    assert not any(report['zero_fit'].values())


def test_v22_rejects_unregistered_model_manifest_before_joblib(tmp_path,monkeypatch):
    source=tmp_path/'input.npz';source.write_bytes(b'trusted input')
    model_manifest=tmp_path/'model.json';model_manifest.write_text('{}')
    monkeypatch.setattr(v22,'restore',lambda *args:pytest.fail('must not load arbitrary joblib'))
    with pytest.raises(ValueError,match='unknown V22'):
        v22.infer(model_manifest,v22.sha(model_manifest),source,v22.sha(source),tmp_path/'out.npz')
