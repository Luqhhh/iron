"""Storage and actual cold-process guards; fixtures contain no official labels."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import bf_tap_r2.ema_reference_artifacts as artifacts


class FrozenPredictor:
    def __init__(self, two_outputs=False):
        self.offset = 4.25
        self.predict_calls = 0
        self.training_object = {'retained': True}
        self.two_outputs = two_outputs

    def predict(self, frame):
        self.predict_calls += 1
        a = frame['x'].to_numpy()+self.offset
        return np.column_stack([a, 2*a]) if self.two_outputs else a

    def fit(self, *args, **kwargs):
        raise AssertionError('The storage/cold path must never fit')


class RefittingPredictor(FrozenPredictor):
    def predict(self, frame):
        self.fit(frame)
        return super().predict(frame)


class CSVReadingPredictor(FrozenPredictor):
    def predict(self, frame):
        pd.read_csv('/never-created/train_samples.csv')
        return super().predict(frame)


class BatchDependentPredictor(FrozenPredictor):
    def predict(self, frame):
        return frame['x'].to_numpy()-frame['x'].mean()


def frame(rows=29):
    return pd.DataFrame({'sample_id':[f'QUERY_{i}' for i in range(rows)],
                         'spout_no':np.arange(rows)%2+1, 'x':np.linspace(1,4,rows)})


def save(tmp_path, model=None, query=None, observed=None, origin=None):
    model = model or FrozenPredictor()
    query = frame() if query is None else query
    if observed is None:
        a=query['x'].to_numpy()+4.25
        observed=np.column_stack([a,2*a]) if model.two_outputs else a
    receipt=artifacts.save_witness(model,query,observed,tmp_path/'witness',
        identity=dict(source_directory=str(Path(origin or tmp_path).resolve()),split_seed=271828,
                      fold=0,trial_id='synthetic-witness',fit_call_id='refit-1'),
        training_ids=['FIT_0','FIT_1'],source_hashes={str(Path(__file__).resolve()):artifacts.sha(__file__),
             str(Path(artifacts.__file__).resolve()):artifacts.sha(artifacts.__file__)},
        full_batch_atol=0,row_atol=1e-12,fit_metadata={'fixture':'preconfigured predictor; no fitting'})
    return tmp_path/'witness',receipt,model


def rewrite_receipt(directory, mutate):
    p=directory/'complete.json';d=json.loads(p.read_text());mutate(d)
    p.write_text(json.dumps(d)+'\n')
    return artifacts.sha(p)


def test_storage_does_not_predict_fit_or_remove_training_attrs(tmp_path):
    directory,receipt,model=save(tmp_path)
    assert model.predict_calls==0 and model.training_object=={'retained':True}
    result=artifacts.audit_witness(directory,receipt)
    assert result['status']=='passed' and result['new_fits']==0
    assert result['differences']=={'full':0,'reverse':0,'chunk':0,'singleton':0}
    assert model.predict_calls==0


def test_actual_independent_python_process_and_multitarget_axis(tmp_path):
    directory,receipt,_=save(tmp_path,FrozenPredictor(two_outputs=True),frame(277))
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(Path(__file__).parent),str(Path(artifacts.__file__).parents[1]),env.get('PYTHONPATH','')])
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    result=subprocess.run([sys.executable,'-m','bf_tap_r2.ema_reference_artifacts',
        '--directory',str(directory),'--receipt-sha256',receipt],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout)
    assert output['rows']==277 and output['differences']['chunk']==0
    assert output['training_csv_reads']==0


@pytest.mark.parametrize('bad',['target','duplicate','overlap'])
def test_targets_and_partition_identity_refused_before_write(tmp_path,bad):
    query=frame()
    if bad=='target':query['tap_time_len']=99.
    elif bad=='duplicate':query.loc[1,'sample_id']=query.loc[0,'sample_id']
    else:query.loc[0,'sample_id']='FIT_0'
    with pytest.raises(ValueError):save(tmp_path,query=query)
    assert not (tmp_path/'witness').exists()


def test_model_bytes_tamper_refused_before_loading(tmp_path):
    directory,receipt,_=save(tmp_path)
    with (directory/'model.pkl').open('ab') as f:f.write(b'extra')
    with pytest.raises(ValueError,match='artifact changed'):
        artifacts.audit_witness(directory,receipt)


def test_external_receipt_refuses_rewritten_local_identity(tmp_path):
    directory,receipt,_=save(tmp_path)
    rewrite_receipt(directory,lambda d:d['identity'].update(split_seed=314159))
    with pytest.raises(ValueError,match='External predictor receipt'):
        artifacts.audit_witness(directory,receipt)


def test_independent_cold_catches_rehashed_fabricated_observed_predictions(tmp_path):
    directory,_,_=save(tmp_path)
    np.save(directory/'observed.npy',np.zeros(29))
    receipt=rewrite_receipt(directory,lambda d:d['hashes'].update({'observed.npy':artifacts.sha(directory/'observed.npy')}))
    with pytest.raises(ValueError,match='prediction mismatch'):
        artifacts.audit_witness(directory,receipt)


def test_independent_cold_rejects_batch_dependent_predictor(tmp_path):
    query=frame();directory,receipt,_=save(tmp_path,BatchDependentPredictor(),query,query.x.to_numpy()-query.x.mean())
    with pytest.raises(ValueError,match='row-independent prediction mismatch'):
        artifacts.audit_witness(directory,receipt)


@pytest.mark.parametrize('model,error',[(RefittingPredictor,'attempted fitting'),(CSVReadingPredictor,'CSV read')])
def test_no_refitting_or_csv_access_during_cold_prediction(tmp_path,model,error):
    directory,receipt,_=save(tmp_path,model())
    with pytest.raises(ValueError,match=error):artifacts.audit_witness(directory,receipt)
    assert not (directory/'cold-audit.json').exists()


def test_local_query_rehash_does_not_change_declared_ids(tmp_path):
    import pickle
    directory,_,_=save(tmp_path)
    query=frame();query.loc[0,'sample_id']='OTHER'
    with (directory/'query.pkl').open('wb') as f:pickle.dump(query,f,protocol=5)
    receipt=rewrite_receipt(directory,lambda d:d['hashes'].update({'query.pkl':artifacts.sha(directory/'query.pkl')}))
    with pytest.raises(ValueError,match='partition identity mismatch'):
        artifacts.audit_witness(directory,receipt)


def test_nonfinite_tolerance_refused_even_with_new_external_receipt(tmp_path):
    directory,_,_=save(tmp_path)
    receipt=rewrite_receipt(directory,lambda d:d.update(full_batch_atol=float('nan')))
    with pytest.raises(ValueError,match='Invalid frozen cold tolerance'):
        artifacts.audit_witness(directory,receipt)


def test_actual_loaded_class_source_is_verified(tmp_path):
    directory,_,_=save(tmp_path)
    receipt=rewrite_receipt(directory,lambda d:d.update(model_source_sha256='0'*64))
    with pytest.raises(ValueError,match='Actual cold predictor class source'):
        artifacts.audit_witness(directory,receipt)


def test_fit_origin_is_preserved_separately_from_witness_location(tmp_path):
    origin=tmp_path/'original-fit-run';origin.mkdir()
    directory,receipt,_=save(tmp_path/'copy-one',origin=origin)
    result=artifacts.audit_witness(directory,receipt)
    second,second_receipt,_=save(tmp_path/'copy-two',origin=origin)
    second_result=artifacts.audit_witness(second,second_receipt)
    assert result['identity']==second_result['identity']
    assert result['identity']['source_directory']==str(origin.resolve())
    assert json.loads((directory/'complete.json').read_text())['artifact_directory']==str(directory.resolve())


def test_repeated_save_and_cold_audit_do_not_overwrite_evidence(tmp_path):
    directory,receipt,_=save(tmp_path)
    complete_before=(directory/'complete.json').read_bytes()
    with pytest.raises(FileExistsError):save(tmp_path)
    assert (directory/'complete.json').read_bytes()==complete_before
    artifacts.audit_witness(directory,receipt)
    before=(directory/'cold-audit.json').read_bytes()
    with pytest.raises(FileExistsError):artifacts.audit_witness(directory,receipt)
    assert (directory/'cold-audit.json').read_bytes()==before


def test_serialization_failure_consumes_directory_and_preserves_failure(tmp_path):
    model=FrozenPredictor();model.unserializable=lambda x:x
    with pytest.raises(Exception):save(tmp_path,model)
    directory=tmp_path/'witness'
    assert (directory/'start.json').exists() and (directory/'failure.json').exists()
    assert not (directory/'complete.json').exists()
    with pytest.raises(FileExistsError):save(tmp_path)
