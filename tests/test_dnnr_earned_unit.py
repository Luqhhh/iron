import json

import numpy as np
import pytest

from bf_tap_r2.dnnr_model import ARMS, Regressor
from bf_tap_r2.dnnr_execution import execute_unit
from bf_tap_r2.dnnr_earned_unit import execute_earned, audit_earned
from bf_tap_r2.dnnr_ledger import ReservationLedger, file_hash
from test_dnnr_model import sample


TASK = dict(target='tap_time_len', seed=7777, fold=0)


@pytest.fixture
def full_and_subsets(tmp_path):
    training = sample(75); query = sample(11, 57322, 200)
    y = 30+training.air_volume.to_numpy()**2+training.hot_air_press.to_numpy()
    full_ledger = ReservationLedger.create(tmp_path/'full-ledger', dict(pair_unit=1, estimator=6, metric_epoch=2, derivative_bank=4))
    full = execute_unit(TASK, training, y, query, tmp_path/'full', full_ledger.root, full_ledger.policy_sha256)
    outputs=[]
    for i,arms in enumerate((ARMS[:2],ARMS[1:])):
        ledger = ReservationLedger.create(tmp_path/f'ledger-{i}', dict(pair_unit=1,estimator=4,metric_epoch=2 if i else 0,derivative_bank=4 if i else 2))
        unit = tmp_path/f'unit-{i}'
        record = execute_earned(TASK,training,y,query,unit,ledger.root,ledger.policy_sha256,arms)
        outputs.append((unit,record,ledger,arms))
    return training,y,query,tmp_path/'full',outputs


def test_omitting_failed_candidate_preserves_all_common_states_and_predictions(full_and_subsets, monkeypatch):
    training,y,query,full,outputs = full_and_subsets
    def no_fit(*args,**kwargs): raise AssertionError('Cold audit attempted estimator fit')
    monkeypatch.setattr(Regressor,'fit',no_fit)
    for unit,record,ledger,arms in outputs:
        values,audit=audit_earned(unit,record['complete_sha256'],TASK,training,y,query,arms,ledger_root=ledger.root)
        assert audit['saved_models']==4 and audit['new_estimator_fits']==0
        for arm in arms:
            np.testing.assert_array_equal(values[arm],np.load(full/f'prediction-{arm}.npy',allow_pickle=False))
            for role in ('inner','outer'):
                with np.load(unit/'models'/f'{role}-{arm}.npz',allow_pickle=False) as one, np.load(full/'models'/f'{role}-{arm}.npz',allow_pickle=False) as two:
                    assert set(one.files)==set(two.files)
                    for name in one.files:
                        if name=='metadata':
                            left,right=json.loads(str(one[name])),json.loads(str(two[name]))
                            # Timing is evidence, not a fitted parameter.
                            left.pop('elapsed_seconds');right.pop('elapsed_seconds')
                            assert left==right
                        else:np.testing.assert_array_equal(one[name],two[name])
        count=ledger.inspect()
        assert count['completed']['estimator']==4 and not any(count['failed'].values())


@pytest.mark.parametrize('defect',['counts','selection','prediction_dtype','extra'])
def test_subset_cold_audit_refuses_rehashed_corrupt_evidence(full_and_subsets,defect):
    training,y,query,_,outputs=full_and_subsets
    unit,record,ledger,arms=outputs[0]
    data=json.loads((unit/'complete.json').read_text())
    if defect=='counts':data['counts']['derivative_local_solutions']+=1
    elif defect=='selection':data['selected_metric_epochs']=1
    elif defect=='prediction_dtype':
        path=unit/'prediction-DNNR_FIXED.npy'
        with path.open('wb') as stream:np.save(stream,np.load(unit/'prediction-KNN_FIXED.npy',allow_pickle=False).astype(np.float32))
        data['prediction_hashes'][path.name]=file_hash(path)
    else:(unit/'extra.json').write_text('{}')
    (unit/'complete.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit_earned(unit,file_hash(unit/'complete.json'),TASK,training,y,query,arms,ledger_root=ledger.root)


def test_subset_refuses_retries_and_illegal_single_arm(full_and_subsets,tmp_path):
    training,y,query,_,outputs=full_and_subsets
    unit,record,ledger,arms=outputs[0]
    before=ledger.inspect()
    with pytest.raises(FileExistsError):execute_earned(TASK,training,y,query,unit,ledger.root,ledger.policy_sha256,arms)
    with pytest.raises(ValueError):execute_earned(TASK,training,y,query,tmp_path/'single',ledger.root,ledger.policy_sha256,('DNNR_FIXED',))
    assert ledger.inspect()==before
