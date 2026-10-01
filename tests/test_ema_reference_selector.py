"""Selector observer fixtures have no optimizer or official targets."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.ema_reference_artifacts import audit_witness,sha
from bf_tap_r2.ema_reference_selector import observe_selector
import bf_tap_r2.ema_reference_selector as observer_module


class SelectorFixture:
    def __init__(self,joint=False):
        self.mean_=np.array([7.,11.]) if joint else 7.
        self.std_=np.array([2.,3.]) if joint else 2.
        self.joint=joint;self.predict_calls=0
        self.training_object={'retained':True}

    def _tensor(self,frame):
        a=torch.tensor(frame.x.to_numpy(copy=True),dtype=torch.float32)
        return torch.stack([a,2*a],dim=1) if self.joint else a

    def _train(self,frame,y,epochs,validation=None):
        if validation is None:return epochs
        pred=self._tensor(validation[0])
        epoch=5
        return 3

    def predict(self,frame):
        self.predict_calls+=1
        return self._tensor(frame).numpy().astype(float)*self.std_+self.mean_


class FailedSelector(SelectorFixture):
    def _train(self,*args,**kwargs):raise RuntimeError('fixture training failure')


def args(tmp_path,model):
    fit=pd.DataFrame({'sample_id':['F0','F1'],'spout_no':[1,2],'x':[0.,1.]})
    cal=pd.DataFrame({'sample_id':['C0','C1','C2'],'spout_no':[1,2,1],
                      'x':[2.,3.,4.],'tap_iron':[99.,99.,99.],'tap_time_len':[99.,99.,99.]})
    identity=dict(source_directory=str(tmp_path.resolve()),split_seed=271828,fold=0,
                  trial_id='selector-fixture',fit_call_id='selector-terminal')
    return fit,cal,dict(identity=identity,source_hashes={str(Path(__file__).resolve()):sha(__file__),
        str(Path(observer_module.__file__).resolve()):sha(observer_module.__file__)},
        full_batch_atol=0,row_atol=0)


@pytest.mark.parametrize('joint',[False,True])
def test_original_best_epoch_and_terminal_validation_tensor_are_both_preserved(tmp_path,joint):
    model=SelectorFixture(joint);fit,cal,kw=args(tmp_path,model);directory=tmp_path/'selector'
    result=observe_selector(SelectorFixture._train,model,fit,np.ones(len(fit)),10,(cal,np.ones(len(cal))),directory,**kw)
    assert result==3 and model.predict_calls==0 and model.training_object=={'retained':True}
    complete=json.loads((directory/'complete.json').read_text())
    assert complete['returned_selected_epoch']==3 and complete['terminal_epoch']==5
    assert complete['extra_warm_predict_calls']==0 and complete['calibration_labels_saved'] is False
    receipt=json.loads((directory/'model-witness/complete.json').read_text())
    assert receipt['training_ids']==['F0','F1'] and receipt['query_ids']==['C0','C1','C2']
    cold=audit_witness(directory/'model-witness',complete['witness_receipt_sha256'])
    assert cold['differences']['full']==0 and model.predict_calls==0


def test_refit_calls_original_without_selector_files_or_observer(tmp_path):
    model=SelectorFixture();fit,cal,kw=args(tmp_path,model);directory=tmp_path/'refit'
    assert observe_selector(SelectorFixture._train,model,fit,np.ones(len(fit)),8,None,directory,**kw)==8
    assert not directory.exists()


def test_original_failure_preserves_failed_directory_and_profile(tmp_path):
    model=FailedSelector();fit,cal,kw=args(tmp_path,model);directory=tmp_path/'failed';before=sys.getprofile()
    with pytest.raises(RuntimeError,match='fixture training failure'):
        observe_selector(FailedSelector._train,model,fit,np.ones(len(fit)),10,(cal,np.ones(len(cal))),directory,**kw)
    assert sys.getprofile() is before and (directory/'start.json').exists() and (directory/'failure.json').exists()
    with pytest.raises(FileExistsError):
        observe_selector(FailedSelector._train,model,fit,np.ones(len(fit)),10,(cal,np.ones(len(cal))),directory,**kw)


def test_existing_profile_is_restored_and_still_receives_original_calls(tmp_path):
    model=SelectorFixture();fit,cal,kw=args(tmp_path,model);calls=[];before=sys.getprofile()
    def prior(frame,event,arg):
        if event=='call' and frame.f_code is SelectorFixture._train.__code__:calls.append('original')
    try:
        sys.setprofile(prior)
        observe_selector(SelectorFixture._train,model,fit,np.ones(len(fit)),10,(cal,np.ones(len(cal))),tmp_path/'selector',**kw)
        assert sys.getprofile() is prior and calls==['original']
    finally:sys.setprofile(before)


def test_unfrozen_original_function_is_refused_before_call(tmp_path):
    model=SelectorFixture();fit,cal,kw=args(tmp_path,model)
    del kw['source_hashes'][str(Path(__file__).resolve())]
    calls=[];before=sys.getprofile()
    def prior(frame,event,arg):
        if event=='call' and frame.f_code is SelectorFixture._train.__code__:calls.append('unexpected')
    try:
        sys.setprofile(prior)
        with pytest.raises(ValueError,match='function source was not frozen'):
            observe_selector(SelectorFixture._train,model,fit,np.ones(len(fit)),10,(cal,np.ones(len(cal))),tmp_path/'unfrozen',**kw)
        assert calls==[]
    finally:sys.setprofile(before)


def test_calibration_overlap_is_refused_before_original_selector(tmp_path):
    model=SelectorFixture();fit,cal,kw=args(tmp_path,model);cal.loc[0,'sample_id']='F0'
    with pytest.raises(ValueError,match='partition is invalid'):
        observe_selector(SelectorFixture._train,model,fit,np.ones(len(fit)),10,(cal,np.ones(len(cal))),tmp_path/'overlap',**kw)
    assert (tmp_path/'overlap/failure.json').exists()
