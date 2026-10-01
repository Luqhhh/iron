"""Observe the original TabM selector without changing its selected epoch.

The original V7/V12 selectors retain their terminal network rather than the
best-epoch network. The witness is explicitly terminal, while the original
returned best epoch remains the fresh-refit instruction. No extra warm
prediction is used: the witness observes the final original validation tensor.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import time

import numpy as np

from .data import TARGETS
from .ema_reference_artifacts import save_witness, sha, write_new


def observe_selector(original, model, frame, y, epochs, validation, directory, *,
                     identity, source_hashes, full_batch_atol, row_atol):
    """Call an unmodified original _train; archive its actual terminal state."""
    if validation is None:
        return original(model, frame, y, epochs, validation)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    old_profile=sys.getprofile(); captured={}
    source=inspect.getsourcefile(original)
    write_new(directory/'start.json',dict(identity=identity,started_ns=time.time_ns(),
        original_function=original.__module__+'.'+original.__qualname__,
        original_source=str(Path(source).resolve()),original_source_sha256=sha(source),
        input_training_ids=frame.sample_id.tolist(),
        input_calibration_ids=validation[0].sample_id.tolist()))
    def profile(current,event,arg):
        if old_profile is not None:
            old_profile(current,event,arg)
        if event!='return' or current.f_code is not original.__code__ or arg is None:
            return
        if not isinstance(arg,(int,np.integer)) or arg<1:
            raise ValueError('Original selector returned no finite selected epoch')
        state=current.f_locals
        if state['self'] is not model or state['validation'] is not validation:
            raise ValueError('Original selector frame identity changed')
        raw=state['pred'].detach().cpu().numpy().astype(float)
        observed=raw*np.asarray(model.std_)+np.asarray(model.mean_)
        stopped=int(state['epoch']);selected=int(arg)
        if selected>stopped or stopped>epochs:
            raise ValueError('Original selector epoch trace is invalid')
        metadata=dict(state_kind='original_terminal_selector_network_not_best_epoch_network',
            returned_selected_epoch=selected,terminal_epoch=stopped,
            observation='last original validation tensor; no additional warm prediction',
            calibration_labels_saved=False,extra_warm_predict_calls=0)
        receipt=save_witness(model,validation[0].drop(columns=list(TARGETS),errors='ignore'),
            observed,directory/'model-witness',identity=identity,
            training_ids=frame.sample_id.tolist(),source_hashes=source_hashes,
            full_batch_atol=full_batch_atol,row_atol=row_atol,fit_metadata=metadata)
        captured.update(metadata,witness_receipt_sha256=receipt)
    try:
        normalized={str(Path(p).resolve()):h for p,h in source_hashes.items()}
        if str(Path(source).resolve()) not in normalized or sha(source)!=normalized[str(Path(source).resolve())]:
            raise ValueError('Original selector function source was not frozen')
        fit_ids=frame.sample_id.tolist();cal_ids=validation[0].sample_id.tolist()
        if (not fit_ids or not cal_ids or len(set(fit_ids))!=len(fit_ids)
                or len(set(cal_ids))!=len(cal_ids) or set(fit_ids)&set(cal_ids)):
            raise ValueError('Original selector fit/calibration ID partition is invalid')
        if not isinstance(epochs,(int,np.integer)) or epochs<1:
            raise ValueError('Original selector epoch cap is invalid')
        sys.setprofile(profile)
        result=original(model,frame,y,epochs,validation)
        sys.setprofile(old_profile)
        if not captured:
            raise ValueError('Original selector validation return was not observed')
        if captured['returned_selected_epoch']!=result:
            raise ValueError('Original selector return changed after capture')
        write_new(directory/'complete.json',dict(identity=identity,**captured,ended_ns=time.time_ns()))
        return result
    except BaseException as error:
        sys.setprofile(old_profile)
        write_new(directory/'failure.json',dict(error=repr(error),identity=identity,ended_ns=time.time_ns()))
        raise
    finally:
        sys.setprofile(old_profile)
