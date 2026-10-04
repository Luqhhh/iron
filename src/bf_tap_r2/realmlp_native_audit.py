"""Observe and audit the immutable RealMLP fit, without replacing its updates."""
from contextlib import contextmanager, ExitStack
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import realmlp_state_adapter as adapter
from .data import FEATURES, TARGETS
from .v3_4_bags import group_safe_inner_folds


def array_digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def row_multiset_digest(a):
    a=np.asarray(a,dtype=np.float32)
    if a.ndim!=2 or not np.isfinite(a).all():
        raise ValueError('Finite two-dimensional preprocessing input required')
    order=np.lexsort(tuple(a[:,i] for i in range(a.shape[1]-1,-1,-1)))
    return array_digest(a[order])


def native_state_digest(estimator):
    return {k:array_digest(v.detach().cpu().numpy())
        for k,v in estimator.alg_interface_.model.model.state_dict().items()}


def last_best_epoch(history):
    if not history or [r['epoch'] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError('Complete ordered validation history required')
    best=float('inf');selected=0
    for row in history:
        value=row['mae']
        if not math.isfinite(value) or value<0:
            raise ValueError('Finite nonnegative native validation metric required')
        if value<=best:
            best,selected=value,row['epoch']
    return selected,best


@contextmanager
def observe_native(directory):
    """Record two original Adam optimizers and callbacks; no extra RNG calls."""
    import torch
    from pytabkit.models.training.lightning_modules import TabNNModule
    from pytabkit.models.training.metrics import Metrics
    from pytabkit.models.nn_models.pipeline import MedianCenterFactory
    directory=Path(directory)
    from .ema_nested_residual import write
    record=dict(optimizers=[],modules=[],preprocessing=[])
    opt_ids={};module_ids={};current_metric=[]
    ctor,step=torch.optim.Adam.__init__,torch.optim.Adam.step
    configure,train,validation=TabNNModule.configure_optimizers,TabNNModule.training_step,TabNNModule.on_validation_epoch_end
    metric,median=Metrics.apply,MedianCenterFactory._fit

    def observe_ctor(self,*args,**kwargs):
        if len(record['optimizers'])>=2:
            raise ValueError('Native Adam constructor budget exceeded')
        role=('selection','refit')[len(record['optimizers'])]
        write(directory/(role+'-optimizer-start.json'),dict(role=role))
        result=ctor(self,*args,**kwargs)
        opt_ids[id(self)]=len(record['optimizers'])
        record['optimizers'].append(dict(role=role,steps=0))
        return result

    def observe_step(self,*args,**kwargs):
        if id(self) not in opt_ids:
            raise ValueError('Unregistered native Adam step')
        value=step(self,*args,**kwargs)
        record['optimizers'][opt_ids[id(self)]]['steps']+=1
        return value

    def observe_configure(self):
        result=configure(self)
        index=opt_ids[id(result.opt)]
        if index!=len(record['modules']):
            raise ValueError('Unexpected native module/optimizer mapping')
        module_ids[id(self)]=index
        record['modules'].append(dict(role=record['optimizers'][index]['role'],
            batch_size=self.train_dl.batch_size,n_batches=len(self.train_dl),
            drop_last=self.train_dl.drop_last,training_rows=self.train_dl.n_samples,
            validation=[],epoch_batches={}))
        return result

    def observe_training(self,batch,batch_idx):
        result=train(self,batch,batch_idx)
        row=record['modules'][module_ids[id(self)]]
        key=str(int(self.current_epoch)+1)
        row['epoch_batches'][key]=row['epoch_batches'].get(key,0)+1
        if not bool(torch.isfinite(result)):
            raise ValueError('Nonfinite native training loss')
        return result

    def observe_metric(y_pred,y,metric_name):
        result=metric(y_pred,y,metric_name)
        if current_metric and metric_name=='mae':
            current_metric[-1].append(float(result.detach().cpu()))
        return result

    def observe_validation(self):
        epoch=int(self.progress.epoch)+1
        values=[];current_metric.append(values)
        try:
            result=validation(self)
        finally:
            current_metric.pop()
        if len(values)!=1:
            raise ValueError('One native MAE per single-model validation epoch required')
        record['modules'][module_ids[id(self)]]['validation'].append(dict(epoch=epoch,mae=values[0],
            native_best_epoch=int(self.best_mean_val_epochs['mae'][0])))
        return result

    def observe_median(self,ds):
        a=ds.tensors['x_cont'].detach().cpu().numpy()
        if a.shape[1]:
            record['preprocessing'].append(dict(rows=len(a),columns=a.shape[1],row_multiset_sha256=row_multiset_digest(a)))
        return median(self,ds)

    with ExitStack() as stack:
        for cls,name,fn in [(torch.optim.Adam,'__init__',observe_ctor),(torch.optim.Adam,'step',observe_step),
            (TabNNModule,'configure_optimizers',observe_configure),(TabNNModule,'training_step',observe_training),
            (TabNNModule,'on_validation_epoch_end',observe_validation),(Metrics,'apply',staticmethod(observe_metric)),
            (MedianCenterFactory,'_fit',observe_median)]:
            stack.enter_context(patch.object(cls,name,fn))
        yield record


def encoded_without_fit(frame):
    a=frame[list(FEATURES)].to_numpy(float)
    medians=np.nanmedian(np.where(np.isfinite(a),a,np.nan),axis=0)
    categories=np.unique(frame.spout_no.to_numpy(float))
    value=np.column_stack([np.where(np.isfinite(a),a,medians),
        frame.spout_no.to_numpy(float)[:,None]==categories[None,:]]).astype(np.float32)
    return value,medians,categories


def verify_trace(record,frame,epoch,horizon):
    mask=group_safe_inner_folds(frame,seed=42)['fold']!=0
    fitting=[frame.loc[mask],frame]
    if len(record['optimizers'])!=2 or len(record['modules'])!=2 or len(record['preprocessing'])!=2:
        raise ValueError('Exactly two native phases and nonempty preprocessing inputs required')
    for i,role in enumerate(('selection','refit')):
        row,opt=record['modules'][i],record['optimizers'][i]
        expected_epochs=horizon if i==0 else epoch
        batches=len(fitting[i])//256
        if (row['role']!=role or opt['role']!=role or row['batch_size']!=256 or not row['drop_last']
                or row['training_rows']!=len(fitting[i]) or row['n_batches']!=batches
                or row['epoch_batches']!={str(j):batches for j in range(1,expected_epochs+1)}
                or opt['steps']!=batches*expected_epochs):
            raise ValueError('Native training steps or partition size differs')
        x,_,_=encoded_without_fit(fitting[i])
        if record['preprocessing'][i]!=dict(rows=len(x),columns=x.shape[1],row_multiset_sha256=row_multiset_digest(x)):
            raise ValueError('Native preprocessing saw the wrong training rows')
    history=record['modules'][0]['validation']
    selected,best=last_best_epoch(history)
    if len(history)!=horizon or selected!=epoch or record['modules'][1]['validation']:
        raise ValueError('Native full horizon / last-best selection / no-validation refit differs')
    for j,row in enumerate(history):
        if row['native_best_epoch']!=last_best_epoch(history[:j+1])[0]:
            raise ValueError('Native selected epoch trajectory differs')
    return best


@contextmanager
def forbid_native_fit():
    import torch
    from pytabkit import RealMLP_TD_Regressor
    from pytabkit.models.training.lightning_modules import TabNNModule
    from pytabkit.models.nn_models.pipeline import MedianCenterFactory
    def forbidden(*args,**kwargs):
        raise ValueError('Cold native process attempted fitting or optimizer construction')
    with ExitStack() as stack:
        for cls,name in [(RealMLP_TD_Regressor,'fit'),(adapter.original.RealMLPRegressor,'fit'),
            (adapter.original.InputEncoder,'fit'),(TabNNModule,'configure_optimizers'),
            (MedianCenterFactory,'_fit'),(torch.optim.Adam,'__init__')]:
            stack.enter_context(patch.object(cls,name,forbidden))
        yield


def cold_audit(directory,frame,query,y,recipe,identity,record,expected,atol=.0005):
    directory=Path(directory);maximum=0.
    mask=group_safe_inner_folds(frame,seed=42)['fold']!=0
    epoch=record['selected_epoch']
    best=verify_trace(record['native_trace'],frame,epoch,recipe['constructor']['n_epochs'])
    with forbid_native_fit():
        for i,role in enumerate(('selection','refit')):
            payload=adapter.load_snapshot(directory/(role+'.pkl'),expected_identity=identity,
                expected_role=role,expected_recipe=recipe)
            h=payload['header'];fitting=frame.loc[mask] if i==0 else frame
            targets=np.asarray(y)[mask] if i==0 else np.asarray(y)
            _,medians,categories=encoded_without_fit(fitting)
            if (h['fitting_rows']!=len(fitting) or h['fitting_ids_digest']!=adapter.digest(fitting.sample_id.tolist())
                    or h['target_digest']!=adapter.digest(targets.tolist())):
                raise ValueError('Saved native fit/target identity differs')
            np.testing.assert_array_equal(h['encoder_medians'],medians)
            np.testing.assert_array_equal(h['encoder_categories'],categories)
            if native_state_digest(payload['estimator'])!=record['parameter_states'][role]:
                raise ValueError('Native saved parameters differ')
            got=adapter.predict_snapshot(payload,query)
            np.testing.assert_array_equal(got,expected[role])
            versions=[adapter.predict_snapshot(payload,query.iloc[::-1])[::-1],
                np.concatenate([adapter.predict_snapshot(payload,query.iloc[j:j+37]) for j in range(0,len(query),37)]),
                adapter.predict_snapshot(payload,query.iloc[:1])]
            for v in versions:
                maximum=max(maximum,float(np.max(abs(v-got[:len(v)]))))
            if role=='selection':
                calibration=frame.loc[~mask].drop(columns=[t for t in TARGETS if t in frame])
                predicted=adapter.predict_snapshot(payload,calibration)
                mae=float(np.mean(np.abs(predicted-np.asarray(y,dtype=np.float32)[~mask])))
                if abs(mae-best)>5e-5:
                    raise ValueError('Saved selector fails native chosen validation metric')
        if maximum>atol:
            raise ValueError('Saved native order/chunk prediction gate failed')
    return dict(status='passed',states=2,selected_epoch=epoch,maximum_order_chunk_difference=maximum,
        selector_validation_mae=best,selector_recomputed_mae=mae,new_fits=0,new_optimizers=0)
