"""Train-only DANet regression, exclusive saved states and fresh cold loading."""
from contextlib import nullcontext
from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from .data import FEATURES
from .dnnr_model import Encoder,validate_frame
from .dnnr_ledger import canonical,file_hash
from .danet_network import Network
from .danet_optimizer import QHAdam

ARMS=('DANET_FIXED','DANET_LEARNED')


@dataclass(frozen=True)
class Settings:
    layers:int=20
    width:int=64
    groups:int=5
    ghost_size:int=256
    dropout:float=.1
    max_epochs:int=240
    batch_size:int=256
    learning_rate:float=.008
    weight_decay:float=1e-5
    beta1:float=.9
    beta2:float=.999
    nu1:float=.8
    nu2:float=1.
    epsilon:float=1e-8
    learning_rate_gamma:float=.95
    learning_rate_step_epochs:int=20
    clip_norm:float=2.
    random_seed:int=42
    loss:str='mae'
    dtype:str='float32'

    def __post_init__(self):
        if (any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in
                ('layers','width','groups','ghost_size','max_epochs','batch_size','learning_rate_step_epochs'))
                or self.layers%2 or self.width%2 or self.ghost_size<2 or self.batch_size<2
                or type(self.random_seed) is not int or not 0<=self.random_seed<2**32
                or self.loss!='mae' or self.dtype!='float32'
                or not np.isfinite([self.dropout,self.learning_rate,self.weight_decay,self.beta1,self.beta2,
                    self.nu1,self.nu2,self.epsilon,self.learning_rate_gamma,self.clip_norm]).all()
                or not 0<=self.dropout<1 or self.learning_rate<=0 or self.weight_decay<0
                or any(not 0<=v<1 for v in (self.beta1,self.beta2)) or any(not 0<=v<=1 for v in (self.nu1,self.nu2))
                or self.epsilon<=0 or not 0<self.learning_rate_gamma<=1 or self.clip_norm<=0):
            raise ValueError('Invalid DANet architecture/training settings')


def state_identity(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(canonical(dict(shape=list(value.shape),dtype=str(value.dtype)))+value.tobytes()).hexdigest()


def source_identity():
    directory=Path(__file__).parent
    return {name:file_hash(directory/name) for name in
        ('danet_terms.py','danet_optimizer.py','danet_network.py','danet_model.py','dnnr_model.py','data.py')}


def make_network(input_dim,arm,settings):
    return Network(input_dim,layers=settings.layers,width=settings.width,groups=settings.groups,
        ghost_size=settings.ghost_size,dropout=settings.dropout,learn_masks=arm=='DANET_LEARNED')


def reserve(observer,kind,metadata):
    context=observer(kind,metadata)
    return nullcontext() if context is None else context


def batch_sizes(rows,settings):
    sizes=[min(settings.batch_size,rows-i) for i in range(0,rows,settings.batch_size)]
    for size in sizes:
        pieces=torch.empty(size).chunk(math.ceil(size/settings.ghost_size))
        if any(len(piece)<2 for piece in pieces):raise ValueError('Training recipe has singleton ghost batch')
    return sizes


class Regressor:
    def __init__(self,arm,settings=None):
        if arm not in ARMS:raise ValueError('Unknown DANet arm')
        self.arm,self.settings=arm,settings or Settings()

    def fit(self,training,y,*,epochs=None,calibration=None,observer=None):
        if getattr(self,'attempted_',False):raise ValueError('DANet estimator is single-use including failed fit')
        self.attempted_=True
        validate_frame(training);y=np.asarray(y,dtype=np.float64)
        epochs=self.settings.max_epochs if epochs is None else epochs
        if (type(epochs) is not int or not 1<=epochs<=self.settings.max_epochs
                or y.shape!=(len(training),) or not np.isfinite(y).all()):raise ValueError('Aligned finite target and declared epochs required')
        batch_sizes(len(training),self.settings)
        self.ids_=training.sample_id.astype(str).tolist()
        if calibration is not None:
            query,target=calibration;validate_frame(query);target=np.asarray(target,dtype=np.float64)
            left=pd.util.hash_pandas_object(training[list(FEATURES)],index=False)
            right=pd.util.hash_pandas_object(query[list(FEATURES)],index=False)
            if (set(query.sample_id.astype(str))&set(self.ids_) or set(left)&set(right)
                    or target.shape!=(len(query),) or not np.isfinite(target).all()):
                raise ValueError('Disjoint group-safe calibration and finite targets required')
        observer=observer or (lambda *_:None)
        with reserve(observer,'estimator',dict(arm=self.arm,rows=len(training),epochs=epochs)):
            started=time.perf_counter();self.encoder_=Encoder().fit(training)
            self.x_=self.encoder_.transform(training).astype(np.float32);self.y_=y.copy()
            self.center_=float(y.mean());self.scale_=float(y.std(ddof=0)) or 1.
            normalized=((y-self.center_)/self.scale_).astype(np.float32)
            self.trace_={};self.calibration_ids_=query.sample_id.astype(str).tolist() if calibration is not None else []
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.settings.random_seed)
                self.model_=make_network(self.x_.shape[1],self.arm,self.settings).cpu().float()
                self.initial_state_digest_=state_identity(np.concatenate([v.detach().cpu().numpy().ravel().astype(np.float64)
                    for v in self.model_.state_dict().values()]))
                with reserve(observer,'optimizer',dict(arm=self.arm,rows=len(training),epochs=epochs)):
                    self._train(normalized,epochs,calibration)
            self.elapsed_seconds_=time.perf_counter()-started;self.fitted_=True
        return self

    def _train(self,normalized,epochs,calibration):
        s=self.settings;model=self.model_;x=torch.from_numpy(self.x_);y=torch.from_numpy(normalized)
        optimizer=QHAdam(model.parameters(),lr=s.learning_rate,betas=(s.beta1,s.beta2),nus=(s.nu1,s.nu2),weight_decay=s.weight_decay,eps=s.epsilon)
        generator=torch.Generator().manual_seed(s.random_seed)
        initial_masks={name:p.detach().clone() for name,p in model.named_parameters() if name.endswith('.logits')}
        trace=dict(training_loss=[],learning_rate=[],gradient_norm=[],mask_change=[],permutations=[],calibration_predictions=[])
        best=math.inf;best_epoch=None;best_state=None
        for epoch in range(1,epochs+1):
            rate=s.learning_rate*s.learning_rate_gamma**((epoch-1)//s.learning_rate_step_epochs)
            for group in optimizer.param_groups:group['lr']=rate
            permutation=torch.randperm(len(x),generator=generator);model.train()
            total=0.;norms=[]
            for indices in permutation.split(s.batch_size):
                optimizer.zero_grad(set_to_none=True)
                prediction=model(x[indices]);loss=(prediction-y[indices]).abs().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite DANet training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),s.clip_norm,error_if_nonfinite=True)
                optimizer.step();norms.append(float(norm));total+=float(loss.detach())*len(indices)
            model.eval();trace['training_loss'].append(total/len(x));trace['learning_rate'].append(rate)
            trace['gradient_norm'].append(float(np.mean(norms)))
            trace['permutations'].append(permutation.numpy().astype(np.int32))
            delta=sum(float((p.detach()-initial_masks[name]).abs().sum()) for name,p in model.named_parameters() if name in initial_masks)
            trace['mask_change'].append(delta)
            if calibration is not None:
                prediction=self._predict(calibration[0]);trace['calibration_predictions'].append(prediction)
                mae=float(np.abs(prediction-np.asarray(calibration[1],float)).mean())
                if mae<best:
                    best,best_epoch=mae,epoch
                    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        self.actual_epochs_=epochs
        self.selected_epoch_=best_epoch if calibration is not None else epochs
        if best_state is not None:model.load_state_dict(best_state)
        self.model_.eval()
        self.trace_={k:np.asarray(v) for k,v in trace.items() if k!='calibration_predictions' or calibration is not None}

    def _predict(self,frame):
        encoded=self.encoder_.transform(frame).astype(np.float32)
        with torch.no_grad():value=self.model_(torch.from_numpy(encoded)).cpu().numpy().astype(np.float64)
        result=self.center_+self.scale_*value
        if not np.isfinite(result).all():raise ValueError('Nonfinite DANet inference; no silent fallback')
        return result

    def predict(self,frame):
        if not getattr(self,'fitted_',False):raise ValueError('DANet is not fitted')
        if set(frame.sample_id.astype(str))&set(self.ids_):raise ValueError('Inference rows overlap training identities')
        return self._predict(frame)

    def arrays(self):
        return {**{'state::'+k:v.detach().cpu().numpy().copy() for k,v in self.model_.state_dict().items()},
            'x':self.x_.copy(),'y':self.y_.copy(),**{'trace::'+k:v.copy() for k,v in self.trace_.items()}}

    def metadata(self,arrays=None):
        if not getattr(self,'fitted_',False):raise ValueError('Only successful DANet fits can be saved')
        arrays=self.arrays() if arrays is None else arrays
        return dict(format='danet-matched-regression-v1',arm=self.arm,settings=asdict(self.settings),source_hashes=source_identity(),
            encoder=self.encoder_.metadata(),target_center=self.center_,target_scale=self.scale_,fit_ids=self.ids_,
            calibration_ids=self.calibration_ids_,actual_epochs=self.actual_epochs_,selected_epoch=self.selected_epoch_,
            selection='first_minimum_raw_MAE' if self.calibration_ids_ else 'fixed_epochs_no_calibration',
            initial_state_digest=self.initial_state_digest_,elapsed_seconds=self.elapsed_seconds_,
            array_digests={k:state_identity(v) for k,v in arrays.items()},postprocessing='none')

    def save(self,path):
        arrays=self.arrays();metadata=self.metadata(arrays)
        with Path(path).open('xb') as stream:np.savez(stream,metadata=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False)),**arrays)
        return file_hash(path)

    @classmethod
    def load(cls,path,sha256):
        if not sha256 or file_hash(path)!=sha256:raise ValueError('Externally anchored DANet model required')
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']));arrays={k:archive[k].copy() for k in archive.files if k!='metadata'}
        if (metadata['format']!='danet-matched-regression-v1' or metadata['source_hashes']!=source_identity()
                or metadata['postprocessing']!='none' or metadata['array_digests']!={k:state_identity(v) for k,v in arrays.items()}):
            raise ValueError('DANet format/source/array identity differs')
        model=cls(metadata['arm'],Settings(**metadata['settings']));model.encoder_=Encoder.from_metadata(metadata['encoder'])
        model.ids_=metadata['fit_ids'];model.center_=metadata['target_center'];model.scale_=metadata['target_scale']
        if (model.ids_!=model.encoder_.fit_ids_ or len(set(model.ids_))!=len(model.ids_)
                or not np.isfinite([model.center_,model.scale_]).all() or model.scale_<=0
                or type(metadata['actual_epochs']) is not int or type(metadata['selected_epoch']) is not int
                or not 1<=metadata['selected_epoch']<=metadata['actual_epochs']<=model.settings.max_epochs):
            raise ValueError('DANet row/target/epoch identity differs')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model.settings.random_seed)
            model.model_=make_network(len(FEATURES)+len(model.encoder_.categories_)+1,model.arm,model.settings).cpu().float()
        template=model.model_.state_dict();states={k[7:]:v for k,v in arrays.items() if k.startswith('state::')}
        traces={k[7:]:v for k,v in arrays.items() if k.startswith('trace::')}
        expected_traces={'training_loss','learning_rate','gradient_norm','mask_change','permutations'}
        if metadata['calibration_ids']:expected_traces.add('calibration_predictions')
        if (set(states)!=set(template) or set(traces)!=expected_traces
                or set(arrays)!={'x','y'}|{'state::'+k for k in states}|{'trace::'+k for k in traces}
                or arrays['x'].shape!=(len(model.ids_),len(FEATURES)+len(model.encoder_.categories_)+1)
                or arrays['x'].dtype!=np.float32 or arrays['y'].shape!=(len(model.ids_),) or arrays['y'].dtype!=np.float64
                or any(not np.isfinite(v).all() for v in arrays.values())):
            raise ValueError('Invalid DANet saved array schema')
        for name,value in states.items():
            expected=template[name].numpy()
            if value.shape!=expected.shape or value.dtype!=expected.dtype:raise ValueError('Invalid DANet state shape/dtype')
        for name,value in traces.items():
            shape=(metadata['actual_epochs'],len(model.ids_)) if name=='permutations' else (
                (metadata['actual_epochs'],len(metadata['calibration_ids'])) if name=='calibration_predictions' else (metadata['actual_epochs'],))
            dtype=np.int32 if name=='permutations' else np.float64
            if value.shape!=shape or value.dtype!=dtype:raise ValueError('Invalid DANet trace shape/dtype')
        model.model_.load_state_dict({k:torch.from_numpy(v) for k,v in states.items()},strict=True);model.model_.eval()
        model.x_,model.y_,model.trace_=arrays['x'],arrays['y'],traces
        model.actual_epochs_=metadata['actual_epochs'];model.selected_epoch_=metadata['selected_epoch']
        model.calibration_ids_=metadata['calibration_ids'];model.initial_state_digest_=metadata['initial_state_digest']
        model.elapsed_seconds_=metadata['elapsed_seconds'];model.saved_metadata_=metadata
        model.fitted_=model.attempted_=True
        return model
